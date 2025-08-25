from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import Dict, Any, List
import google.cloud.vision as vision
import google.generativeai as genai
import json
import logging
import re
import tempfile
import decimal
from decimal import Decimal, ROUND_HALF_UP
from sbase_connect import send_to_supabase, supabase  # Import supabase client and functions
from openai_service import openai_service


app = FastAPI()

# Add CORS middleware
app.add_middleware(
   CORSMiddleware,
   allow_origins=[
       "http://0.0.0.0:7000", 
       "http://localhost:7000", 
       "http://127.0.0.1:7000"
   ],
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)

# Add Google Cloud credentials from environment variable
creds_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
if creds_json:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(json.loads(creds_json), f)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f.name
    print("Google Cloud credentials loaded from environment variable")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/heic", "image/webp", 
    "image/gif", "image/bmp", "image/tiff", "image/svg+xml",
    # Sometimes browsers send these
    "application/octet-stream", None
}

# ========================================
# API KEYS AND AUTHENTICATION SETUP
# ========================================

# Option 1: Hardcode your API keys here (for development/testing)

# Option 2: Use environment variables (recommended for production)
# Check for required environment variables
from dotenv import load_dotenv
load_dotenv()  # Add this at the top after imports

# Use only environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found! Please either:")
    logger.error("1. Set HARDCODED_GEMINI_API_KEY in server.py, OR")
    logger.error("2. Set GEMINI_API_KEY environment variable")
    raise ValueError("GEMINI_API_KEY is required")

genai.configure(api_key=GEMINI_API_KEY)
logger.info("Gemini API configured successfully")

# Set up Google Cloud authentication
if GOOGLE_CREDENTIALS_PATH:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS_PATH
    logger.info(f"Google Cloud credentials set to: {GOOGLE_CREDENTIALS_PATH}")
else:
    logger.warning("Google Cloud credentials not found! Please either:")
    logger.warning("1. Set HARDCODED_GOOGLE_CREDENTIALS_PATH in server.py, OR")
    logger.warning("2. Set GOOGLE_APPLICATION_CREDENTIALS environment variable")

# Check Google Cloud authentication
try:
    # Try to create a Vision client to test authentication
    test_client = vision.ImageAnnotatorClient()
    logger.info("Google Cloud Vision client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Google Cloud Vision client: {e}")
    logger.error("Make sure your service account JSON file path is correct and the file exists")
    logger.error("Also ensure the Vision API is enabled in your Google Cloud project")

app = FastAPI(title="Grocery Price Tag Processor")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/test")
async def test_connection():
    """Simple test endpoint to verify connectivity"""
    return {"status": "Backend is working!", "timestamp": "2025"}


class ProductData(BaseModel):
    product_name: str
    price: str
    size: str
    store: str
    categories: list = []  # Add categories field


class ProductCategorizeRequest(BaseModel):
    product_name: str


class ProductCategorizeResponse(BaseModel):
    category: str


@app.post("/process-image")
async def process_image(
    image: UploadFile = File(...),
):
    """Receive an image (from a mobile app, for example) and return structured
    product data extracted from the image.
    """
    try:
        # 1. Ensure the image type is supported
        print("Successfully done: 1. Received image from Flutter front-end")
        logger.info(f"Received image: {image.filename}, Content-Type: {image.content_type}")
        
        if image.content_type not in SUPPORTED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported image type: {image.content_type}. Supported types: {list(SUPPORTED_IMAGE_TYPES)}"
            )

        image_bytes = await image.read()
        logger.info(f"Image bytes read: {len(image_bytes)} bytes")

        # 2. OCR via Google Vision
        print("Successfully done: 2. Sends image to Google Cloud Vision OCR")
        logger.info("Starting OCR processing...")
        ocr_text = extract_text_from_bytes(image_bytes)
        print("Successfully done: 3. Receives text from Google Cloud Vision OCR")
        logger.info(f"OCR extracted text: {ocr_text[:200]}..." if len(ocr_text) > 200 else f"OCR extracted text: {ocr_text}")
        
        if not ocr_text:
            raise HTTPException(status_code=422, detail="No text detected in the image")

        # 3. Parse structured data via Gemini
        print("Successfully done: 4. Sends text to Gemini API")
        logger.info("Starting Gemini processing...")
        product_data = call_gemini_api(ocr_text)
        print("Successfully done: 5. Receives filtered text from Gemini API (Product Name, Size, Price)")
        logger.info(f"Gemini response: {product_data}")

        try:
            # Add empty store and categories fields - they will be filled in by the frontend
            product_data["store"] = ""
            product_data["categories"] = []
            print("Successfully done: 6. Sends text back to front end")
            return ProductData(**product_data)
        except Exception as e:
            logger.error(f"Failed to create ProductData from Gemini response: {e}")
            logger.error(f"Gemini response was: {product_data}")
            # Return partial data with empty strings for missing fields
            safe_data = {
                "product_name": product_data.get("product_name", ""),
                "price": product_data.get("price", ""),
                "size": product_data.get("size", ""),
                "store": "",  # Add empty store field
                "categories": []  # Add empty categories field
            }
            print("Successfully done: 6. Sends text back to front end (with partial data)")
            return ProductData(**safe_data)
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process_image: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/validate-update")
async def validate_update(product: ProductData):
    """Validate product data for UPDATE mode before allowing submission"""
    try:
        logger.info(f"Validating update for product: {product.dict()}")
        
        # Convert price string to decimal for comparison
        try:
            # Clean the price string - remove currency symbols, spaces, and keep only digits and decimal separators
            import re
            price_str = product.price.strip()
            
            # Remove currency symbols and common prefixes
            price_str = re.sub(r'^(Rs?\.?\s*|R\.?S\.?\s*|R\.?P\.?\s*)', '', price_str, flags=re.IGNORECASE)
            
            # Remove any non-digit characters except comma and dot
            price_str = re.sub(r'[^\d,.]', '', price_str)
            
            # Handle comma as decimal separator (common in some formats)
            if ',' in price_str and '.' not in price_str:
                # If there's only one comma and it's not at the end, treat it as decimal separator
                if price_str.count(',') == 1 and not price_str.endswith(','):
                    price_str = price_str.replace(',', '.')
                else:
                    # Multiple commas or comma at end - remove all commas
                    price_str = price_str.replace(',', '')
            
            # Ensure we have a valid decimal string
            if not price_str or price_str == '.' or price_str == ',':
                raise ValueError("No valid price found")
                
            # Convert to Decimal and round to 2 decimal places
            price_decimal = Decimal(price_str).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            
            if price_decimal > Decimal('9999.99'):
                raise ValueError("Price must be less than 10000.00")
                
            logger.info(f"Converted price '{product.price}' to {price_decimal}")
            
        except (ValueError, decimal.ConversionSyntax) as e:
            logger.error(f"Invalid price format: '{product.price}' - Error: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid price format. Expected format: 'Rs XX,XX' or 'XX.XX' (max 9999,99), got: '{product.price}'. Please enter a valid price."
            )

        # Search for similar products in database
        from difflib import SequenceMatcher
        
        # Get all products from database
        existing_products = supabase.table('products').select('*').execute()
        
        if not existing_products.data:
            raise HTTPException(
                status_code=400,
                detail="Product not found in database. Please try adding it instead."
            )
        
        # Find similar products (exact match or very similar titles)
        similar_products = []
        product_name_lower = product.product_name.lower().strip()
        
        for existing_product in existing_products.data:
            existing_name_lower = existing_product['product'].lower().strip()
            
            # Calculate similarity ratio
            similarity = SequenceMatcher(None, product_name_lower, existing_name_lower).ratio()
            
            # Consider it similar if:
            # 1. Exact match (similarity = 1.0)
            # 2. Very similar (similarity >= 0.8) - allows for 1-2 character differences
            if similarity >= 0.8:
                similar_products.append({
                    'product': existing_product,
                    'similarity': similarity
                })
        
        if not similar_products:
            raise HTTPException(
                status_code=400,
                detail="Product not found in database. Please try adding it instead."
            )
        
        # Find the most similar product
        best_match = max(similar_products, key=lambda x: x['similarity'])
        matching_product = best_match['product']
        
        logger.info(f"Found matching product: {matching_product['product']} (similarity: {best_match['similarity']:.2f})")
        
        # Check if name and price are exactly the same
        existing_price = Decimal(str(matching_product['price'])).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        
        if best_match['similarity'] == 1.0 and price_decimal == existing_price:
            raise HTTPException(
                status_code=400,
                detail="This product with the same price already exists in the database."
            )
        
        # If name is the same but price is different, check store and category
        primary_category = product.categories[0] if product.categories else "miscellaneous"
        
        if best_match['similarity'] == 1.0 and price_decimal != existing_price:
            # Check if store and category match
            if (matching_product['store'] == product.store and 
                matching_product['category'] == primary_category):
                # Same name, same store, same category - allow the price update
                return {
                    "status": "valid", 
                    "message": "Product found with different price. Update allowed.",
                    "existing_product": {
                        "id": matching_product['id'],
                        "name": matching_product['product'],
                        "price": float(existing_price),
                        "store": matching_product['store'],
                        "category": matching_product['category']
                    },
                    "new_price": float(price_decimal)
                }
            else:
                # Same name but different store or category
                raise HTTPException(
                    status_code=400,
                    detail=f"Product '{matching_product['product']}' exists but with different store/category. Expected store: '{matching_product['store']}', category: '{matching_product['category']}'."
                )
        
        # If we get here, name is similar and price is different - allow update
        return {
            "status": "valid", 
            "message": "Similar product found with different price. Update allowed.",
            "existing_product": {
                "id": matching_product['id'],
                "name": matching_product['product'],
                "price": float(existing_price),
                "store": matching_product['store'],
                "category": matching_product['category']
            },
            "new_price": float(price_decimal)
        }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating update: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update-product")
async def update_product(product: ProductData):
    """Update existing product in Supabase database - for UPDATE mode only"""
    try:
        logger.info(f"Updating existing product: {product.dict()}")
        
        # First validate that the product exists and can be updated
        validation_response = await validate_update(product)
        
        if validation_response["status"] != "valid":
            raise HTTPException(status_code=400, detail="Product validation failed")
        
        existing_product_id = validation_response["existing_product"]["id"]
        
        # Convert price string to decimal
        try:
            import re
            price_str = product.price.strip()
            
            # Remove currency symbols and common prefixes
            price_str = re.sub(r'^(Rs?\.?\s*|R\.?S\.?\s*|R\.?P\.?\s*)', '', price_str, flags=re.IGNORECASE)
            
            # Remove any non-digit characters except comma and dot
            price_str = re.sub(r'[^\d,.]', '', price_str)
            
            # Handle comma as decimal separator (common in some formats)
            if ',' in price_str and '.' not in price_str:
                # If there's only one comma and it's not at the end, treat it as decimal separator
                if price_str.count(',') == 1 and not price_str.endswith(','):
                    price_str = price_str.replace(',', '.')
                else:
                    # Multiple commas or comma at end - remove all commas
                    price_str = price_str.replace(',', '')
            
            # Ensure we have a valid decimal string
            if not price_str or price_str == '.' or price_str == ',':
                raise ValueError("No valid price found")
                
            # Convert to Decimal and round to 2 decimal places
            price_decimal = Decimal(price_str).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            
            if price_decimal > Decimal('9999.99'):
                raise ValueError("Price must be less than 10000.00")
                
            logger.info(f"Converted price '{product.price}' to {price_decimal}")
            
        except (ValueError, decimal.ConversionSyntax) as e:
            logger.error(f"Invalid price format: '{product.price}' - Error: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid price format. Expected format: 'Rs XX,XX' or 'XX.XX' (max 9999,99), got: '{product.price}'. Please enter a valid price."
            )
        
        # Update the existing product by ID
        primary_category = product.categories[0] if product.categories else "miscellaneous"
        
        # Use the original product name from database for similar matches
        # Only use the new name if it's an exact match
        original_product_name = validation_response["existing_product"]["name"]
        
        # Check if this was an exact match or similar match
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, product.product_name.lower().strip(), original_product_name.lower().strip()).ratio()
        
        # Use original name for similar matches (< 1.0), new name only for exact matches (1.0)
        final_product_name = original_product_name if similarity < 1.0 else product.product_name
        
        logger.info(f"Similarity: {similarity:.3f} - Using name: '{final_product_name}' (original: '{original_product_name}', input: '{product.product_name}')")
        
        updated_data = {
            "product": final_product_name,
            "price": float(price_decimal),
            "size": product.size,
            "store": product.store,
            "category": primary_category,
        }
        
        logger.info(f"Updating product ID {existing_product_id} with data: {updated_data}")
        
        result = supabase.table('products').update(updated_data).eq('id', existing_product_id).execute()
        
        logger.info(f"Update result: {result}")
        
        if result.data:
            logger.info(f"Successfully updated product ID {existing_product_id}")
            return {"status": "success", "data": result.data, "message": "Product updated successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to update product - no data returned")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating product: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/submit-product")
async def submit_product(product: ProductData):
    """Submit product data to Supabase database"""
    try:
        logger.info(f"Received product data: {product.dict()}")
        
        # Convert price string (e.g. "Rs 12,50") to decimal
        try:
            # Clean the price string - remove currency symbols, spaces, and keep only digits and decimal separators
            import re
            price_str = product.price.strip()
            
            # Remove currency symbols and common prefixes
            price_str = re.sub(r'^(Rs?\.?\s*|R\.?S\.?\s*|R\.?P\.?\s*)', '', price_str, flags=re.IGNORECASE)
            
            # Remove any non-digit characters except comma and dot
            price_str = re.sub(r'[^\d,.]', '', price_str)
            
            # Handle comma as decimal separator (common in some formats)
            if ',' in price_str and '.' not in price_str:
                # If there's only one comma and it's not at the end, treat it as decimal separator
                if price_str.count(',') == 1 and not price_str.endswith(','):
                    price_str = price_str.replace(',', '.')
                else:
                    # Multiple commas or comma at end - remove all commas
                    price_str = price_str.replace(',', '')
            
            # Ensure we have a valid decimal string
            if not price_str or price_str == '.' or price_str == ',':
                raise ValueError("No valid price found")
                
            # Convert to Decimal and round to 2 decimal places
            price_decimal = Decimal(price_str).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            
            if price_decimal > Decimal('9999.99'):
                raise ValueError("Price must be less than 10000.00")
                
            logger.info(f"Converted price '{product.price}' to {price_decimal}")
            
        except (ValueError, decimal.ConversionSyntax) as e:
            logger.error(f"Invalid price format: '{product.price}' - Error: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid price format. Expected format: 'Rs XX,XX' or 'XX.XX' (max 9999,99), got: '{product.price}'. Please enter a valid price."
            )
        
        # Send to Supabase with categories (will use first category only)
        primary_category = product.categories[0] if product.categories else "miscellaneous"
        result = send_to_supabase(
            store=product.store,
            price=float(price_decimal),  # Convert to float for JSON serialization
            name=product.product_name,
            size=product.size,
            category=primary_category
        )
        
        logger.info(f"Supabase response: {result}")
        
        if result["success"]:
            return {"status": "success", "data": result["data"]}
        else:
            # If it's a table not found error, return a more helpful message
            if "table not found" in str(result["error"]).lower():
                raise HTTPException(
                    status_code=500,
                    detail="Supabase table 'products' not found. Please create the table first."
                )
            raise HTTPException(status_code=400, detail=result["error"])
            
    except ValueError as e:
        logger.error(f"Price conversion error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting product: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-product")
async def add_product(
    product_name: str = Form(...),
    size: str = Form(...),
    price: str = Form(...),
    store: str = Form(...),
    categories: str = Form(...),
    product_image: UploadFile = File(...)
):
    """Add a new product with image upload to Supabase"""
    try:
        logger.info(f"Received add product request: {product_name}, {price}, {size}, {store}")
        
        # Parse categories from JSON string
        try:
            categories_list = json.loads(categories)
        except json.JSONDecodeError:
            categories_list = []
        
        # Use first category as primary, fallback to miscellaneous
        primary_category = categories_list[0] if categories_list else "miscellaneous"
        logger.info(f"Using primary category: {primary_category} from categories: {categories_list}")
        
        # Convert price string (e.g. "Rs 12,50") to decimal
        try:
            # Clean the price string - remove currency symbols, spaces, and keep only digits and decimal separators
            import re
            price_str = price.strip()
            
            # Remove currency symbols and common prefixes
            price_str = re.sub(r'^(Rs?\.?\s*|R\.?S\.?\s*|R\.?P\.?\s*)', '', price_str, flags=re.IGNORECASE)
            
            # Remove any non-digit characters except comma and dot
            price_str = re.sub(r'[^\d,.]', '', price_str)
            
            # Handle comma as decimal separator (common in some formats)
            if ',' in price_str and '.' not in price_str:
                # If there's only one comma and it's not at the end, treat it as decimal separator
                if price_str.count(',') == 1 and not price_str.endswith(','):
                    price_str = price_str.replace(',', '.')
                else:
                    # Multiple commas or comma at end - remove all commas
                    price_str = price_str.replace(',', '')
            
            # Ensure we have a valid decimal string
            if not price_str or price_str == '.' or price_str == ',':
                raise ValueError("No valid price found")
                
            # Convert to Decimal and round to 2 decimal places
            price_decimal = Decimal(price_str).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            
            if price_decimal > Decimal('9999.99'):
                raise ValueError("Price must be less than 10000.00")
                
            logger.info(f"Converted price '{price}' to {price_decimal}")
            
        except (ValueError, decimal.InvalidOperation) as e:
            logger.error(f"Invalid price format: '{price}' - Error: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid price format. Expected format: 'Rs XX,XX' or 'XX.XX' (max 9999,99), got: '{price}'. Please enter a valid price."
            )
        
        # Step 1: Insert product info into products table (without image)
        product_data = {
            "product": product_name,
            "price": float(price_decimal),
            "size": size,
            "store": store,
            "category": primary_category,
        }
        
        logger.info(f"Inserting product: {product_data}")
        
        # Insert product and get the ID using the same method as existing code
        try:
            # First, try to find if product already exists
            existing_product = supabase.table('products').select('id').eq('product', product_name).execute()
            
            if existing_product.data and len(existing_product.data) > 0:
                # Product exists, update it
                product_id = existing_product.data[0]['id']
                result = supabase.table('products').update(product_data).eq('id', product_id).execute()
                logger.info(f"Updated existing product with ID: {product_id}")
            else:
                # Product doesn't exist, insert new
                result = supabase.table('products').insert(product_data).execute()
                if result.data and len(result.data) > 0:
                    product_id = result.data[0]['id']
                    logger.info(f"Inserted new product with ID: {product_id}")
                else:
                    raise Exception("Failed to insert new product - no ID returned")
            logger.info(f"Operation result: {result}")
                    
        except Exception as e:
            logger.error(f"Error inserting product: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to insert product: {str(e)}")
        
        # Step 2: Upload product image to bucket
        if product_image.content_type not in SUPPORTED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported image type: {product_image.content_type}"
            )
        
        image_bytes = await product_image.read()
        file_name = f"{product_id}.jpg"
        
        logger.info(f"Uploading image: {file_name}")
        
        # Upload to Supabase storage
        storage_result = supabase.storage.from_('product-images').upload(
            file_name, 
            image_bytes,
            file_options={"content-type": "image/jpeg"}
        )
        
        if hasattr(storage_result, 'error') and storage_result.error:
            logger.error(f"Storage upload error: {storage_result.error}")
            raise HTTPException(status_code=500, detail=f"Failed to upload image: {storage_result.error}")
        
        # Step 3: Update product with image path
        image_path = f"product-images/{file_name}"
        update_result = supabase.table('products').update({"images": image_path}).eq('id', product_id).execute()
        
        if not update_result.data:
            logger.warning("Product image path update may have failed")
        
        logger.info(f"Product {product_id} updated with image path: {image_path} and category: {primary_category}")
        
        return {
            "status": "success", 
            "product_id": product_id,
            "image_path": image_path,
            "category": primary_category,
            "categories": categories_list,
            "message": "Product added successfully with image and category"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding product: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/categorize-product", response_model=ProductCategorizeResponse)
async def categorize_product(request: ProductCategorizeRequest):
    """Categorize a product using OpenAI API"""
    try:
        logger.info(f"Categorizing product: {request.product_name}")
        
        # Get available categories from frontend
        available_categories = ['dairy', 'liquid', 'wheat', 'meat', 'grown', 'frozen', 'snacks', 'miscellaneous']
        
        # Use the OpenAI service to categorize the product
        category = await openai_service.categorize_product(request.product_name, available_categories)
        
        logger.info(f"Product '{request.product_name}' categorized as: {category}")
        return ProductCategorizeResponse(category=category)
        
    except Exception as e:
        logger.error(f"Error categorizing product '{request.product_name}': {e}")
        # Always return miscellaneous on error, never fail
        return ProductCategorizeResponse(category="miscellaneous")


@app.get("/debug-openai")
async def debug_openai():
    """Comprehensive OpenAI API debugging endpoint"""
    import os
    
    debug_info = {
        "timestamp": "2025-01-13",
        "credentials_status": {},
        "api_tests": {},
        "recommendations": []
    }
    
    try:
        # Check credentials
        api_key = os.getenv("OPENAI_API_KEY")
        
        debug_info["credentials_status"] = {
            "api_key_present": bool(api_key),
            "api_key_length": len(api_key) if api_key else 0,
            "api_key_preview": api_key[:8] + "..." if api_key and len(api_key) > 8 else "NOT_SET"
        }
        
        if not api_key:
            debug_info["recommendations"].append("❌ Set OPENAI_API_KEY in .env file")
            return debug_info
        
        # Test categorization
        logger.info("🔍 Starting OpenAI debug tests...")
        try:
            test_products = ["milk", "coca cola", "bread"]
            test_results = {}
            
            for product in test_products:
                category = await openai_service.categorize_product(product)
                test_results[product] = {
                    "category": category,
                    "is_fallback": category == "miscellaneous"
                }
            
            debug_info["api_tests"]["categorization"] = {
                "success": True,
                "test_results": test_results,
                "all_fallback": all(result["is_fallback"] for result in test_results.values())
            }
            
            if debug_info["api_tests"]["categorization"]["all_fallback"]:
                debug_info["recommendations"].append("❌ All categorizations returned miscellaneous - API may not be working properly")
            else:
                debug_info["recommendations"].append("✅ Categorization working correctly!")
                
        except Exception as e:
            debug_info["api_tests"]["categorization"] = {
                "success": False,
                "error": str(e)
            }
            debug_info["recommendations"].append(f"❌ Categorization test exception: {str(e)}")
        
        # Add general recommendations
        if not debug_info["recommendations"]:
            debug_info["recommendations"].append("✅ All tests passed! OpenAI API should be working.")
        
        debug_info["recommendations"].extend([
            "💡 Check server logs for detailed error messages",
            "💡 Verify OpenAI API key is valid and has sufficient credits",
            "💡 Check OpenAI API status if requests are failing"
        ])
        
        return debug_info
        
    except Exception as e:
        logger.error(f"Debug endpoint error: {e}")
        debug_info["error"] = str(e)
        debug_info["recommendations"].append(f"❌ Debug endpoint failed: {str(e)}")
        return debug_info


@app.get("/test-openai")
async def test_openai():
    """Simple OpenAI API connection test"""
    try:
        # Test with a simple product
        test_product = "milk"
        category = await openai_service.categorize_product(test_product)
        return {
            "status": "success" if category != "miscellaneous" else "warning",
            "message": f"OpenAI API test completed. Product '{test_product}' categorized as: {category}",
            "test_product": test_product,
            "test_category": category,
            "note": "If category is 'miscellaneous', check /debug-openai for detailed analysis"
        }
    except Exception as e:
        logger.error(f"OpenAI API test error: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/categories")
async def get_all_categories():
    """Get all available categories (hardcoded list)"""
    try:
        # Return the predefined categories
        categories = ['dairy', 'liquid', 'wheat', 'meat', 'grown', 'frozen', 'snacks', 'miscellaneous']
        return {
            "status": "success",
            "categories": categories
        }
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/test-supabase")
async def test_supabase():
    """Test Supabase connection"""
    try:
        # Try to query the products table
        result = supabase.table('products').select("*").limit(1).execute()
        return {
            "status": "success",
            "message": "Successfully connected to Supabase",
            "table_exists": True
        }
    except Exception as e:
        logger.error(f"Supabase connection error: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

###########################
# Helper Functions        #
###########################

def extract_text_from_bytes(image_bytes: bytes) -> str:
    """Uses Google Vision API to perform OCR on image bytes."""
    try:
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=image_bytes)
        response = client.text_detection(image=image)

        if response.error.message:
            logger.error(f"Vision API error: {response.error.message}")
            raise HTTPException(status_code=502, detail=f"Vision API error: {response.error.message}")

        annotations = response.text_annotations
        extracted_text = annotations[0].description if annotations else ""
        logger.info(f"Vision API extracted {len(extracted_text)} characters")
        return extracted_text
        
    except Exception as e:
        logger.error(f"Error in extract_text_from_bytes: {e}")
        raise HTTPException(status_code=502, detail=f"OCR processing failed: {str(e)}")


def preprocess_ocr_text(ocr_text: str) -> dict:
    import re
    # 1. Find all prices with Rs, R.S, or R.P (case-insensitive)
    price_pattern = r'(?:RS|R\.S|R\.P)[\s:]*([0-9]+[\.,][0-9]{2})'
    prices = re.findall(price_pattern, ocr_text, re.IGNORECASE)
    # Convert to float for comparison, replace comma with dot if needed
    price_values = []
    for p in prices:
        try:
            price_values.append(float(p.replace(",", ".")))
        except:
            continue
    min_price = min(price_values) if price_values else None
    # 2. Find candidate product name lines
    lines = [l.strip() for l in ocr_text.split("\n") if l.strip()]
    # Exclude lines with unwanted words/numbers
    blacklist = ["MARKETING", "CO", "LTD", "CDT", "IMPORTS", "FOODS", "WING&CO", "COMPANY", "DISTRIBUTORS", "MANUFACTURERS", "&", "LTD.", "CO."]
    def is_valid_title(line):
        if any(word in line.upper() for word in blacklist):
            return False
        if re.search(r'\d{4,}', line):  # many numbers
            return False
        if re.search(r'\d{2,}', line) and len(line.split()) <= 2:
            return False
        return True
    valid_lines = [l for l in lines if is_valid_title(l)]
    # Pick the longest valid line
    product_name = max(valid_lines, key=len) if valid_lines else ""
    product_name = product_name.upper()
    # 3. Return filtered info
    return {
        "filtered_price": f"Rs {min_price:,.2f}".replace(",", ",") if min_price is not None else "",
        "filtered_product_name": product_name
    }


def call_gemini_api(ocr_text: str) -> Dict[str, Any]:
    """Calls Gemini to convert raw OCR text into structured fields."""
    # Preprocess OCR text for best candidates
    filtered = preprocess_ocr_text(ocr_text)
    prompt = f"""
You are a data extraction assistant. Your task is to extract structured product information from OCR'd grocery price tag text.

Extract product information from this grocery price tag OCR text and return ONLY a JSON object with these exact fields:

OCR Text: \"{ocr_text}\"

IMPORTANT RULES:
1. If you find more than one price with Rs, R.S, or R.P, always pick the smallest one. If you see a price, always format it as Rs XX,XX (use a comma as decimal separator).

2. For the product name, use this as a strong hint: \"{filtered['filtered_product_name']}\". The product name should be the longest line of text, in all capital letters, and must not include words like marketing, co, ltd, cdt, &, many numbers, or any manufacturer-like words. Only use this line if it looks like a real product name a customer would say in a store.

3. CRITICAL: The product name should NEVER include size information. Extract size information separately and put it in the "size" field. Common size indicators to look for and extract:
   - Weight: kg, g, lb, oz, etc.
   - Volume: L, ml, fl oz, etc.
   - Count: pieces, pcs, units, etc.
   - Dimensions: cm, inches, etc.
   - Any numbers followed by units of measurement

4. Examples of proper separation:
   - "COCA COLA 2L" → product_name: "COCA COLA", size: "2L"
   - "MILK 1L" → product_name: "MILK", size: "1L"
   - "BREAD 500G" → product_name: "BREAD", size: "500G"
   - "CHIPS 100G" → product_name: "CHIPS", size: "100G"

Return format (respond with ONLY valid JSON, no other text):
{{
  \"product_name\": \"extracted product name (without size)\",
  \"price\": \"extracted price with currency\",
  \"size\": \"extracted size/quantity\"
}}

If any field can't be identified confidently, return an empty string \"\". Do not include any explanation or text outside of the JSON.
"""

    try:
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        response = model.generate_content(prompt)
        logger.info(f"Raw Gemini response: {response.text}")
        
        # Gemini can sometimes wrap JSON in markdown; strip backticks etc.
        content = response.text.strip()
        
        # Remove markdown code block formatting if present
        if content.startswith("```json"):
            content = content[7:]  # Remove ```json
        elif content.startswith("```"):
            content = content[3:]   # Remove ```
            
        if content.endswith("```"):
            content = content[:-3]  # Remove trailing ```
            
        content = content.strip()
        
        # Try to parse JSON
        try:
            parsed_data = json.loads(content)
            logger.info(f"Successfully parsed JSON: {parsed_data}")
            return parsed_data
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Content that failed to parse: {content}")
            
            # Try to extract JSON from the response using regex
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, content)
            
            for match in matches:
                try:
                    parsed_data = json.loads(match)
                    logger.info(f"Successfully extracted JSON from regex: {parsed_data}")
                    return parsed_data
                except json.JSONDecodeError:
                    continue
            
            # If all else fails, return empty structure
            logger.warning("Could not parse any JSON from Gemini response, returning empty structure")
            return {"product_name": "", "price": "", "size": ""}
            
    except Exception as e:
        logger.error(f"Error in call_gemini_api: {e}")
        raise HTTPException(status_code=502, detail=f"Gemini API error: {str(e)}")