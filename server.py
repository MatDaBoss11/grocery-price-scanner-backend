from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import Dict, Any
import google.cloud.vision as vision
import google.generativeai as genai
import json
import logging
import re
import tempfile
from decimal import Decimal, ROUND_HALF_UP
from sbase_connect import send_to_supabase, supabase  # Import supabase client directly


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
    store: str  # Add store field


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
            # Add empty store field - it will be filled in by the frontend
            product_data["store"] = ""
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
                "store": ""  # Add empty store field
            }
            print("Successfully done: 6. Sends text back to front end (with partial data)")
            return ProductData(**safe_data)
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process_image: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/submit-product")
async def submit_product(product: ProductData):
    """Submit product data to Supabase database"""
    try:
        logger.info(f"Received product data: {product.dict()}")
        
        # Convert price string (e.g. "Rs 12,50") to decimal
        try:
            # Remove 'Rs ' and replace comma with dot
            price_str = product.price.replace("Rs ", "").replace(",", ".")
            # Convert to Decimal and round to 2 decimal places
            price_decimal = Decimal(price_str).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            
            if price_decimal > Decimal('9999.99'):
                raise ValueError("Price must be less than 10000.00")
                
            logger.info(f"Converted price {product.price} to {price_decimal}")
            
        except ValueError as e:
            logger.error(f"Invalid price format: {product.price}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid price format or value too large. Expected format: 'Rs XX,XX' (max 9999,99), got: {product.price}"
            )
        
        # Send to Supabase
        result = send_to_supabase(
            store=product.store,
            price=float(price_decimal),  # Convert to float for JSON serialization
            name=product.product_name,
            size=product.size
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

If you find more than one price with Rs, R.S, or R.P, always pick the smallest one. If you see a price, always format it as Rs XX,XX (use a comma as decimal separator).

For the product name, use this as a strong hint: \"{filtered['filtered_product_name']}\". The product name should be the longest line of text, in all capital letters, and must not include words like marketing, co, ltd, cdt, &, many numbers, or any manufacturer-like words. Only use this line if it looks like a real product name a customer would say in a store.

Return format (respond with ONLY valid JSON, no other text):
{{
  \"product_name\": \"extracted product name\",
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