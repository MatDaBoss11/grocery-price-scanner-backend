from supabase import create_client, Client
import os
from datetime import datetime
import logging
from decimal import Decimal, ROUND_HALF_UP

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Supabase client
import os
from dotenv import load_dotenv
load_dotenv()

url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_SERVICE_KEY')

if not url or not key:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables are required")
# For production, use environment variables:
# url = os.getenv('SUPABASE_URL')
# key = os.getenv('SUPABASE_SERVICE_KEY')

supabase: Client = create_client(url, key)

def verify_table_exists():
    """Verify that the products table exists and has the correct schema"""
    try:
        # Try to get table info
        result = supabase.table('products').select("*").limit(1).execute()
        logger.info("Successfully connected to products table")
        return True
    except Exception as e:
        logger.error(f"Error verifying products table: {str(e)}")
        return False



def send_to_supabase(store: str, price: float, name: str, size: str, category: str = None) -> dict:
    """
    Send product data to Supabase database with optional category
    
    Args:
        store (str): Store name (required)
        price (float): Product price (required)
        name (str): Product name (required)
        size (str): Product size (required)
        category (str): Category name (optional)
        
    Returns:
        dict: Response with success status and data/error
    """
    try:
        # Validate inputs
        if not all([store, price, name, size]):
            return {"success": False, "error": "All fields are required"}
            
        # Verify table exists
        if not verify_table_exists():
            return {"success": False, "error": "products table not found in Supabase"}
            
        # Log the data being sent
        logger.info(f"Sending to Supabase: store={store}, price={price}, name={name}, size={size}, category={category}")
            
        # Prepare the data
        product_data = {
            "product": name,
            "price": float(price),
            "size": size,
            "store": store,
            "category": category if category else "miscellaneous",
            "created_at": datetime.utcnow().isoformat()  # Add timestamp
        }
        
        logger.info(f"Prepared data: {product_data}")
        
        # Insert into products table - check if exists first
        existing_product = supabase.table('products').select('id').eq('product', name).execute()
        
        if existing_product.data and len(existing_product.data) > 0:
            # Product exists, update it
            product_id = existing_product.data[0]['id']
            result = supabase.table('products').update(product_data).eq('id', product_id).execute()
            logger.info(f"Updated existing product '{name}' with ID: {product_id}")
        else:
            # Product doesn't exist, insert new
            result = supabase.table('products').insert(product_data).execute()
            if result.data and len(result.data) > 0:
                product_id = result.data[0]['id']
                logger.info(f"Inserted new product '{name}' with ID: {product_id}")
            else:
                logger.error(f"Failed to insert new product '{name}' - no data returned")
                return {"success": False, "error": "Failed to insert product - no data returned"}
        
        # Check if we got a product_id from either update or insert
        if 'product_id' in locals():
            logger.info(f"✅ Successfully processed product: {name} with ID: {product_id} and category: {category}")
            return {"success": True, "data": result.data if result.data else [{"id": product_id}]}
        else:
            logger.error(f"❌ Failed to process: {name}")
            return {"success": False, "error": "Failed to process product"}
            
    except Exception as e:
        logger.error(f"❌ Error inserting {name}: {str(e)}")
        return {"success": False, "error": str(e)}


# Verify table exists on module load
if not verify_table_exists():
    logger.error("""
    ❌ Products table not found in Supabase!
    Please create a table named 'products' with the following columns:
    - product (text, required)
    - price (float, required)
    - size (text, required)
    - store (text, required)
    - category (text, optional)
    - created_at (timestamp with timezone, default: now())
    """)