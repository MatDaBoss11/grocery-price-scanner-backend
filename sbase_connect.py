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

def send_to_supabase(store: str, price: float, name: str, size: str) -> dict:
    """
    Send product data to Supabase database
    
    Args:
        store (str): Store name (required)
        price (float): Product price (required)
        name (str): Product name (required)
        size (str): Product size (required)
        
    Returns:
        dict: Response with success status and data/error
    """
    try:
        # Validate inputs
        if not all([store, price, name, size]):
            return {"success": False, "error": "All fields are required"}
            
        # Verify table exists
        if not verify_table_exists():
            return {"success": False, "error": "Products table not found in Supabase"}
            
        # Log the data being sent
        logger.info(f"Sending to Supabase: store={store}, price={price}, name={name}, size={size}")
            
        # Prepare the data
        product_data = {
            "product": name,
            "price": float(price),
            "size": size,
            "store": store,
            "created_at": datetime.utcnow().isoformat()  # Add timestamp
        }
        
        logger.info(f"Prepared data: {product_data}")
        
        # Insert into Supabase
        result = supabase.table('products').insert(product_data).execute()
        
        if result.data:
            logger.info(f"✅ Successfully inserted: {name}")
            return {"success": True, "data": result.data}
        else:
            logger.error(f"❌ Failed to insert: {name}")
            return {"success": False, "error": "No data returned"}
            
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
    - created_at (timestamp with timezone, default: now())
    """)