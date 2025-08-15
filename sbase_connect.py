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

def get_categories() -> dict:
    """
    Get all categories from the categories table
    
    Returns:
        dict: Response with success status and categories data
    """
    try:
        result = supabase.table('categories').select("*").execute()
        
        if result.data:
            logger.info(f"✅ Successfully retrieved {len(result.data)} categories")
            return {"success": True, "data": result.data}
        else:
            logger.error("❌ Failed to retrieve categories")
            return {"success": False, "error": "No categories found"}
            
    except Exception as e:
        logger.error(f"❌ Error retrieving categories: {str(e)}")
        return {"success": False, "error": str(e)}

def insert_product_categories(product_id: int, category_ids: list) -> dict:
    """
    Insert product-category relationships into junction table
    
    Args:
        product_id (int): Product ID from products table
        category_ids (list): List of category IDs
        
    Returns:
        dict: Response with success status and data/error
    """
    try:
        if not product_id or not category_ids:
            return {"success": False, "error": "Product ID and category IDs are required"}
            
        # Prepare junction table data
        junction_data = []
        for category_id in category_ids:
            junction_data.append({
                "product_id": product_id,
                "category_id": category_id
            })
        
        logger.info(f"Inserting {len(junction_data)} product-category relationships")
        
        # Insert into junction table
        result = supabase.table('product_categories').insert(junction_data).execute()
        
        if result.data:
            logger.info(f"✅ Successfully inserted product-category relationships")
            return {"success": True, "data": result.data}
        else:
            logger.error("❌ Failed to insert product-category relationships")
            return {"success": False, "error": "No data returned"}
            
    except Exception as e:
        logger.error(f"❌ Error inserting product-category relationships: {str(e)}")
        return {"success": False, "error": str(e)}

def send_to_supabase(store: str, price: float, name: str, size: str, categories: list = None) -> dict:
    """
    Send product data to Supabase database with optional categories
    
    Args:
        store (str): Store name (required)
        price (float): Product price (required)
        name (str): Product name (required)
        size (str): Product size (required)
        categories (list): List of category names (optional)
        
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
        logger.info(f"Sending to Supabase: store={store}, price={price}, name={name}, size={size}, categories={categories}")
            
        # Prepare the data
        product_data = {
            "product": name,
            "price": float(price),
            "size": size,
            "store": store,
            "created_at": datetime.utcnow().isoformat()  # Add timestamp
        }
        
        logger.info(f"Prepared data: {product_data}")
        
        # Insert into products table
        result = supabase.table('products').insert(product_data).execute()
        
        if result.data and len(result.data) > 0:
            product_id = result.data[0]['id']
            logger.info(f"✅ Successfully inserted product: {name} with ID: {product_id}")
            
            # Handle categories if provided
            if categories:
                # Get all available categories
                categories_result = get_categories()
                if categories_result["success"]:
                    available_categories = {cat['name']: cat['id'] for cat in categories_result["data"]}
                    
                    # Find category IDs for the provided category names
                    category_ids = []
                    for category_name in categories:
                        if category_name in available_categories:
                            category_ids.append(available_categories[category_name])
                        else:
                            logger.warning(f"Category '{category_name}' not found in database")
                    
                    # Insert into junction table if we have valid category IDs
                    if category_ids:
                        junction_result = insert_product_categories(product_id, category_ids)
                        if not junction_result["success"]:
                            logger.error(f"Failed to insert category relationships: {junction_result['error']}")
            
            return {"success": True, "data": result.data}
        else:
            logger.error(f"❌ Failed to insert: {name}")
            return {"success": False, "error": "No data returned"}
            
    except Exception as e:
        logger.error(f"❌ Error inserting {name}: {str(e)}")
        return {"success": False, "error": str(e)}

def update_categories_in_database():
    """
    Helper function to update/migrate category data in Supabase
    Note: This should be run when the database structure needs to be updated
    """
    try:
        logger.info("Checking for category table updates...")
        
        # Expected categories based on new structure
        expected_categories = [
            'dairy', 'liquid', 'wheat', 'meat', 'grown', 'frozen', 'snacks', 'miscellaneous'
        ]
        
        # Get current categories
        categories_result = get_categories()
        if categories_result["success"]:
            existing_categories = [cat['name'] for cat in categories_result["data"]]
            logger.info(f"Existing categories: {existing_categories}")
            
            # Check if we need to add new categories
            missing_categories = [cat for cat in expected_categories if cat not in existing_categories]
            if missing_categories:
                logger.info(f"Missing categories that should be added: {missing_categories}")
                
                # Note: You'll need to manually add these to your Supabase categories table:
                for cat in missing_categories:
                    logger.info(f"  INSERT INTO categories (name) VALUES ('{cat}');")
            
            # Check for old categories that need migration
            old_categories = ['vegetables', 'fruits']
            found_old = [cat for cat in old_categories if cat in existing_categories]
            if found_old:
                logger.info(f"Found old categories that need migration: {found_old}")
                logger.info("Manual migration needed:")
                logger.info("  1. Update product_categories entries for 'vegetables' and 'fruits' to 'grown'")
                logger.info("  2. Remove old 'vegetables' and 'fruits' categories after migration")
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking categories: {e}")
        return False

# Verify table exists on module load and check categories
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
else:
    # Check category structure
    update_categories_in_database()