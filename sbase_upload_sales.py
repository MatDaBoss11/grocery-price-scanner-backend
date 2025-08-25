from supabase import create_client, Client
from datetime import datetime

# Initialize Supabase client
url = "https://reuhsokiceymokjwgwjg.supabase.co"  # Replace with your Supabase project URL
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJldWhzb2tpY2V5bW9randnd2pnIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDM3ODEwNywiZXhwIjoyMDY5OTU0MTA3fQ.m0ay6ykLxgxWaLNgjYO3d9h5pgCcfRXUajB9QiIF9QQ"  # Replace with your Supabase anon/service key
supabase: Client = create_client(url, key)

# Product data
products = [
    {
        "product_name": "FARMLAND FULL CREAM MILK POWDER 1KG",
        "old_price": 293.00,
        "new_price": 277.95,
        "discount_percentage": 5.14,
        "category": "Dairy"
    },
    {
        "product_name": "THE FOREST BEE HONEY FLIP GLASS 1KG",
        "old_price": 390.00,
        "new_price": 249.95,
        "discount_percentage": 35.91,
        "category": "Miscellaneous"
    },
    {
        "product_name": "LA VACHE QUI RIT FROMAGE FONDU X 24 PORTIONS 336G",
        "old_price": 160.00,
        "new_price": 136.95,
        "discount_percentage": 14.41,
        "category": "Dairy"
    },
    {
        "product_name": "KRAFT CHEDDAR CHEESE 250G",
        "old_price": 97.95,
        "new_price": 86.50,
        "discount_percentage": 11.69,
        "category": "Dairy"
    },
    {
        "product_name": "OVALTINE POUCH 150G",
        "old_price": 121.00,
        "new_price": 89.95,
        "discount_percentage": 25.66,
        "category": "Liquid"
    },
    {
        "product_name": "NUTRIFI' WHOLEWHEAT BISCUITS CEREAL 450G",
        "old_price": 199.90,
        "new_price": 169.95,
        "discount_percentage": 14.98,
        "category": "Wheat"
    },
    {
        "product_name": "FRISCO CAFE ORIGINAL 250G",
        "old_price": 199.90,
        "new_price": 169.95,
        "discount_percentage": 14.98,
        "category": "Liquid"
    },
    {
        "product_name": "CADBURY DRINKING CHOCOLATE JAR 450G",
        "old_price": 225.00,
        "new_price": 179.95,
        "discount_percentage": 20.02,
        "category": "Liquid"
    },
    {
        "product_name": "BIOLYS VENTRE PLAT FENOUIL HIBISCUS / DETOX PISSENLIT X 24",
        "old_price": 235.95,
        "new_price": 182.95,
        "discount_percentage": 22.46,
        "category": "Liquid"
    },
    {
        "product_name": "CELLIFLORE INFUSION THE MINCEUR",
        "old_price": 287.50,
        "new_price": 209.95,
        "discount_percentage": 26.99,
        "category": "Liquid"
    },
    {
        "product_name": "ARLA FULL CREAM UHT MILK 1LT",
        "old_price": 77.05,
        "new_price": 62.95,
        "discount_percentage": 18.30,
        "category": "Dairy"
    },
    {
        "product_name": "WINNERS INSTANT FULL CREAM MILK POWDER 1KG",
        "old_price": 240.00,
        "new_price": 239.95,
        "discount_percentage": 0.02,
        "category": "Dairy"
    },
    {
        "product_name": "ARLA FULL CREAM MILK POWDER 1KG",
        "old_price": 310.00,
        "new_price": 239.95,
        "discount_percentage": 22.60,
        "category": "Dairy"
    },
    {
        "product_name": "DAIRYLITE LOW FAT MILK POWDER 1KG",
        "old_price": 242.00,
        "new_price": 219.95,
        "discount_percentage": 9.11,
        "category": "Dairy"
    },
    {
        "product_name": "EMCO INSTANT WHITE OATS 500G (3 REFS)",
        "old_price": 95.00,
        "new_price": 79.95,
        "discount_percentage": 15.84,
        "category": "Wheat"
    },
    {
        "product_name": "SAFA QUICK COOKING OAT POUCH 500G",
        "old_price": 83.79,
        "new_price": 64.95,
        "discount_percentage": 22.47,
        "category": "Wheat"
    },
    {
        "product_name": "FRANCE LAIT CEREALE LACTEE 250G",
        "old_price": 115.00,
        "new_price": 92.95,
        "discount_percentage": 19.17,
        "category": "Wheat"
    },
    {
        "product_name": "CAMPO CEREALS ORIGINAL 500G",
        "old_price": 149.00,
        "new_price": 120.95,
        "discount_percentage": 18.83,
        "category": "Wheat"
    },
    {
        "product_name": "OATED WHOLE ROLL / INSTANT OATS JAR 1KG",
        "old_price": 156.00,
        "new_price": 129.95,
        "discount_percentage": 16.70,
        "category": "Wheat"
    }
]

# Prepare data for Supabase (rename fields to match table schema)
current_timestamp = datetime.now().isoformat()

supabase_data = []
for product in products:
    supabase_data.append({
        "product_name": product["product_name"],
        "previous_price": product["old_price"],  # Renamed from old_price
        "new_price": product["new_price"],
        "discount_percentage": product["discount_percentage"],
        "timestamp": current_timestamp,
        "category": product["category"]
    })

# Upload to Supabase
try:
    response = supabase.table("winners_promotions").insert(supabase_data).execute()
    print(f"Successfully uploaded {len(supabase_data)} products to Supabase!")
    print(f"Response: {response}")
except Exception as e:
    print(f"Error uploading to Supabase: {e}")

# Optional: Upload one by one if batch upload fails
# for item in supabase_data:
#     try:
#         response = supabase.table("winners_promotions").insert(item).execute()
#         print(f"Uploaded: {item['product_name']}")
#     except Exception as e:
#         print(f"Error uploading {item['product_name']}: {e}")