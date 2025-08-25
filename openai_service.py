import os
import logging
from typing import Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class OpenAIService:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = None
        
        if not self.api_key:
            logger.warning("OpenAI API key not configured")
        else:
            self.client = AsyncOpenAI(api_key=self.api_key)
            logger.info("OpenAI client initialized successfully")

    async def categorize_product(self, product_name: str, available_categories: list = None) -> str:
        """Categorize a product using OpenAI's GPT-4o-mini model"""
        try:
            if not self.client:
                logger.error("OpenAI client not initialized - missing API key")
                return self.fallback_categorize(product_name, available_categories)
            
            # Default categories if none provided
            if not available_categories:
                available_categories = ['dairy', 'liquid', 'wheat', 'meat', 'grown', 'frozen', 'snacks', 'miscellaneous']
            
            # Create the prompt
            prompt = f"""You are an expert product categorization assistant for a French grocery store. The product names are in FRENCH. You need to analyze the French product name carefully, including any brand names, and categorize it into the most appropriate category.

PRODUCT NAME (FRENCH): "{product_name}"

AVAILABLE CATEGORIES:
- dairy
- liquid
- wheat
- meat
- grown
- frozen
- snacks
- miscellaneous

IMPORTANT INSTRUCTIONS:
1. LE NOM DU PRODUIT EST EN FRANÇAIS - il peut contenir des marques (comme Danone, Nestlé, Carrefour, etc.)
2. CONCENTREZ-VOUS SUR CE QU'EST RÉELLEMENT LE PRODUIT, pas seulement la marque
3. Exemples français:
   - "LAIT NESTLÉ" → dairy (Nestlé est la marque, lait est le produit)
   - "COCA COLA" → liquid (Coca Cola est une marque de boisson)
   - "YAOURT DANONE" → dairy (Danone est la marque, yaourt est le produit)
   - "BISCUITS OREO" → snacks (Oreo est la marque, biscuits sont des snacks)
   - "CORN FLAKES KELLOGG'S" → wheat (Kellogg's est la marque, corn flakes sont des céréales)
   - "FROMAGE PRÉSIDENT" → dairy (Président est la marque, fromage est laitier)
   - "JUS D'ORANGE TROPICANA" → liquid (Tropicana est la marque, jus est liquide)
4. Utilisez votre connaissance des marques françaises et internationales
5. Si le nom contient plusieurs mots, analysez le produit principal
6. Choisissez la catégorie LA PLUS APPROPRIÉE
7. Répondez avec SEULEMENT le nom de la catégorie (exactement comme listé ci-dessus)
8. Essayes le plus possible de trouver la catégorie la plus appropriée, même si le produit n'est pas dans la liste des catégories.

Category:"""

            logger.info(f"Categorizing product: '{product_name}' using OpenAI GPT-4o-mini")
            
            # Make the API call using GPT-4o-mini (cheapest and fastest)
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.1,  # Low temperature for consistent results
                timeout=10.0  # Fast timeout for quick responses
            )
            
            # Extract the response
            category_response = response.choices[0].message.content.strip().lower()
            logger.info(f"OpenAI response: '{category_response}'")
            
            # Validate the response is a valid category
            if category_response in [c.lower() for c in available_categories]:
                logger.info(f"Successfully categorized '{product_name}' as: {category_response}")
                return category_response
            else:
                # Try to extract category from response if it contains extra text
                for category in available_categories:
                    if category.lower() in category_response:
                        logger.info(f"Extracted category '{category}' from response: '{category_response}'")
                        return category.lower()
                
                logger.warning(f"OpenAI returned invalid category: {category_response}")
                return self.fallback_categorize(product_name, available_categories)
                
        except Exception as e:
            logger.error(f"Error categorizing product '{product_name}' with OpenAI: {e}")
            return self.fallback_categorize(product_name, available_categories)

    def fallback_categorize(self, product_name: str, available_categories: list = None) -> str:
        """Enhanced fallback categorization using keyword matching and brand recognition when OpenAI fails"""
        logger.info(f"Using fallback categorization for: '{product_name}'")
        
        if not available_categories:
            available_categories = ['dairy', 'liquid', 'wheat', 'meat', 'grown', 'frozen', 'snacks', 'miscellaneous']
        
        product_lower = product_name.lower()
        
        # Brand-to-category mapping for French brands and international brands in France
        brand_categories = {
            # French Dairy brands
            "danone": "dairy", "yoplait": "dairy", "activia": "dairy", "lactalis": "dairy", 
            "président": "dairy", "president": "dairy", "kiri": "dairy", "babybel": "dairy",
            "nestle": "dairy", "nestlé": "dairy", "la laitière": "dairy", "laitiere": "dairy",
            "candia": "dairy", "lactel": "dairy", "bridel": "dairy", "elle & vire": "dairy",
            "bongrain": "dairy", "roquefort": "dairy", "camembert": "dairy",
            
            # French Beverage brands
            "evian": "liquid", "perrier": "liquid", "vittel": "liquid", "contrex": "liquid",
            "badoit": "liquid", "cristaline": "liquid", "mont roucous": "liquid",
            "coca cola": "liquid", "pepsi": "liquid", "sprite": "liquid", "fanta": "liquid",
            "orangina": "liquid", "schweppes": "liquid", "lipton": "liquid", "tropicana": "liquid",
            "minute maid": "liquid", "oasis": "liquid", "pulco": "liquid",
            
            # French Snack brands
            "lu": "snacks", "belin": "snacks", "michel et augustin": "snacks",
            "oreo": "snacks", "kitkat": "snacks", "mars": "snacks", "snickers": "snacks",
            "toblerone": "snacks", "milka": "snacks", "lindt": "snacks", "ferrero": "snacks",
            "haribo": "snacks", "cadbury": "snacks", "pringles": "snacks", "lays": "snacks",
            "vico": "snacks", "charal": "snacks", "bénédictine": "snacks",
            
            # French Cereal/wheat brands
            "kelloggs": "wheat", "kellogg's": "wheat", "nesquik": "wheat", "chocapic": "wheat",
            "lion": "wheat", "jordans": "wheat", "quaker": "wheat", "cheerios": "wheat",
            "corn flakes": "wheat", "special k": "wheat", "cruesli": "wheat",
            "pain de mie": "wheat", "harry's": "wheat", "harrys": "wheat", "jacquet": "wheat",
            
            # French Frozen brands
            "häagen-dazs": "frozen", "haagen dazs": "frozen", "ben jerry": "frozen", "ben & jerry's": "frozen",
            "magnum": "frozen", "carte d'or": "frozen", "carte dor": "frozen",
            "picard": "frozen", "findus": "frozen", "marie": "frozen",
            
            # French Meat brands
            "fleury michon": "meat", "herta": "meat", "aoste": "meat", "jean caby": "meat",
            "madrange": "meat", "bordeau chesnel": "meat", "père dodu": "meat", "pere dodu": "meat",
            "spam": "meat", "knacki": "meat", "knaki": "meat",
            
            # French Store brands
            "carrefour": "miscellaneous", "auchan": "miscellaneous", "leclerc": "miscellaneous",
            "monoprix": "miscellaneous", "casino": "miscellaneous", "système u": "miscellaneous"
        }
        
        # Check for brand matches first
        brand_score = 0
        brand_category = None
        for brand, category in brand_categories.items():
            if brand in product_lower:
                brand_score = 10  # High score for brand matches
                brand_category = category
                logger.info(f"Brand detected: '{brand}' suggests category: {category}")
                break
        
        # Enhanced keyword categorization
        category_keywords = {
            "dairy": [
                "milk", "cheese", "yogurt", "yoghurt", "butter", "cream", "dairy",
                "mozzarella", "cheddar", "cottage", "ricotta", "parmesan", "gouda",
                "ice cream", "gelato",
                "lait", "fromage", "yaourt", "beurre", "crème", "laitier", "glace"
            ],
            "liquid": [
                "water", "juice", "drink", "beverage", "soda", "soft drink", "tea",
                "coffee", "wine", "beer", "liquid", "smoothie", "cola", "energy drink",
                "lemonade", "cocktail", "syrup",
                "eau", "jus", "boisson", "thé", "café", "vin", "bière", "liquide"
            ],
            "wheat": [
                "bread", "flour", "pasta", "wheat", "cereal", "biscuit", "crackers",
                "noodles", "grain", "oats", "rice", "quinoa", "barley", "corn",
                "cookies", "cake", "muffin", "croissant", "bagel",
                "pain", "farine", "pâtes", "blé", "céréale", "riz", "avoine", "gâteau"
            ],
            "meat": [
                "meat", "chicken", "beef", "pork", "fish", "salmon", "tuna", "egg",
                "turkey", "ham", "sausage", "bacon", "seafood", "lamb", "duck",
                "protein", "steak", "fillet", "wings",
                "viande", "poulet", "bœuf", "porc", "poisson", "œuf", "jambon"
            ],
            "grown": [
                "fruit", "vegetable", "apple", "banana", "orange", "grape", "berry",
                "strawberry", "tomato", "potato", "carrot", "lettuce", "spinach",
                "fresh", "organic", "produce", "onion", "pepper", "cucumber", "avocado",
                "légume", "pomme", "banane", "tomate", "carotte", "frais", "bio"
            ],
            "frozen": [
                "frozen", "ice cream", "popsicle", "sorbet", "ice", "frost",
                "frozen meal", "freezer", "gelato",
                "congelé", "glace", "sorbet", "surgelé"
            ],
            "snacks": [
                "candy", "chocolate", "dessert", "cake", "cookie", "chips", "snack",
                "sweet", "sugar", "fried", "donut", "pastry", "nuts", "popcorn",
                "bonbon", "chocolat", "gâteau", "biscuit", "sucré", "frit", "noix"
            ],
            "miscellaneous": [
                "spice", "sauce", "condiment", "oil", "vinegar", "seasoning",
                "extract", "powder", "salt", "pepper", "herbs", "dressing",
                "épice", "huile", "vinaigre", "assaisonnement", "sel", "poivre"
            ]
        }
        
        # Score each category
        category_scores = {}
        for category in available_categories:
            if category in category_keywords:
                score = 0
                keywords = category_keywords[category]
                
                # Add brand bonus if applicable
                if brand_category == category:
                    score += brand_score
                
                for keyword in keywords:
                    if keyword in product_lower:
                        base_score = 1
                        # Extra points for exact word matches
                        words = product_lower.split()
                        if keyword in words:
                            base_score += 3
                        # Bonus for longer keywords (more specific)
                        if len(keyword) > 4:
                            base_score += 1
                        score += base_score
                
                category_scores[category] = score
        
        # Find the category with highest score
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            if best_category[1] > 0:
                logger.info(f"Fallback categorized '{product_name}' as '{best_category[0]}' (score: {best_category[1]})")
                return best_category[0]
        
        # Default to miscellaneous
        logger.info(f"No fallback match for '{product_name}', returning miscellaneous")
        return "miscellaneous"


# Global instance
openai_service = OpenAIService()

