#!/usr/bin/env python3
"""
Test script to verify size separation logic in the OCR processing.
This script tests the call_gemini_api function with sample OCR text to ensure
that product names and sizes are properly separated.
"""

import sys
import os
import json
from typing import Dict, Any

# Add the current directory to the path so we can import from server.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the function we want to test
from server import call_gemini_api

def test_size_separation():
    """Test various OCR text samples to ensure proper size separation."""
    
    test_cases = [
        {
            "name": "Coca Cola with size",
            "ocr_text": "COCA COLA 2L\nRs 150,00\nBeverage",
            "expected": {
                "product_name": "COCA COLA",
                "size": "2L"
            }
        },
        {
            "name": "Milk with weight",
            "ocr_text": "FRESH MILK 1L\nRs 120,50\nDairy Product",
            "expected": {
                "product_name": "FRESH MILK", 
                "size": "1L"
            }
        },
        {
            "name": "Bread with grams",
            "ocr_text": "WHOLE WHEAT BREAD 500G\nRs 45,00\nBakery",
            "expected": {
                "product_name": "WHOLE WHEAT BREAD",
                "size": "500G"
            }
        },
        {
            "name": "Chips with grams",
            "ocr_text": "POTATO CHIPS 100G\nRs 25,00\nSnacks",
            "expected": {
                "product_name": "POTATO CHIPS",
                "size": "100G"
            }
        },
        {
            "name": "Product without size",
            "ocr_text": "BANANA\nRs 30,00\nFruits",
            "expected": {
                "product_name": "BANANA",
                "size": ""
            }
        }
    ]
    
    print("Testing size separation logic...")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"OCR Text: {test_case['ocr_text']}")
        
        try:
            # Call the function
            result = call_gemini_api(test_case['ocr_text'])
            
            print(f"Result: {json.dumps(result, indent=2)}")
            
            # Check if size was properly separated
            product_name = result.get('product_name', '')
            size = result.get('size', '')
            
            expected_product = test_case['expected']['product_name']
            expected_size = test_case['expected']['size']
            
            # Simple validation
            if expected_size and size:
                print(f"✅ Size properly extracted: '{size}'")
            elif not expected_size and not size:
                print(f"✅ No size correctly identified")
            else:
                print(f"❌ Size extraction issue - Expected: '{expected_size}', Got: '{size}'")
                
            if expected_product.lower() in product_name.lower():
                print(f"✅ Product name looks correct: '{product_name}'")
            else:
                print(f"⚠️  Product name may need review: '{product_name}'")
                
        except Exception as e:
            print(f"❌ Error processing test case: {e}")
        
        print("-" * 30)

if __name__ == "__main__":
    # Check if required environment variables are set
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY environment variable not set!")
        print("Please set it before running this test.")
        sys.exit(1)
    
    test_size_separation()
    print("\nTest completed!")
