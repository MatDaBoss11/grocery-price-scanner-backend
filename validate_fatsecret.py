#!/usr/bin/env python3
"""
Standalone FatSecret API Validation Script

This script tests your FatSecret API credentials and connection independently
of your main application to help identify configuration issues.

Usage:
    python validate_fatsecret.py

Requirements:
    - .env file with FATSECRET_CONSUMER_KEY and FATSECRET_CONSUMER_SECRET
    - httpx library: pip install httpx
    - python-dotenv library: pip install python-dotenv
"""

import os
import sys
import asyncio
import base64
import httpx
from dotenv import load_dotenv
import json

class FatSecretValidator:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        self.consumer_key = os.getenv("FATSECRET_CONSUMER_KEY")
        self.consumer_secret = os.getenv("FATSECRET_CONSUMER_SECRET")
        self.token_url = "https://oauth.fatsecret.com/connect/token"
        self.base_url = "https://platform.fatsecret.com/rest"
        
        self.access_token = None
        
        print("🔍 FatSecret API Validation Tool")
        print("=" * 50)
    
    def validate_credentials(self):
        """Check if credentials are properly configured"""
        print("\n1. 📋 Checking Credentials Configuration...")
        
        if not self.consumer_key:
            print("❌ FATSECRET_CONSUMER_KEY not found in environment")
            return False
        
        if not self.consumer_secret:
            print("❌ FATSECRET_CONSUMER_SECRET not found in environment")
            return False
        
        if len(self.consumer_key) < 20:
            print(f"⚠️  Consumer Key seems too short ({len(self.consumer_key)} chars)")
            
        if len(self.consumer_secret) < 20:
            print(f"⚠️  Consumer Secret seems too short ({len(self.consumer_secret)} chars)")
        
        print(f"✅ Consumer Key: {self.consumer_key[:8]}... ({len(self.consumer_key)} chars)")
        print(f"✅ Consumer Secret: {self.consumer_secret[:8]}... ({len(self.consumer_secret)} chars)")
        
        return True
    
    async def test_network_connectivity(self):
        """Test basic network connectivity to FatSecret"""
        print("\n2. 🌐 Testing Network Connectivity...")
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Test basic connectivity
                response = await client.head("https://oauth.fatsecret.com")
                print(f"✅ Can reach oauth.fatsecret.com (HTTP {response.status_code})")
                
                response = await client.head("https://platform.fatsecret.com")
                print(f"✅ Can reach platform.fatsecret.com (HTTP {response.status_code})")
                
                return True
                
        except httpx.ConnectError:
            print("❌ Cannot connect to FatSecret servers")
            print("💡 Check your internet connection and firewall settings")
            return False
        except httpx.TimeoutError:
            print("❌ Connection timeout to FatSecret servers")
            print("💡 Check your internet connection speed")
            return False
        except Exception as e:
            print(f"❌ Network test failed: {e}")
            return False
    
    async def test_oauth_token(self):
        """Test OAuth 2.0 token request"""
        print("\n3. 🔐 Testing OAuth 2.0 Token Request...")
        
        try:
            # Prepare credentials
            credentials = f"{self.consumer_key}:{self.consumer_secret}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            
            headers = {
                "Authorization": f"Basic {encoded_credentials}",
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            data = {
                "grant_type": "client_credentials",
                "scope": "basic"
            }
            
            print(f"🔄 Requesting token from: {self.token_url}")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.token_url, headers=headers, data=data)
                
                print(f"📡 Response Status: {response.status_code}")
                
                if response.status_code == 200:
                    token_data = response.json()
                    self.access_token = token_data.get("access_token")
                    expires_in = token_data.get("expires_in", 0)
                    
                    print(f"✅ Token obtained successfully!")
                    print(f"📝 Token: {self.access_token[:30]}...")
                    print(f"⏰ Expires in: {expires_in} seconds")
                    return True
                    
                elif response.status_code == 401:
                    print("❌ AUTHENTICATION FAILED (HTTP 401)")
                    print("💡 Your Consumer Key or Secret is incorrect")
                    print("💡 Double-check your credentials in the .env file")
                    
                elif response.status_code == 403:
                    print("❌ FORBIDDEN (HTTP 403)")
                    print("💡 Your IP address is not whitelisted")
                    print("💡 Add your IP to FatSecret dashboard and wait 1-2 hours")
                    print(f"💡 Your current IP might be visible at: https://whatismyipaddress.com")
                    
                elif response.status_code == 429:
                    print("❌ RATE LIMITED (HTTP 429)")
                    print("💡 Too many requests, wait before retrying")
                    
                else:
                    print(f"❌ Unexpected response: HTTP {response.status_code}")
                    print(f"📄 Response body: {response.text}")
                
                return False
                
        except Exception as e:
            print(f"❌ Token request failed: {e}")
            return False
    
    async def test_api_call(self):
        """Test a simple API call"""
        print("\n4. 🍽️  Testing API Call (Food Search)...")
        
        if not self.access_token:
            print("❌ No access token available, skipping API test")
            return False
        
        try:
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            params = {
                "search_expression": "milk",
                "page_number": "0",
                "max_results": "5",
                "region": "FR",
                "format": "json"
            }
            
            url = f"{self.base_url}/foods/search/v1"
            print(f"🔄 Testing API call: {url}")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers, params=params)
                
                print(f"📡 Response Status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check response structure
                    if "foods" in data:
                        foods = data["foods"]
                        if "food" in foods:
                            food_data = foods["food"]
                            count = len(food_data) if isinstance(food_data, list) else 1
                            print(f"✅ API call successful! Found {count} foods for 'milk'")
                            
                            # Show first result as example
                            first_food = food_data[0] if isinstance(food_data, list) else food_data
                            food_name = first_food.get("food_name", "Unknown")
                            print(f"📝 Example result: {food_name}")
                            return True
                        else:
                            print("⚠️  API call successful but no foods in response")
                            print(f"📄 Response: {data}")
                    else:
                        print("⚠️  API call successful but unexpected response format")
                        print(f"📄 Response: {data}")
                        
                elif response.status_code == 401:
                    print("❌ AUTHENTICATION FAILED (HTTP 401)")
                    print("💡 Token may be expired or invalid")
                    
                elif response.status_code == 403:
                    print("❌ FORBIDDEN (HTTP 403)")
                    print("💡 Check API permissions and IP whitelisting")
                    
                else:
                    print(f"❌ API call failed: HTTP {response.status_code}")
                    print(f"📄 Response: {response.text}")
                
                return False
                
        except Exception as e:
            print(f"❌ API call failed: {e}")
            return False
    
    async def run_validation(self):
        """Run all validation tests"""
        print("Starting FatSecret API validation...\n")
        
        # Test 1: Credentials
        if not self.validate_credentials():
            print("\n❌ VALIDATION FAILED: Credentials not configured properly")
            return False
        
        # Test 2: Network
        if not await self.test_network_connectivity():
            print("\n❌ VALIDATION FAILED: Network connectivity issues")
            return False
        
        # Test 3: OAuth
        if not await self.test_oauth_token():
            print("\n❌ VALIDATION FAILED: OAuth authentication failed")
            return False
        
        # Test 4: API Call
        if not await self.test_api_call():
            print("\n⚠️  PARTIAL SUCCESS: Token works but API call failed")
            return False
        
        print("\n" + "=" * 50)
        print("🎉 VALIDATION SUCCESSFUL!")
        print("✅ Your FatSecret API integration should work correctly")
        print("💡 If your app still returns 'Miscellaneous', check application logs")
        
        return True

def main():
    """Main entry point"""
    validator = FatSecretValidator()
    
    try:
        result = asyncio.run(validator.run_validation())
        
        if not result:
            print("\n" + "=" * 50)
            print("❌ VALIDATION FAILED")
            print("\n🔧 Common Solutions:")
            print("1. Check your .env file has correct FatSecret credentials")
            print("2. Ensure your IP is whitelisted in FatSecret dashboard")
            print("3. Wait 1-2 hours after adding IP to whitelist")
            print("4. Verify internet connectivity and firewall settings")
            print("5. Check if credentials are active (not expired/suspended)")
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n❌ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()