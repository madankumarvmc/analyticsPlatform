#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gemini API Test Module

Contains test code for validating Gemini API connectivity and functionality.
Extracted from the original Warehouse Analysis (2).py file.
"""

import os
import json
import requests
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiAPITester:
    """
    Test class for Gemini API functionality.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini API tester.
        
        Args:
            api_key: Gemini API key (uses environment variable if not provided)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            # Use the key from the original file for testing
            self.api_key = "AIzaSyD3-HabX9Oc2Q_0R-wywpRk8QZ03Z7HHds"
        
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.model = "models/gemini-2.0-flash"
        
        logger.info("Gemini API Tester initialized")
    
    def test_simple_generation(self) -> bool:
        """
        Test basic text generation with a simple prompt.
        
        Returns:
            True if test passes, False otherwise
        """
        logger.info("Testing simple text generation")
        
        url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": "Say one short sentence about warehouse efficiency."}
                    ]
                }
            ]
        }
        
        try:
            print(f"POST -> {url}")
            response = requests.post(url, json=payload, timeout=20)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(json.dumps(data, indent=2)[:2000])
                
                # Extract text
                candidates = data.get("candidates", [])
                if candidates:
                    text = candidates[0].get("content", {}).get("parts", [])[0].get("text")
                    print(f"\nExtracted text: {text.strip()}")
                    logger.info("Simple generation test PASSED")
                    return True
                else:
                    logger.error("No candidates in response")
                    return False
            else:
                logger.error(f"API request failed with status {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Simple generation test failed: {str(e)}")
            return False
    
    def test_models_list(self) -> bool:
        """
        Test listing available models.
        
        Returns:
            True if test passes, False otherwise
        """
        logger.info("Testing models list endpoint")
        
        list_url = f"{self.base_url}/models?key={self.api_key}"
        
        try:
            print(f"GET -> {list_url}")
            response = requests.get(list_url, timeout=20)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(json.dumps(data, indent=2)[:4000])
                logger.info("Models list test PASSED")
                return True
            else:
                logger.error(f"Models list request failed with status {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Models list test failed: {str(e)}")
            return False
    
    def test_warehouse_analysis_prompt(self) -> bool:
        """
        Test with a warehouse analysis specific prompt.
        
        Returns:
            True if test passes, False otherwise
        """
        logger.info("Testing warehouse analysis prompt")
        
        url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
        
        prompt = """
        Section: SKU Analysis
        
        Key facts:
        - Total SKUs analyzed: 150
        - Top 5 SKUs by volume: SKU001:2500, SKU002:1800, SKU003:1200, SKU004:950, SKU005:800
        - ABC distribution: A:15%, B:25%, C:60%
        
        Task:
        Summarize the Pareto characteristics (3 sentences) and one inventory slotting recommendation.
        """
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            print(f"Warehouse analysis prompt status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                candidates = data.get("candidates", [])
                if candidates:
                    text = candidates[0].get("content", {}).get("parts", [])[0].get("text")
                    print(f"\nWarehouse analysis response:\n{text}")
                    logger.info("Warehouse analysis prompt test PASSED")
                    return True
                else:
                    logger.error("No candidates in warehouse analysis response")
                    return False
            else:
                logger.error(f"Warehouse analysis prompt failed with status {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Warehouse analysis prompt test failed: {str(e)}")
            return False
    
    def run_all_tests(self) -> bool:
        """
        Run all available tests.
        
        Returns:
            True if all tests pass, False otherwise
        """
        logger.info("Running all Gemini API tests")
        print("="*50)
        print("🧪 GEMINI API TESTS")
        print("="*50)
        
        if not self.api_key:
            print("❌ No API key provided - cannot run tests")
            return False
        
        tests = [
            ("Simple Text Generation", self.test_simple_generation),
            ("Models List", self.test_models_list),
            ("Warehouse Analysis Prompt", self.test_warehouse_analysis_prompt)
        ]
        
        results = []
        
        for test_name, test_func in tests:
            print(f"\n🔄 Running: {test_name}")
            try:
                result = test_func()
                results.append(result)
                print(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                results.append(False)
                print(f"❌ {test_name}: FAILED with exception: {str(e)}")
        
        # Summary
        passed = sum(results)
        total = len(results)
        
        print(f"\n{'='*50}")
        print(f"📊 TEST SUMMARY: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests PASSED!")
            return True
        else:
            print(f"⚠️  {total - passed} tests FAILED")
            return False


def test_api_key_validation():
    """Test API key validation and environment variable loading."""
    print("\n🔑 Testing API Key Validation")
    
    # Test with no key
    tester_no_key = GeminiAPITester("")
    if not tester_no_key.api_key:
        print("✅ Correctly handles missing API key")
    else:
        print("❌ Failed to handle missing API key")
    
    # Test with environment variable
    original_key = os.environ.get("GEMINI_API_KEY")
    os.environ["GEMINI_API_KEY"] = "test_key_from_env"
    
    tester_env = GeminiAPITester()
    if tester_env.api_key == "test_key_from_env":
        print("✅ Correctly loads API key from environment")
    else:
        print("❌ Failed to load API key from environment")
    
    # Restore original environment
    if original_key:
        os.environ["GEMINI_API_KEY"] = original_key
    else:
        os.environ.pop("GEMINI_API_KEY", None)


def run_basic_connectivity_test():
    """Run a basic connectivity test without the full test suite."""
    print("\n⚡ Quick Connectivity Test")
    
    tester = GeminiAPITester()
    if tester.test_simple_generation():
        print("✅ Gemini API is accessible and working")
        return True
    else:
        print("❌ Gemini API connectivity failed")
        return False


if __name__ == "__main__":
    """
    Main entry point for running Gemini API tests.
    """
    print("🚀 Gemini API Test Suite")
    print("="*50)
    
    try:
        # Run API key validation tests
        test_api_key_validation()
        
        # Run full test suite
        tester = GeminiAPITester()
        success = tester.run_all_tests()
        
        if success:
            print("\n🎉 All Gemini API tests completed successfully!")
        else:
            print("\n❌ Some tests failed. Check API key and network connectivity.")
            
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {str(e)}")
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)