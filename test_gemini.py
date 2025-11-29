#!/usr/bin/env python3
"""
Simple test to verify Gemini API is working
"""

import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print(f"API Key present: {bool(api_key)}")
print(f"API Key length: {len(api_key) if api_key else 0}")

if not api_key:
    print("ERROR: No API key found")
    exit(1)

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.0-flash-exp')

print("\n" + "="*60)
print("Testing Gemini API with simple prompt...")
print("="*60)

try:
    response = model.generate_content("Say hello in JSON format with a 'message' field")
    
    print(f"\nResponse object: {response}")
    print(f"\nCandidates: {response.candidates}")
    print(f"\nFinish reason: {response.candidates[0].finish_reason}")
    
    if response.text:
        print(f"\nResponse text: {response.text}")
        print("\n✅ API is working!")
    else:
        print("\n❌ No text in response")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
