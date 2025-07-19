#!/usr/bin/env python3
"""
Hugging Face Authentication Setup for Gemma Models
==================================================

This script helps you set up authentication to access Gemma models.
You need to:
1. Accept the model terms at: https://huggingface.co/google/gemma-2b
2. Get your access token from: https://huggingface.co/settings/tokens
"""

import os
import subprocess
import sys

def main():
    print("🔐 Hugging Face Authentication Setup")
    print("=" * 50)
    
    # Check if already authenticated
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        print(f"✓ Already authenticated as: {user}")
        return True
    except Exception:
        print("✗ Not authenticated")
    
    print("\n📋 Steps to authenticate:")
    print("1. Go to https://huggingface.co/google/gemma-2b")
    print("2. Click 'Accept terms and conditions'")
    print("3. Go to https://huggingface.co/settings/tokens")
    print("4. Create a new token (read access is sufficient)")
    print("5. Copy the token")
    
    token = input("\n🔑 Enter your Hugging Face token: ").strip()
    
    if not token:
        print("❌ No token provided. Please run this script again with a valid token.")
        return False
    
    try:
        # Login using the token
        subprocess.run([
            sys.executable, "-m", "huggingface_hub", "login", "--token", token
        ], check=True)
        
        print("✅ Authentication successful!")
        
        # Test access to Gemma model
        print("\n🧪 Testing model access...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
        print("✅ Successfully loaded Gemma tokenizer!")
        
        return True
        
    except subprocess.CalledProcessError:
        print("❌ Authentication failed. Please check your token.")
        return False
    except Exception as e:
        print(f"❌ Model access test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 