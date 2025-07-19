#!/usr/bin/env python3
"""
Hugging Face Authentication Setup Script
Sets up authentication for accessing gated models like Gemma.
"""

import os
import subprocess
import sys
from pathlib import Path

def setup_hf_auth(token):
    """Set up Hugging Face authentication with the provided token."""
    
    print("🔐 Setting up Hugging Face authentication...")
    
    # Method 1: Set environment variable
    os.environ['HF_TOKEN'] = token
    print("✅ Set HF_TOKEN environment variable")
    
    # Method 2: Use huggingface-cli login
    try:
        result = subprocess.run([
            'huggingface-cli', 'login', '--token', token
        ], capture_output=True, text=True, check=True)
        print("✅ Successfully logged in with huggingface-cli")
        print(f"Output: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  huggingface-cli login failed: {e}")
        print(f"Error output: {e.stderr}")
    except FileNotFoundError:
        print("⚠️  huggingface-cli not found, skipping CLI login")
    
    # Method 3: Create .huggingface/token file
    hf_dir = Path.home() / '.huggingface'
    hf_dir.mkdir(exist_ok=True)
    
    token_file = hf_dir / 'token'
    with open(token_file, 'w') as f:
        f.write(token)
    
    # Set proper permissions
    token_file.chmod(0o600)
    print(f"✅ Created token file at {token_file}")
    
    # Method 4: Set HF_HUB_TOKEN environment variable
    os.environ['HF_HUB_TOKEN'] = token
    print("✅ Set HF_HUB_TOKEN environment variable")
    
    print("\n🎉 Hugging Face authentication setup complete!")
    print("\nYou can now access gated models like Gemma.")
    print("\nTo test the setup, you can run:")
    print("python -c \"from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b', token='your_token_here')\"")

def test_authentication():
    """Test if authentication is working."""
    print("\n🧪 Testing authentication...")
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            'google/gemma-2b',
            token=os.environ.get('HF_TOKEN') or os.environ.get('HF_HUB_TOKEN')
        )
        print("✅ Successfully loaded Gemma tokenizer!")
        return True
    except Exception as e:
        print(f"❌ Authentication test failed: {e}")
        return False

if __name__ == "__main__":
    # Your Hugging Face token
    HF_TOKEN = "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"
    
    setup_hf_auth(HF_TOKEN)
    
    # Test the authentication
    if test_authentication():
        print("\n🎯 Authentication is working correctly!")
        print("You're ready to run rhyme probe experiments with Gemma models.")
    else:
        print("\n⚠️  Authentication test failed. Please check your token and try again.") 