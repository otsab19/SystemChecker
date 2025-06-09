#!/usr/bin/env python3
"""
Setup script for AI System Administrator Assistant
"""

import os
import sys
import subprocess
from pathlib import Path


def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def create_directories():
    """Create necessary directories"""
    directories = [
        "data/vector_db",
        "logs"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")


def create_env_file():
    """Create .env file template"""
    env_content = """# AI System Administrator Assistant Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Customize settings
# COLLECTION_INTERVAL_HOURS=1
# CHUNK_SIZE=1000
# CHUNK_OVERLAP=200
"""

    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_content)
        print("Created .env file template")
        print("Please edit .env and add your Gemini API key")
    else:
        print(".env file already exists")


def main():
    print("Setting up AI System Administrator Assistant...")

    try:
        install_requirements()
        create_directories()
        create_env_file()

        print("\n✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Edit .env file and add your Gemini API key")
        print("2. Run: python main.py")

    except Exception as e:
        print(f"❌ Setup failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()