#!/usr/bin/env python3
"""Test script for configuration loading."""

# Now we can import the config module
from config import load_config, setup_project_paths

def main():
    """Main test function."""
    try:
        print("Setting up project paths...")
        setup_project_paths()
        
        print("Loading configuration...")
        cfg = load_config()
        
        print("Configuration loaded successfully!")
        print(f"Environment config: {cfg.environment}")
        print(f"Data config: {cfg.data}")
        print(f"Model config: {cfg.model}")
        
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())