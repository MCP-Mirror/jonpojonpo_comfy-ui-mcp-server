# __init__.py

import asyncio
from .server import main as server_main

def main():
    """Entry point for the package."""
    try:
        asyncio.run(server_main())
    except Exception as e:
        print(f"Error running server: {e}")
        raise