"""
Initialize the database and create all tables.
Run once: python scripts/init_db.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import engine, Base

def init():
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Done. Database initialized.")

    # Create storage directories
    os.makedirs("./storage/models", exist_ok=True)
    os.makedirs("./storage/optimized", exist_ok=True)
    os.makedirs("./storage/reports", exist_ok=True)
    print("Storage directories created.")

if __name__ == "__main__":
    init()
