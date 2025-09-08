#!/usr/bin/env python3
"""
Database Setup Script for TalkHeal

This script initializes the required databases for the TalkHeal application.
Run this script once to set up the database schema.

Usage:
    python setup_database.py
"""

import sqlite3
import os
from auth.auth_utils import init_db

def setup_journals_db():
    """Initialize the journals database"""
    conn = sqlite3.connect("journals.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS journal_entries (
        id TEXT PRIMARY KEY,
        email TEXT,
        entry TEXT,
        sentiment TEXT,
        date TEXT
    )
    """)
    conn.commit()
    conn.close()
    print("✅ Journals database initialized successfully")

def main():
    """Main setup function"""
    print("🚀 Setting up TalkHeal databases...")
    
    # Initialize users database
    try:
        init_db()
        print("✅ Users database initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing users database: {e}")
    
    # Initialize journals database
    try:
        setup_journals_db()
    except Exception as e:
        print(f"❌ Error initializing journals database: {e}")
    
    print("\n🎉 Database setup complete!")
    print("📝 Note: These .db files are automatically created when the app runs.")
    print("🔒 They are ignored by git to prevent conflicts and protect user data.")

if __name__ == "__main__":
    main()
