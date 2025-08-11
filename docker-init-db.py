#!/usr/bin/env python3
"""
Database initialization script for Docker containers
Builds the equipment database if it doesn't exist
"""

import os
import sys
import subprocess
from pathlib import Path

def init_database():
    """Initialize equipment database for Docker container"""
    
    print("🔧 Docker Database Initialization")
    print("=" * 40)
    
    # Get database path from environment
    db_path = os.getenv('DB_PATH', '/app/data/equipment.db')
    excel_file = '/app/Tescom_new_list.xlsx'
    
    print(f"Target database: {db_path}")
    print(f"Excel source: {excel_file}")
    
    # Check if database already exists
    if os.path.exists(db_path):
        # Get database size and record count
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM equipment;")
            count = cursor.fetchone()[0]
            conn.close()
            
            size = os.path.getsize(db_path)
            print(f"✅ Database exists: {count} records, {size} bytes")
            
            if count > 0:
                print("✅ Database is ready!")
                return True
            else:
                print("⚠️  Database exists but is empty, rebuilding...")
        except Exception as e:
            print(f"⚠️  Database exists but seems corrupted: {e}")
    else:
        print("❌ Database not found, building from Excel...")
    
    # Check if Excel file exists
    if not os.path.exists(excel_file):
        print(f"❌ Excel file not found: {excel_file}")
        print("💡 Solutions:")
        print("   1. Ensure Tescom_new_list.xlsx is in the Docker build context")
        print("   2. Mount an existing database volume to /app/data")
        print("   3. Copy database from host to container")
        return False
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Build database using the build script
    try:
        print("🔨 Building database from Excel...")
        
        cmd = [
            sys.executable, 'build_equipment_db.py',
            '--excel', excel_file,
            '--db', db_path,
            '--rebuild',
            '--report'
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            cwd='/app',
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print("✅ Database built successfully!")
            print("Build output:")
            print(result.stdout)
            
            # Verify the built database
            if os.path.exists(db_path):
                import sqlite3
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM equipment;")
                count = cursor.fetchone()[0]
                conn.close()
                
                print(f"✅ Verification: {count} records in database")
                return True
            else:
                print("❌ Database file not created despite successful build")
                return False
        else:
            print("❌ Database build failed!")
            print("Error output:")
            print(result.stderr)
            print("Standard output:")
            print(result.stdout)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Database build timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Database build error: {e}")
        return False

def test_database_access():
    """Test database access after initialization"""
    print("\n🧪 Testing Database Access")
    print("-" * 30)
    
    try:
        # Run our diagnostic script
        result = subprocess.run([
            sys.executable, 'test_docker_db.py'
        ], cwd='/app', capture_output=True, text=True, timeout=60)
        
        print("Test output:")
        print(result.stdout)
        
        if result.stderr:
            print("Test errors:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting Docker database initialization...")
    
    # Initialize database
    db_success = init_database()
    
    if db_success:
        # Test the database
        test_success = test_database_access()
        
        if test_success:
            print("\n🎉 Database initialization completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Database test failed")
            sys.exit(1)
    else:
        print("\n❌ Database initialization failed")
        sys.exit(1)
