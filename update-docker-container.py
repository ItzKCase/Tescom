#!/usr/bin/env python3
"""
Container update script for development workflow
Handles rebuilding and restarting Docker containers with code changes
"""

import subprocess
import sys
import time
import os

def run_command(cmd, description=""):
    """Run a command and show output"""
    if description:
        print(f"🔧 {description}")
    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_excel_changes():
    """Check if Excel file has been modified since last build"""
    excel_path = "Tescom_new_list.xlsx"
    if not os.path.exists(excel_path):
        return False
    
    # Check if Excel file is newer than database
    db_path = "equipment.db"
    if os.path.exists(db_path):
        excel_time = os.path.getmtime(excel_path)
        db_time = os.path.getmtime(db_path)
        return excel_time > db_time
    
    return True

def quick_update():
    """Quick update: rebuild and restart container"""
    print("🚀 Quick Container Update")
    print("=" * 40)
    
    # Check if we need to rebuild database
    if check_excel_changes():
        print("📊 Excel file has changed - database will be rebuilt")
        print("💡 This ensures your latest equipment data is used")
    
    steps = [
        (["docker-compose", "down"], "Stopping current container"),
        (["docker-compose", "build", "--no-cache"], "Rebuilding image with changes"),
        (["docker-compose", "up", "-d"], "Starting updated container")
    ]
    
    for cmd, desc in steps:
        if not run_command(cmd, desc):
            print(f"\n❌ Update failed at: {desc}")
            return False
        print()
    
    print("✅ Container updated successfully!")
    print("🌐 App available at: http://localhost:8080")
    
    if check_excel_changes():
        print("📊 Note: Database will be automatically rebuilt from latest Excel data")
    
    return True

def development_update():
    """Development update with cache optimization"""
    print("🛠️  Development Container Update")
    print("=" * 40)
    
    # Check if we need to rebuild database
    if check_excel_changes():
        print("📊 Excel file has changed - database will be rebuilt")
        print("💡 This ensures your latest equipment data is used")
    
    steps = [
        (["docker-compose", "down"], "Stopping current container"),
        (["docker-compose", "build"], "Rebuilding image (with cache)"),
        (["docker-compose", "up", "-d"], "Starting updated container")
    ]
    
    for cmd, desc in steps:
        if not run_command(cmd, desc):
            print(f"\n❌ Update failed at: {desc}")
            return False
        print()
    
    print("✅ Container updated successfully!")
    print("🌐 App available at: http://localhost:8080")
    
    if check_excel_changes():
        print("📊 Note: Database will be automatically rebuilt from latest Excel data")
    
    return True

def force_rebuild():
    """Force complete rebuild from scratch"""
    print("🔥 Force Rebuild (Clean Build)")
    print("=" * 40)
    
    steps = [
        (["docker-compose", "down", "-v"], "Stopping and removing volumes"),
        (["docker", "system", "prune", "-f"], "Cleaning Docker cache"),
        (["docker-compose", "build", "--no-cache", "--pull"], "Force rebuilding from scratch"),
        (["docker-compose", "up", "-d"], "Starting fresh container")
    ]
    
    for cmd, desc in steps:
        if not run_command(cmd, desc):
            print(f"\n❌ Rebuild failed at: {desc}")
            return False
        print()
    
    print("✅ Container completely rebuilt!")
    print("🌐 App available at: http://localhost:8080")
    print("📊 Database will be automatically rebuilt from latest Excel data")
    return True

def rebuild_database_only():
    """Rebuild just the database without restarting the app"""
    print("📊 Database Rebuild Only")
    print("=" * 40)
    
    if not check_excel_changes():
        print("ℹ️  Excel file hasn't changed - database is up to date")
        return True
    
    print("📊 Excel file has changed - rebuilding database...")
    
    steps = [
        (["docker-compose", "--profile", "rebuild-db", "up", "db-rebuilder"], "Rebuilding database from Excel"),
        (["docker-compose", "--profile", "rebuild-db", "down"], "Cleaning up rebuild service")
    ]
    
    for cmd, desc in steps:
        if not run_command(cmd, desc):
            print(f"\n❌ Database rebuild failed at: {desc}")
            return False
        print()
    
    print("✅ Database rebuilt successfully!")
    print("💡 App continues running with updated data")
    return True

def show_logs():
    """Show container logs"""
    print("📄 Container Logs")
    print("=" * 20)
    print("Press Ctrl+C to stop viewing logs")
    print()
    
    try:
        subprocess.run(["docker-compose", "logs", "-f", "tescom-agent"])
    except KeyboardInterrupt:
        print("\n⏹️  Stopped viewing logs")

def check_status():
    """Check container status"""
    print("📊 Container Status")
    print("=" * 20)
    
    # Check if container is running
    try:
        result = subprocess.run(
            ["docker-compose", "ps"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        print(result.stdout)
        
        # Test if app is responding
        print("🌐 Testing app connectivity...")
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:8080", timeout=5)
            print("✅ App is responding at http://localhost:8080")
        except:
            print("❌ App is not responding")
        
        # Check database status
        print("\n📊 Checking database status...")
        try:
            result = subprocess.run(
                ["docker", "exec", "tescom-agent", "ls", "-la", "/app/data/"],
                capture_output=True,
                text=True,
                check=True
            )
            if "equipment.db" in result.stdout:
                print("✅ Database exists in container")
            else:
                print("❌ Database not found in container")
        except:
            print("❌ Could not check database status")
            
    except subprocess.CalledProcessError:
        print("❌ Docker Compose not running")

def main():
    """Main interactive menu"""
    while True:
        print("\n" + "="*50)
        print("🔄 DOCKER CONTAINER UPDATE MENU")
        print("="*50)
        print("1. Quick Update (recommended)")
        print("2. Development Update (faster, uses cache)")
        print("3. Force Rebuild (clean slate)")
        print("4. Rebuild Database Only (Excel changes)")
        print("5. Check Container Status")
        print("6. View Logs")
        print("7. Stop Container")
        print("8. Exit")
        print("-"*50)
        
        choice = input("Select option (1-8): ").strip()
        
        try:
            if choice == "1":
                quick_update()
            elif choice == "2":
                development_update()
            elif choice == "3":
                confirm = input("⚠️  This will do a complete rebuild. Continue? (y/n): ")
                if confirm.lower().startswith('y'):
                    force_rebuild()
            elif choice == "4":
                rebuild_database_only()
            elif choice == "5":
                check_status()
            elif choice == "6":
                show_logs()
            elif choice == "7":
                run_command(["docker-compose", "down"], "Stopping container")
            elif choice == "8":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-8.")
                
        except KeyboardInterrupt:
            print("\n⏹️  Operation interrupted")
            continue
        
        if choice != "6":  # Don't pause after logs (they're interactive)
            input("\n⏸️  Press Enter to continue...")

if __name__ == "__main__":
    print("🔄 Docker Container Update Tool")
    print("=" * 40)
    
    # Quick check if docker-compose.yml exists
    if not os.path.exists("docker-compose.yml"):
        print("❌ docker-compose.yml not found")
        print("💡 Please run this from the project root directory")
        sys.exit(1)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
        sys.exit(0)
