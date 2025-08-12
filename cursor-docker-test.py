#!/usr/bin/env python3
"""
Interactive Docker testing script for Cursor IDE
Provides a menu-driven interface for testing Docker deployment
"""

import os
import sys
import subprocess
import time
import threading
from pathlib import Path

class DockerTester:
    def __init__(self):
        self.container_name = "tescom-agent-test"
        self.image_name = "tescom-agent"
        
    def run_command(self, cmd, capture_output=False, timeout=None):
        """Run a command and display output"""
        print(f"🔧 Running: {' '.join(cmd)}")
        print("-" * 50)
        
        try:
            if capture_output:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                print(result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)
                return result.returncode == 0, result.stdout
            else:
                result = subprocess.run(cmd, timeout=timeout)
                return result.returncode == 0, ""
        except subprocess.TimeoutExpired:
            print("❌ Command timed out")
            return False, ""
        except Exception as e:
            print(f"❌ Error: {e}")
            return False, ""
    
    def check_docker_running(self):
        """Check if Docker is running"""
        print("🔍 Checking Docker status...")
        success, _ = self.run_command(["docker", "version"], capture_output=True, timeout=10)
        if success:
            print("✅ Docker is running")
            return True
        else:
            print("❌ Docker is not running or not accessible")
            print("💡 Please start Docker Desktop and try again")
            return False
    
    def build_image(self):
        """Build the Docker image"""
        print("🏗️  Building Docker image...")
        success, _ = self.run_command(["docker", "build", "-t", self.image_name, "."])
        if success:
            print("✅ Docker image built successfully")
        else:
            print("❌ Docker build failed")
        return success
    
    def test_database_init(self):
        """Test database initialization"""
        print("🗄️  Testing database initialization...")
        success, output = self.run_command([
            "docker", "run", "--rm",
            "-e", "OPENAI_API_KEY=test_key",
            "-e", "SERPER_API_KEY=test_key", 
            self.image_name,
            "python", "docker-init-db.py"
        ], capture_output=True, timeout=120)
        
        if success:
            print("✅ Database initialization successful")
        else:
            print("❌ Database initialization failed")
        return success
    
    def start_app_test(self):
        """Start the app for testing"""
        print("🚀 Starting app for testing...")
        print("📝 You'll need to provide API keys...")
        
        # Get API keys
        openai_key = input("Enter OpenAI API key (or 'test' for demo): ").strip()
        serper_key = input("Enter Serper API key (or 'test' for demo): ").strip()
        
        if not openai_key:
            openai_key = "test_key"
        if not serper_key:
            serper_key = "test_key"
        
        print(f"🌐 Starting app on http://localhost:8080")
        print("⏹️  Press Ctrl+C to stop...")
        
        # Run in foreground so user can stop it
        success, _ = self.run_command([
            "docker", "run", "--rm",
            "-p", "8080:7860",
            "-e", f"OPENAI_API_KEY={openai_key}",
            "-e", f"SERPER_API_KEY={serper_key}",
            "--name", self.container_name,
            self.image_name
        ])
        
        return success
    
    def deploy_with_compose(self):
        """Deploy using docker-compose"""
        print("🐳 Deploying with Docker Compose...")
        
        # Check if .env file exists
        if not os.path.exists(".env"):
            print("⚠️  No .env file found")
            create_env = input("Create .env file with test keys? (y/n): ").lower().startswith('y')
            
            if create_env:
                with open(".env", "w") as f:
                    f.write("OPENAI_API_KEY=your_openai_key_here\n")
                    f.write("SERPER_API_KEY=your_serper_key_here\n")
                print("✅ Created .env file - please edit it with real API keys")
                return False
        
        success, _ = self.run_command(["docker-compose", "up", "-d"])
        
        if success:
            print("✅ App deployed successfully")
            print("🌐 Access at: http://localhost:8080")
            print("📝 View logs: docker-compose logs -f")
            print("⏹️  Stop: docker-compose down")
        else:
            print("❌ Deployment failed")
        
        return success
    
    def view_logs(self):
        """View container logs"""
        print("📄 Viewing container logs...")
        self.run_command(["docker-compose", "logs", "-f", "tescom-agent"])
    
    def cleanup(self):
        """Clean up Docker resources"""
        print("🧹 Cleaning up Docker resources...")
        
        # Stop compose
        self.run_command(["docker-compose", "down", "-v"], capture_output=True)
        
        # Remove test containers
        self.run_command(["docker", "rm", "-f", self.container_name], capture_output=True)
        
        # Remove test image
        remove = input("Remove Docker image? (y/n): ").lower().startswith('y')
        if remove:
            self.run_command(["docker", "rmi", self.image_name], capture_output=True)
        
        print("✅ Cleanup complete")
    
    def show_menu(self):
        """Show interactive menu"""
        while True:
            print("\n" + "="*60)
            print("🐳 TESCOM DOCKER TESTING MENU")
            print("="*60)
            print("1. Check Docker Status")
            print("2. Build Docker Image")
            print("3. Test Database Initialization")
            print("4. Start App (Interactive Test)")
            print("5. Deploy with Docker Compose")
            print("6. View Logs")
            print("7. Cleanup Docker Resources")
            print("8. Full Test Suite (1→2→3→5)")
            print("9. Exit")
            print("-"*60)
            
            choice = input("Select option (1-9): ").strip()
            
            try:
                if choice == "1":
                    self.check_docker_running()
                elif choice == "2":
                    if self.check_docker_running():
                        self.build_image()
                elif choice == "3":
                    if self.check_docker_running():
                        self.test_database_init()
                elif choice == "4":
                    if self.check_docker_running():
                        self.start_app_test()
                elif choice == "5":
                    if self.check_docker_running():
                        self.deploy_with_compose()
                elif choice == "6":
                    self.view_logs()
                elif choice == "7":
                    self.cleanup()
                elif choice == "8":
                    # Full test suite
                    if self.check_docker_running():
                        if self.build_image():
                            if self.test_database_init():
                                self.deploy_with_compose()
                elif choice == "9":
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please select 1-9.")
                    
            except KeyboardInterrupt:
                print("\n⏹️  Operation interrupted")
                continue
            
            input("\n⏸️  Press Enter to continue...")

def main():
    """Main function"""
    print("🚀 Tescom Docker Testing Tool for Cursor IDE")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("Dockerfile"):
        print("❌ Dockerfile not found in current directory")
        print("💡 Please run this from the Tescom project root")
        sys.exit(1)
    
    tester = DockerTester()
    
    try:
        tester.show_menu()
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
        sys.exit(0)

if __name__ == "__main__":
    main()
