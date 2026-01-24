#!/usr/bin/env python3
"""
GitLab Integration CLI Testing Tool

Interactive command-line tool for testing GitLab integration functionality.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from utils.gitlab_issue_manager import GitLabIssueManager
    from utils.gitlab_auto_issue import auto_create_issue, analyze_user_input

    # Create a global manager instance for CLI use
    try:
        gitlab_manager = GitLabIssueManager()
    except (ValueError, ImportError):
        gitlab_manager = None
    GITLAB_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    gitlab_manager = None
    GITLAB_AVAILABLE = False

def check_configuration():
    """Check GitLab configuration status."""
    print("🔍 Checking GitLab Configuration")
    print("-" * 40)
    
    # Check environment variables
    token = os.getenv('GITLAB_TOKEN')
    project_id = os.getenv('GITLAB_PROJECT_ID')
    gitlab_url = os.getenv('GITLAB_URL', 'https://gitlab.com')
    
    print(f"GitLab URL: {gitlab_url}")
    print(f"Project ID: {project_id if project_id else '❌ Not set'}")
    print(f"Token: {'✅ Set' if token else '❌ Not set'}")
    
    if not GITLAB_AVAILABLE:
        print("❌ GitLab integration not available")
        return False
    
    # Check connection
    if gitlab_manager.project:
        print(f"✅ Connected to project: {gitlab_manager.project.name}")
        print(f"🔗 Project URL: {gitlab_manager.project.web_url}")
        return True
    else:
        print("❌ Not connected to GitLab")
        return False

def test_issue_detection():
    """Test automatic issue detection."""
    print("\n🧪 Testing Issue Detection")
    print("-" * 40)
    
    test_cases = [
        "The GUI crashes when I click the start button",
        "Add support for exporting simulation results to CSV format",
        "There's a critical memory leak in the GPU acceleration code",
        "I would like to implement better error handling for network connections",
        "The cathode model is giving incorrect power calculations"
    ]
    
    for i, description in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {description}")
        
        analysis = analyze_user_input(description)
        
        print(f"   Type: {analysis['type']} (confidence: {analysis['confidence']:.2f})")
        print(f"   Priority: {analysis['priority']}")
        print(f"   Suggested title: {analysis['suggested_title']}")

def create_test_bug():
    """Create a test bug issue."""
    print("\n🐛 Creating Test Bug Issue")
    print("-" * 40)
    
    if not gitlab_manager.project:
        print("❌ Not connected to GitLab")
        return
    
    description = """
    Test bug created by GitLab integration testing tool.
    
    This is a test issue to verify that the GitLab integration is working correctly.
    
    Steps to reproduce:
    1. Run the test_gitlab_cli.py script
    2. Select option to create test bug
    3. Observe that this issue is created
    
    Expected behavior:
    Issue should be created successfully with proper formatting and labels.
    """
    
    issue_id = auto_create_issue(description, force_type='bug')
    
    if issue_id:
        print(f"✅ Created test bug issue #{issue_id}")
    else:
        print("❌ Failed to create test bug issue")

def create_test_enhancement():
    """Create a test enhancement issue."""
    print("\n✨ Creating Test Enhancement Issue")
    print("-" * 40)
    
    if not gitlab_manager.project:
        print("❌ Not connected to GitLab")
        return
    
    description = """
    Test enhancement created by GitLab integration testing tool.
    
    This is a test feature request to verify that the GitLab integration is working correctly.
    
    Proposed implementation:
    1. Add automated testing for GitLab integration
    2. Create CLI tool for manual testing
    3. Implement issue template validation
    4. Add integration with CI/CD pipeline
    
    This would improve the development workflow and ensure reliable issue management.
    """
    
    issue_id = auto_create_issue(description, force_type='enhancement')
    
    if issue_id:
        print(f"✅ Created test enhancement issue #{issue_id}")
    else:
        print("❌ Failed to create test enhancement issue")

def interactive_issue_creation():
    """Interactive issue creation."""
    print("\n📝 Interactive Issue Creation")
    print("-" * 40)
    
    if not gitlab_manager.project:
        print("❌ Not connected to GitLab")
        return
    
    print("Enter your issue description (or 'quit' to exit):")
    print("You can describe a bug or feature request in natural language.")
    print()
    
    while True:
        try:
            description = input("Description: ").strip()
            
            if description.lower() in ['quit', 'exit', 'q']:
                break
            
            if not description:
                continue
            
            # Analyze the input
            analysis = analyze_user_input(description)
            
            print("\n🔍 Analysis:")
            print(f"   Type: {analysis['type']} (confidence: {analysis['confidence']:.2f})")
            print(f"   Priority: {analysis['priority']}")
            print(f"   Title: {analysis['suggested_title']}")
            
            # Ask for confirmation
            confirm = input(f"\nCreate {analysis['type']} issue? (y/n/s to skip): ").lower()
            
            if confirm == 'y':
                issue_id = auto_create_issue(description)
                if issue_id:
                    print(f"✅ Created issue #{issue_id}")
                else:
                    print("❌ Failed to create issue")
            elif confirm == 's':
                print("⏭️  Skipped")
            else:
                print("❌ Cancelled")
            
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def run_full_test_suite():
    """Run the complete test suite."""
    print("\n🧪 Running Full Test Suite")
    print("-" * 40)
    
    try:
        from tests.test_gitlab_integration import run_tests
        success = run_tests()
        return success
    except ImportError:
        print("❌ Test suite not available")
        return False

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="GitLab Integration Testing Tool")
    parser.add_argument('--check', action='store_true', help='Check configuration only')
    parser.add_argument('--test-detection', action='store_true', help='Test issue detection')
    parser.add_argument('--create-bug', action='store_true', help='Create test bug issue')
    parser.add_argument('--create-enhancement', action='store_true', help='Create test enhancement issue')
    parser.add_argument('--interactive', action='store_true', help='Interactive issue creation')
    parser.add_argument('--run-tests', action='store_true', help='Run full test suite')
    
    args = parser.parse_args()
    
    print("🛠️  GitLab Integration Testing Tool")
    print("=" * 50)
    
    # Always check configuration first
    config_ok = check_configuration()
    
    if args.check:
        return 0 if config_ok else 1
    
    if not GITLAB_AVAILABLE:
        print("\n❌ GitLab integration not available")
        return 1
    
    if args.test_detection:
        test_issue_detection()
    
    elif args.create_bug:
        if config_ok:
            create_test_bug()
        else:
            print("❌ Cannot create issues - configuration invalid")
    
    elif args.create_enhancement:
        if config_ok:
            create_test_enhancement()
        else:
            print("❌ Cannot create issues - configuration invalid")
    
    elif args.interactive:
        if config_ok:
            interactive_issue_creation()
        else:
            print("❌ Cannot create issues - configuration invalid")
    
    elif args.run_tests:
        success = run_full_test_suite()
        return 0 if success else 1
    
    else:
        # Default: show menu
        if not config_ok:
            print("\n⚠️  Configuration issues detected. Some features may not work.")
        
        print("\n📋 Available Options:")
        print("1. Test issue detection")
        print("2. Create test bug issue")
        print("3. Create test enhancement issue") 
        print("4. Interactive issue creation")
        print("5. Run full test suite")
        print("6. Check configuration")
        
        try:
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == '1':
                test_issue_detection()
            elif choice == '2':
                if config_ok:
                    create_test_bug()
                else:
                    print("❌ Configuration required for issue creation")
            elif choice == '3':
                if config_ok:
                    create_test_enhancement()
                else:
                    print("❌ Configuration required for issue creation")
            elif choice == '4':
                if config_ok:
                    interactive_issue_creation()
                else:
                    print("❌ Configuration required for issue creation")
            elif choice == '5':
                success = run_full_test_suite()
                return 0 if success else 1
            elif choice == '6':
                check_configuration()
            else:
                print("❌ Invalid option")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())