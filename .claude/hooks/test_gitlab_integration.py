#!/usr/bin/env python3
"""
Test script for GitLab API integration.

This script demonstrates how to configure and test the GitLab integration
for Claude Code hooks.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the hooks directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from utils.gitlab_client import (
    test_gitlab_connection,
    load_gitlab_config,
    create_issue,
    get_project_info,
    get_current_branch
)

def main():
    load_dotenv() # will search for .env file in local folder and load variables 
    """Test GitLab integration functionality."""
    print("🔧 GitLab API Integration Test")
    print("=" * 50)
    
    # Test configuration loading
    print("\n1. Testing Configuration Loading...")
    config = load_gitlab_config()
    print(f"   ✓ Config loaded: {config}")
    
    # Test environment variable detection
    print("\n2. Testing Environment Variables...")
    env_vars = {
        "GITLAB_TOKEN": os.getenv("GITLAB_TOKEN"),
        "GITLAB_URL": os.getenv("GITLAB_URL"),
        "GITLAB_PROJECT_ID": os.getenv("GITLAB_PROJECT_ID")
    }
    for var, value in env_vars.items():
        status = "✓ Set" if value else "✗ Not set"
        print(f"   {status}: {var}")
    
    # Test connection
    print("\n3. Testing GitLab Connection...")
    success = test_gitlab_connection()
    
    if success:
        print("   ✅ GitLab integration is working!")
        
        # Test project info
        print("\n4. Testing Project Information...")
        project_info = get_project_info()
        if project_info:
            print(f"   ✓ Project: {project_info['name']}")
            print(f"   ✓ URL: {project_info['web_url']}")
            print(f"   ✓ Default branch: {project_info['default_branch']}")
        
        # Test current branch
        print("\n5. Testing Current Branch...")
        branch = get_current_branch()
        print(f"   ✓ Current branch: {branch}")
        
        # Test issue creation (dry run)
        print("\n6. Testing Issue Creation (dry run)...")
        print("   ℹ️  To test issue creation, uncomment the following lines:")
        print("   # issue = create_issue('Test Issue', 'This is a test issue created by the GitLab integration')")
        print("   # print(f'Created issue: {issue['web_url']}')")
        
    else:
        print("   ❌ GitLab integration not working")
        print("\n📋 Setup Instructions:")
        print("   1. Set environment variables:")
        print("      export GITLAB_TOKEN='your-gitlab-token'")
        print("      export GITLAB_URL='https://gitlab-runner.tail301d0a.ts.net'")
        print("      export GITLAB_PROJECT_ID='your-project-id'")
        print("   2. Or configure in .claude/settings.json:")
        print("      {")
        print("        'gitlab': {")
        print("          'enabled': true,")
        print("          'url': 'https://gitlab-runner.tail301d0a.ts.net',")
        print("          'token': 'your-gitlab-token',")
        print("          'project_id': 'your-project-id'")
        print("        }")
        print("      }")
    
    print("\n🎯 Integration Features Available:")
    print("   • Automatic issue creation on hook failures")
    print("   • Auto-MR creation when multiple commits accumulate")
    print("   • Commit commenting and status updates")
    print("   • Project information and branch management")
    
    print("\n✨ Test completed!")

if __name__ == "__main__":