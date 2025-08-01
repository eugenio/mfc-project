#!/usr/bin/env python3
"""
Hook Functionality Test Script

This script tests if hooks are operational by triggering various actions
that should generate hook events and then checking for hook logs.
"""

import os
import subprocess
import time
import json
from datetime import datetime

def check_hook_logs():
    """Check for recent hook log entries."""
    print("🔍 Checking for hook log directories...")
    
    hook_log_paths = [
        "./logs",
        "./data/logs", 
        "./src/logs",
        "./reports/report-20250709/logs"
    ]
    
    recent_logs = []
    cutoff_time = time.time() - 3600  # Last hour
    
    for log_path in hook_log_paths:
        if os.path.exists(log_path):
            print(f"  Found log directory: {log_path}")
            
            for session_dir in os.listdir(log_path):
                session_path = os.path.join(log_path, session_dir)
                if os.path.isdir(session_path):
                    # Check for hook files
                    hook_files = [
                        "pre_tool_use.json",
                        "post_tool_use.json", 
                        "user_prompt_submit.json",
                        "notification.json"
                    ]
                    
                    for hook_file in hook_files:
                        hook_file_path = os.path.join(session_path, hook_file)
                        if os.path.exists(hook_file_path):
                            mtime = os.path.getmtime(hook_file_path)
                            if mtime > cutoff_time:
                                recent_logs.append({
                                    'file': hook_file_path,
                                    'modified': datetime.fromtimestamp(mtime),
                                    'type': hook_file.replace('.json', '')
                                })
    
    return recent_logs

def trigger_test_tool_use():
    """Trigger a simple tool use that should generate hooks."""
    print("\n🔧 Triggering test tool use...")
    
    # Create a test file that should trigger file creation hooks
    test_file = "hook_test_output.txt"
    content = f"""Hook Test Output
Generated at: {datetime.now()}
This file was created to test hook functionality.

Test commands executed:
1. File creation (this file)
2. Directory listing
3. File content reading

If hooks are operational, this action should be logged.
"""
    
    with open(test_file, 'w') as f:
        f.write(content)
    
    print(f"  Created test file: {test_file}")
    
    # Execute some commands that might trigger hooks
    commands = [
        "ls -la hook_test_output.txt",
        "wc -l hook_test_output.txt",
        "head -5 hook_test_output.txt"
    ]
    
    for cmd in commands:
        print(f"  Executing: {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"    ✅ Success: {result.stdout.strip()}")
            else:
                print(f"    ❌ Error: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print("    ⏱️ Timeout")
        except Exception as e:
            print(f"    ❌ Exception: {e}")

def analyze_hook_logs(recent_logs):
    """Analyze recent hook logs for patterns."""
    print(f"\n📊 Analyzing {len(recent_logs)} recent hook log entries...")
    
    if not recent_logs:
        print("  ❌ No recent hook logs found")
        return False
    
    # Group by type
    hook_types = {}
    for log_entry in recent_logs:
        hook_type = log_entry['type']
        if hook_type not in hook_types:
            hook_types[hook_type] = []
        hook_types[hook_type].append(log_entry)
    
    print("  Hook activity summary:")
    for hook_type, entries in hook_types.items():
        print(f"    {hook_type}: {len(entries)} entries")
        
        # Show most recent entry
        latest = max(entries, key=lambda x: x['modified'])
        print(f"      Latest: {latest['modified']} ({latest['file']})")
    
    # Try to read a recent hook file
    if recent_logs:
        latest_log = max(recent_logs, key=lambda x: x['modified'])
        print(f"\n📖 Sample from latest hook log ({latest_log['file']}):")
        
        try:
            with open(latest_log['file'], 'r') as f:
                content = f.read()
                if content.strip():
                    # Try to parse as JSON
                    try:
                        data = json.loads(content)
                        if isinstance(data, list) and len(data) > 0:
                            print(f"    Entries: {len(data)}")
                            latest_entry = data[-1]  # Get last entry
                            if 'hook_event_name' in latest_entry:
                                print(f"    Latest event: {latest_entry['hook_event_name']}")
                            if 'tool_name' in latest_entry:
                                print(f"    Tool: {latest_entry['tool_name']}")
                            if 'cwd' in latest_entry:
                                print(f"    Working dir: {latest_entry['cwd']}")
                        else:
                            print(f"    Content: {content[:200]}...")
                    except json.JSONDecodeError:
                        print(f"    Raw content: {content[:200]}...")
                else:
                    print("    (Empty file)")
        except Exception as e:
            print(f"    Error reading file: {e}")
    
    return len(recent_logs) > 0

def test_hook_configuration():
    """Test if hook configuration files exist."""
    print("\n⚙️ Checking hook configuration...")
    
    config_paths = [
        "~/.claude/settings.json",
        "~/.claude/hooks.json", 
        "./.claude/hooks.json",
        "./hooks.json",
        "./claude-code-hooks.json"
    ]
    
    found_configs = []
    
    for config_path in config_paths:
        expanded_path = os.path.expanduser(config_path)
        if os.path.exists(expanded_path):
            found_configs.append(expanded_path)
            print(f"  ✅ Found config: {expanded_path}")
            
            # Try to read config
            try:
                with open(expanded_path, 'r') as f:
                    content = f.read()
                    if 'hook' in content.lower():
                        print("    Contains hook configuration")
                    else:
                        print("    No hook references found")
            except Exception as e:
                print(f"    Error reading: {e}")
        else:
            print(f"  ❌ Not found: {expanded_path}")
    
    return len(found_configs) > 0

def main():
    """Run hook functionality test."""
    print("🧪 Hook Functionality Test")
    print("=" * 50)
    
    # Step 1: Check for existing hook logs  
    print("\n1. Checking for existing hook logs...")
    initial_logs = check_hook_logs()
    initial_count = len(initial_logs)
    print(f"   Found {initial_count} recent hook log entries")
    
    # Step 2: Check hook configuration
    print("\n2. Checking hook configuration...")
    config_exists = test_hook_configuration()
    
    # Step 3: Trigger test actions
    print("\n3. Triggering test actions...")
    trigger_test_tool_use()
    
    # Step 4: Wait and check for new logs
    print("\n4. Waiting for hook processing...")
    time.sleep(2)  # Give hooks time to process
    
    print("\n5. Checking for new hook logs...")
    final_logs = check_hook_logs()
    final_count = len(final_logs)
    new_logs = final_count - initial_count
    
    # Step 6: Analyze results
    print("\n6. Analysis Results:")
    print("=" * 30)
    
    hooks_operational = analyze_hook_logs(final_logs)
    
    if hooks_operational:
        print("\n✅ HOOKS APPEAR TO BE OPERATIONAL")
        print("   - Recent hook activity detected")
        print(f"   - {final_count} total recent hook log entries")
        if new_logs > 0:
            print(f"   - {new_logs} new entries generated during test")
    else:
        print("\n❌ HOOKS DO NOT APPEAR TO BE OPERATIONAL")
        print("   - No recent hook activity detected")
        print("   - No hook logs found in expected locations")
    
    # Configuration status
    if config_exists:
        print("   - Hook configuration files found")
    else:
        print("   - No hook configuration files found")
    
    # Cleanup
    if os.path.exists("hook_test_output.txt"):
        os.remove("hook_test_output.txt")
        print("\n🧹 Cleaned up test file")
    
    print("\n📋 Test Summary:")
    print(f"   Hook Logs Found: {final_count}")
    print(f"   Configuration Files: {'Yes' if config_exists else 'No'}")
    print(f"   Status: {'OPERATIONAL' if hooks_operational else 'NOT OPERATIONAL'}")
    
    return 0 if hooks_operational else 1

if __name__ == "__main__":
    exit(main())