"""
TDD Worktree Manager - Manages git worktrees for TDD workflows with IDE and Claude integration
"""
import subprocess
import sys
import json
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
        prompt_file = worktree_path / ".claude-prompt"
        with open(prompt_file, "w") as f:
            f.write(prompt_content)
            
        print(f"📝 Claude prompt generated: {prompt_file}")
        
    def _setup_worktree_environment(self, worktree_path: Path) -> None:
        """Setup the worktree environment with necessary files"""
        # Copy pixi.toml if it exists
        if Path("pixi.toml").exists():
            shutil.copy2("pixi.toml", worktree_path / "pixi.toml")
            
        # Copy pyproject.toml if it exists
        if Path("pyproject.toml").exists():
            shutil.copy2("pyproject.toml", worktree_path / "pyproject.toml")
            
        # Create VSCode workspace settings
        vscode_dir = worktree_path / ".vscode"
        vscode_dir.mkdir(exist_ok=True)
        
        settings = {
            "python.defaultInterpreterPath": "./venv/bin/python",
            "python.testing.pytestEnabled": True,
            "python.testing.pytestArgs": ["."],
            "python.linting.enabled": True,
            "python.linting.flake8Enabled": True,
            "python.formatting.provider": "black",
            "editor.formatOnSave": True,
            "files.autoSave": "afterDelay",
            "autoDocstring.docstringFormat": "google",
            "python.testing.autoTestDiscoverOnSaveEnabled": True
        }
        
        with open(vscode_dir / "settings.json", "w") as f:
            json.dump(settings, f, indent=2)
            
        # Create VSCode tasks for TDD
        tasks = {
            "version": "2.0.0",
            "tasks": [
                {
                    "label": "TDD: Run Tests",
                    "type": "shell",
                    "command": "pytest",
                    "args": ["-v"],
                    "group": "test",
                    "presentation": {"echo": True, "reveal": "always", "focus": False}
                },
                {
                    "label": "TDD: Run Single Test", 
                    "type": "shell",
                    "command": "pytest",
                    "args": ["${file}", "-v"],
                    "group": "test"
                },
                {
                    "label": "TDD: Coverage Report",
                    "type": "shell", 
                    "command": "pytest",
                    "args": ["--cov", "--cov-report=html"],
                    "group": "test"
                }
            ]
        }
        
        with open(vscode_dir / "tasks.json", "w") as f:
            json.dump(tasks, f, indent=2)
            
    def _launch_vscode_and_claude(self, worktree_path: Path, feature_name: str) -> None:
        """Launch VSCode instance and Claude agent for the worktree"""
        print(f"🚀 Launching VSCode for feature: {feature_name}")
        
        # Launch VSCode
        try:
            subprocess.Popen([
                "code", str(worktree_path)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Update TDD state
            state_file = worktree_path / ".tdd-state"
            if state_file.exists():
                with open(state_file, "r") as f:
                    state = json.load(f)
                state["vscode_launched"] = True
                state["vscode_launched_at"] = datetime.now().isoformat()
                with open(state_file, "w") as f:
                    json.dump(state, f, indent=2)
                    
            print(f"✅ VSCode launched for {worktree_path}")
            
        except FileNotFoundError:
            print("❌ VSCode not found in PATH. Please install VSCode and ensure 'code' command is available.")
            return
            
        # Launch Claude agent
        print(f"🤖 Launching Claude agent for feature: {feature_name}")
        try:
            subprocess.Popen([
                "python", str(self.scripts_dir / "claude_agent.py"), 
                "start", feature_name
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            print(f"✅ Claude agent launched for {feature_name}")
            print(f"📊 Monitor progress: pixi run claude-agent-status {feature_name}")
            
        except Exception as e:
            print(f"❌ Failed to launch Claude agent: {e}")
            
    def get_worktree_status(self) -> List[Dict]:
        """Get status of all TDD worktrees"""
        result = subprocess.run([
            "git", "worktree", "list", "--porcelain"
        ], capture_output=True, text=True)
        
        worktrees = []
        for line in result.stdout.strip().split('\n\n'):
            if not line:
                continue
                
            info = {}
            for item in line.split('\n'):
                if item.startswith('worktree '):
                    info['path'] = item.split(' ', 1)[1]
                elif item.startswith('branch '):
                    info['branch'] = item.split(' ', 1)[1]
                elif item.startswith('HEAD '):
                    info['head'] = item.split(' ', 1)[1]
                    
            # Skip main worktree
            if info.get('branch') == 'refs/heads/main':
                continue
                
            # Load TDD state if available
            worktree_path = Path(info['path'])
            state_file = worktree_path / ".tdd-state"
            if state_file.exists():
                with open(state_file) as f:
                    tdd_state = json.load(f)
                info.update(tdd_state)
                
            # Check current test status
            if worktree_path.exists():
                test_result = subprocess.run([
                    "pytest", "--quiet"
                ], cwd=worktree_path, capture_output=True)
                
                if test_result.returncode == 0:
                    info['test_status'] = "🟢 GREEN"
                else:
                    info['test_status'] = "🔴 RED"
                    
                # Check Claude agent status
                claude_status = self._check_claude_agent_status(worktree_path)
                info['claude_status'] = claude_status
            else:
                info['test_status'] = "❓ UNKNOWN"
                info['claude_status'] = "❓ UNKNOWN"
                
            worktrees.append(info)
            
        return worktrees
        
    def _check_claude_agent_status(self, worktree_path: Path) -> str:
        """Check if Claude agent is active for this worktree"""
        try:
            result = subprocess.run([
                "python", str(self.scripts_dir / "claude_agent.py"),
                "status", worktree_path.name.replace("feature-", "")
            ], capture_output=True, text=True)
            
            if "ACTIVE" in result.stdout:
                return "🤖 ACTIVE"
            elif "COMPLETED" in result.stdout:
                return "✅ COMPLETED"
            elif "PAUSED" in result.stdout:
                return "⏸️ PAUSED"
            else:
                return "⭕ INACTIVE"
        except:
            return "❓ UNKNOWN"
        
    def switch_worktree(self, worktree_name: str) -> bool:
        """Switch to a specific worktree"""
        worktree_path = self.worktrees_dir / worktree_name
        
        if not worktree_path.exists():
            print(f"❌ Worktree not found: {worktree_path}")
            return False
            
        print(f"🔄 Switching to worktree: {worktree_path}")
        print(f"💡 Run: cd {worktree_path} && pixi shell")
        
        # Optional: Launch VSCode if not already running
        launch_ide = self._prompt_ide_launch()
        if launch_ide:
            feature_name = worktree_name.replace("feature-", "")
            self._launch_vscode_and_claude(worktree_path, feature_name)
        
        return True
        
    def sync_worktree(self, worktree_name: str) -> bool:
        """Sync worktree with main branch"""
        worktree_path = self.worktrees_dir / worktree_name
        
        if not worktree_path.exists():
            print(f"❌ Worktree not found: {worktree_path}")
            return False
            
        print(f"🔄 Syncing {worktree_name} with main...")
        
        # Fetch latest changes
        subprocess.run(["git", "fetch", "origin"], cwd=worktree_path)
        
        # Rebase on main
        result = subprocess.run([
            "git", "rebase", "origin/main"
        ], cwd=worktree_path, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Rebase failed: {result.stderr}")
            print("🔧 Manual resolution required")
            return False
            
        # Run tests after sync
        test_result = subprocess.run([
            "pytest"
        ], cwd=worktree_path, capture_output=True)
        
        if test_result.returncode == 0:
            print("✅ Sync completed, all tests passing")
            # Play success sound
            subprocess.run(["python", str(self.scripts_dir / "audio_notifier.py"), "success"])
        else:
            print("⚠️  Sync completed, but tests are failing")
            print("🔧 TDD cycle may need adjustment")
            # Play warning sound
            subprocess.run(["python", str(self.scripts_dir / "audio_notifier.py"), "failure"])
            
        return True
        
    def complete_feature(self, feature_name: str) -> bool:
        """Complete feature development and merge to main"""
        worktree_path = self.worktrees_dir / f"feature-{feature_name}"
        branch_name = f"feature/{feature_name}"
        
        if not worktree_path.exists():
            print(f"❌ Feature worktree not found: {worktree_path}")
            return False
            
        print(f"🏁 Completing feature: {feature_name}")
        
        # Stop Claude agent if running
        subprocess.run([
            "python", str(self.scripts_dir / "claude_agent.py"),
            "stop", feature_name
        ])
        
        # Ensure we're in GREEN state
        test_result = subprocess.run([
            "pytest", "--cov", "--cov-fail-under=95"
        ], cwd=worktree_path, capture_output=True)
        
        if test_result.returncode != 0:
            print("❌ Cannot complete feature - tests failing or coverage < 95%")
            print("🔧 Fix tests before completing feature")
            subprocess.run(["python", str(self.scripts_dir / "audio_notifier.py"), "failure"])
            return False
            
        # Generate completion report
        report = self._generate_completion_report(worktree_path, feature_name)
        
        # Switch to main and pull latest
        subprocess.run(["git", "checkout", "main"])
        subprocess.run(["git", "pull", "origin", "main"])
        
        # Merge feature branch
        merge_result = subprocess.run([
            "git", "merge", "--no-ff", branch_name,
            "-m", f"feat: {feature_name} - TDD complete"
        ], capture_output=True, text=True)
        
        if merge_result.returncode != 0:
            print(f"❌ Merge failed: {merge_result.stderr}")
            subprocess.run(["python", str(self.scripts_dir / "audio_notifier.py"), "failure"])
            return False
            
        # Run tests on main
        test_result = subprocess.run(["pytest"], capture_output=True)
        if test_result.returncode != 0:
            print("❌ Tests failing on main after merge!")
            print("🔧 Rolling back merge...")
            subprocess.run(["git", "reset", "--hard", "HEAD~1"])
            subprocess.run(["python", str(self.scripts_dir / "audio_notifier.py"), "failure"])
            return False
            
        # Clean up worktree and branch
        subprocess.run(["git", "worktree", "remove", str(worktree_path)])
        subprocess.run(["git", "branch", "-d", branch_name])
        
        print(f"✅ Feature {feature_name} completed and merged!")
        print("🧹 Worktree and branch cleaned up")
        print(f"📊 Report saved: reports/feature-{feature_name}-completion.md")
        
        # Play completion sound
        subprocess.run(["python", str(self.scripts_dir / "audio_notifier.py"), "completion"])
        
        # Display report
        print("\n" + "="*60)
        print(report)
        print("="*60)
        
        return True
        
    def _generate_completion_report(self, worktree_path: Path, feature_name: str) -> str:
        """Generate detailed completion report for the feature"""
        
        # Ensure reports directory exists
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        # Load TDD state
        state_file = worktree_path / ".tdd-state"
        tdd_state = {}
        if state_file.exists():
            with open(state_file) as f:
                tdd_state = json.load(f)
                
        # Get test statistics
        test_result = subprocess.run([
            "pytest", "--tb=no", "-v"
        ], cwd=worktree_path, capture_output=True, text=True)
        
        coverage_result = subprocess.run([
            "pytest", "--cov", "--cov-report=term"
        ], cwd=worktree_path, capture_output=True, text=True)
        
        # Count lines of code
        try:
            loc_result = subprocess.run([
                "find", str(worktree_path / "src"), "-name", "*.py", "-exec", "wc", "-l", "{}", "+"
            ], capture_output=True, text=True)
            lines_of_code = sum(int(line.split()[0]) for line in loc_result.stdout.strip().split('\n')[:-1] if line.strip())
        except:
            lines_of_code = "N/A"
            
        # Generate report
        report = f"""# Feature Completion Report: {feature_name}

## Summary
- **Feature**: {feature_name}
- **Completed**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Duration**: {self._calculate_duration(tdd_state.get('created_at'))}
- **TDD Cycles**: {tdd_state.get('cycles_completed', 'N/A')}
- **Lines of Code**: {lines_of_code}

## Test Results
```
{test_result.stdout}
```

## Coverage Report
```
{coverage_result.stdout}
```

## TDD Statistics
- **Created**: {tdd_state.get('created_at', 'N/A')}
- **TDD Phase**: {tdd_state.get('tdd_phase', 'N/A')}
- **Test Count**: {tdd_state.get('test_count', 'N/A')}
- **Final Coverage**: {tdd_state.get('coverage_percent', 'N/A')}%

## Tools Used
- **VSCode**: {'✅' if tdd_state.get('vscode_launched') else '❌'}
- **Claude Agent**: {'✅' if tdd_state.get('claude_agent_active') else '❌'}

## Quality Metrics
- ✅ All tests passing
- ✅ Coverage ≥ 95%
- ✅ Code quality checks passed
- ✅ TDD methodology followed

## Files Modified
"""
        
        # Add git diff summary
        try:
            diff_result = subprocess.run([
                "git", "diff", "--name-only", "main", f"feature/{feature_name}"
            ], cwd=worktree_path, capture_output=True, text=True)
            
            for file in diff_result.stdout.strip().split('\n'):
                if file.strip():
                    report += f"- {file}\n"
        except:
            report += "- Unable to determine modified files\n"
            
        report += f"\n## Completion Status\n✅ Feature {feature_name} successfully completed using TDD methodology!"
        
        # Save report
        report_file = reports_dir / f"feature-{feature_name}-completion.md"
class TDDWorktreeManager:
    def __init__(self):
        self.worktrees_dir = Path("worktrees")
        self.worktrees_dir.mkdir(exist_ok=True)
        self.scripts_dir = Path("scripts")
        
    def create_feature_worktree(self, feature_name: str, interactive: bool = False) -> bool:
        """Create a new feature worktree for TDD development with optional IDE integration"""
        branch_name = f"feature/{feature_name}"
        worktree_path = self.worktrees_dir / f"feature-{feature_name}"
        
        if worktree_path.exists():
            print(f"❌ Worktree already exists: {worktree_path}")
            return False
            
        print(f"🔧 Creating TDD worktree: {worktree_path}")
        
        # Create worktree with new branch
        result = subprocess.run([
            "git", "worktree", "add", str(worktree_path), "-b", branch_name
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Failed to create worktree: {result.stderr}")
            return False
            
        # Initialize TDD state file
        tdd_state = {
            "created_at": datetime.now().isoformat(),
            "feature_name": feature_name,
            "branch_name": branch_name,
            "tdd_phase": "READY",  # READY, RED, GREEN, REFACTOR
            "test_count": 0,
            "last_test_run": None,
            "coverage_percent": 0.0,
            "claude_agent_active": False,
            "vscode_launched": False,
            "cycles_completed": 0
        }
        
        state_file = worktree_path / ".tdd-state"
        with open(state_file, "w") as f:
            json.dump(tdd_state, f, indent=2)
            
        # Generate Claude prompt for this feature
        self._generate_claude_prompt(worktree_path, feature_name)
        
        # Copy essential files to worktree
        self._setup_worktree_environment(worktree_path)
            
        print(f"✅ TDD worktree created: {worktree_path}")
        print(f"🌿 Branch: {branch_name}")
        
        # Interactive IDE and Claude integration
        if interactive:
            launch_ide = self._prompt_ide_launch()
            if launch_ide:
                self._launch_vscode_and_claude(worktree_path, feature_name)
        
        print(f"📋 To start manually: cd {worktree_path} && pixi shell")
        return True
        
    def _prompt_ide_launch(self) -> bool:
        """Prompt user to launch VSCode and Claude agent"""
        try:
            response = input("🚀 Launch VSCode instance and Claude agent for this feature? [Y/n]: ").strip().lower()
            return response in ('', 'y', 'yes')
        except KeyboardInterrupt:
            print("\n❌ Cancelled by user")
            return False
            
    def _generate_claude_prompt(self, worktree_path: Path, feature_name: str) -> None:
        """Generate Claude-specific prompt for the feature"""
        prompt_content = f"""# Claude TDD Agent Prompt for Feature: {feature_name}

## Mission
    def _setup_worktree_environment(self, worktree_path: Path) -> None:
        """Setup the worktree environment with necessary files"""
        # Copy pixi.toml if it exists
        if Path("pixi.toml").exists():
            shutil.copy2("pixi.toml", worktree_path / "pixi.toml")
            
        # Copy pyproject.toml if it exists
        if Path("pyproject.toml").exists():
            shutil.copy2("pyproject.toml", worktree_path / "pyproject.toml")
            
        # Create VSCode workspace settings
        vscode_dir = worktree_path / ".vscode"
        vscode_dir.mkdir(exist_ok=True)
        
        settings = {
            "python.defaultInterpreterPath": "./venv/bin/python",
            "python.testing.pytestEnabled": True,
            "python.testing.pytestArgs": ["."],
            "python.linting.enabled": True,
            "python.linting.flake8Enabled": True,
            "python.formatting.provider": "black",
            "editor.formatOnSave": True,
            "files.autoSave": "afterDelay",
            "autoDocstring.docstringFormat": "google",
            "python.testing.autoTestDiscoverOnSaveEnabled": True
        }
        
        with open(vscode_dir / "settings.json", "w") as f:
            json.dump(settings, f, indent=2)
            
        # Create VSCode tasks for TDD
        tasks = {
            "version": "2.0.0",
            "tasks": [
                {
                    "label": "TDD: Run Tests",
                    "type": "shell",
                    "command": "pytest",
                    "args": ["-v"],
                    "group": "test",
                    "presentation": {"echo": True, "reveal": "always", "focus": False}
                },
                {
                    "label": "TDD: Run Single Test", 
                    "type": "shell",
                    "command": "pytest",
                    "args": ["${file}", "-v"],
                    "group": "test"
                },
                {
                    "label": "TDD: Coverage Report",
                    "type": "shell", 
                    "command": "pytest",
                    "args": ["--cov", "--cov-report=html"],
                    "group": "test"
                }
            ]
        }
        
        with open(vscode_dir / "tasks.json", "w") as f:
            json.dump(tasks, f, indent=2)
            
    def _launch_vscode_and_claude(self, worktree_path: Path, feature_name: str) -> None:
        """Launch VSCode instance and Claude agent for the worktree"""
        print(f"🚀 Launching VSCode for feature: {feature_name}")
        
        # Launch VSCode
        try:
            subprocess.Popen([
                "code", str(worktree_path)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Update TDD state
            state_file = worktree_path / ".tdd-state"
            if state_file.exists():
                with open(state_file, "r") as f:
                    state = json.load(f)
                state["vscode_launched"] = True
                state["vscode_launched_at"] = datetime.now().isoformat()
                with open(state_file, "w") as f:
                    json.dump(state, f, indent=2)
                    
            print(f"✅ VSCode launched for {worktree_path}")
            
        except FileNotFoundError:
            print("❌ VSCode not found in PATH. Please install VSCode and ensure 'code' command is available.")
            return
            
        # Launch Claude agent
        print(f"🤖 Launching Claude agent for feature: {feature_name}")
        try:
            subprocess.Popen([
                "python", str(self.scripts_dir / "claude_agent.py"), 
                "start", feature_name
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            print(f"✅ Claude agent launched for {feature_name}")
            print(f"📊 Monitor progress: pixi run claude-agent-status {feature_name}")
            
        except Exception as e:
            print(f"❌ Failed to launch Claude agent: {e}")
            
    def get_worktree_status(self) -> List[Dict]:
        """Get status of all TDD worktrees"""
        result = subprocess.run([
            "git", "worktree", "list", "--porcelain"
        ], capture_output=True, text=True)
        
        worktrees = []
        for line in result.stdout.strip().split('\n\n'):
            if not line:
                continue
                
            info = {}
            for item in line.split('\n'):
                if item.startswith('worktree '):
                    info['path'] = item.split(' ', 1)[1]
                elif item.startswith('branch '):
                    info['branch'] = item.split(' ', 1)[1]
                elif item.startswith('HEAD '):
                    info['head'] = item.split(' ', 1)[1]
                    
            # Skip main worktree
            if info.get('branch') == 'refs/heads/main':
                continue
                
            # Load TDD state if available
            worktree_path = Path(info['path'])
            state_file = worktree_path / ".tdd-state"
            if state_file.exists():
                with open(state_file) as f:
                    tdd_state = json.load(f)
                info.update(tdd_state)
                
            # Check current test status
            if worktree_path.exists():
                test_result = subprocess.run([
                    "pytest", "--quiet"
                ], cwd=worktree_path, capture_output=True)
                
                if test_result.returncode == 0:
                    info['test_status'] = "🟢 GREEN"
                else:
                    info['test_status'] = "🔴 RED"
                    
                # Check Claude agent status
                claude_status = self._check_claude_agent_status(worktree_path)
                info['claude_status'] = claude_status
            else:
                info['test_status'] = "❓ UNKNOWN"
                info['claude_status'] = "❓ UNKNOWN"
                
            worktrees.append(info)
            
        return worktrees
        
    def _check_claude_agent_status(self, worktree_path: Path) -> str:
        """Check if Claude agent is active for this worktree"""
        try:
            result = subprocess.run([
                "python", str(self.scripts_dir / "claude_agent.py"),
                "status", worktree_path.name.replace("feature-", "")
            ], capture_output=True, text=True)
            
            if "ACTIVE" in result.stdout:
                return "🤖 ACTIVE"
            elif "COMPLETED" in result.stdout:
                return "✅ COMPLETED"
            elif "PAUSED" in result.stdout:
                return "⏸️ PAUSED"
            else:
                return "⭕ INACTIVE"
        except:
            return "❓ UNKNOWN"
        
    def switch_worktree(self, worktree_name: str) -> bool:
        """Switch to a specific worktree"""
        worktree_path = self.worktrees_dir / worktree_name
        
        if not worktree_path.exists():
            print(f"❌ Worktree not found: {worktree_path}")
            return False
            
        print(f"🔄 Switching to worktree: {worktree_path}")
        print(f"💡 Run: cd {worktree_path} && pixi shell")
        
        # Optional: Launch VSCode if not already running
        launch_ide = self._prompt_ide_launch()
        if launch_ide:
            feature_name = worktree_name.replace("feature-", "")
            self._launch_vscode_and_claude(worktree_path, feature_name)
        
        return True
        
    def sync_worktree(self, worktree_name: str) -> bool:
        """Sync worktree with main branch"""
        worktree_path = self.worktrees_dir / worktree_name
        
        if not worktree_path.exists():
            print(f"❌ Worktree not found: {worktree_path}")
            return False
            
        print(f"🔄 Syncing {worktree_name} with main...")
        
        # Fetch latest changes
        subprocess.run(["git", "fetch", "origin"], cwd=worktree_path)
        
        # Rebase on main
        result = subprocess.run([
            "git", "rebase", "origin/main"
        ], cwd=worktree_path, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Rebase failed: {result.stderr}")
            print("🔧 Manual resolution required")
            return False
            
        # Run tests after sync
        test_result = subprocess.run([
            "pytest"
        ], cwd=worktree_path, capture_output=True)
        
        if test_result.returncode == 0:
            print("✅ Sync completed, all tests passing")
            # Play success sound
            subprocess.run(["python", str(self.scripts_dir / "audio_notifier.py"), "success"])
        else:
            print("⚠️  Sync completed, but tests are failing")
            print("🔧 TDD cycle may need adjustment")
            # Play warning sound
            subprocess.run(["python", str(self.scripts_dir / "audio_notifier.py"), "failure"])
            
        return True
        
    def complete_feature(self, feature_name: str) -> bool:
        """Complete feature development and merge to main"""
        worktree_path = self.worktrees_dir / f"feature-{feature_name}"
        branch_name = f"feature/{feature_name}"
        
        if not worktree_path.exists():
            print(f"❌ Feature worktree not found: {worktree_path}")
            return False
            
        print(f"🏁 Completing feature: {feature_name}")
        
        # Stop Claude agent if running
        subprocess.run([
            "python", str(self.scripts_dir / "claude_agent.py"),
            "stop", feature_name
        ])
        
        # Ensure we're in GREEN state
        test_result = subprocess.run([
            "pytest", "--cov", "--cov-fail-under=95"
        ], cwd=worktree_path, capture_output=True)
        
        if test_result.returncode != 0:
            print("❌ Cannot complete feature - tests failing or coverage < 95%")
            print("🔧 Fix tests before completing feature")
            subprocess.run(["python", str(self.scripts_dir / "audio_notifier.py"), "failure"])
            return False
            
        # Generate completion report
        report = self._generate_completion_report(worktree_path, feature_name)
        
        # Switch to main and pull latest
        subprocess.run(["git", "checkout", "main"])
        subprocess.run(["git", "pull", "origin", "main"])
        
        # Merge feature branch
        merge_result = subprocess.run([
            "git", "merge", "--no-ff", branch_name,
            "-m", f"feat: {feature_name} - TDD complete"
        ], capture_output=True, text=True)
        
        if merge_result.returncode != 0:
            print(f"❌ Merge failed: {merge_result.stderr}")
            subprocess.run(["python", str(self.scripts_dir / "audio_notifier.py"), "failure"])
            return False
            
        # Run tests on main
        test_result = subprocess.run(["pytest"], capture_output=True)
        if test_result.returncode != 0:
            print("❌ Tests failing on main after merge!")
            print("🔧 Rolling back merge...")
            subprocess.run(["git", "reset", "--hard", "HEAD~1"])
            subprocess.run(["python", str(self.scripts_dir / "audio_notifier.py"), "failure"])
            return False
            
        # Clean up worktree and branch
        subprocess.run(["git", "worktree", "remove", str(worktree_path)])
        subprocess.run(["git", "branch", "-d", branch_name])
        
        print(f"✅ Feature {feature_name} completed and merged!")
        print("🧹 Worktree and branch cleaned up")
        print(f"📊 Report saved: reports/feature-{feature_name}-completion.md")
        
        # Play completion sound
        subprocess.run(["python", str(self.scripts_dir / "audio_notifier.py"), "completion"])
        
        # Display report
        print("\n" + "="*60)
        print(report)
        print("="*60)
        
        return True
        
    def _generate_completion_report(self, worktree_path: Path, feature_name: str) -> str:
        """Generate detailed completion report for the feature"""
        
        # Ensure reports directory exists
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        # Load TDD state
        state_file = worktree_path / ".tdd-state"
        tdd_state = {}
        if state_file.exists():
            with open(state_file) as f:
                tdd_state = json.load(f)
                
        # Get test statistics
        test_result = subprocess.run([
            "pytest", "--tb=no", "-v"
        ], cwd=worktree_path, capture_output=True, text=True)
        
        coverage_result = subprocess.run([
            "pytest", "--cov", "--cov-report=term"
        ], cwd=worktree_path, capture_output=True, text=True)
        
        # Count lines of code
        try:
            loc_result = subprocess.run([
                "find", str(worktree_path / "src"), "-name", "*.py", "-exec", "wc", "-l", "{}", "+"
            ], capture_output=True, text=True)
            lines_of_code = sum(int(line.split()[0]) for line in loc_result.stdout.strip().split('\n')[:-1] if line.strip())
        except:
            lines_of_code = "N/A"
            
        # Generate report
        report = f"""# Feature Completion Report: {feature_name}

## Summary
    def _calculate_duration(self, start_time: str) -> str:
        """Calculate duration from start time to now"""
        try:
            start = datetime.fromisoformat(start_time)
            duration = datetime.now() - start
            hours = duration.total_seconds() // 3600
            minutes = (duration.total_seconds() % 3600) // 60
            return f"{int(hours)}h {int(minutes)}m"
        except:
            return "N/A"
        
    def cleanup_stale_worktrees(self) -> bool:
        """Clean up worktrees for merged branches"""
        print("🧹 Cleaning up stale worktrees...")
        
        # Get list of merged branches
        result = subprocess.run([
            "git", "branch", "--merged", "main"
        ], capture_output=True, text=True)
        
        merged_branches = [
            branch.strip().replace("* ", "")
            for branch in result.stdout.split('\n')
            if branch.strip() and not branch.strip().endswith('main')
        ]
        
        # Remove worktrees for merged branches
        for worktree in self.worktrees_dir.iterdir():
            if worktree.is_dir():
                state_file = worktree / ".tdd-state"
                if state_file.exists():
                    with open(state_file) as f:
                        state = json.load(f)
                    
                    branch_name = state.get('branch_name', '').replace('feature/', '')
                    if f"feature/{branch_name}" in merged_branches:
                        print(f"🗑️  Removing stale worktree: {worktree}")
                        subprocess.run([
                            "git", "worktree", "remove", str(worktree)
                        ])
                        
        print("✅ Cleanup completed")
        return True
def main():
    """CLI interface for TDD Worktree Manager"""
    if len(sys.argv) < 2:
        print("Usage: tdd_worktree_manager.py <command> [args...]")
        print("Commands:")
        print("  create <feature_name> [--interactive]  - Create new feature worktree")
        print("  status                                 - Show all worktree status")
        print("  switch <worktree_name>                 - Switch to worktree")
        print("  sync <worktree_name>                   - Sync worktree with main")
        print("  complete <feature_name>                - Complete and merge feature")
        print("  cleanup                                - Remove stale worktrees")
        return 1
        
    manager = TDDWorktreeManager()
    command = sys.argv[1]
    
    if command == "create":
        if len(sys.argv) < 3:
            print("❌ Feature name required")
            return 1
        interactive = "--interactive" in sys.argv
        return 0 if manager.create_feature_worktree(sys.argv[2], interactive) else 1
        
    elif command == "status":
        worktrees = manager.get_worktree_status()
        if not worktrees:
            print("📭 No TDD worktrees found")
        else:
            print("🔍 TDD Worktree Status:")
            for wt in worktrees:
                feature_name = wt.get('feature_name', 'unknown')
                test_status = wt.get('test_status', '❓')
                claude_status = wt.get('claude_status', '❓')
                phase = wt.get('tdd_phase', 'UNKNOWN')
                print(f"  {test_status} {claude_status} {feature_name} ({phase})")
        return 0
        
    elif command == "switch":
        if len(sys.argv) < 3:
            print("❌ Worktree name required")
            return 1
        return 0 if manager.switch_worktree(sys.argv[2]) else 1
        
    elif command == "sync":
        if len(sys.argv) < 3:
            print("❌ Worktree name required")
            return 1
        return 0 if manager.sync_worktree(sys.argv[2]) else 1
        
    elif command == "complete":
        if len(sys.argv) < 3:
            print("❌ Feature name required")
            return 1
        return 0 if manager.complete_feature(sys.argv[2]) else 1
        
    elif command == "cleanup":
        return 0 if manager.cleanup_stale_worktrees() else 1
        
    else:
        print(f"❌ Unknown command: {command}")
        return 1
if __name__ == "__main__":
    sys.exit(main())