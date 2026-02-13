"""
Claude Agent - Autonomous TDD development using Claude API
"""
import subprocess
import sys
import json
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
class ClaudeAgent:
    def __init__(self):
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.model = "claude-sonnet-4-20250514"
        self.worktrees_dir = Path("worktrees")
        
    def start_agent(self, feature_name: str) -> bool:
        """Start Claude agent for feature development"""
        worktree_path = self.worktrees_dir / f"feature-{feature_name}"
        
        if not worktree_path.exists():
            print(f"❌ Worktree not found: {worktree_path}")
            return False
            
        prompt_file = worktree_path / ".claude-prompt"
        if not prompt_file.exists():
            print(f"❌ Claude prompt not found: {prompt_file}")
            return False
            
        print(f"🤖 Starting Claude agent for feature: {feature_name}")
        
        # Load prompt
        with open(prompt_file) as f:
            prompt = f.read()
            
        # Update TDD state
        state_file = worktree_path / ".tdd-state"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            state["claude_agent_active"] = True
            state["claude_agent_started"] = datetime.now().isoformat()
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
                
        # Start autonomous development loop
        self._development_loop(worktree_path, feature_name, prompt)
        
        return True
        
    def _development_loop(self, worktree_path: Path, feature_name: str, initial_prompt: str) -> None:
        """Main development loop for Claude agent"""
        iteration = 0
        max_iterations = 50  # Prevent infinite loops
        
        print(f"🔄 Starting development loop for {feature_name}")
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n📍 TDD Iteration {iteration}")
            
            # Check for stop signals
            if self._check_stop_signal(worktree_path):
                print("🛑 Stop signal detected, halting agent")
                break
                
            # Check for pause signal
            if self._check_pause_signal(worktree_path):
                print("⏸️ Pause signal detected, waiting...")
                time.sleep(30)
                continue
                
            # Analyze current state
            current_state = self._analyze_worktree_state(worktree_path)
            
            # Check if feature is complete
            if current_state.get("tests_passing") and current_state.get("coverage") >= 95:
                print("✅ Feature appears complete!")
                if self._verify_completion(worktree_path):
                    break
                    
            # Generate next action prompt
            action_prompt = self._generate_action_prompt(current_state, initial_prompt)
            
            # Get Claude's response
            response = self._call_claude_api(action_prompt)
            if not response:
                print("❌ Failed to get Claude response")
                time.sleep(10)
                continue
                
            # Execute Claude's instructions
            self._execute_claude_instructions(worktree_path, response)
            
            # Update progress
            self._update_progress(worktree_path, iteration)
            
            # Brief pause between iterations
            time.sleep(5)
            
        print(f"🏁 Development loop completed after {iteration} iterations")
        self._finalize_agent(worktree_path, feature_name)
        
    def _check_stop_signal(self, worktree_path: Path) -> bool:
        """Check if stop signal file exists"""
        return (worktree_path / ".claude-stop").exists()
        
    def _check_pause_signal(self, worktree_path: Path) -> bool:
        """Check if pause signal file exists"""
        return (worktree_path / ".claude-pause").exists()
        
    def _analyze_worktree_state(self, worktree_path: Path) -> Dict:
        """Analyze current state of the worktree"""
        state = {
            "timestamp": datetime.now().isoformat(),
            "tests_passing": False,
            "coverage": 0.0,
            "files_modified": [],
            "last_error": None
        }
        
        # Run tests
        test_result = subprocess.run([
            "pytest", "--tb=short"
        ], cwd=worktree_path, capture_output=True, text=True)
        
        state["tests_passing"] = test_result.returncode == 0
        if test_result.returncode != 0:
            state["last_error"] = test_result.stdout + test_result.stderr
            
        # Check coverage
        coverage_result = subprocess.run([
            "pytest", "--cov", "--cov-report=term"
        ], cwd=worktree_path, capture_output=True, text=True)
        
        # Extract coverage percentage
        for line in coverage_result.stdout.split('\n'):
            if 'TOTAL' in line:
                try:
                    coverage_str = line.split()[-1].replace('%', '')
                    state["coverage"] = float(coverage_str)
                except:
                    pass
                    
        return state
        
    def _generate_action_prompt(self, current_state: Dict, initial_prompt: str) -> str:
        """Generate prompt for next action based on current state"""
        prompt = f"""
    def _call_claude_api(self, prompt: str) -> Optional[str]:
        """Call Claude API with the given prompt"""
        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": 2000,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                },
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["content"][0]["text"]
            else:
                print(f"❌ Claude API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Claude API call failed: {e}")
            return None
            
    def _execute_claude_instructions(self, worktree_path: Path, response: str) -> None:
        """Execute Claude's instructions (placeholder for now)"""
        # This would need to parse Claude's response and execute code changes
        # For now, just log the response
        log_file = worktree_path / ".claude-log"
        with open(log_file, "a") as f:
            f.write(f"\n{datetime.now().isoformat()}\n")
            f.write(response)
            f.write("\n" + "="*50 + "\n")
            
        print("📝 Claude instructions logged")
        
    def _update_progress(self, worktree_path: Path, iteration: int) -> None:
        """Update progress in TDD state file"""
        state_file = worktree_path / ".tdd-state"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            state["claude_iterations"] = iteration
            state["last_claude_update"] = datetime.now().isoformat()
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
                
    def _verify_completion(self, worktree_path: Path) -> bool:
        """Verify that the feature is truly complete"""
        # Run comprehensive checks
        checks = [
            (["pytest"], "All tests must pass"),
            (["pytest", "--cov", "--cov-fail-under=95"], "Coverage must be ≥95%"),
            (["black", "--check", "."], "Code must be formatted"),
            (["isort", "--check-only", "."], "Imports must be sorted"),
            (["flake8"], "No linting errors"),
            (["mypy", "src/"], "Type checking must pass")
        ]
        
        for command, description in checks:
            result = subprocess.run(command, cwd=worktree_path, capture_output=True)
            if result.returncode != 0:
                print(f"❌ {description}")
                return False
                
        print("✅ All completion checks passed!")
        return True
        
    def _finalize_agent(self, worktree_path: Path, feature_name: str) -> None:
        """Finalize the Claude agent work"""
        # Update TDD state
        state_file = worktree_path / ".tdd-state"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            state["claude_agent_active"] = False
            state["claude_agent_completed"] = datetime.now().isoformat()
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
                
        print(f"🎉 Claude agent completed work on {feature_name}")
        
        # Play completion sound
        try:
            subprocess.run(["python", "scripts/audio_notifier.py", "completion"])
        except:
            pass
            
    def get_status(self, feature_name: str) -> str:
        """Get current status of Claude agent"""
        worktree_path = self.worktrees_dir / f"feature-{feature_name}"
        state_file = worktree_path / ".tdd-state"
        
        if not state_file.exists():
            return "UNKNOWN"
            
        with open(state_file) as f:
            state = json.load(f)
            
        if state.get("claude_agent_active"):
            return "ACTIVE"
        elif state.get("claude_agent_completed"):
            return "COMPLETED"
        else:
            return "INACTIVE"
            
    def stop_agent(self, feature_name: str) -> bool:
        """Stop Claude agent for feature"""
        worktree_path = self.worktrees_dir / f"feature-{feature_name}"
        stop_file = worktree_path / ".claude-stop"
        
        with open(stop_file, "w") as f:
            f.write(f"Stopped at {datetime.now().isoformat()}")
            
        print(f"🛑 Stop signal sent to Claude agent for {feature_name}")
        return True
def main():
    """CLI interface for Claude Agent"""
    if len(sys.argv) < 3:
        print("Usage: claude_agent.py <command> <feature_name>")
        print("Commands:")
        print("  start <feature_name>   - Start Claude agent")
        print("  status <feature_name>  - Get agent status")
        print("  stop <feature_name>    - Stop agent")
        return 1
        
    agent = ClaudeAgent()
    command = sys.argv[1]
    feature_name = sys.argv[2]
    
    if command == "start":
        return 0 if agent.start_agent(feature_name) else 1
    elif command == "status":
        status = agent.get_status(feature_name)
        print(f"Claude agent status for {feature_name}: {status}")
        return 0
    elif command == "stop":
        return 0 if agent.stop_agent(feature_name) else 1
    else:
        print(f"❌ Unknown command: {command}")
        return 1
if __name__ == "__main__":
    sys.exit(main())