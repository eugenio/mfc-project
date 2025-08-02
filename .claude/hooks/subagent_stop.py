#!/usr/bin/env -S pixi run python
# /// script
# requires-python = ">=3.8"
# ///

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from cchooks import SubagentStopContext, create_context
from utils.constants import ensure_session_log_dir

# Create context
c = create_context()
assert isinstance(c, SubagentStopContext)

def get_tts_script_path():
    """
    Determine which TTS script to use based on available API keys.
    Priority order: ElevenLabs > OpenAI > pyttsx3
    """
    # Get current script directory and construct utils/tts path
    script_dir = Path(__file__).parent
    tts_dir = script_dir / "utils" / "tts"

    # Check for ElevenLabs API key (highest priority)
    if os.getenv("ELEVENLABS_API_KEY"):
        elevenlabs_script = tts_dir / "elevenlabs_tts.py"
        if elevenlabs_script.exists():
            return str(elevenlabs_script)

    # Check for OpenAI API key (second priority)
    if os.getenv("OPENAI_API_KEY"):
        openai_script = tts_dir / "openai_tts.py"
        if openai_script.exists():
            return str(openai_script)

    # Fall back to pyttsx3 (no API key required)
    pyttsx3_script = tts_dir / "pyttsx3_tts.py"
    if pyttsx3_script.exists():
        return str(pyttsx3_script)

    return None

def announce_subagent_completion(subagent_type):
    """Announce subagent completion using TTS."""
    try:
        tts_script = get_tts_script_path()
        if not tts_script:
            return  # No TTS scripts available

        # Create subagent-specific message
        message = f"{subagent_type} agent finished"

        # Call the TTS script with the message
        subprocess.run(
            ["pixi", "run", tts_script, message],
            capture_output=True,  # Suppress output
            timeout=10,  # 10-second timeout
        )

    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        # Fail silently if TTS encounters issues
        pass
    except Exception:
        # Fail silently for any other errors
        pass

# Main hook logic
# Extract fields
session_id = c.session_id or ""
stop_hook_active = c.stop_hook_active
subagent_type = getattr(c, 'subagent_type', 'subagent')  # Get subagent type if available

# Ensure session log directory exists
log_dir = ensure_session_log_dir(session_id)
log_path = log_dir / "subagent_stop.json"

# Read existing log data or initialize empty list
log_data = []
if log_path.exists():
    try:
        with open(log_path) as f:
            log_data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        log_data = []

# Append new data
log_entry = {
    'session_id': session_id,
    'stop_hook_active': stop_hook_active,
    'subagent_type': subagent_type,
    'transcript_path': c.transcript_path,
    'timestamp': datetime.now().isoformat()
}
log_data.append(log_entry)

# Write back to file with formatting
with open(log_path, "w") as f:
    json.dump(log_data, f, indent=2)

# Handle --chat switch (same as stop.py)
if "--chat" in sys.argv and c.transcript_path:
    transcript_path = c.transcript_path
    # Convert relative paths to absolute paths
    if not os.path.isabs(transcript_path):
        transcript_path = os.path.abspath(transcript_path)
    if os.path.exists(transcript_path):
        # Read .jsonl file and convert to JSON array
        chat_data = []
        try:
            with open(transcript_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            chat_data.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass  # Skip invalid lines

            # Write to logs/subagent_chat.json (different from main chat.json)
            chat_file = log_dir / "subagent_chat.json"
            with open(chat_file, "w") as f:
                json.dump(chat_data, f, indent=2)
        except Exception:
            pass  # Fail silently

# Announce subagent completion via TTS
announce_subagent_completion(subagent_type)

# Exit success
c.output.exit_success()
