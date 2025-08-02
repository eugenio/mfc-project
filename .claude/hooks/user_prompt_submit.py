#!/usr/bin/env -S pixi run python

import json
from datetime import datetime

from cchooks import UserPromptSubmitContext, create_context
from utils.constants import ensure_session_log_dir

c = create_context()
assert isinstance(c, UserPromptSubmitContext)

def log_user_prompt(session_id, prompt):
    log_dir = ensure_session_log_dir(session_id)
    log_file = log_dir / "user_prompt_submit.json"

    log_data = []
    if log_file.exists():
        try:
            with open(log_file) as f:
                log_data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            log_data = []

    log_entry = {
        'prompt': prompt,
        'session_id': session_id,
        'timestamp': datetime.now().isoformat()
    }
    log_data.append(log_entry)

    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)

session_id = c.session_id or "unknown"
prompt = c.prompt

log_user_prompt(session_id, prompt)

c.output.exit_success()
