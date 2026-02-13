# Claude Code Hooks Migration Guide

This guide documents how to migrate the Claude Code hooks system from this project to a separate, dedicated repository. The hooks system is infrastructure tooling for Claude Code and should be maintained independently from the MFC application.

## Overview

The `.claude/hooks/` directory contains Python scripts and utilities that integrate with Claude Code's hook system. These hooks provide:

- Edit and file creation thresholds with chunking
- Security and guardian functionality
- Notification and event sending
- LLM integration for summarization
- Text-to-speech capabilities

This infrastructure is not specific to the MFC project and should be maintained in a separate repository for reuse across multiple Claude Code projects.

## Files to Migrate

### Core Hook Scripts

These are the main entry points called by Claude Code:

| File | Purpose |
|------|---------|
| `pre_tool_use.py` | Runs before any tool is executed |
| `post_tool_use.py` | Runs after any tool completes |
| `user_prompt_submit.py` | Processes user prompts before submission |
| `notification.py` | Handles notification events |
| `stop.py` | Executes when Claude Code stops |
| `subagent_stop.py` | Executes when a subagent stops |
| `send_event.py` | Sends events to external services |
| `enhanced_file_chunking.py` | Handles chunked file creation |
| `enhanced_issue_detection.py` | Detects and reports issues |

### Utility Modules

Supporting modules used by the hook scripts:

| File | Purpose |
|------|---------|
| `utils/constants.py` | Shared constants and configuration |
| `utils/enhanced_security_guardian.py` | Security validation and guardrails |
| `utils/summarizer.py` | Text summarization utilities |
| `utils/llm/anth.py` | Anthropic API integration |
| `utils/llm/oai.py` | OpenAI API integration |
| `utils/tts/elevenlabs_tts.py` | ElevenLabs TTS integration |
| `utils/tts/openai_tts.py` | OpenAI TTS integration |
| `utils/tts/pyttsx3_tts.py` | Local TTS using pyttsx3 |

### Documentation

README files documenting the hooks system:

| File | Purpose |
|------|---------|
| `README_AUTOMATED_ISSUES.md` | Automated issue detection docs |
| `README_edit_thresholds.md` | Edit threshold configuration |
| `README_CHUNKED_FILE_CREATION.md` | Chunked file creation docs |

### Configuration Files

These should be adapted for the new repository:

| Source File | Purpose |
|-------------|---------|
| `.claude/settings.json` | Hook configurations (hooks section) |
| `.claude/settings.json` | Edit thresholds (`edit_thresholds` section) |
| `.claude/settings.json` | File creation thresholds (`file_creation_thresholds` section) |

## Creating a New Hooks Repository

### Step 1: Create the Repository

```bash
# Create a new repository (e.g., claude-code-hooks)
mkdir claude-code-hooks
cd claude-code-hooks
git init
```

### Step 2: Set Up Directory Structure

```
claude-code-hooks/
├── hooks/
│   ├── __init__.py
│   ├── pre_tool_use.py
│   ├── post_tool_use.py
│   ├── user_prompt_submit.py
│   ├── notification.py
│   ├── stop.py
│   ├── subagent_stop.py
│   ├── send_event.py
│   ├── enhanced_file_chunking.py
│   └── enhanced_issue_detection.py
├── utils/
│   ├── __init__.py
│   ├── constants.py
│   ├── enhanced_security_guardian.py
│   ├── summarizer.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── anth.py
│   │   └── oai.py
│   └── tts/
│       ├── __init__.py
│       ├── elevenlabs_tts.py
│       ├── openai_tts.py
│       └── pyttsx3_tts.py
├── docs/
│   ├── README_AUTOMATED_ISSUES.md
│   ├── README_edit_thresholds.md
│   └── README_CHUNKED_FILE_CREATION.md
├── templates/
│   └── settings.json.example
├── pyproject.toml
├── pixi.toml
└── README.md
```

### Step 3: Copy Files from MFC Project

```bash
# From the MFC project root
cp -r .claude/hooks/* /path/to/claude-code-hooks/hooks/
mv /path/to/claude-code-hooks/hooks/utils /path/to/claude-code-hooks/utils
mv /path/to/claude-code-hooks/hooks/README_*.md /path/to/claude-code-hooks/docs/
```

### Step 4: Create pyproject.toml

```toml
[project]
name = "claude-code-hooks"
version = "0.1.0"
description = "Hook system for Claude Code agent infrastructure"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.49.0",
    "openai>=1.0.0",
    "pyttsx3>=2.90",
    "requests>=2.31.0",
]

[project.optional-dependencies]
tts = [
    "elevenlabs>=0.2.0",
]
```

### Step 5: Create pixi.toml

```toml
[project]
name = "claude-code-hooks"
version = "0.1.0"
channels = ["conda-forge"]
platforms = ["linux-64", "osx-arm64", "win-64"]

[tasks]
# Core hook tasks
pre-tool-hook = "python hooks/pre_tool_use.py"
post-tool-hook = "python hooks/post_tool_use.py"
notification-hook = "python hooks/notification.py"
stop-hook = "python hooks/stop.py"
subagent-stop-hook = "python hooks/subagent_stop.py"
user-prompt-submit-hook = "python hooks/user_prompt_submit.py"
send-event-hook = "python hooks/send_event.py"

[dependencies]
python = ">=3.11"
```

### Step 6: Update Import Paths

After moving files, update import statements in the hook scripts:

```python
# Before (in MFC project)
from utils.constants import SOME_CONSTANT

# After (in hooks repository)
from utils.constants import SOME_CONSTANT  # Same, if utils/ is at root
```

## Integrating the Hooks Repository with Projects

### Option 1: Git Submodule

Add the hooks repository as a submodule:

```bash
cd your-project
git submodule add https://your-git-host/claude-code-hooks.git .claude/hooks-repo
```

Then symlink to `.claude/hooks/`:

```bash
ln -s hooks-repo/hooks .claude/hooks
```

### Option 2: Symlink from Shared Location

Clone the hooks repository to a shared location:

```bash
git clone https://your-git-host/claude-code-hooks.git ~/tools/claude-code-hooks
```

Symlink in each project:

```bash
cd your-project
ln -s ~/tools/claude-code-hooks/hooks .claude/hooks
```

### Option 3: Install as Python Package

If the hooks repository is published as a package:

```bash
pip install claude-code-hooks
# or with pixi
pixi add claude-code-hooks
```

Then reference the installed hooks in your settings.json.

## Updating settings.json

After migration, update your project's `.claude/settings.json` to reference the hooks:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "pixi run -e default pre-tool-hook"
          }
        ]
      }
    ]
  }
}
```

If using a submodule or symlink, update the paths in your pixi.toml tasks:

```toml
[tasks]
pre-tool-hook = "cd .claude/hooks && python pre_tool_use.py"
```

## Removing Hooks from MFC Project

After confirming the new hooks repository works:

1. **Do NOT remove `.claude/hooks/` directory** - It may be symlinked or contain project-specific overrides
1. **Update pixi.toml** - Point hook tasks to the new repository location
1. **Update settings.json** - Ensure paths are correct
1. **Test all hooks** - Verify they work with the new setup

## Checklist

- [ ] Create new hooks repository
- [ ] Copy all hook files and utilities
- [ ] Update import paths in hook scripts
- [ ] Create pyproject.toml and pixi.toml
- [ ] Set up CI/CD for the hooks repository
- [ ] Choose integration method (submodule, symlink, or package)
- [ ] Update project settings to use new hooks
- [ ] Test all hook functionality
- [ ] Document any project-specific customizations

## Additional Resources

- [Claude Code Hooks Documentation](https://docs.anthropic.com/claude-code/hooks)
- Hook configuration is in `.claude/settings.json`
- Edit thresholds documentation: See `docs/README_edit_thresholds.md` in hooks repo
