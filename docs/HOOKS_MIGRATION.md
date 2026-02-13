# Claude Code Hooks Migration Guide

This document describes the migration of Claude Code hooks to a separate repository and how they are integrated into the MFC project.

## Overview

The Claude Code hooks have been extracted from `.claude/hooks/` into a dedicated GitLab repository:
- **Repository**: `gitlab-runner.tail301d0a.ts.net/uge/claude-code-hooks`
- **Local path**: `.claude/hooks-repo` (git submodule)
- **Symlink**: `.claude/hooks` -> `.claude/hooks-repo/hooks`

## Repository Structure

```
claude-code-hooks/
├── hooks/              # Main hook scripts
│   ├── pre_tool_use.py
│   ├── post_tool_use.py
│   ├── notification.py
│   ├── stop.py
│   ├── subagent_stop.py
│   ├── user_prompt_submit.py
│   ├── send_event.py
│   ├── enhanced_file_chunking.py
│   └── enhanced_issue_detection.py
├── utils/              # Utility modules
│   ├── constants.py
│   ├── enhanced_security_guardian.py
│   ├── git_guardian.py
│   ├── summarizer.py
│   ├── llm/           # LLM integrations
│   │   ├── anth.py
│   │   └── oai.py
│   └── tts/           # Text-to-speech
├── docs/               # Documentation
├── templates/          # Example configurations
├── pixi.toml          # Pixi environment configuration
└── pyproject.toml     # Python project configuration
```

## Integration with MFC Project

### Submodule Setup

The hooks repository is added as a git submodule:

```bash
git submodule add git@gitlab-runner.tail301d0a.ts.net:uge/claude-code-hooks.git .claude/hooks-repo
```

### Pixi Task Configuration

The pixi.toml includes tasks that set the PYTHONPATH to include the hooks-repo:

```toml
pre-tool-hook = "PYTHONPATH=/home/uge/mfc-project/.claude/hooks-repo python /home/uge/mfc-project/.claude/hooks-repo/hooks/pre_tool_use.py"
post-tool-hook = "PYTHONPATH=/home/uge/mfc-project/.claude/hooks-repo python /home/uge/mfc-project/.claude/hooks-repo/hooks/post_tool_use.py"
# ... other hook tasks
```

### Settings Configuration

The `.claude/settings.json` references hooks via pixi tasks:

```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "",
      "hooks": [{
        "type": "command",
        "command": "pixi run pre-tool-hook"
      }]
    }]
  }
}
```

## Updating Hooks

To update the hooks submodule to the latest version:

```bash
cd .claude/hooks-repo
git pull origin main
cd ../..
git add .claude/hooks-repo
git commit -m "Update hooks submodule to latest"
```

## Testing Hooks

Hook tests are located in `q-learning-mfcs/tests/hooks/`:

```bash
pixi run pytest q-learning-mfcs/tests/hooks/ -v
```

The tests add both hooks-repo paths to sys.path:
```python
sys.path.insert(0, str(project_root / '.claude' / 'hooks-repo'))
sys.path.insert(0, str(project_root / '.claude' / 'hooks-repo' / 'hooks'))
```

## Original Hooks Backup

The original hooks are preserved at `.claude/hooks-original/` for reference.

## Troubleshooting

### Import Errors

If you see import errors like `ModuleNotFoundError: No module named 'utils'`:
1. Ensure the PYTHONPATH includes `.claude/hooks-repo`
2. Check that the submodule is properly initialized: `git submodule update --init`

### Hook Not Executing

1. Verify pixi tasks are defined in pixi.toml
2. Check `.claude/settings.json` hook configuration
3. Run `pixi run pre-tool-hook` manually to test
