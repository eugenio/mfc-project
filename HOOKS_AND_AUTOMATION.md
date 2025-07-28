# Hooks and Automation System
## Overview

The MFC project features a sophisticated automation system built around Claude Code hooks. This system provides intelligent file handling, automated issue tracking, event logging, and development workflow enhancements.
## Hook System Architecture

### Hook Types and Execution Order

```
User Action
    │
    ├─► UserPromptSubmit Hook
    │       ├─► Log user input
    │       └─► Send event
    │
    ├─► PreToolUse Hook
    │       ├─► Validate operation
    │       ├─► Check thresholds
    │       ├─► Chunk large files
    │       └─► Send event
    │
    ├─► Tool Execution
    │
    ├─► PostToolUse Hook
    │       ├─► Log results
    │       ├─► Check for issues
    │       └─► Send event
    │
    └─► Notification Hook (if needed)
            ├─► Create notifications
            └─► Send event
```
