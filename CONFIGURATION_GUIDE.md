# Configuration System Guide
## Overview

The MFC project uses a sophisticated hierarchical configuration system that supports multiple profiles, environment variable substitution, validation, and runtime overrides. This guide covers all aspects of configuration management.
## Configuration Structure

### Main Configuration File
**Location**: `.claude/settings.json`

```json
{
  "hooks": {
    "PreToolUse": [...],
    "PostToolUse": [...],
    "Notification": [...],
    "Stop": [...],
    "SubagentStop": [...],
    "PreCompact": [...],
    "UserPromptSubmit": [...]
  },
  "gitlab": {
    "enabled": true,
    "url": "https://gitlab.com",
    "project_id": null,
    "default_branch": "main",
    "features": {
      "auto_issue_on_test_failure": true,
      "auto_issue_on_build_failure": true,
      "auto_issue_on_performance_regression": true,
      "auto_issue_on_documentation_gap": true,
      "auto_issue_on_hook_failure": true,
      "auto_mr_on_multiple_commits": false,
      "commit_threshold_for_mr": 5
    }
  },
  "chunked_file_creation": {
    "enabled": true,
    "max_new_file_lines": 50,
    "max_lines_per_chunk": 25,
    "min_segments_for_chunking": 3,
    "commit_message_prefix": "Auto-commit: ",
    "supported_extensions": [".py", ".js", ".ts", ".jsx", ".tsx", ".mojo", ".🔥", ".md", ".markdown"]
  }
}
```
## Hook Configuration

### Pre-Tool Use Hooks
Executes before any tool operation:
```json
"PreToolUse": [
  {
    "matcher": "",
    "hooks": [
      {
        "type": "command",
        "command": "pixi run /home/uge/mfc-project/.claude/hooks/pre_tool_use.py"
      },
      {
        "type": "command",
        "command": "pixi run /home/uge/mfc-project/.claude/hooks/send_event.py --source-app mfc-project --event-type PreToolUse --summarize"
      }
    ]
  }
]
```

### Hook Types
- **PreToolUse**: Before tool execution
- **PostToolUse**: After tool execution
- **Notification**: System notifications
- **Stop**: Session termination
- **SubagentStop**: Subagent termination
- **PreCompact**: Before compacting conversation
- **UserPromptSubmit**: User input processing
## GitLab Integration

### Configuration Options
```json
"gitlab": {
  "enabled": true,
  "url": "https://gitlab.com",
  "project_id": null,
  "default_branch": "main",
  "features": {
    "auto_issue_on_test_failure": true,
    "auto_issue_on_build_failure": true,
    "auto_issue_on_performance_regression": true,
    "auto_issue_on_documentation_gap": true,
    "auto_issue_on_hook_failure": true,
    "auto_mr_on_multiple_commits": false,
    "commit_threshold_for_mr": 5
  }
}
```

### Feature Flags
- **auto_issue_on_test_failure**: Create issues for test failures
- **auto_issue_on_build_failure**: Create issues for build failures
- **auto_issue_on_performance_regression**: Track performance issues
- **auto_issue_on_documentation_gap**: Flag missing documentation
- **auto_issue_on_hook_failure**: Report hook execution failures
- **auto_mr_on_multiple_commits**: Auto-create merge requests
- **commit_threshold_for_mr**: Number of commits before MR
## File Chunking Configuration

### Settings
```json
"chunked_file_creation": {
  "enabled": true,
  "max_new_file_lines": 50,
  "max_lines_per_chunk": 25,
  "min_segments_for_chunking": 3,
  "commit_message_prefix": "Auto-commit: ",
  "supported_extensions": [".py", ".js", ".ts", ".jsx", ".tsx", ".mojo", ".🔥", ".md", ".markdown"]
}
```

### Parameters
- **enabled**: Enable/disable chunking system
- **max_new_file_lines**: Minimum lines to trigger chunking
- **max_lines_per_chunk**: Maximum lines per chunk
- **min_segments_for_chunking**: Minimum logical segments required
- **commit_message_prefix**: Prefix for auto-generated commits
- **supported_extensions**: File types that support chunking
## Q-Learning Configuration

### Profile System
**Location**: `q-learning-mfcs/src/config/`

#### Conservative Profile
```yaml
# conservative_control.yaml
name: "Conservative Control Profile"
description: "Stable, conservative parameters for long-term operation"

biological:
  species: "geobacter"
  substrate: "acetate"
  max_growth_rate: 0.35  # h^-1 (conservative)
  half_saturation_constant: 3.0  # mM (higher for stability)
  yield_coefficient: 0.45  # g_biomass/g_substrate
  decay_rate: 0.015  # h^-1
  max_biofilm_thickness: 2.5  # μm (limited)

control:
  q_learning:
    learning_rate: 0.05  # Conservative learning
    discount_factor: 0.99  # Long-term focus
    epsilon_initial: 0.1  # Limited exploration
    epsilon_decay: 0.999
    epsilon_min: 0.01
    
  flow_control:
    min_flow_rate: 0.3  # mL/h
    max_flow_rate: 0.8  # mL/h
    flow_step_size: 0.1
```

#### Research Profile
```yaml
# research_optimization.yaml
name: "Research Optimization Profile"
description: "Aggressive parameters for maximum performance"

biological:
  species: "geobacter"
  substrate: "acetate"
  max_growth_rate: 0.46  # h^-1 (literature maximum)
  half_saturation_constant: 2.5  # mM (optimized)
  yield_coefficient: 0.5  # g_biomass/g_substrate
  decay_rate: 0.01  # h^-1
  max_biofilm_thickness: 3.0  # μm (optimal)

control:
  q_learning:
    learning_rate: 0.1  # Faster learning
    discount_factor: 0.95  # Balanced
    epsilon_initial: 0.3  # More exploration
    epsilon_decay: 0.995
    epsilon_min: 0.05
```
## Environment Variables

### Supported Variables
```bash
# GitLab credentials
export GITLAB_TOKEN="your-token-here"
export GITLAB_PROJECT_ID="12345"

# GPU selection
export CUDA_VISIBLE_DEVICES="0"
export ROCM_VISIBLE_DEVICES="0"

# Configuration overrides
export MFC_CONFIG_PROFILE="research"
export MFC_USE_GPU="true"
export MFC_LOG_LEVEL="DEBUG"
```

### Variable Substitution in YAML
```yaml
gitlab:
  token: ${GITLAB_TOKEN}
  project_id: ${GITLAB_PROJECT_ID}
  
compute:
  gpu_device: ${CUDA_VISIBLE_DEVICES:-0}
  use_gpu: ${MFC_USE_GPU:-true}
```
## Configuration Manager API

### Loading Configurations
```python
from config.config_manager import ConfigurationManager

# Initialize manager
config = ConfigurationManager()

# Load specific profile
config.load_profile('research')

# Load from file
config.load_profile_from_file('custom', 'path/to/config.yaml')

# Get configuration section
bio_params = config.get_configuration('biological')
```

### Runtime Overrides
```python
# Override specific values
config.set_override('biological.max_growth_rate', 0.5)

# Bulk overrides
overrides = {
    'biological': {
        'max_growth_rate': 0.5,
        'decay_rate': 0.008
    }
}
config.apply_overrides(overrides)
```

### Validation
```python
# Validate configuration
try:
    config.validate_configuration()
except ConfigurationError as e:
    print(f"Invalid configuration: {e}")
```
## Configuration Schema

### Biological Parameters
```yaml
biological:
  species: str  # geobacter, shewanella, mixed
  substrate: str  # acetate, lactate, glucose
  max_growth_rate: float  # 0.1-1.0 h^-1
  half_saturation_constant: float  # 0.5-10.0 mM
  yield_coefficient: float  # 0.1-0.8 g/g
  decay_rate: float  # 0.001-0.1 h^-1
  max_biofilm_thickness: float  # 0.5-5.0 μm
```

### Control Parameters
```yaml
control:
  q_learning:
    learning_rate: float  # 0.001-0.5
    discount_factor: float  # 0.8-0.99
    epsilon_initial: float  # 0.05-0.5
    epsilon_decay: float  # 0.99-0.999
    epsilon_min: float  # 0.01-0.1
    
  pid:
    kp: float  # 0.1-10.0
    ki: float  # 0.01-1.0
    kd: float  # 0.001-0.1
```

### Visualization Parameters
```yaml
visualization:
  plot_interval: int  # 1-100 steps
  save_plots: bool
  plot_format: str  # png, svg, pdf
  figure_dpi: int  # 72-300
  theme: str  # default, dark, publication
```
## Best Practices

### 1. Profile Selection
- Use **conservative** for production systems
- Use **research** for performance optimization
- Use **precision** for laboratory validation

### 2. Configuration Hierarchy
```
1. Default values (hardcoded)
2. Profile configuration (YAML)
3. Environment variables
4. Runtime overrides
```

### 3. Version Control
- Commit configuration files
- Document parameter changes
- Use meaningful profile names

### 4. Validation
Always validate after loading:
```python
config = ConfigurationManager()
config.load_profile('custom')
if not config.validate_configuration():
    raise ValueError("Invalid configuration")
```
## Common Configuration Tasks

### Creating Custom Profile
```python
# Create custom configuration
custom_config = {
    'name': 'my_custom_profile',
    'biological': {
        'species': 'geobacter',
        'max_growth_rate': 0.4
    },
    'control': {
        'q_learning': {
            'learning_rate': 0.08
        }
    }
}

# Save to file
import yaml
with open('configs/my_custom.yaml', 'w') as f:
    yaml.dump(custom_config, f)
```

### Switching Profiles at Runtime
```python
# Start with conservative
model = MFCModel(config_profile='conservative')

# Switch to research after stabilization
if model.is_stable():
    config.load_profile('research')
    model.update_configuration(config)
```

### Debugging Configuration Issues
```python
# Enable debug logging
config.set_log_level('DEBUG')

# Print effective configuration
print(config.get_effective_configuration())

# Check inheritance chain
print(config.get_inheritance_chain())
```
## Configuration Files Reference

### Hook Scripts
- `.claude/hooks/pre_tool_use.py`: Pre-execution monitoring
- `.claude/hooks/post_tool_use.py`: Post-execution logging
- `.claude/hooks/enhanced_file_chunking.py`: File chunking logic
- `.claude/hooks/gitlab_issue_manager.py`: Issue automation
- `.claude/hooks/send_event.py`: Event logging

### Q-Learning Configs
- `configs/conservative_control.yaml`: Stable operation
- `configs/research_optimization.yaml`: Performance focus
- `configs/precision_control.yaml`: High accuracy
- `configs/species/`: Species-specific parameters
- `configs/substrates/`: Substrate-specific parameters
## Troubleshooting

### Configuration Not Loading
```bash
# Check file exists
ls -la q-learning-mfcs/src/config/

# Verify YAML syntax
python -c "import yaml; yaml.load(open('config.yaml'))"

# Check permissions
chmod 644 config.yaml
```

### Environment Variables Not Working
```bash
# Verify variables are set
env | grep MFC_

# Check variable substitution
python -c "import os; print(os.path.expandvars('${MFC_CONFIG_PROFILE}'))"
```

### Validation Errors
```python
# Get detailed validation errors
from config.validation_schemas import validate_biological_params

try:
    validate_biological_params(bio_config)
except ValueError as e:
    print(f"Validation error: {e}")
```

This configuration guide provides comprehensive coverage of all configuration aspects in the MFC project. The hierarchical system with validation ensures robust and flexible configuration management.