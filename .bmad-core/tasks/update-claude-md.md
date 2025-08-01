# CLAUDE.md Standardization Task

## Task Metadata
- **Created**: 2025-07-31
- **Type**: CLAUDE.md File Enhancement & Standardization
- **Integration**: Documentation Agent, BMAD Framework
- **Purpose**: Enhance and standardize CLAUDE.md files across project and global scope

## Task Overview

Standardizes CLAUDE.md files to include documentation agent integration guidelines and ensures consistent documentation practices across the MFC project.

## Prerequisites

- Access to existing CLAUDE.md files (global and project-specific)
- Documentation agent infrastructure deployed
- BMAD framework integration capabilities
- GitLab API access for tracking changes

## Workflow Steps

### Step 1: Analyze Existing CLAUDE.md Files

**Action**: Analyze current CLAUDE.md configuration and guidelines
**Input**: Existing CLAUDE.md files in project and global scope
**Output**: Analysis report of current guidelines and integration points

**Analysis Targets**:
- Global CLAUDE.md at `/home/uge/.claude/CLAUDE.md`
- Project CLAUDE.md at `/home/uge/mfc-project/CLAUDE.md`
- Documentation-related guidelines already present
- Integration points with existing agents

**Assessment Criteria**:
```python
def analyze_claude_md_files():
    """Analyze existing CLAUDE.md files for documentation integration."""
    analysis = {
        'current_documentation_guidelines': [],
        'agent_integration_present': False,
        'git_workflow_integration': False,
        'standardization_rules': [],
        'enhancement_opportunities': []
    }
    return analysis
```

### Step 2: Design Enhanced Documentation Guidelines

**Action**: Design comprehensive documentation guidelines for CLAUDE.md integration
**Input**: Current CLAUDE.md analysis
**Output**: Enhanced documentation section for CLAUDE.md files

**Enhanced Guidelines Structure**:
```markdown
## Documentation Standards and Automation

### Documentation Agent Integration
- Use `doc-agent` (Alexandra) for documentation standardization tasks
- Apply standardized templates for all technical documentation
- Maintain scientific accuracy during format standardization
- Integrate documentation changes with git-commit-guardian workflow

### Documentation Quality Standards
- Apply consistent metadata headers to all documents
- Use standardized templates for document types (technical-spec, api-doc, user-guide, architecture)
- Preserve all scientific references and technical accuracy
- Validate documentation against quality standards before committing

### Automated Documentation Workflows
- Stage documentation files individually for changes >25 lines
- Use standardized commit message formats for documentation
- Update related GitLab issues with documentation progress
- Trigger automated validation and quality checks

### Documentation Types and Templates
- **Technical Specifications**: Use technical-spec-template.md
- **API Documentation**: Use api-doc-template.md  
- **User Guides**: Use user-guide-template.md
- **Architecture Documentation**: Use architecture-doc-template.md
```

### Step 3: Update Global CLAUDE.md

**Action**: Enhance global CLAUDE.md with documentation agent integration
**Target**: `/home/uge/.claude/CLAUDE.md`
**Integration**: Add documentation standards without disrupting existing guidelines

**Enhancement Strategy**:
1. Preserve all existing user guidelines
2. Add documentation agent integration section
3. Include standardization workflow guidelines
4. Maintain backward compatibility with existing patterns

**Implementation Approach**:
```python
def update_global_claude_md():
    """Update global CLAUDE.md with documentation integration."""
    
    # Read existing content
    existing_content = read_file('/home/uge/.claude/CLAUDE.md')
    
    # Prepare documentation section
    doc_section = generate_documentation_guidelines()
    
    # Insert documentation section appropriately
    enhanced_content = insert_documentation_section(
        existing_content, 
        doc_section
    )
    
    # Validate integration
    validate_claude_md_integration(enhanced_content)
    
    return enhanced_content
```

### Step 4: Update Project CLAUDE.md

**Action**: Enhance project-specific CLAUDE.md with MFC-specific documentation guidelines
**Target**: `/home/uge/mfc-project/CLAUDE.md`
**Integration**: Add MFC-specific documentation standards and scientific accuracy requirements

**MFC-Specific Enhancements**:
```markdown
### MFC Documentation Standards

#### Scientific Documentation Requirements
- Preserve all mathematical formulas and scientific notation
- Maintain literature references and citations
- Validate parameter ranges and units
- Ensure experimental data accuracy

#### MFC-Specific Templates
- Use MFC parameter documentation standards
- Include validation data and literature sources
- Apply consistent unit formats (V, A, W, °C, g/L, S/m)
- Maintain biofilm and electrochemical terminology consistency

#### Integration with Existing Workflows
- Coordinate with existing pixi environment management
- Follow established git workflow patterns
- Integrate with existing test infrastructure
- Maintain compatibility with protein modeling components
```

### Step 5: Validate CLAUDE.md Integration

**Action**: Validate enhanced CLAUDE.md files for functionality and compatibility
**Input**: Updated CLAUDE.md files
**Output**: Validation report and integration verification

**Validation Checks**:
- [ ] Syntax and formatting validation
- [ ] Integration with existing agent workflows
- [ ] Compatibility with git-commit-guardian
- [ ] Documentation agent activation testing
- [ ] No disruption to existing functionality

**Validation Script**:
```python
def validate_claude_md_integration():
    """Validate CLAUDE.md integration with documentation agent."""
    
    validation_results = {
        'syntax_valid': validate_claude_md_syntax(),
        'agent_integration': test_agent_activation(),
        'git_integration': test_git_workflow_compatibility(),
        'backward_compatibility': test_existing_functionality()
    }
    
    return validation_results
```

### Step 6: Deploy and Test Integration

**Action**: Deploy enhanced CLAUDE.md files and test documentation agent integration
**Integration**: Live testing with documentation agent activation
**Output**: Deployment verification and functionality confirmation

**Deployment Steps**:
1. Backup existing CLAUDE.md files
2. Deploy enhanced versions
3. Test documentation agent activation
4. Verify git-commit-guardian integration
5. Confirm existing workflow compatibility

**Integration Testing**:
```bash
# Test documentation agent activation
claude-code activate doc-agent

# Test documentation standardization
claude-code standardize-docs --dry-run

# Test git integration
git status
git add .
git commit -m "test: documentation agent integration"
```

## Output Specifications

### Enhanced Global CLAUDE.md Structure
```markdown
# Existing global guidelines (preserved)
[...existing content...]

## Documentation Standards and Automation
- Documentation agent integration guidelines
- Quality standards and validation rules  
- Automated workflow integration
- Template usage guidelines

## Documentation Agent Integration
- Agent activation procedures
- Standardization workflow steps
- Git integration requirements
- Quality validation processes
```

### Enhanced Project CLAUDE.md Structure
```markdown
# Existing project guidelines (preserved)
[...existing content...]

## MFC Documentation Standards
- Scientific accuracy requirements
- MFC-specific templates and standards
- Parameter documentation guidelines
- Literature reference standards

## Documentation Automation
- Pixi task integration
- Git workflow coordination
- GitLab issue management
- Quality validation procedures
```

## Success Criteria

- [ ] Global CLAUDE.md enhanced with documentation agent integration
- [ ] Project CLAUDE.md updated with MFC-specific documentation standards
- [ ] Documentation agent successfully activated through CLAUDE.md guidelines
- [ ] Git-commit-guardian integration working with documentation workflows
- [ ] Existing functionality preserved and unaffected
- [ ] Scientific accuracy requirements properly integrated

## Error Handling

### CLAUDE.md Syntax Errors
**Response**: Validate syntax before deployment, provide correction guidance

### Integration Conflicts
**Response**: Preserve existing functionality, resolve conflicts through careful integration

### Agent Activation Failures
**Response**: Verify agent infrastructure, provide fallback documentation procedures

## Maintenance Requirements

- Regular review of documentation guidelines effectiveness
- Updates to match evolving project requirements
- Synchronization with BMAD framework enhancements
- Integration testing with new agent capabilities

---

**Integration Points**:
- Documentation Agent (Alexandra)
- git-commit-guardian workflow
- GitLab API integration
- Pixi task system
- BMAD framework standards