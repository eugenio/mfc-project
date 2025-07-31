# Documentation Synchronization Task

## Task Metadata
- **Created**: 2025-07-31
- **Type**: Git Integration & Documentation Sync
- **Integration**: git-commit-guardian, GitLab API
- **Purpose**: Synchronize documentation with code changes and version control

## Task Overview

Synchronizes documentation changes with the git-commit-guardian workflow, ensuring all documentation updates follow project standards and are properly tracked.

## Prerequisites

- Access to git-commit-guardian agent in `.claude/agents/`
- Integration with existing CLAUDE.md workflow patterns
- GitLab API access for issue management
- Valid git repository with staging capabilities

## Workflow Steps

### Step 1: Pre-Commit Documentation Validation

**Action**: Validate documentation before committing
**Integration Point**: git-commit-guardian pre-commit validation
**Input**: Staged documentation files
**Output**: Validation report and approval/rejection

```bash
# Automatic validation via existing hook system
# Called by git-commit-guardian before processing commits
```

**Validation Checks**:
- [ ] Metadata completeness (required fields present)
- [ ] Template structure compliance
- [ ] Link validation (internal links functional)
- [ ] Scientific accuracy preservation
- [ ] Formatting consistency
- [ ] Reference completeness

### Step 2: Documentation Change Classification

**Action**: Classify documentation changes for appropriate handling
**Input**: Git diff of documentation files
**Output**: Change classification and handling strategy

**Change Types**:
1. **Content Updates**: Scientific data, formulas, or technical content changes
2. **Structural Updates**: Template application, reorganization, metadata addition
3. **Formatting Updates**: Style, markup, or presentation changes
4. **Reference Updates**: Bibliography, citations, or link updates

**Classification Logic**:
```python
def classify_doc_changes(git_diff):
    """Classify documentation changes for appropriate processing."""
    change_types = []
    
    if contains_scientific_content(git_diff):
        change_types.append("content_update")
    if contains_metadata_changes(git_diff):
        change_types.append("structural_update") 
    if contains_formatting_only(git_diff):
        change_types.append("formatting_update")
    if contains_reference_changes(git_diff):
        change_types.append("reference_update")
    
    return change_types
```

### Step 3: Git-Commit-Guardian Integration

**Action**: Integrate with existing git-commit-guardian workflow
**Integration**: Extend existing `.claude/agents/git-commit-guardian`
**Input**: Classified documentation changes
**Output**: Standardized commits with appropriate messages

**Integration Pattern**:
```markdown
# Documentation-specific commit message patterns
docs: standardize {document_name} format
docs: update {document_name} content  
docs: add {document_name} documentation
docs: fix {document_name} references
```

**Commit Message Template**:
```
docs: {action} {document_name}

{change_description}
- {specific_change_1}
- {specific_change_2}
- {specific_change_3}

{scientific_validation_note}
{template_compliance_note}

🤖 Generated with [Claude Code](https://claude.ai/code)
Co-Authored-By: Claude <noreply@anthropic.com>
```

### Step 4: Documentation Staging Strategy

**Action**: Stage documentation files according to CLAUDE.md guidelines
**Rule**: "stage ONE file per commit" for modifications >25 lines
**Input**: Multiple documentation changes
**Output**: Staged commits following project guidelines

**Staging Logic**:
```python
def stage_documentation_changes(doc_changes):
    """Stage documentation changes according to project rules."""
    staged_commits = []
    
    for doc_file, changes in doc_changes.items():
        change_size = count_significant_changes(changes)
        
        if change_size > 25:  # Large changes - separate commits
            # Break into logical chunks
            chunks = chunk_changes_logically(changes)
            for chunk in chunks:
                staged_commits.append({
                    'files': [doc_file],
                    'changes': chunk,
                    'message': generate_commit_message(doc_file, chunk)
                })
        else:  # Small changes - single commit
            staged_commits.append({
                'files': [doc_file],
                'changes': changes,
                'message': generate_commit_message(doc_file, changes)
            })
    
    return staged_commits
```

### Step 5: GitLab Issue Integration

**Action**: Update GitLab issues related to documentation changes
**Integration**: GitLab API via existing hook infrastructure
**Input**: Committed documentation changes
**Output**: Updated GitLab issues and progress tracking

**Issue Update Logic**:
```python
def update_documentation_issues(commit_info, gitlab_client):
    """Update GitLab issues related to documentation changes."""
    
    # Find related issues
    related_issues = find_documentation_issues(commit_info['files'])
    
    for issue in related_issues:
        # Update issue with progress
        update_message = generate_progress_update(commit_info)
        gitlab_client.add_issue_comment(issue.id, update_message)
        
        # Check if issue can be closed
        if documentation_complete(issue, commit_info):
            gitlab_client.close_issue(issue.id, "Documentation completed")
```

### Step 6: Automated Documentation Tasks

**Action**: Trigger automated documentation maintenance tasks
**Integration**: Pixi task system
**Input**: Completed documentation commits
**Output**: Automated maintenance and validation

**Automated Tasks**:
```bash
# Triggered after documentation commits
pixi run validate-docs  # Validate documentation consistency
pixi run lint-markdown  # Lint markdown formatting  
pixi run build-docs     # Rebuild documentation artifacts
```

## Integration Configuration

### Git-Commit-Guardian Enhancement

**File**: `.claude/agents/git-commit-guardian`
**Enhancement**: Add documentation-specific validation

```yaml
# Addition to existing git-commit-guardian configuration
documentation_handling:
  enabled: true
  validation_rules:
    - metadata_completeness
    - template_compliance
    - scientific_accuracy_preservation
    - reference_validation
  
  commit_patterns:
    standardization: "docs: standardize {filename} format"
    content_update: "docs: update {filename} content"
    new_documentation: "docs: add {filename} documentation"
    reference_fix: "docs: fix {filename} references"
  
  staging_rules:
    large_changes_threshold: 25  # lines
    separate_commits: true
    logical_chunking: true
```

### CLAUDE.md Integration

**Enhancement**: Add documentation-specific guidelines to project CLAUDE.md

```markdown
## Documentation Guidelines

### Documentation Agent Integration
- Use `doc-agent` for documentation standardization tasks
- Follow standardized templates for all technical documentation
- Maintain scientific accuracy during format standardization
- Integrate documentation changes with git-commit-guardian workflow

### Documentation Commit Standards
- Stage documentation files individually for changes >25 lines
- Use standardized commit message formats for documentation
- Validate documentation before committing
- Update related GitLab issues with documentation progress
```

## Automation Workflows

### Hook Integration

**File**: `.claude/hooks/post_tool_use.py`
**Enhancement**: Add documentation synchronization

```python
def handle_documentation_changes(modified_files):
    """Handle documentation changes in post-tool-use hook."""
    doc_files = [f for f in modified_files if f.startswith('docs/')]
    
    if doc_files:
        # Validate documentation changes
        validation_results = validate_documentation_changes(doc_files)
        
        # Update GitLab issues
        update_related_issues(doc_files, validation_results)
        
        # Trigger automated tasks
        if validation_results['needs_rebuild']:
            trigger_pixi_task('build-docs')
```

### Pixi Task Integration

**File**: `pixi.toml`
**Enhancement**: Add documentation synchronization tasks

```toml
[tasks]
# Existing tasks...

# Documentation synchronization
sync-docs = "python .bmad-core/utils/doc-sync.py"
validate-docs-git = "python .bmad-core/utils/validate-docs-pre-commit.py"
update-doc-issues = "python .bmad-core/utils/update-gitlab-doc-issues.py"
```

## Error Handling and Recovery

### Validation Failures

**Scenario**: Documentation fails validation checks
**Response**: 
1. Block commit until validation passes
2. Provide specific error messages and guidance
3. Suggest corrections using documentation templates
4. Allow override with explicit user confirmation

### Git Integration Failures

**Scenario**: git-commit-guardian integration fails
**Response**:
1. Fall back to manual commit process
2. Log integration failure for investigation
3. Notify user of manual intervention required
4. Provide guidance for manual documentation commit

### GitLab API Failures

**Scenario**: GitLab issue updates fail
**Response**:
1. Log API failure details
2. Continue with git operations
3. Queue issue updates for retry
4. Notify user of issue tracking limitation

## Success Criteria

- [ ] Documentation changes integrate seamlessly with git-commit-guardian
- [ ] All documentation commits follow standardized message formats
- [ ] GitLab issues automatically updated with documentation progress
- [ ] Large documentation changes properly staged into separate commits
- [ ] Scientific accuracy preserved during all synchronization operations
- [ ] Automated validation prevents non-compliant documentation commits

## Output Deliverables

1. **Enhanced git-commit-guardian**: Extended to handle documentation changes
2. **Documentation Sync Utilities**: Python scripts for automation
3. **Pixi Task Integration**: Documentation-specific pixi tasks
4. **GitLab Issue Automation**: Automated issue tracking for documentation
5. **Validation Framework**: Pre-commit validation for documentation changes

## Post-Implementation Verification

### Integration Testing
- Test documentation commits through git-commit-guardian
- Verify GitLab issue updates
- Validate pixi task integration
- Confirm error handling behavior

### Workflow Testing
- Test large documentation changes (>25 lines)
- Verify staging and commit separation
- Test scientific accuracy preservation
- Validate template compliance checking

### User Experience Testing
- Ensure seamless integration with existing workflows
- Verify clear error messages and guidance
- Test recovery from failure scenarios
- Confirm minimal disruption to development workflow