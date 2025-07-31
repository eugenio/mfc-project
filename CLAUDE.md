- Before running any Mojo file, run `mojo format` to the file in question, and fix obvious errors
- From now stage one file per commit, and if there's any modification of any file done by you or me with more than 50 lines, do a commit for each 50 lines of modifications
- From now on write all the commands you want to execute, in a way that all the commands are visible in the prompt area, and if a command is blocked by the hooks you have to stop, print the command in full and ask me to review it. If I think it's ok, I'll execute the command in another terminal and then I'll tell you to continue
- Do not use hardcoded values unless it's pysical or biological constants derived from literature
- If you write in a file metadata "created_at" you should add today's date. If you modify an existing file metadata use the tag "last_modified_at"

## MFC Documentation Standards

### Documentation Agent Integration
- Use `doc-agent` (Alexandra) for documentation standardization tasks
- Apply standardized templates for all technical documentation  
- Maintain scientific accuracy during format standardization
- Integrate documentation changes with git-commit-guardian workflow

### Scientific Documentation Requirements
- Preserve all mathematical formulas and scientific notation
- Maintain literature references and citations
- Validate parameter ranges and units (V, A, W, °C, g/L, S/m)
- Ensure experimental data accuracy

### MFC-Specific Templates
- Use MFC parameter documentation standards
- Include validation data and literature sources
- Apply consistent unit formats for electrochemical parameters
- Maintain biofilm and electrochemical terminology consistency

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
