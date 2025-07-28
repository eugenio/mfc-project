#!/usr/bin/env python3
"""
Enhanced file chunking system with improved commit messages for large file creation.

This module provides intelligent code segmentation and incremental file building
to break large file creation into smaller, logical commits with meaningful messages
and comprehensive final summaries.

Supports Python, JavaScript/TypeScript, Mojo, and Markdown files.
"""

import os
import re
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Any

def analyze_code_structure(content: str, file_path: str) -> Dict[str, Any]:
    """
    Analyze code structure to identify logical segments for chunking.
    
    Args:
        content: File content to analyze
        file_path: Path to the file for type detection
        
    Returns:
        dict: Code structure analysis with segments
    """
    file_ext = Path(file_path).suffix.lower()
    lines = content.splitlines()
    
    structure = {
        "file_type": file_ext,
        "total_lines": len(lines),
        "segments": [],
        "imports": [],
        "classes": [],
        "functions": [],
        "comments": [],
        "constants": []
    }
    
    if file_ext == '.py':
        structure.update(_analyze_python_structure(lines))
    elif file_ext in ['.js', '.ts', '.jsx', '.tsx']:
        structure.update(_analyze_javascript_structure(lines))
    elif file_ext in ['.mojo', '.🔥']:
        structure.update(_analyze_mojo_structure(lines))
    elif file_ext in ['.md', '.markdown']:
        structure.update(_analyze_markdown_structure(lines))
    else:
        structure.update(_analyze_generic_structure(lines))
    
    return structure

def _analyze_python_structure(lines: List[str]) -> Dict[str, Any]:
    """Analyze Python code structure for logical chunking."""
    segments = []
    imports = []
    classes = []
    functions = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Module docstring at the beginning
        if i < 5 and (line.startswith('"""') or line.startswith("'''")):
            start_i = i
            quote = line[:3]
            # Find end of docstring
            if line.count(quote) < 2:  # Multi-line docstring
                i += 1
                while i < len(lines) and quote not in lines[i]:
                    i += 1
            
            segments.append({
                "type": "module_docstring",
                "start_line": start_i,
                "end_line": i,
                "priority": 1,
                "description": "Module documentation and metadata"
            })
        
        # Imports section
        elif line.startswith(('import ', 'from ')):
            start_i = i
            while i < len(lines) and (lines[i].strip().startswith(('import ', 'from ')) or lines[i].strip() == ''):
                if lines[i].strip():
                    imports.append(lines[i].strip())
                i += 1
            i -= 1  # Back up one
            
            segments.append({
                "type": "imports",
                "start_line": start_i,
                "end_line": i,
                "priority": 2,
                "description": f"Import statements ({len(imports)} imports)"
            })
        
        # Class definitions
        elif line.startswith('class '):
            class_name = line.split('class ')[1].split('(')[0].split(':')[0].strip()
            start_i = i
            
            # Find end of class (next class/function at root level or end of file)
            i += 1
            indent_level = len(lines[start_i]) - len(lines[start_i].lstrip())
            while i < len(lines):
                current_line = lines[i]
                current_indent = len(current_line) - len(current_line.lstrip())
                
                if (current_line.strip() and 
                    current_indent <= indent_level and 
                    (current_line.strip().startswith(('class ', 'def ', '@')) or 
                     not current_line.startswith(' '))):
                    break
                i += 1
            i -= 1  # Back up one
            
            classes.append(class_name)
            segments.append({
                "type": "class",
                "name": class_name,
                "start_line": start_i,
                "end_line": i,
                "priority": 3,
                "description": f"Class {class_name} definition"
            })
        
        # Function definitions (at module level)
        elif line.startswith('def ') and not line.startswith('    '):
            func_name = line.split('def ')[1].split('(')[0].strip()
            start_i = i
            
            # Find end of function
            i += 1
            while i < len(lines):
                current_line = lines[i]
                if (current_line.strip() and 
                    not current_line.startswith(' ') and 
                    current_line.strip().startswith(('def ', 'class ', '@'))):
                    break
                i += 1
            i -= 1  # Back up one
            
            functions.append(func_name)
            segments.append({
                "type": "function",
                "name": func_name,
                "start_line": start_i,
                "end_line": i,
                "priority": 4,
                "description": f"Function {func_name}"
            })
        
        # Constants and module-level variables
        elif ('=' in line and line[0].isupper() and 
              not line.startswith(' ') and 
              not line.startswith('#')):
            const_name = line.split('=')[0].strip()
            segments.append({
                "type": "constant",
                "name": const_name,
                "start_line": i,
                "end_line": i,
                "priority": 2,
                "description": f"Constant {const_name}"
            })
        
        i += 1
    
    return {
        "segments": segments,
        "imports": imports,
        "classes": classes,
        "functions": functions
    }

def _analyze_markdown_structure(lines: List[str]) -> Dict[str, Any]:
    """Analyze Markdown structure for logical chunking."""
    segments = []
    headings = []
    code_blocks = []
    tables = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Title/Header (highest priority - goes first)
        if i < 5 and line.startswith('# ') and not line.startswith('## '):
            title = line[2:].strip()
            segments.append({
                "type": "title",
                "name": title,
                "start_line": i,
                "end_line": i,
                "priority": 1,
                "description": f"Document title: {title}"
            })
            headings.append(title)
        
        # H2 sections (major sections)
        elif line.startswith('## '):
            section_title = line[3:].strip()
            start_i = i
            
            # Find end of section (next H2 or end of file)
            i += 1
            while i < len(lines):
                current_line = lines[i].strip()
                if current_line.startswith('## '):
                    break
                i += 1
            i -= 1  # Back up one
            
            segments.append({
                "type": "section",
                "name": section_title,
                "start_line": start_i,
                "end_line": i,
                "priority": 2,
                "description": f"Section: {section_title}"
            })
            headings.append(section_title)
        
        # H3 subsections
        elif line.startswith('### '):
            subsection_title = line[4:].strip()
            start_i = i
            
            # Find end of subsection (next H3/H2 or end of file)
            i += 1
            while i < len(lines):
                current_line = lines[i].strip()
                if current_line.startswith(('### ', '## ')):
                    break
                i += 1
            i -= 1  # Back up one
            
            segments.append({
                "type": "subsection",
                "name": subsection_title,
                "start_line": start_i,
                "end_line": i,
                "priority": 3,
                "description": f"Subsection: {subsection_title}"
            })
            headings.append(subsection_title)
        
        # Code blocks
        elif line.startswith('```'):
            start_i = i
            language = line[3:].strip() if len(line) > 3 else "code"
            
            # Find end of code block
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                i += 1
            
            if i < len(lines):  # Found closing ```
                segments.append({
                    "type": "code_block",
                    "name": language,
                    "start_line": start_i,
                    "end_line": i,
                    "priority": 4,
                    "description": f"Code block ({language})"
                })
                code_blocks.append(language)
        
        # Tables (markdown tables starting with |)
        elif line.startswith('|') and '|' in line[1:]:
            start_i = i
            
            # Find end of table
            while i < len(lines) and lines[i].strip().startswith('|'):
                i += 1
            i -= 1  # Back up one
            
            segments.append({
                "type": "table",
                "start_line": start_i,
                "end_line": i,
                "priority": 4,
                "description": "Table data"
            })
            tables.append(f"table_{len(tables)+1}")
        
        i += 1
    
    return {
        "segments": segments,
        "headings": headings,
        "code_blocks": code_blocks,
        "tables": tables
    }

def _analyze_javascript_structure(lines: List[str]) -> Dict[str, Any]:
    """Analyze JavaScript/TypeScript code structure."""
    segments = []
    imports = []
    classes = []
    functions = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Import statements
        if line.startswith(('import ', 'export ', 'const ')) and 'require' in line:
            start_i = i
            while i < len(lines) and (lines[i].strip().startswith(('import ', 'export ')) or lines[i].strip() == ''):
                if lines[i].strip():
                    imports.append(lines[i].strip())
                i += 1
            i -= 1
            
            segments.append({
                "type": "imports",
                "start_line": start_i,
                "end_line": i,
                "priority": 2,
                "description": f"Import/export statements"
            })
        
        # Class definitions
        elif line.startswith('class '):
            class_name = line.split('class ')[1].split(' ')[0].split('{')[0].strip()
            classes.append(class_name)
            # Simple implementation - could be enhanced for proper brace matching
            segments.append({
                "type": "class",
                "name": class_name,
                "start_line": i,
                "end_line": i + 10,  # Simplified
                "priority": 3,
                "description": f"Class {class_name}"
            })
        
        i += 1
    
    return {
        "segments": segments,
        "imports": imports,
        "classes": classes,
        "functions": functions
    }

def _analyze_mojo_structure(lines: List[str]) -> Dict[str, Any]:
    """Analyze Mojo code structure."""
    segments = []
    imports = []
    structs = []
    functions = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Import statements
        if line.startswith('from ') and 'import' in line:
            imports.append(line)
            segments.append({
                "type": "import",
                "start_line": i,
                "end_line": i,
                "priority": 2,
                "description": "Import statement"
            })
        
        # Function definitions
        elif line.startswith('fn '):
            func_name = line.split('fn ')[1].split('(')[0].strip()
            functions.append(func_name)
            segments.append({
                "type": "function",
                "name": func_name,
                "start_line": i,
                "end_line": i + 5,  # Simplified
                "priority": 4,
                "description": f"Function {func_name}"
            })
        
        # Struct definitions
        elif line.startswith('struct '):
            struct_name = line.split('struct ')[1].split(':')[0].split('(')[0].strip()
            structs.append(struct_name)
            segments.append({
                "type": "struct",
                "name": struct_name,
                "start_line": i,
                "end_line": i + 10,  # Simplified
                "priority": 3,
                "description": f"Struct {struct_name}"
            })
        
        i += 1
    
    return {
        "segments": segments,
        "imports": imports,
        "structs": structs,
        "functions": functions
    }

def _analyze_generic_structure(lines: List[str]) -> Dict[str, Any]:
    """Generic structure analysis for unknown file types."""
    segments = []
    
    # Simple line-based chunking for unknown types
    chunk_size = 25
    for i in range(0, len(lines), chunk_size):
        end_line = min(i + chunk_size - 1, len(lines) - 1)
        segments.append({
            "type": "generic_chunk",
            "start_line": i,
            "end_line": end_line,
            "priority": 3,
            "description": f"Code chunk {i//chunk_size + 1}"
        })
    
    return {"segments": segments}

def create_logical_chunks(structure: Dict[str, Any], content: str, max_lines_per_chunk: int = 25) -> List[Dict[str, Any]]:
    """
    Create logical chunks from code structure analysis.
    
    Args:
        structure: Code structure analysis
        content: Original file content
        max_lines_per_chunk: Maximum lines per chunk
        
    Returns:
        List of chunks with content and metadata
    """
    lines = content.splitlines()
    chunks = []
    
    # Sort segments by priority and start line
    segments = sorted(structure.get("segments", []), key=lambda x: (x["priority"], x["start_line"]))
    
    current_chunk_lines = []
    current_chunk_segments = []
    
    for segment in segments:
        start_line = segment["start_line"]
        end_line = segment["end_line"]
        segment_lines = lines[start_line:end_line + 1]
        
        # If adding this segment would exceed max lines, finalize current chunk
        if (len(current_chunk_lines) + len(segment_lines) > max_lines_per_chunk and 
            current_chunk_lines):
            
            chunks.append({
                "content": "\n".join(current_chunk_lines),
                "segments": current_chunk_segments,
                "line_count": len(current_chunk_lines),
                "description": _generate_chunk_description(current_chunk_segments)
            })
            
            current_chunk_lines = []
            current_chunk_segments = []
        
        # Add segment to current chunk
        current_chunk_lines.extend(segment_lines)
        current_chunk_segments.append(segment)
    
    # Add final chunk if there's remaining content
    if current_chunk_lines:
        chunks.append({
            "content": "\n".join(current_chunk_lines),
            "segments": current_chunk_segments,
            "line_count": len(current_chunk_lines),
            "description": _generate_chunk_description(current_chunk_segments)
        })
    
    return chunks

def _generate_chunk_description(segments: List[Dict[str, Any]]) -> str:
    """Generate descriptive text for a chunk based on its segments."""
    if not segments:
        return "content"
    
    types = {}
    names = []
    
    for segment in segments:
        seg_type = segment["type"]
        types[seg_type] = types.get(seg_type, 0) + 1
        
        if "name" in segment:
            names.append(segment["name"])
    
    parts = []
    
    # Markdown-specific descriptions
    if types.get("title"):
        parts.append("document title")
    
    if types.get("section"):
        section_names = [name for seg in segments if seg["type"] == "section" for name in [seg.get("name")] if name]
        if section_names:
            parts.append(f"section: {', '.join(section_names[:2])}")
        else:
            parts.append(f"{types['section']} sections")
    
    if types.get("subsection"):
        subsection_names = [name for seg in segments if seg["type"] == "subsection" for name in [seg.get("name")] if name]
        if subsection_names:
            parts.append(f"subsection: {', '.join(subsection_names[:2])}")
        else:
            parts.append(f"{types['subsection']} subsections")
    
    if types.get("code_block"):
        parts.append(f"{types['code_block']} code blocks")
    
    if types.get("table"):
        parts.append(f"{types['table']} tables")
    
    # Code-specific descriptions
    if types.get("imports"):
        parts.append(f"{types['imports']} imports")
    
    if types.get("class"):
        if names:
            class_names = [name for seg in segments if seg["type"] == "class" for name in [seg.get("name")] if name]
            parts.append(f"class {', '.join(class_names[:2])}")
        else:
            parts.append(f"{types['class']} classes")
    
    if types.get("function"):
        if len(names) <= 2:
            func_names = [name for seg in segments if seg["type"] == "function" for name in [seg.get("name")] if name]
            parts.append(f"functions: {', '.join(func_names)}")
        else:
            parts.append(f"{types['function']} functions")
    
    if types.get("constant"):
        parts.append(f"{types['constant']} constants")
    
    if types.get("module_docstring"):
        parts.append("module documentation")
    
    return " | ".join(parts) if parts else "content"

def _generate_detailed_commit_message(chunk: Dict[str, Any], chunk_index: int, total_chunks: int, file_name: str, config: Dict[str, Any]) -> str:
    """Generate a detailed, meaningful commit message for a chunk."""
    prefix = config.get('commit_message_prefix', 'Auto-commit: ')
    
    # Extract key information from the chunk
    segments = chunk.get('segments', [])
    line_count = chunk['line_count']
    
    # Build commit message parts
    header = f"{prefix}chunk {chunk_index + 1}/{total_chunks} - {file_name}"
    
    # Generate detailed description based on segment types
    segment_details = []
    
    for segment in segments:
        seg_type = segment['type']
        seg_desc = segment.get('description', '')
        
        if seg_type == 'class' and 'name' in segment:
            segment_details.append(f"Added class '{segment['name']}' with its methods")
        elif seg_type == 'function' and 'name' in segment:
            segment_details.append(f"Implemented function '{segment['name']}'")
        elif seg_type == 'imports':
            segment_details.append("Added import statements and dependencies")
        elif seg_type == 'module_docstring':
            segment_details.append("Added module documentation and metadata")
        elif seg_type == 'section' and 'name' in segment:
            segment_details.append(f"Added section '{segment['name']}'")
        elif seg_type == 'title':
            segment_details.append("Added document title and introduction")
        elif seg_type == 'code_block':
            lang = segment.get('name', 'code')
            segment_details.append(f"Added {lang} code example")
        elif seg_type == 'table':
            segment_details.append("Added data table")
        elif seg_type == 'constant' and 'name' in segment:
            segment_details.append(f"Defined constant '{segment['name']}'")
    
    # Fallback to generic description if no specific details
    if not segment_details:
        segment_details.append(chunk['description'])
    
    # Combine into final message
    message = f"{header} ({line_count} lines)\n\n"
    message += "This chunk includes:\n"
    for detail in segment_details[:5]:  # Limit to 5 items
        message += f"- {detail}\n"
    
    if len(segment_details) > 5:
        message += f"- ... and {len(segment_details) - 5} more items\n"
    
    return message.strip()

def _generate_final_summary_message(chunks: List[Dict[str, Any]], file_path: str, structure: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Generate a comprehensive summary commit message for the entire file."""
    prefix = config.get('commit_message_prefix', 'Auto-commit: ')
    file_name = os.path.basename(file_path)
    
    # Build summary statistics
    total_lines = structure['total_lines']
    total_chunks = len(chunks)
    
    # Count different types of content
    all_types = {}
    all_names = {'classes': [], 'functions': [], 'sections': []}
    
    for chunk in chunks:
        for segment in chunk.get('segments', []):
            seg_type = segment['type']
            all_types[seg_type] = all_types.get(seg_type, 0) + 1
            
            if seg_type == 'class' and 'name' in segment:
                all_names['classes'].append(segment['name'])
            elif seg_type == 'function' and 'name' in segment:
                all_names['functions'].append(segment['name'])
            elif seg_type == 'section' and 'name' in segment:
                all_names['sections'].append(segment['name'])
    
    # Build the summary message
    message = f"{prefix}Final summary - Completed {file_name} ({total_lines} lines in {total_chunks} chunks)\n\n"
    
    message += "📄 File Overview:\n"
    message += f"- Total lines: {total_lines}\n"
    message += f"- Chunks created: {total_chunks}\n"
    message += f"- File type: {structure['file_type']}\n"
    
    # Content summary
    message += "\n📋 Content Summary:\n"
    
    if all_types.get('module_docstring'):
        message += "- Module documentation\n"
    
    if all_types.get('imports'):
        message += f"- {all_types['imports']} import sections\n"
    
    if all_names['classes']:
        message += f"- {len(all_names['classes'])} classes: {', '.join(all_names['classes'][:5])}"
        if len(all_names['classes']) > 5:
            message += f" (and {len(all_names['classes']) - 5} more)"
        message += "\n"
    
    if all_names['functions']:
        message += f"- {len(all_names['functions'])} functions: {', '.join(all_names['functions'][:5])}"
        if len(all_names['functions']) > 5:
            message += f" (and {len(all_names['functions']) - 5} more)"
        message += "\n"
    
    if all_names['sections']:
        message += f"- {len(all_names['sections'])} sections: {', '.join(all_names['sections'][:5])}"
        if len(all_names['sections']) > 5:
            message += f" (and {len(all_names['sections']) - 5} more)"
        message += "\n"
    
    if all_types.get('code_block'):
        message += f"- {all_types['code_block']} code examples\n"
    
    if all_types.get('table'):
        message += f"- {all_types['table']} tables\n"
    
    # Add purpose based on file type
    message += "\n🎯 Purpose: "
    if structure['file_type'] in ['.md', '.markdown']:
        message += "Documentation file providing "
        if 'guide' in file_name.lower() or 'tutorial' in file_name.lower():
            message += "guidance and tutorials"
        elif 'readme' in file_name.lower():
            message += "project overview and instructions"
        elif 'api' in file_name.lower():
            message += "API reference documentation"
        else:
            message += "comprehensive documentation"
    elif structure['file_type'] == '.py':
        if 'test' in file_name.lower():
            message += "Test suite for validating functionality"
        elif 'model' in file_name.lower():
            message += "Data models and business logic"
        elif 'api' in file_name.lower() or 'view' in file_name.lower():
            message += "API endpoints and request handlers"
        else:
            message += "Python module implementing core functionality"
    else:
        message += "Implementation of application logic"
    
    message += "\n\n✅ All chunks successfully committed and file creation completed."
    
    return message

def should_use_chunked_creation(file_path: str, content: str, config: Dict[str, Any]) -> bool:
    """
    Determine if a file creation should use chunked approach.
    
    Args:
        file_path: Path to the file being created
        content: File content
        config: Configuration settings
        
    Returns:
        bool: True if chunked creation should be used
    """
    if not config.get('enabled', True):
        return False
    
    lines = content.splitlines()
    line_count = len(lines)
    
    # Check size threshold
    max_lines = config.get('max_new_file_lines', 50)
    if line_count <= max_lines:
        return False
    
    # Check file type (now includes Markdown)
    file_ext = Path(file_path).suffix.lower()
    supported_types = config.get('supported_extensions', ['.py', '.js', '.ts', '.jsx', '.tsx', '.mojo', '.🔥', '.md', '.markdown'])
    
    if file_ext not in supported_types:
        return False
    
    # Analyze structure complexity
    structure = analyze_code_structure(content, file_path)
    segment_count = len(structure.get('segments', []))
    
    # Use chunking if file has multiple logical segments
    min_segments = config.get('min_segments_for_chunking', 3)
    
    return segment_count >= min_segments

def load_chunking_config() -> Dict[str, Any]:
    """
    Load chunking configuration from settings.json.
    
    Returns:
        dict: Chunking configuration with defaults
    """
    default_config = {
        "enabled": True,
        "max_new_file_lines": 50,
        "max_lines_per_chunk": 25,
        "min_segments_for_chunking": 3,
        "commit_message_prefix": "Auto-commit: ",
        "supported_extensions": [".py", ".js", ".ts", ".jsx", ".tsx", ".mojo", ".🔥", ".md", ".markdown"]
    }
    
    try:
        settings_path = Path(__file__).parent.parent / "settings.json"
        if settings_path.exists():
            import json
            with open(settings_path, 'r') as f:
                settings = json.load(f)
                chunking_config = settings.get("chunked_file_creation", {})
                # Merge with defaults
                default_config.update(chunking_config)
    except Exception:
        pass
    
    return default_config

def perform_chunked_file_creation(file_path: str, content: str, config: Dict[str, Any]) -> bool:
    """
    Create a large file by breaking it into logical chunks and committing each separately.
    
    Args:
        file_path: Path to the file to create
        content: Complete file content
        config: Configuration settings
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"🔨 Starting chunked file creation for {file_path}", file=sys.stderr)
        
        # Analyze code structure
        structure = analyze_code_structure(content, file_path)
        print(f"📊 Analysis: {structure['total_lines']} lines, {len(structure.get('segments', []))} segments", file=sys.stderr)
        
        # Create logical chunks
        max_lines = config.get('max_lines_per_chunk', 25)
        chunks = create_logical_chunks(structure, content, max_lines)
        
        print(f"📦 Created {len(chunks)} logical chunks", file=sys.stderr)
        
        # Create file incrementally
        file_dir = os.path.dirname(file_path)
        if file_dir and not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)
        
        accumulated_content = ""
        file_name = os.path.basename(file_path)
        
        for i, chunk in enumerate(chunks):
            print(f"🔧 Processing chunk {i+1}/{len(chunks)}: {chunk['description']}", file=sys.stderr)
            
            # Add chunk content to accumulated content
            if accumulated_content and not accumulated_content.endswith('\n'):
                accumulated_content += '\n'
            accumulated_content += chunk['content']
            
            # Write current accumulated content to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(accumulated_content)
            
            # Stage and commit this chunk
            result = subprocess.run(['git', 'add', file_path], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ git add failed: {result.stderr}", file=sys.stderr)
                return False
            
            # Generate detailed commit message for individual chunk
            commit_msg = _generate_detailed_commit_message(chunk, i, len(chunks), file_name, config)
            
            result = subprocess.run(['git', 'commit', '-m', commit_msg], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ git commit failed: {result.stderr}", file=sys.stderr)
                return False
            
            print(f"✅ Committed chunk {i+1}/{len(chunks)}: {chunk['description']}", file=sys.stderr)
        
        # Create final comprehensive summary commit
        print("📝 Creating final summary commit...", file=sys.stderr)
        
        # Generate comprehensive summary message
        summary_message = _generate_final_summary_message(chunks, file_path, structure, config)
        
        # Stage file again for summary commit
        result = subprocess.run(['git', 'add', file_path], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ git add for summary failed: {result.stderr}", file=sys.stderr)
            return False
        
        # Create summary commit
        result = subprocess.run(['git', 'commit', '--amend', '-m', summary_message], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Summary commit failed: {result.stderr}", file=sys.stderr)
            return False
        
        print("✅ Final summary commit created successfully", file=sys.stderr)
        print(f"🎉 Chunked file creation completed successfully for {file_path}", file=sys.stderr)
        return True
        
    except Exception as e:
        print(f"❌ Chunked file creation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return False

def check_chunked_file_creation(tool_name: str, tool_input: Dict[str, Any]) -> bool:
    """
    Check if file creation should use chunked approach and perform it if needed.
    
    Args:
        tool_name: Name of the tool being used
        tool_input: Tool input parameters
        
    Returns:
        bool: True if operation was handled by chunking (blocks original), False otherwise
    """
    if tool_name != 'Write':
        return False
    
    file_path = tool_input.get('file_path', '')
    content = tool_input.get('content', '')
    
    if not file_path or not content:
        return False
    
    # Check if file already exists (not a new creation)
    if os.path.exists(file_path):
        return False
    
    config = load_chunking_config()
    
    if should_use_chunked_creation(file_path, content, config):
        print(f"🚀 Large file creation detected: {file_path} ({len(content.splitlines())} lines)", file=sys.stderr)
        print("📦 Using chunked file creation approach", file=sys.stderr)
        
        # Perform chunked creation
        if perform_chunked_file_creation(file_path, content, config):
            print("✅ Chunked file creation completed successfully", file=sys.stderr)
            return True  # Block original Write operation
        else:
            print("❌ Chunked file creation failed, falling back to normal creation", file=sys.stderr)
            return False
    
    return False