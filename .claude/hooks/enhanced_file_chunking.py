#!/usr/bin/env python3
"""
Enhanced file chunking system with Markdown support for large file creation.

This module provides intelligent code segmentation and incremental file building
to break large file creation into smaller, logical commits.

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
            
            # Generate meaningful commit message
            file_name = os.path.basename(file_path)
            commit_msg = (f"{config.get('commit_message_prefix', 'Auto-commit: ')}"
                         f"chunk {i+1}/{len(chunks)} - {file_name} ({chunk['line_count']} lines) - {chunk['description']}")
            
            result = subprocess.run(['git', 'commit', '-m', commit_msg], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ git commit failed: {result.stderr}", file=sys.stderr)
                return False
            
            print(f"✅ Committed chunk {i+1}/{len(chunks)}: {chunk['description']}", file=sys.stderr)
        
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