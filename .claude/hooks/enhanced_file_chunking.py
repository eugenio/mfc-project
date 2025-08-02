#!/usr/bin/env python3
"""
Enhanced file chunking system v2 with improved commit messages and better structure analysis.

This module provides intelligent code segmentation and incremental file building
to break large file creation into smaller, logical commits with meaningful messages
and comprehensive final summaries.

Improvements in v2:
- Better commit message generation with contextual information
- Improved structure analysis for all supported languages
- Enhanced error handling and recovery
- Better support for mixed content files
- Smarter chunk boundary detection
- Support for configuration files and data files
- Better handling of edge cases

Supports: Python, JavaScript/TypeScript, Mojo, Markdown, YAML, JSON, TOML, and more.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from utils.enhanced_security_guardian import secure_chunked_file_creation
from utils.git_guardian import fallback_to_direct_commit, request_guardian_commit


def analyze_code_structure(content: str, file_path: str) -> dict[str, Any]:
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
        "constants": [],
        "metadata": {},
        "has_tests": False,
        "has_main": False,
        "complexity_score": 0
    }

    # Detect if it's a test file
    file_name = Path(file_path).name.lower()
    structure["has_tests"] = 'test' in file_name or 'spec' in file_name

    # Language-specific analysis
    if file_ext in ['.py', '.pyi']:
        structure.update(_analyze_python_structure(lines))
    elif file_ext in ['.js', '.ts', '.jsx', '.tsx']:
        structure.update(_analyze_javascript_structure(lines))
    elif file_ext in ['.mojo', '.🔥']:
        structure.update(_analyze_mojo_structure(lines))
    elif file_ext in ['.md', '.markdown']:
        structure.update(_analyze_markdown_structure(lines))
    elif file_ext in ['.yaml', '.yml']:
        structure.update(_analyze_yaml_structure(lines))
    elif file_ext in ['.json']:
        structure.update(_analyze_json_structure(lines))
    elif file_ext in ['.toml']:
        structure.update(_analyze_toml_structure(lines))
    elif file_ext in ['.c', '.cpp', '.cc', '.h', '.hpp']:
        structure.update(_analyze_c_cpp_structure(lines))
    elif file_ext in ['.rs']:
        structure.update(_analyze_rust_structure(lines))
    elif file_ext in ['.go']:
        structure.update(_analyze_go_structure(lines))
    else:
        structure.update(_analyze_generic_structure(lines))

    # Calculate complexity score
    structure["complexity_score"] = _calculate_complexity_score(structure)

    return structure

def _calculate_complexity_score(structure: dict[str, Any]) -> int:
    """Calculate a complexity score based on code structure."""
    score = 0
    score += len(structure.get("classes", [])) * 10
    score += len(structure.get("functions", [])) * 5
    score += len(structure.get("imports", [])) * 2
    score += len(structure.get("segments", [])) * 3
    score += structure.get("total_lines", 0) // 50
    return score

def _analyze_python_structure(lines: list[str]) -> dict[str, Any]:
    """Analyze Python code structure for logical chunking."""
    segments = []
    imports = []
    classes = []
    functions = []
    decorators = []
    has_main = False

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
                "description": "Module documentation and metadata",
                "content_preview": lines[start_i][:50] + "..." if len(lines[start_i]) > 50 else lines[start_i]
            })

        # Type annotations and future imports (high priority)
        elif line.startswith('from __future__'):
            segments.append({
                "type": "future_import",
                "start_line": i,
                "end_line": i,
                "priority": 0,  # Highest priority
                "description": "Future imports",
                "content_preview": line
            })

        # Imports section
        elif line.startswith(('import ', 'from ')):
            start_i = i
            import_group = []
            while i < len(lines) and (lines[i].strip().startswith(('import ', 'from ')) or
                                     lines[i].strip() == '' or
                                     lines[i].strip().startswith('#')):
                if lines[i].strip() and not lines[i].strip().startswith('#'):
                    import_group.append(lines[i].strip())
                    imports.append(lines[i].strip())
                i += 1
            i -= 1  # Back up one

            # Group imports by type
            stdlib_imports = [imp for imp in import_group if _is_stdlib_import(imp)]
            third_party_imports = [imp for imp in import_group if not _is_stdlib_import(imp) and not _is_local_import(imp)]
            local_imports = [imp for imp in import_group if _is_local_import(imp)]

            import_desc = []
            if stdlib_imports:
                import_desc.append(f"{len(stdlib_imports)} stdlib")
            if third_party_imports:
                import_desc.append(f"{len(third_party_imports)} third-party")
            if local_imports:
                import_desc.append(f"{len(local_imports)} local")

            segments.append({
                "type": "imports",
                "start_line": start_i,
                "end_line": i,
                "priority": 2,
                "description": f"Import statements ({', '.join(import_desc)})",
                "import_count": len(import_group),
                "imports": import_group[:5]  # First 5 for preview
            })

        # Decorators (often used with classes/functions)
        elif line.startswith('@'):
            decorator_name = line[1:].split('(')[0].strip()
            decorators.append(decorator_name)
            # Don't create segment for decorator alone, include with next item

        # Class definitions
        elif line.startswith('class '):
            class_match = re.match(r'class\s+(\w+)', line)
            if class_match:
                class_name = class_match.group(1)
                start_i = i

                # Include any decorators before the class
                if decorators:
                    # Look back for decorators
                    temp_i = i - 1
                    while temp_i >= 0 and (lines[temp_i].strip().startswith('@') or lines[temp_i].strip() == ''):
                        if lines[temp_i].strip():
                            start_i = temp_i
                        temp_i -= 1

                # Find end of class
                i += 1
                indent_level = len(lines[start_i]) - len(lines[start_i].lstrip())
                method_count = 0
                class_lines = []

                while i < len(lines):
                    current_line = lines[i]
                    current_indent = len(current_line) - len(current_line.lstrip())

                    # Count methods
                    if current_line.strip().startswith('def ') and current_indent > indent_level:
                        method_count += 1

                    # Check for end of class
                    if (current_line.strip() and
                        current_indent <= indent_level and
                        not current_line.strip().startswith('#')):
                        break

                    class_lines.append(current_line)
                    i += 1

                i -= 1  # Back up one

                classes.append(class_name)
                segments.append({
                    "type": "class",
                    "name": class_name,
                    "start_line": start_i,
                    "end_line": i,
                    "priority": 3,
                    "description": f"Class {class_name} with {method_count} methods",
                    "decorators": decorators.copy(),
                    "method_count": method_count,
                    "is_test_class": class_name.startswith('Test') or class_name.endswith('Test'),
                    "line_count": i - start_i + 1
                })
                decorators.clear()

        # Function definitions (at module level)
        elif line.startswith('def ') and not line.startswith('    '):
            func_match = re.match(r'def\s+(\w+)', line)
            if func_match:
                func_name = func_match.group(1)
                start_i = i

                # Include decorators
                if decorators:
                    temp_i = i - 1
                    while temp_i >= 0 and (lines[temp_i].strip().startswith('@') or lines[temp_i].strip() == ''):
                        if lines[temp_i].strip():
                            start_i = temp_i
                        temp_i -= 1

                # Find end of function
                i += 1
                func_lines = []
                has_yield = False

                while i < len(lines):
                    current_line = lines[i]
                    if 'yield' in current_line:
                        has_yield = True

                    if (current_line.strip() and
                        not current_line.startswith(' ') and
                        not current_line.strip().startswith('#')):
                        break
                    func_lines.append(current_line)
                    i += 1

                i -= 1  # Back up one

                # Check if it's main function
                if func_name == 'main':
                    has_main = True

                functions.append(func_name)
                segments.append({
                    "type": "function",
                    "name": func_name,
                    "start_line": start_i,
                    "end_line": i,
                    "priority": 4,
                    "description": f"{'Generator' if has_yield else 'Function'} {func_name}",
                    "decorators": decorators.copy(),
                    "is_test": func_name.startswith('test_'),
                    "is_main": func_name == 'main',
                    "is_generator": has_yield,
                    "line_count": i - start_i + 1
                })
                decorators.clear()

        # Constants and module-level variables
        elif ('=' in line and not line.startswith(' ') and
              not line.startswith('#') and not line.startswith('if')):
            const_match = re.match(r'(\w+)\s*=', line)
            if const_match:
                const_name = const_match.group(1)
                is_constant = const_name.isupper()

                segments.append({
                    "type": "constant" if is_constant else "variable",
                    "name": const_name,
                    "start_line": i,
                    "end_line": i,
                    "priority": 2,
                    "description": f"{'Constant' if is_constant else 'Variable'} {const_name}",
                    "is_configuration": 'CONFIG' in const_name or 'SETTING' in const_name
                })

        # Main execution block
        elif line == 'if __name__ == "__main__":' or line == "if __name__ == '__main__':":
            start_i = i
            i += 1
            while i < len(lines) and (lines[i].startswith(' ') or lines[i].strip() == ''):
                i += 1
            i -= 1

            segments.append({
                "type": "main_block",
                "start_line": start_i,
                "end_line": i,
                "priority": 5,
                "description": "Main execution block"
            })

        i += 1

    return {
        "segments": segments,
        "imports": imports,
        "classes": classes,
        "functions": functions,
        "has_main": has_main
    }

def _is_stdlib_import(import_line: str) -> bool:
    """Check if import is from Python standard library."""
    stdlib_modules = {
        'os', 'sys', 'json', 're', 'math', 'random', 'datetime', 'time',
        'pathlib', 'collections', 'itertools', 'functools', 'typing',
        'subprocess', 'threading', 'multiprocessing', 'asyncio', 'unittest'
    }

    if import_line.startswith('import '):
        module = import_line.split()[1].split('.')[0]
        return module in stdlib_modules
    elif import_line.startswith('from '):
        module = import_line.split()[1].split('.')[0]
        return module in stdlib_modules
    return False

def _is_local_import(import_line: str) -> bool:
    """Check if import is local (relative)."""
    return 'from .' in import_line or import_line.startswith('from .')

def _analyze_markdown_structure(lines: list[str]) -> dict[str, Any]:
    """Analyze Markdown structure for logical chunking."""
    segments = []
    headings = []
    code_blocks = []
    tables: list[str] = []
    toc_found = False

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

        # Table of Contents
        elif (line.lower() in ['## table of contents', '## contents', '## toc'] or
              (i < 20 and 'table of contents' in line.lower())):
            start_i = i
            toc_found = True
            i += 1
            while i < len(lines) and (lines[i].strip().startswith('-') or
                                     lines[i].strip().startswith('*') or
                                     lines[i].strip().startswith('1.') or
                                     lines[i].strip() == ''):
                i += 1
            i -= 1

            segments.append({
                "type": "toc",
                "start_line": start_i,
                "end_line": i,
                "priority": 2,
                "description": "Table of Contents"
            })

        # H2 sections (major sections)
        elif line.startswith('## '):
            section_title = line[3:].strip()
            start_i = i
            section_content = {"code_blocks": 0, "subsections": 0, "lists": 0}

            # Find end of section
            i += 1
            while i < len(lines):
                current_line = lines[i].strip()
                if current_line.startswith('## '):
                    break
                elif current_line.startswith('### '):
                    section_content["subsections"] += 1
                elif current_line.startswith('```'):
                    section_content["code_blocks"] += 1
                elif current_line.startswith(('- ', '* ', '1. ')):
                    section_content["lists"] += 1
                i += 1
            i -= 1

            content_desc = []
            if section_content["subsections"]:
                content_desc.append(f"{section_content['subsections']} subsections")
            if section_content["code_blocks"]:
                content_desc.append(f"{section_content['code_blocks']} code examples")

            segments.append({
                "type": "section",
                "name": section_title,
                "start_line": start_i,
                "end_line": i,
                "priority": 3,
                "description": f"Section: {section_title}",
                "content_summary": ', '.join(content_desc) if content_desc else "text content"
            })
            headings.append(section_title)

        # Code blocks
        elif line.startswith('```'):
            start_i = i
            language = line[3:].strip() if len(line) > 3 else "code"
            code_lines = []

            # Find end of code block
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1

            if i < len(lines):  # Found closing ```
                segments.append({
                    "type": "code_block",
                    "name": language,
                    "start_line": start_i,
                    "end_line": i,
                    "priority": 4,
                    "description": f"Code block ({language})",
                    "line_count": len(code_lines),
                    "is_example": True
                })
                code_blocks.append(language)

        i += 1

    return {
        "segments": segments,
        "headings": headings,
        "code_blocks": code_blocks,
        "tables": tables,
        "has_toc": toc_found
    }

def _analyze_yaml_structure(lines: list[str]) -> dict[str, Any]:
    """Analyze YAML structure for logical chunking."""
    segments = []
    current_section = None
    section_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Top-level keys (no indentation)
        if stripped and not line.startswith(' ') and ':' in line and not stripped.startswith('#'):
            if current_section:
                segments.append({
                    "type": "yaml_section",
                    "name": current_section,
                    "start_line": section_start,
                    "end_line": i - 1,
                    "priority": 3,
                    "description": f"Configuration section: {current_section}"
                })

            current_section = stripped.split(':')[0].strip()
            section_start = i

    # Add final section
    if current_section:
        segments.append({
            "type": "yaml_section",
            "name": current_section,
            "start_line": section_start,
            "end_line": len(lines) - 1,
            "priority": 3,
            "description": f"Configuration section: {current_section}"
        })

    return {"segments": segments}

def _analyze_json_structure(lines: list[str]) -> dict[str, Any]:
    """Analyze JSON structure for logical chunking."""
    # For JSON, we'll treat the entire file as one segment
    # since it needs to be valid JSON
    segments = [{
        "type": "json_data",
        "start_line": 0,
        "end_line": len(lines) - 1,
        "priority": 3,
        "description": "JSON data structure"
    }]

    return {"segments": segments}

def _analyze_toml_structure(lines: list[str]) -> dict[str, Any]:
    """Analyze TOML structure for logical chunking."""
    segments = []
    current_section = None
    section_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Section headers [section]
        if stripped.startswith('[') and stripped.endswith(']'):
            if current_section:
                segments.append({
                    "type": "toml_section",
                    "name": current_section,
                    "start_line": section_start,
                    "end_line": i - 1,
                    "priority": 3,
                    "description": f"Configuration section: {current_section}"
                })

            current_section = stripped[1:-1]
            section_start = i

    # Add final section
    if current_section:
        segments.append({
            "type": "toml_section",
            "name": current_section,
            "start_line": section_start,
            "end_line": len(lines) - 1,
            "priority": 3,
            "description": f"Configuration section: {current_section}"
        })

    return {"segments": segments}

def _analyze_c_cpp_structure(lines: list[str]) -> dict[str, Any]:
    """Analyze C/C++ code structure."""
    segments = []
    includes = []
    functions = []
    classes: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Include statements
        if line.startswith('#include'):
            includes.append(line)
            segments.append({
                "type": "include",
                "start_line": i,
                "end_line": i,
                "priority": 2,
                "description": "Include directive"
            })

        # Function definitions (simple detection)
        elif re.match(r'^(static\s+)?(\w+\s+)+\w+\s*\([^)]*\)\s*{?', line):
            func_match = re.search(r'(\w+)\s*\([^)]*\)', line)
            if func_match:
                func_name = func_match.group(1)
                functions.append(func_name)

                # Find function end (simple brace counting)
                start_i = i
                brace_count = line.count('{') - line.count('}')
                i += 1

                while i < len(lines) and brace_count > 0:
                    brace_count += lines[i].count('{') - lines[i].count('}')
                    i += 1

                segments.append({
                    "type": "function",
                    "name": func_name,
                    "start_line": start_i,
                    "end_line": i - 1,
                    "priority": 4,
                    "description": f"Function {func_name}"
                })
                i -= 1

        i += 1

    return {
        "segments": segments,
        "includes": includes,
        "functions": functions,
        "classes": classes
    }

def _analyze_rust_structure(lines: list[str]) -> dict[str, Any]:
    """Analyze Rust code structure."""
    segments = []
    functions = []
    structs = []
    impls: list[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Function definitions
        if stripped.startswith('fn ') or stripped.startswith('pub fn '):
            func_match = re.match(r'(?:pub\s+)?fn\s+(\w+)', stripped)
            if func_match:
                func_name = func_match.group(1)
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
        elif stripped.startswith('struct ') or stripped.startswith('pub struct '):
            struct_match = re.match(r'(?:pub\s+)?struct\s+(\w+)', stripped)
            if struct_match:
                struct_name = struct_match.group(1)
                structs.append(struct_name)
                segments.append({
                    "type": "struct",
                    "name": struct_name,
                    "start_line": i,
                    "end_line": i + 10,  # Simplified
                    "priority": 3,
                    "description": f"Struct {struct_name}"
                })

    return {
        "segments": segments,
        "functions": functions,
        "structs": structs,
        "impls": impls
    }

def _analyze_go_structure(lines: list[str]) -> dict[str, Any]:
    """Analyze Go code structure."""
    segments = []
    functions = []
    types: list[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Package declaration
        if stripped.startswith('package '):
            package_name = stripped.split()[1]
            segments.append({
                "type": "package",
                "name": package_name,
                "start_line": i,
                "end_line": i,
                "priority": 1,
                "description": f"Package {package_name}"
            })

        # Import statements
        elif stripped.startswith('import'):
            segments.append({
                "type": "imports",
                "start_line": i,
                "end_line": i,
                "priority": 2,
                "description": "Import statements"
            })

        # Function definitions
        elif stripped.startswith('func '):
            func_match = re.match(r'func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)', stripped)
            if func_match:
                func_name = func_match.group(1)
                functions.append(func_name)
                segments.append({
                    "type": "function",
                    "name": func_name,
                    "start_line": i,
                    "end_line": i + 10,  # Simplified
                    "priority": 4,
                    "description": f"Function {func_name}"
                })

    return {
        "segments": segments,
        "functions": functions,
        "types": types
    }

def _analyze_javascript_structure(lines: list[str]) -> dict[str, Any]:
    """Analyze JavaScript/TypeScript code structure."""
    segments = []
    imports = []
    classes = []
    functions = []
    exports: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Import statements
        if line.startswith('import ') or (line.startswith('const ') and 'require(' in line):
            start_i = i
            import_group = []

            while i < len(lines) and (lines[i].strip().startswith(('import ', 'const ')) or
                                     lines[i].strip() == '' or
                                     lines[i].strip().startswith('//')):
                if lines[i].strip() and not lines[i].strip().startswith('//'):
                    import_group.append(lines[i].strip())
                    imports.append(lines[i].strip())
                i += 1
            i -= 1

            segments.append({
                "type": "imports",
                "start_line": start_i,
                "end_line": i,
                "priority": 2,
                "description": f"Import/require statements ({len(import_group)} imports)"
            })

        # Class definitions (ES6)
        elif line.startswith('class ') or line.startswith('export class '):
            class_match = re.match(r'(?:export\s+)?class\s+(\w+)', line)
            if class_match:
                class_name = class_match.group(1)
                classes.append(class_name)

                # Find class end (simple brace counting)
                start_i = i
                brace_count = line.count('{') - line.count('}')
                i += 1

                while i < len(lines) and brace_count > 0:
                    brace_count += lines[i].count('{') - lines[i].count('}')
                    i += 1

                segments.append({
                    "type": "class",
                    "name": class_name,
                    "start_line": start_i,
                    "end_line": i - 1,
                    "priority": 3,
                    "description": f"Class {class_name}",
                    "is_exported": 'export' in lines[start_i]
                })
                i -= 1

        # Function declarations
        elif (line.startswith('function ') or
              line.startswith('async function ') or
              line.startswith('export function ') or
              line.startswith('export async function ')):
            func_match = re.match(r'(?:export\s+)?(?:async\s+)?function\s+(\w+)', line)
            if func_match:
                func_name = func_match.group(1)
                functions.append(func_name)

                # Find function end
                start_i = i
                brace_count = line.count('{') - line.count('}')
                i += 1

                while i < len(lines) and brace_count > 0:
                    brace_count += lines[i].count('{') - lines[i].count('}')
                    i += 1

                segments.append({
                    "type": "function",
                    "name": func_name,
                    "start_line": start_i,
                    "end_line": i - 1,
                    "priority": 4,
                    "description": f"Function {func_name}",
                    "is_async": 'async' in lines[start_i],
                    "is_exported": 'export' in lines[start_i]
                })
                i -= 1

        # Arrow functions and const functions
        elif (re.match(r'(?:export\s+)?const\s+\w+\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=]+)\s*=>', line) or
              re.match(r'(?:export\s+)?const\s+\w+\s*=\s*(?:async\s+)?function', line)):
            const_match = re.match(r'(?:export\s+)?const\s+(\w+)', line)
            if const_match:
                func_name = const_match.group(1)
                functions.append(func_name)

                segments.append({
                    "type": "arrow_function",
                    "name": func_name,
                    "start_line": i,
                    "end_line": i + 5,  # Simplified
                    "priority": 4,
                    "description": f"Arrow function {func_name}",
                    "is_exported": 'export' in line
                })

        # React components (special case)
        elif ('React.Component' in line or 'extends Component' in line or
              'export default function' in line or 'export const' in line):
            if 'export default function' in line:
                comp_match = re.match(r'export\s+default\s+function\s+(\w+)', line)
                if comp_match:
                    comp_name = comp_match.group(1)
                    segments.append({
                        "type": "react_component",
                        "name": comp_name,
                        "start_line": i,
                        "end_line": i + 10,  # Simplified
                        "priority": 3,
                        "description": f"React component {comp_name}"
                    })

        i += 1

    return {
        "segments": segments,
        "imports": imports,
        "classes": classes,
        "functions": functions,
        "exports": exports
    }

def _analyze_mojo_structure(lines: list[str]) -> dict[str, Any]:
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
            func_match = re.match(r'fn\s+(\w+)', line)
            if func_match:
                func_name = func_match.group(1)
                functions.append(func_name)

                # Find function end (indentation-based)
                start_i = i
                i += 1
                base_indent = len(lines[start_i]) - len(lines[start_i].lstrip())

                while i < len(lines):
                    current_line = lines[i]
                    if current_line.strip() and len(current_line) - len(current_line.lstrip()) <= base_indent:
                        break
                    i += 1

                segments.append({
                    "type": "function",
                    "name": func_name,
                    "start_line": start_i,
                    "end_line": i - 1,
                    "priority": 4,
                    "description": f"Function {func_name}"
                })
                i -= 1

        # Struct definitions
        elif line.startswith('struct '):
            struct_match = re.match(r'struct\s+(\w+)', line)
            if struct_match:
                struct_name = struct_match.group(1)
                structs.append(struct_name)

                # Find struct end
                start_i = i
                i += 1
                base_indent = len(lines[start_i]) - len(lines[start_i].lstrip())

                while i < len(lines):
                    current_line = lines[i]
                    if current_line.strip() and len(current_line) - len(current_line.lstrip()) <= base_indent:
                        break
                    i += 1

                segments.append({
                    "type": "struct",
                    "name": struct_name,
                    "start_line": start_i,
                    "end_line": i - 1,
                    "priority": 3,
                    "description": f"Struct {struct_name}"
                })
                i -= 1

        i += 1

    return {
        "segments": segments,
        "imports": imports,
        "structs": structs,
        "functions": functions
    }

def _analyze_generic_structure(lines: list[str]) -> dict[str, Any]:
    """Generic structure analysis for unknown file types."""
    segments = []

    # Look for common patterns
    has_shebang = len(lines) > 0 and lines[0].startswith('#!')

    # Simple line-based chunking with pattern detection
    chunk_size = 25
    for i in range(0, len(lines), chunk_size):
        end_line = min(i + chunk_size - 1, len(lines) - 1)

        # Analyze chunk content
        chunk_lines = lines[i:end_line + 1]
        chunk_info = {
            "has_comments": any(line.strip().startswith('#') for line in chunk_lines),
            "has_functions": any('function' in line or 'def' in line for line in chunk_lines),
            "has_braces": any('{' in line or '}' in line for line in chunk_lines)
        }

        desc_parts = [f"Lines {i+1}-{end_line+1}"]
        if chunk_info["has_comments"]:
            desc_parts.append("with comments")
        if chunk_info["has_functions"]:
            desc_parts.append("with functions")

        segments.append({
            "type": "generic_chunk",
            "start_line": i,
            "end_line": end_line,
            "priority": 3,
            "description": " ".join(desc_parts)
        })

    return {
        "segments": segments,
        "has_shebang": has_shebang
    }

def create_logical_chunks(structure: dict[str, Any], content: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Create logical chunks from code structure analysis with improved grouping.
    
    Args:
        structure: Code structure analysis
        content: Original file content
        config: Configuration settings
        
    Returns:
        List of chunks with content and metadata
    """
    lines = content.splitlines()
    chunks = []

    # Get configuration
    max_lines_per_chunk = config.get('max_lines_per_chunk', 25)
    intelligent_grouping = config.get('intelligent_grouping', True)

    # Sort segments by priority and start line
    segments = sorted(structure.get("segments", []), key=lambda x: (x["priority"], x["start_line"]))

    # Group related segments if intelligent grouping is enabled
    if intelligent_grouping:
        segments = _group_related_segments(segments, structure, max_lines_per_chunk)

    current_chunk_lines: list[str] = []
    current_chunk_segments: list[dict[str, Any]] = []
    current_chunk_start = 0

    for segment in segments:
        start_line = segment["start_line"]
        end_line = segment["end_line"]
        segment_lines = lines[start_line:end_line + 1]

        # Check if we need to start a new chunk
        should_start_new_chunk = False

        # Reasons to start a new chunk:
        # 1. Adding this segment would exceed max lines
        if len(current_chunk_lines) + len(segment_lines) > max_lines_per_chunk and current_chunk_lines:
            should_start_new_chunk = True

        # 2. Priority change (e.g., moving from imports to classes)
        if (current_chunk_segments and
            abs(segment["priority"] - current_chunk_segments[-1]["priority"]) > 1):
            should_start_new_chunk = True

        # 3. Logical boundary (e.g., end of all imports, start of main content)
        if _is_logical_boundary(segment, current_chunk_segments):
            should_start_new_chunk = True

        if should_start_new_chunk:
            # Finalize current chunk
            chunks.append({
                "content": "\n".join(current_chunk_lines),
                "segments": current_chunk_segments,
                "line_count": len(current_chunk_lines),
                "start_line": current_chunk_start,
                "end_line": current_chunk_start + len(current_chunk_lines) - 1,
                "description": _generate_chunk_description(current_chunk_segments),
                "chunk_type": _determine_chunk_type(current_chunk_segments)
            })

            current_chunk_lines = []
            current_chunk_segments = []
            current_chunk_start = start_line

        # Add segment to current chunk
        current_chunk_lines.extend(segment_lines)
        current_chunk_segments.append(segment)

    # Add final chunk if there's remaining content
    if current_chunk_lines:
        chunks.append({
            "content": "\n".join(current_chunk_lines),
            "segments": current_chunk_segments,
            "line_count": len(current_chunk_lines),
            "start_line": current_chunk_start,
            "end_line": current_chunk_start + len(current_chunk_lines) - 1,
            "description": _generate_chunk_description(current_chunk_segments),
            "chunk_type": _determine_chunk_type(current_chunk_segments)
        })

    return chunks

def _group_related_segments(segments: list[dict[str, Any]], structure: dict[str, Any],
                           max_lines: int) -> list[dict[str, Any]]:
    """Group related segments together for better logical chunking."""
    grouped = []
    i = 0

    while i < len(segments):
        current = segments[i]

        # Try to group small related items
        if current["type"] in ["constant", "variable", "import", "include"]:
            # Group consecutive items of the same type
            group = [current]
            j = i + 1
            total_lines = current["end_line"] - current["start_line"] + 1

            while j < len(segments):
                next_seg = segments[j]
                next_lines = next_seg["end_line"] - next_seg["start_line"] + 1

                if (next_seg["type"] == current["type"] and
                    next_seg["priority"] == current["priority"] and
                    total_lines + next_lines <= max_lines):
                    group.append(next_seg)
                    total_lines += next_lines
                    j += 1
                else:
                    break

            if len(group) > 1:
                # Create a grouped segment
                grouped_segment = {
                    "type": f"grouped_{current['type']}s",
                    "start_line": group[0]["start_line"],
                    "end_line": group[-1]["end_line"],
                    "priority": current["priority"],
                    "description": f"{len(group)} {current['type']}s",
                    "items": [g.get("name", "") for g in group if "name" in g]
                }
                grouped.append(grouped_segment)
                i = j
            else:
                grouped.append(current)
                i += 1
        else:
            grouped.append(current)
            i += 1

    return grouped

def _is_logical_boundary(segment: dict[str, Any], previous_segments: list[dict[str, Any]]) -> bool:
    """Determine if this segment represents a logical boundary."""
    if not previous_segments:
        return False

    last_segment = previous_segments[-1]

    # Moving from imports to main content
    if last_segment["type"] in ["imports", "import", "include"] and \
       segment["type"] not in ["imports", "import", "include", "constant", "variable"]:
        return True

    # Moving from documentation to code
    if last_segment["type"] in ["module_docstring", "title", "toc"] and \
       segment["type"] not in ["module_docstring", "title", "toc"]:
        return True

    # Starting main execution block
    if segment["type"] == "main_block":
        return True

    return False

def _determine_chunk_type(segments: list[dict[str, Any]]) -> str:
    """Determine the overall type of a chunk based on its segments."""
    if not segments:
        return "mixed"

    types = [s["type"] for s in segments]

    # Foundation chunks (imports, constants, setup)
    if all(t in ["imports", "import", "include", "constant", "variable", "module_docstring",
                 "title", "future_import"] for t in types):
        return "foundation"

    # Implementation chunks (classes, functions)
    if any(t in ["class", "function", "struct", "arrow_function"] for t in types):
        return "implementation"

    # Documentation chunks
    if all(t in ["title", "section", "subsection", "toc", "code_block"] for t in types):
        return "documentation"

    # Configuration chunks
    if any(t in ["yaml_section", "toml_section", "json_data"] for t in types):
        return "configuration"

    # Test chunks
    if any(s.get("is_test", False) or s.get("is_test_class", False) for s in segments):
        return "tests"

    return "mixed"

def _generate_chunk_description(segments: list[dict[str, Any]]) -> str:
    """Generate descriptive text for a chunk based on its segments."""
    if not segments:
        return "content"

    chunk_type = _determine_chunk_type(segments)
    types: dict[str, int] = {}
    names = []

    for segment in segments:
        seg_type = segment["type"]
        types[seg_type] = types.get(seg_type, 0) + 1

        if "name" in segment:
            names.append(segment["name"])

    # Generate type-specific descriptions
    if chunk_type == "foundation":
        parts = []
        if types.get("module_docstring"):
            parts.append("module docs")
        if types.get("imports") or types.get("import"):
            import_count = sum(types.get(t, 0) for t in ["imports", "import"])
            parts.append(f"{import_count} import groups")
        if types.get("constant") or types.get("variable"):
            const_count = types.get("constant", 0) + types.get("variable", 0)
            parts.append(f"{const_count} definitions")
        return "Foundation: " + ", ".join(parts)

    elif chunk_type == "implementation":
        parts = []
        if types.get("class"):
            class_names = [s["name"] for s in segments if s["type"] == "class" and "name" in s]
            if len(class_names) <= 2:
                parts.append(f"class {', '.join(class_names)}")
            else:
                parts.append(f"{types['class']} classes")

        if types.get("function") or types.get("arrow_function"):
            func_count = types.get("function", 0) + types.get("arrow_function", 0)
            func_names = [s["name"] for s in segments if s["type"] in ["function", "arrow_function"] and "name" in s]
            if len(func_names) <= 3:
                parts.append(f"functions: {', '.join(func_names)}")
            else:
                parts.append(f"{func_count} functions")

        return "Implementation: " + ", ".join(parts)

    elif chunk_type == "documentation":
        parts = []
        for segment in segments:
            if segment["type"] == "title":
                parts.append("title")
            elif segment["type"] == "section":
                parts.append(f"section '{segment['name']}'")
            elif segment["type"] == "code_block":
                parts.append(f"{segment['name']} example")
        return "Documentation: " + ", ".join(parts[:3])

    elif chunk_type == "tests":
        test_count = sum(1 for s in segments if s.get("is_test") or s.get("is_test_class"))
        return f"Tests: {test_count} test cases"

    else:
        # Generic description
        return f"{chunk_type}: {len(segments)} components"

def _generate_detailed_commit_message(chunk: dict[str, Any], chunk_index: int,
                                     total_chunks: int, file_name: str,
                                     config: dict[str, Any]) -> str:
    """Generate a detailed, meaningful commit message for a chunk."""
    prefix = config.get('commit_message_prefix', 'feat: ')
    use_conventional_commits = config.get('use_conventional_commits', True)

    # Determine commit type based on chunk content
    chunk_type = chunk.get('chunk_type', 'mixed')
    segments = chunk.get('segments', [])

    if use_conventional_commits:
        # Use conventional commit format
        if chunk_type == "tests":
            commit_type = "test"
        elif chunk_type == "documentation":
            commit_type = "docs"
        elif chunk_type == "configuration":
            commit_type = "chore"
        elif any(s.get("is_test", False) for s in segments):
            commit_type = "test"
        else:
            commit_type = "feat"

        prefix = f"{commit_type}: "

    # Build the header
    if total_chunks == 1:
        header = f"{prefix}add {file_name}"
    else:
        header = f"{prefix}add {file_name} (part {chunk_index + 1}/{total_chunks})"

    # Build detailed description
    description_lines = []

    # Add chunk type description
    if chunk_type == "foundation":
        description_lines.append("Add foundational components:")
    elif chunk_type == "implementation":
        description_lines.append("Implement core functionality:")
    elif chunk_type == "documentation":
        description_lines.append("Add documentation:")
    elif chunk_type == "tests":
        description_lines.append("Add test cases:")
    elif chunk_type == "configuration":
        description_lines.append("Add configuration:")

    # Add specific details about what's in this chunk
    for segment in segments[:5]:  # Limit to first 5 segments
        seg_type = segment['type']

        if seg_type == 'class' and 'name' in segment:
            methods = segment.get('method_count', 0)
            desc = f"- Class `{segment['name']}`"
            if methods > 0:
                desc += f" with {methods} methods"
            description_lines.append(desc)

        elif seg_type == 'function' and 'name' in segment:
            desc = f"- Function `{segment['name']}`"
            if segment.get('is_async'):
                desc = f"- Async function `{segment['name']}`"
            elif segment.get('is_generator'):
                desc = f"- Generator `{segment['name']}`"
            description_lines.append(desc)

        elif seg_type == 'imports':
            import_count = segment.get('import_count', 0)
            if import_count > 0:
                description_lines.append(f"- {import_count} import statements")

        elif seg_type == 'section' and 'name' in segment:
            description_lines.append(f"- Section: {segment['name']}")

        elif seg_type == 'module_docstring':
            description_lines.append("- Module documentation")

    if len(segments) > 5:
        description_lines.append(f"- ...and {len(segments) - 5} more components")

    # Add line count
    description_lines.append(f"\nAdds {chunk['line_count']} lines")

    # Combine into final message
    if len(description_lines) > 1:
        message = header + "\n\n" + "\n".join(description_lines)
    else:
        message = header

    return message.strip()

def _generate_final_summary_message(chunks: list[dict[str, Any]], file_path: str,
                                   structure: dict[str, Any], config: dict[str, Any]) -> str:
    """Generate a comprehensive summary commit message for the entire file."""
    use_conventional_commits = config.get('use_conventional_commits', True)
    file_name = os.path.basename(file_path)
    file_ext = Path(file_path).suffix

    # Determine overall commit type
    if structure.get("has_tests") or 'test' in file_name.lower():
        commit_type = "test"
    elif file_ext in ['.md', '.markdown', '.rst', '.txt']:
        commit_type = "docs"
    elif file_ext in ['.yml', '.yaml', '.json', '.toml', '.ini', '.cfg']:
        commit_type = "chore"
    else:
        commit_type = "feat"

    if use_conventional_commits:
        prefix = f"{commit_type}: "
    else:
        prefix = config.get('commit_message_prefix', 'Auto-commit: ')

    # Build the summary message
    header = f"{prefix}complete {file_name} implementation"

    # Build summary statistics
    total_lines = structure['total_lines']
    total_chunks = len(chunks)
    complexity = structure.get('complexity_score', 0)

    # Count different types of content
    component_counts = {
        'classes': len(structure.get('classes', [])),
        'functions': len(structure.get('functions', [])),
        'imports': len(structure.get('imports', [])),
        'tests': sum(1 for c in chunks if c.get('chunk_type') == 'tests')
    }

    # Build the description
    lines = [
        header,
        "",
        f"Completed implementation of {file_name} with {total_lines} lines in {total_chunks} commits.",
        ""
    ]

    # Add purpose based on analysis
    if structure.get("has_tests"):
        lines.append("Purpose: Test suite for validating functionality")
    elif structure.get("has_main"):
        lines.append("Purpose: Executable script with main entry point")
    elif component_counts['classes'] > 0:
        lines.append(f"Purpose: Module defining {component_counts['classes']} classes")
    elif file_ext in ['.md', '.markdown']:
        lines.append("Purpose: Documentation")
    else:
        lines.append("Purpose: Utility module")

    lines.append("")
    lines.append("Components added:")

    # List components
    if component_counts['classes'] > 0:
        class_names = structure.get('classes', [])[:5]
        lines.append(f"- {component_counts['classes']} classes: {', '.join(class_names)}")
        if len(structure.get('classes', [])) > 5:
            lines.append(f"  (and {len(structure.get('classes', [])) - 5} more)")

    if component_counts['functions'] > 0:
        func_names = structure.get('functions', [])[:5]
        lines.append(f"- {component_counts['functions']} functions: {', '.join(func_names)}")
        if len(structure.get('functions', [])) > 5:
            lines.append(f"  (and {len(structure.get('functions', [])) - 5} more)")

    if component_counts['imports'] > 0:
        lines.append(f"- {component_counts['imports']} imports")

    # Add chunk summary
    lines.extend([
        "",
        "Implementation approach:",
        f"- Split into {total_chunks} logical chunks",
        f"- Complexity score: {complexity}",
        f"- File type: {file_ext or 'unknown'}"
    ])

    # Add completion marker
    lines.extend([
        "",
        "✅ File creation completed successfully"
    ])

    return "\n".join(lines)

def should_use_chunked_creation(file_path: str, content: str, config: dict[str, Any]) -> bool:
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

    # Check file type
    file_ext = Path(file_path).suffix.lower()
    supported_types = config.get('supported_extensions', [
        '.py', '.js', '.ts', '.jsx', '.tsx', '.mojo', '.🔥',
        '.md', '.markdown', '.c', '.cpp', '.h', '.hpp',
        '.rs', '.go', '.java', '.yaml', '.yml', '.json', '.toml'
    ])

    if file_ext not in supported_types:
        return False

    # Analyze structure complexity
    structure = analyze_code_structure(content, file_path)
    segment_count = len(structure.get('segments', []))

    # Use chunking if file has multiple logical segments
    min_segments = config.get('min_segments_for_chunking', 3)

    # Special cases
    if structure.get('complexity_score', 0) > 50:
        return True  # High complexity files should always be chunked

    if structure.get('has_tests') and line_count > 30:
        return True  # Test files benefit from chunking

    return segment_count >= min_segments

def load_chunking_config() -> dict[str, Any]:
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
        "commit_message_prefix": "feat: ",
        "use_conventional_commits": True,
        "intelligent_grouping": True,
        "supported_extensions": [
            ".py", ".js", ".ts", ".jsx", ".tsx", ".mojo", ".🔥",
            ".md", ".markdown", ".c", ".cpp", ".h", ".hpp",
            ".rs", ".go", ".java", ".yaml", ".yml", ".json", ".toml"
        ]
    }

    try:
        settings_path = Path(__file__).parent.parent / "settings.json"
        if settings_path.exists():
            with open(settings_path) as f:
                settings = json.load(f)
                chunking_config = settings.get("chunked_file_creation", {})
                # Merge with defaults
                for key, value in chunking_config.items():
                    default_config[key] = value
    except Exception as e:
        print(f"Warning: Could not load settings: {e}", file=sys.stderr)

    return default_config

def perform_chunked_file_creation(file_path: str, content: str, config: dict[str, Any]) -> bool:
    """
    Create a large file using enhanced security guardian with cross-fragment validation.
    
    Args:
        file_path: Path to the file to create
        content: Complete file content
        config: Configuration settings
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"🔨 Starting enhanced secure chunked file creation for {file_path}", file=sys.stderr)

    # Use enhanced security guardian for secure chunked file creation
    try:
        success = secure_chunked_file_creation(file_path, content, config)
        if success:
            print("✅ Enhanced secure chunked file creation completed successfully", file=sys.stderr)
        else:
            print("❌ Enhanced secure chunked file creation failed security validation", file=sys.stderr)
            # Fallback to original implementation
            print("⚠️  Falling back to original chunked file creation implementation", file=sys.stderr)
            success = _fallback_chunked_file_creation(file_path, content, config)
        return success

    except Exception as e:
        print(f"ERROR: Enhanced security guardian failed: {e}", file=sys.stderr)
        # Fallback to original implementation for emergency cases
        print("⚠️  Falling back to original chunked file creation implementation", file=sys.stderr)
        return _fallback_chunked_file_creation(file_path, content, config)


def _fallback_chunked_file_creation(file_path: str, content: str, config: dict[str, Any]) -> bool:
    """Fallback chunked file creation implementation."""
    try:
        print(f"🔨 Starting fallback chunked file creation for {file_path}", file=sys.stderr)

        # Analyze code structure
        structure = analyze_code_structure(content, file_path)
        print("📊 Analysis complete:", file=sys.stderr)
        print(f"   - Total lines: {structure['total_lines']}", file=sys.stderr)
        print(f"   - Segments found: {len(structure.get('segments', []))}", file=sys.stderr)
        print(f"   - Complexity score: {structure.get('complexity_score', 0)}", file=sys.stderr)

        # Create logical chunks
        chunks = create_logical_chunks(structure, content, config)

        print(f"📦 Created {len(chunks)} logical chunks:", file=sys.stderr)
        for i, chunk in enumerate(chunks):
            print(f"   - Chunk {i+1}: {chunk['description']} ({chunk['line_count']} lines)",
                  file=sys.stderr)

        # Create file incrementally
        file_dir = os.path.dirname(file_path)
        if file_dir and not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)

        accumulated_content = ""
        file_name = os.path.basename(file_path)

        # Create progress tracking
        total_lines_written = 0

        for i, chunk in enumerate(chunks):
            print(f"\n🔧 Processing chunk {i+1}/{len(chunks)}", file=sys.stderr)

            # Add chunk content to accumulated content
            if accumulated_content and not accumulated_content.endswith('\n'):
                accumulated_content += '\n'
            accumulated_content += chunk['content']

            # Write current accumulated content to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(accumulated_content)

            # Stage the file
            result = subprocess.run(['git', 'add', file_path],
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ git add failed: {result.stderr}", file=sys.stderr)
                return False

            # Generate detailed commit message
            commit_msg = _generate_detailed_commit_message(chunk, i, len(chunks),
                                                          file_name, config)

            # Use git-commit-guardian for secure, validated commits
            commit_success = request_guardian_commit(
                files=[file_path],
                commit_message=commit_msg,
                change_type="create",
                auto_generated=True
            )

            if not commit_success:
                print("🛡️  Git-commit-guardian failed, attempting fallback", file=sys.stderr)
                # Fallback to direct commit if guardian fails
                commit_success = fallback_to_direct_commit([file_path], commit_msg)

            if not commit_success:
                print(f"❌ Failed to commit chunk {i+1}", file=sys.stderr)
                return False

            total_lines_written += chunk['line_count']
            progress = (total_lines_written / structure['total_lines']) * 100
            print(f"✅ Committed chunk {i+1}/{len(chunks)} - Progress: {progress:.1f}%",
                  file=sys.stderr)

        # Create final comprehensive summary commit
        print("\n📝 Creating final summary commit...", file=sys.stderr)

        # Generate comprehensive summary message
        summary_message = _generate_final_summary_message(chunks, file_path, structure, config)

        # Note: Git commit amend is handled directly as it's a special case
        # The git-commit-guardian doesn't handle --amend operations
        result = subprocess.run(['git', 'commit', '--amend', '-m', summary_message],
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"⚠️ Summary commit amendment failed: {result.stderr}", file=sys.stderr)
            # Non-fatal - the file was still created successfully
        else:
            print("✅ Final summary commit created successfully", file=sys.stderr)

        print("\n🎉 Enhanced chunked file creation completed successfully!", file=sys.stderr)
        print(f"   - File: {file_path}", file=sys.stderr)
        print(f"   - Total lines: {structure['total_lines']}", file=sys.stderr)
        print(f"   - Chunks created: {len(chunks)}", file=sys.stderr)
        print(f"   - Commits made: {len(chunks)}", file=sys.stderr)

        return True

    except Exception as e:
        print(f"❌ Chunked file creation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return False

def check_chunked_file_creation(tool_name: str, tool_input: dict[str, Any]) -> bool:
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
        print(f"🚀 Large file creation detected: {file_path}", file=sys.stderr)
        print(f"   - Lines: {len(content.splitlines())}", file=sys.stderr)
        print(f"   - Size: {len(content)} bytes", file=sys.stderr)
        print("📦 Using enhanced chunked file creation approach v2", file=sys.stderr)

        # Perform chunked creation
        if perform_chunked_file_creation(file_path, content, config):
            print("✅ Chunked file creation completed successfully", file=sys.stderr)
            return True  # Block original Write operation
        else:
            print("❌ Chunked file creation failed, falling back to normal creation",
                  file=sys.stderr)
            return False

    return False

# For testing
if __name__ == "__main__":
    # Test the analysis functions
    test_content = '''#!/usr/bin/env python3
"""Test module for chunking system."""

import os
import sys
from pathlib import Path

MAX_SIZE = 1000
DEBUG = True

class DataProcessor:
    """Process data."""
    
    def __init__(self):
        self.data = []
    
    def process(self):
        return "processed"

def main():
    """Main entry point."""
    processor = DataProcessor()
    print(processor.process())

if __name__ == "__main__":
    main()
'''

    structure = analyze_code_structure(test_content, "test.py")
    print(json.dumps(structure, indent=2))
