import ast
import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from multiprocessing import Value, Manager
from typing import Tuple, Dict, Any


class TimeoutException(Exception):
    """Exception raised when code execution times out."""

    pass


def timeout_handler(signum, frame):
    """Signal handler for timeouts."""
    raise TimeoutException("Code execution timed out")


def extract_imports_from_prompt(prompt):
    """Extract import statements from the prompt text."""
    if not prompt:
        return ""

    # Find all import statements in the prompt
    import_patterns = [
        r"^from\s+[\w\.]+\s+import\s+.*$",  # from module import ...
        r"^import\s+[\w\.,\s]+$",  # import module, module2
    ]

    imports = []
    lines = prompt.split("\n")

    for line in lines:
        line = line.strip()
        for pattern in import_patterns:
            if re.match(pattern, line):
                imports.append(line)
                break

    # Remove duplicates while preserving order
    seen = set()
    unique_imports = []
    for imp in imports:
        if imp not in seen:
            seen.add(imp)
            unique_imports.append(imp)

    return "\n".join(unique_imports)


def cleanup_code(code):
    """
    Extract function definitions from code that may contain explanatory text,
    markdown, and other non-executable content.
    """
    if not code or not isinstance(code, str):
        return ""

    # Step 1: Remove markdown code blocks but keep the content
    code = re.sub(r"```python\s*\n?", "", code)
    code = re.sub(r"```\s*\n?", "", code)

    # Step 2: Split into lines for processing
    lines = code.split("\n")

    # Step 3: Find and extract function definitions
    function_blocks = []
    current_function = []
    in_function = False
    base_indent = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check if this line starts a function definition
        if re.match(r"^(\s*)def\s+\w+\s*\(", line):
            # If we were already in a function, save it
            if in_function and current_function:
                function_blocks.append("\n".join(current_function))

            # Start new function
            current_function = [line]
            in_function = True
            base_indent = len(line) - len(line.lstrip())

        elif in_function:
            # We're inside a function definition
            if stripped == "":
                # Empty line - include it
                current_function.append(line)
            elif line.startswith(" " * (base_indent + 1)) or line.startswith("\t"):
                # This line is indented more than the function def - it's part of the function
                current_function.append(line)
            elif stripped.startswith('"""') or stripped.startswith("'''"):
                # Handle docstrings that might be at the same level as def
                current_function.append(line)
                # Look for closing docstring
                quote_type = '"""' if stripped.startswith('"""') else "'''"
                if not (stripped.endswith(quote_type) and len(stripped) > 3):
                    # Multi-line docstring, find the end
                    i += 1
                    while i < len(lines):
                        current_function.append(lines[i])
                        if quote_type in lines[i]:
                            break
                        i += 1
            else:
                # This line is not part of the function (back to base level or less)
                # End current function
                if current_function:
                    function_blocks.append("\n".join(current_function))
                current_function = []
                in_function = False

                # Check if this line starts a new function
                if re.match(r"^(\s*)def\s+\w+\s*\(", line):
                    current_function = [line]
                    in_function = True
                    base_indent = len(line) - len(line.lstrip())

        i += 1

    # Don't forget the last function if we ended while still in one
    if in_function and current_function:
        function_blocks.append("\n".join(current_function))

    # Step 4: Join all function blocks
    result = "\n\n".join(function_blocks)

    # Step 5: Final cleanup - remove any remaining explanatory text that might have slipped through
    # Remove lines that look like natural language explanations
    lines = result.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        # Skip lines that are clearly natural language (but keep comments and docstrings)
        if (
            stripped
            and not stripped.startswith("#")
            and not stripped.startswith('"""')
            and not stripped.startswith("'''")
            and not any(
                keyword in line
                for keyword in [
                    "def ",
                    "return ",
                    "if ",
                    "for ",
                    "while ",
                    "=",
                    "(",
                    ")",
                    "[",
                    "]",
                ]
            )
            and re.match(r"^[A-Z][a-z].*[.!?:]$", stripped)
        ):
            continue
        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines).strip()

    # Step 6: Validate that we have actual function definitions
    if not re.search(r"def\s+\w+\s*\(", result):
        return ""

    return result


# ============================================================================
# BigCodeBench-specific cleanup and extraction functions
# ============================================================================

def cleanup_code_bigcodebench(code: str) -> str:
    """
    Clean up model output for BigCodeBench tasks.
    
    Handles cases where:
    - Output contains markdown code blocks
    - Output starts with function parameters (prompt ended with "def func(")
    - Output contains explanatory text before/after code
    """
    if not code or not isinstance(code, str):
        return ""
    
    # Step 1: Remove markdown code blocks
    code = re.sub(r"```python\s*\n?", "", code)
    code = re.sub(r"```\s*\n?", "", code)
    
    # Step 2: Remove common boilerplate phrases
    boilerplate_patterns = [
        r"^Here's?\s*(the|a|my)?\s*(implementation|solution|code|function).*?:\s*\n?",
        r"^Sure[,!]?\s*(here's?|I'll).*?:\s*\n?",
        r"^I'll\s*(write|create|implement).*?:\s*\n?",
        r"^Let me\s*(write|create|implement).*?:\s*\n?",
        r"^The\s+(aux|helper|main)\s+function.*?:\s*\n?",
    ]
    for pattern in boilerplate_patterns:
        code = re.sub(pattern, "", code, flags=re.IGNORECASE | re.MULTILINE)
    
    # Step 3: Remove trailing explanations
    # Find where actual code ends (last line with code-like content)
    lines = code.split("\n")
    code_end_idx = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped and (
            stripped.startswith(("def ", "return ", "if ", "for ", "while ", "class ", "#", '"""', "'''"))
            or "=" in stripped
            or stripped.startswith((")", "]", "}"))
            or re.match(r"^\s+\S", lines[i])  # indented line
        ):
            code_end_idx = i + 1
            break
    
    code = "\n".join(lines[:code_end_idx])
    
    return code.strip()


def extract_aux_function_bigcodebench(raw_output: str, prompt_ended_with_def: bool = True) -> str:
    """
    Extract aux function from model output for BigCodeBench.
    
    Args:
        raw_output: Raw model output
        prompt_ended_with_def: If True, prompt ended with "def aux(" so output starts with params
    
    Returns:
        Complete aux function definition
    """
    if not raw_output:
        return ""
    
    cleaned = cleanup_code_bigcodebench(raw_output)
    if not cleaned:
        return ""
    
    # Case 1: Output already contains "def aux("
    if re.search(r"def\s+aux\s*\(", cleaned):
        return _extract_function_by_name(cleaned, "aux")
    
    # Case 2: Prompt ended with "def aux(" so output starts with parameters
    if prompt_ended_with_def:
        # Prepend "def aux(" to complete the function
        completed = "def aux(" + cleaned
        # Try to extract the function
        extracted = _extract_function_by_name(completed, "aux")
        if extracted:
            return extracted
        # If extraction failed, the output might just be the body
        # Try to construct a minimal function
        return _construct_function_from_body(cleaned, "aux")
    
    return ""


def extract_main_function_bigcodebench(
    raw_output: str, 
    entry_point: str,
    func_signature: str = "",
    prompt_ended_with_signature: bool = True
) -> str:
    """
    Extract main function from model output for BigCodeBench.
    
    Args:
        raw_output: Raw model output
        entry_point: Name of the main function (e.g., "task_func")
        func_signature: Full function signature (e.g., "def task_func(x, y):")
        prompt_ended_with_signature: If True, prompt ended with signature so output is body only
    
    Returns:
        Complete main function definition
    """
    if not raw_output:
        return ""
    
    cleaned = cleanup_code_bigcodebench(raw_output)
    if not cleaned:
        return ""
    
    # Case 1: Output already contains the function definition
    if re.search(rf"def\s+{re.escape(entry_point)}\s*\(", cleaned):
        return _extract_function_by_name(cleaned, entry_point)
    
    # Case 2: Output is just the function body
    if prompt_ended_with_signature and func_signature:
        # Ensure signature ends with colon
        sig = func_signature.strip()
        if not sig.endswith(":"):
            sig += ":"
        
        # Indent the body if not already indented
        body_lines = cleaned.split("\n")
        indented_body = []
        for line in body_lines:
            if line.strip():  # Non-empty line
                if not line.startswith(("    ", "\t")):
                    line = "    " + line
            indented_body.append(line)
        
        completed = sig + "\n" + "\n".join(indented_body)
        return completed.strip()
    
    return ""


def extract_chaining_output_bigcodebench(
    raw_output: str,
    entry_point: str
) -> tuple:
    """
    Extract both aux and main functions from chaining mode output for BigCodeBench.
    
    In chaining mode, Agent 2 outputs both the predicted aux function and the main function.
    
    Args:
        raw_output: Raw model output containing both functions
        entry_point: Name of the main function
    
    Returns:
        Tuple of (predicted_aux_code, main_code)
    """
    if not raw_output:
        return "", ""
    
    # Prompt ended with "def aux(" so prepend it
    cleaned = "def aux(" + cleanup_code_bigcodebench(raw_output)
    
    # Extract aux function
    aux_code = _extract_function_by_name(cleaned, "aux")
    
    # Extract main function
    main_code = _extract_function_by_name(cleaned, entry_point)
    
    return aux_code, main_code


def _extract_function_by_name(code: str, function_name: str) -> str:
    """
    Extract a complete function definition by name from code.
    More robust than extract_specific_function for BigCodeBench.
    """
    if not code or not function_name:
        return ""
    
    lines = code.split("\n")
    function_lines = []
    in_function = False
    base_indent = 0
    paren_depth = 0
    in_signature = False
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Check for function start
        match = re.match(rf"^(\s*)def\s+{re.escape(function_name)}\s*\(", line)
        if match and not in_function:
            in_function = True
            in_signature = True
            base_indent = len(match.group(1))
            function_lines = [line]
            
            # Count parentheses to handle multi-line signatures
            paren_depth = line.count("(") - line.count(")")
            if paren_depth == 0 and ":" in line:
                in_signature = False
            
            i += 1
            continue
        
        if in_function:
            if in_signature:
                # Continue collecting signature lines
                function_lines.append(line)
                paren_depth += line.count("(") - line.count(")")
                if paren_depth <= 0 and ":" in line:
                    in_signature = False
            else:
                # Check if still in function body
                current_indent = len(line) - len(line.lstrip()) if stripped else base_indent + 4
                
                if stripped == "":
                    # Empty line - include if we have content after
                    function_lines.append(line)
                elif current_indent > base_indent:
                    # Indented content - part of function
                    function_lines.append(line)
                elif stripped.startswith(("#", '"""', "'''")):
                    # Comment or docstring at function level
                    function_lines.append(line)
                else:
                    # Back to base indent or new definition - end of function
                    break
        
        i += 1
    
    # Remove trailing empty lines
    while function_lines and not function_lines[-1].strip():
        function_lines.pop()
    
    result = "\n".join(function_lines)
    
    # Validate we got a complete function
    if not re.search(rf"def\s+{re.escape(function_name)}\s*\(", result):
        return ""
    
    return result


def _construct_function_from_body(body: str, function_name: str) -> str:
    """
    Construct a function from just the body when we can't parse it properly.
    Used as a fallback when the model output is malformed.
    """
    if not body:
        return ""
    
    lines = body.split("\n")
    
    # Try to find if there's a return statement to infer it's function body
    has_return = any("return " in line for line in lines)
    
    if not has_return:
        return ""
    
    # Indent body if needed
    indented_lines = []
    for line in lines:
        if line.strip():
            if not line.startswith(("    ", "\t")):
                line = "    " + line
        indented_lines.append(line)
    
    # Create minimal function
    return f"def {function_name}():\n" + "\n".join(indented_lines)


def extract_specific_function(code, function_name):
    """
    Extract a specific function by name from the code.
    This is useful when you know exactly which function you want.
    """
    if not code or not function_name:
        return ""

    # First clean the code
    cleaned = cleanup_code(code)

    # Then extract the specific function
    lines = cleaned.split("\n")
    function_lines = []
    in_target_function = False
    base_indent = 0

    for line in lines:
        if re.match(rf"^(\s*)def\s+{re.escape(function_name)}\s*\(", line):
            function_lines = [line]
            in_target_function = True
            base_indent = len(line) - len(line.lstrip())
        elif in_target_function:
            if (
                line.strip() == ""
                or line.startswith(" " * (base_indent + 1))
                or line.startswith("\t")
            ):
                function_lines.append(line)
            else:
                # End of function
                break

    return "\n".join(function_lines).strip()


def check_function_definition(code, function_name, description):
    """Check if a function is properly defined with return statement."""
    # Extract just the specific function
    func_code = extract_specific_function(code, function_name)

    if not func_code:
        return False, f"{description} not defined"

    # Remove docstrings before checking for return statement
    # This prevents matching "return" in docstring text like "return string representation"
    func_code_no_docstring = _remove_docstrings(func_code)
    
    # Check if function has return statement (outside of docstrings)
    has_return = re.search(r"return\s+", func_code_no_docstring) is not None

    if not has_return:
        return False, f"{description} missing return statement"

    return True, f"{description} properly defined with return statement"


def _remove_docstrings(code):
    """Remove docstrings from code to avoid false positives in return statement detection."""
    if not code:
        return code
    
    # Remove triple-quoted strings (docstrings)
    # Handle both """ and ''' styles
    result = re.sub(r'"""[\s\S]*?"""', '', code)
    result = re.sub(r"'''[\s\S]*?'''", '', result)
    
    return result


def check_syntax(code, description):
    """Check syntax of code."""
    try:
        ast.parse(code)
        return True, f"{description} syntax OK"
    except SyntaxError as e:
        return False, f"{description} has syntax error: {str(e)}"


def concatenate_functions(aux_completion, main_completion, imports=""):
    """Concatenate imports, aux and main functions."""
    aux_clean = cleanup_code(aux_completion)
    main_clean = cleanup_code(main_completion)

    # Build the combined code with imports first
    parts = []

    if imports:
        parts.append(imports)

    if aux_clean:
        parts.append(aux_clean)

    if main_clean:
        parts.append(main_clean)

    combined_code = "\n\n".join(parts)
    return combined_code


def extract_test_cases(test_code, entry_point):
    """Extract individual test cases from the test code and replace candidate with entry_point."""
    if not test_code or not entry_point:
        return []

    # Find the check function
    check_match = re.search(
        r"def check\(candidate\):(.*?)(?=\n\ndef|\Z)", test_code, re.DOTALL
    )
    if not check_match:
        return []

    check_body = check_match.group(1)

    # Find all assert statements - using a more general approach
    # Pattern 1: Direct candidate calls with ==
    direct_equality_pattern = r"assert\s+candidate\([^)]*\)\s*==\s*[^\n]+"

    # Pattern 2: Simple abs difference pattern
    simple_abs_pattern = r"assert\s+abs\(candidate\([^)]*\)\s*-\s*[^)]+\)\s*<\s*[^\n]+"

    # Pattern 3: Math.fabs or other function calls containing candidate
    math_fabs_pattern = r"assert\s+math\.fabs\([^)]*candidate[^)]*\)\s*<\s*[^\n]+"

    # Pattern 4: General assert statements that contain "candidate" - catch-all
    general_candidate_pattern = r"assert\s+[^\n]*candidate[^\n]*"

    # Try patterns in order of specificity (most specific first)
    patterns = [
        direct_equality_pattern,
        simple_abs_pattern,
        math_fabs_pattern,
        general_candidate_pattern,
    ]

    all_matches = []
    check_lines = check_body.strip().split("\n")

    # Process each line to find assert statements
    for line in check_lines:
        line = line.strip()
        if line.startswith("assert") and "candidate" in line:
            # Check if this line matches any of our specific patterns
            matched = False
            for pattern in patterns[:-1]:  # Don't use general pattern in this loop
                if re.match(pattern, line):
                    all_matches.append(line)
                    matched = True
                    break

            # If no specific pattern matched, use the general pattern as fallback
            if not matched and re.match(general_candidate_pattern, line):
                all_matches.append(line)

    # Remove duplicates while preserving order
    seen = set()
    unique_matches = []
    for match in all_matches:
        if match not in seen:
            seen.add(match)
            unique_matches.append(match)

    # Replace 'candidate' with actual entry_point name
    test_cases = []
    for match in unique_matches:
        test_case = match.replace("candidate", entry_point)
        test_cases.append(test_case)

    return test_cases


def check_aux_function_usage(main_code, aux_function_name="aux"):
    """Check if the main function calls the auxiliary function, including indirect usage."""
    if not main_code or not aux_function_name:
        return False

    # Pattern 1: Direct function calls - aux(...)
    direct_call_pattern = rf"\b{re.escape(aux_function_name)}\s*\("

    # Pattern 2: Aux function used in assignments - var = aux(...)
    assignment_pattern = rf"=\s*{re.escape(aux_function_name)}\s*\("

    # Pattern 3: Aux function used in expressions - return aux(...) + something
    expression_pattern = rf"\b{re.escape(aux_function_name)}\s*\([^)]*\)"

    # Check for any of these patterns
    patterns = [direct_call_pattern, assignment_pattern, expression_pattern]

    for pattern in patterns:
        if re.search(pattern, main_code):
            return True

    return False


def is_wrapper_function(main_code, aux_function_name="aux"):
    """
    Check if the main function is just a simple wrapper around the aux function.
    A wrapper is defined as a function that:
    1. Calls aux function once
    2. Directly returns the result without significant processing
    """
    if not main_code or not aux_function_name:
        return False

    # Remove comments and docstrings for cleaner analysis
    lines = main_code.split("\n")
    code_lines = []
    in_docstring = False

    for line in lines:
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            continue

        # Handle docstrings
        if stripped.startswith('"""') or stripped.startswith("'''"):
            quote_type = '"""' if stripped.startswith('"""') else "'''"
            if stripped.endswith(quote_type) and len(stripped) > 3:
                # Single line docstring, skip it
                continue
            else:
                # Multi-line docstring start
                in_docstring = True
                continue
        elif in_docstring:
            if '"""' in stripped or "'''" in stripped:
                in_docstring = False
            continue

        # Skip comments
        if stripped.startswith("#"):
            continue

        # Skip function definition line
        if stripped.startswith("def "):
            continue

        code_lines.append(stripped)

    # Now analyze the actual code content
    if len(code_lines) == 0:
        return True  # Empty function body is considered a wrapper

    # Count aux function calls
    aux_call_pattern = rf"\b{re.escape(aux_function_name)}\s*\("
    aux_calls = sum(1 for line in code_lines if re.search(aux_call_pattern, line))

    # Pattern 1: Single line that calls aux and returns directly
    # e.g., "return aux(...)"
    if len(code_lines) == 1:
        line = code_lines[0]
        if line.startswith("return ") and aux_function_name in line:
            return True

    # Pattern 2: Two lines - assign aux result and return it
    # e.g., "result = aux(...)" followed by "return result"
    elif len(code_lines) == 2:
        first_line = code_lines[0]
        second_line = code_lines[1]

        # Check if first line assigns aux result to a variable
        aux_assignment_pattern = rf"(\w+)\s*=\s*{re.escape(aux_function_name)}\s*\("
        assignment_match = re.search(aux_assignment_pattern, first_line)

        if assignment_match:
            var_name = assignment_match.group(1)
            # Check if second line just returns that variable
            if second_line == f"return {var_name}":
                return True

    return False


def check_aux_call_without_assignment(main_code, aux_function_name="aux"):
    """
    Check if aux function is called but its return value is ignored (not assigned).
    Uses bad patterns approach - looks for standalone aux calls that waste the return value.

    Returns:
        bool: True if aux is called without assignment (should deduct points)
        list: List of problematic aux calls found
    """

    if not main_code or aux_function_name not in main_code:
        return False, []

    # Find all aux function calls
    aux_call_pattern = rf"{re.escape(aux_function_name)}\s\([^)]\)"

    problematic_calls = []
    lines = main_code.split("\n")

    for line_num, line in enumerate(lines, 1):
        stripped_line = line.strip()

        # Skip empty lines and comments
        if not stripped_line or stripped_line.startswith("#"):
            continue

        # If line contains aux call, check if it's a bad pattern
        if re.search(aux_call_pattern, stripped_line):

            # Bad patterns (aux result is wasted):
            bad_patterns = [
                # Standalone aux call (just aux(...) on its own line)
                rf"^{re.escape(aux_function_name)}\s\([^)]\)$",
                # Aux call followed by semicolon (if someone uses semicolons)
                rf"^{re.escape(aux_function_name)}\s\([^)]\);?\s$",
                # Aux call in expression statement that doesn't use the result
                # (This is the most common bad pattern)
                rf"^\s{re.escape(aux_function_name)}\s\([^)]\)\s*$",
            ]

            # Check if this line matches any bad pattern
            is_bad_usage = any(
                re.search(pattern, stripped_line) for pattern in bad_patterns
            )

            if is_bad_usage:
                problematic_calls.append(f"Line {line_num}: {stripped_line}")

    return len(problematic_calls) > 0, problematic_calls


def check_aux_call_without_usage(main_code, aux_function_name="aux"):
    """
    Check if aux function is called but its return value is ignored (not assigned).
    Uses bad patterns approach - looks for standalone aux calls that waste the return value.

    Returns:
        bool: True if aux is called without assignment (should deduct points)
        list: List of problematic aux calls found
    """
    if not main_code or aux_function_name not in main_code:
        return False, []

    # AST parse may fail on incomplete or special-token-laden generations
    try:
        tree = ast.parse(main_code)
    except SyntaxError:
        return False, []

    analyzer = AuxUsageAnalyzer(aux_function_name, main_code)
    analyzer.visit(tree)
    problematic_calls = analyzer.get_problematic_calls()

    return len(problematic_calls) > 0, problematic_calls


class AuxUsageAnalyzer(ast.NodeVisitor):
    def __init__(self, function_name, code):
        self.function_name = function_name
        self.code = code
        self.parent_map = {}
        self.problematic_calls = []
        self.variable_usage = {}  # Track variable usage

    def visit(self, node):
        # Build parent-child relationships
        for child in ast.iter_child_nodes(node):
            self.parent_map[child] = node
        return super().visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == self.function_name:

            # Check the context of this aux call
            context = self._get_call_context(node)

            if context["is_problematic"]:
                self.problematic_calls.append(
                    {
                        "line": node.lineno,
                        "reason": context["reason"],
                        "variable": context.get("variable"),
                    }
                )

        self.generic_visit(node)

    def visit_Name(self, node):
        # Track variable usage (when variable is being read)
        if isinstance(node.ctx, ast.Load):
            if node.id not in self.variable_usage:
                self.variable_usage[node.id] = []
            self.variable_usage[node.id].append(node.lineno)

        self.generic_visit(node)

    def _get_call_context(self, call_node):
        """Determine if aux call usage is problematic."""
        parent = self.parent_map.get(call_node)

        if parent is None:
            return {"is_problematic": True, "reason": "standalone call (not assigned)"}

        # Good contexts - aux return value is used
        if isinstance(parent, ast.Return):
            return {"is_problematic": False, "reason": "used as return value"}

        if isinstance(parent, ast.Call):
            return {"is_problematic": False, "reason": "used as function argument"}

        if isinstance(parent, (ast.BinOp, ast.UnaryOp, ast.Compare)):
            return {"is_problematic": False, "reason": "used in expression"}

        if isinstance(parent, (ast.If, ast.While)) and parent.test == call_node:
            return {"is_problematic": False, "reason": "used in condition"}

        if isinstance(parent, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
            return {"is_problematic": False, "reason": "used in collection"}

        # Assignment context - need to check if assigned variable is used
        if isinstance(parent, ast.Assign):
            for target in parent.targets:
                if isinstance(target, ast.Name):
                    var_name = target.id

                    # Check if variable is never used
                    if self._is_variable_unused(var_name, parent.lineno):
                        return {
                            "is_problematic": True,
                            "reason": f"assigned to {var_name} but never used",
                            "variable": var_name,
                        }

                    # Check if variable is reassigned before use
                    reassign_line = self._get_reassignment_before_use(
                        var_name, parent.lineno
                    )
                    if reassign_line:
                        return {
                            "is_problematic": True,
                            "reason": f"assigned to {var_name} but reassigned on line {reassign_line} before use",
                            "variable": var_name,
                        }

            return {"is_problematic": False, "reason": "assigned and used"}

        # Standalone expression (aux() by itself)
        if isinstance(parent, ast.Expr):
            return {"is_problematic": True, "reason": "standalone call (not assigned)"}

        # Default to safe (not problematic)
        return {"is_problematic": False, "reason": "used in other context"}

    def _is_variable_unused(self, var_name, assignment_line):
        """Check if variable is never used after assignment."""
        if var_name not in self.variable_usage:
            return True

        # Check if there are any usages after the assignment line
        usage_lines = [
            line for line in self.variable_usage[var_name] if line > assignment_line
        ]
        return len(usage_lines) == 0

    def _get_reassignment_before_use(self, var_name, assignment_line):
        """Check if variable is reassigned before being used."""
        if var_name not in self.variable_usage:
            return None

        # Find reassignments and usages after the aux assignment
        reassignments = []
        usages = []

        try:
            tree = ast.parse(self.code)
        except SyntaxError:
            # If code cannot be parsed, we cannot analyze reassignments safely
            return None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and hasattr(node, "lineno")
                and node.lineno > assignment_line
            ):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        reassignments.append(node.lineno)

            if (
                isinstance(node, ast.Name)
                and node.id == var_name
                and isinstance(node.ctx, ast.Load)
                and hasattr(node, "lineno")
                and node.lineno > assignment_line
            ):
                usages.append(node.lineno)

        if not reassignments:
            return None

        first_reassignment = min(reassignments)
        first_usage = min(usages) if usages else float("inf")

        return first_reassignment if first_reassignment < first_usage else None

    def get_problematic_calls(self):
        """Return list of problematic calls with formatted messages."""
        formatted_calls = []
        for call in self.problematic_calls:
            line = call["line"]
            reason = call["reason"]
            formatted_calls.append(f"Line {line}: {reason}")

        return formatted_calls


def check_main_defines_aux_internally(main_code, aux_function_name="aux"):
    """
    Check if the main function code defines its own aux function internally.
    This is a violation of collaboration - main should use the externally provided aux function,
    not define its own.

    Args:
        main_code: The code containing the main function (completion2)
        aux_function_name: Name of the aux function to check for (default: "aux")

    Returns:
        Tuple[bool, str]: (violation_detected, message)
            - violation_detected: True if main defines its own aux function
            - message: Description of the violation or success
    """
    if not main_code or not isinstance(main_code, str):
        return False, "No main code provided"

    # Pattern to detect aux function definition in main code
    # Matches: def aux(...) with various spacing
    aux_def_pattern = rf"^\s*def\s+{re.escape(aux_function_name)}\s*\("
    
    # Search line by line for aux function definition
    lines = main_code.split("\n")
    for line_num, line in enumerate(lines, 1):
        if re.match(aux_def_pattern, line):
            return True, f"Main code defines its own '{aux_function_name}' function at line {line_num}"
    
    # Also check using AST for more robust detection
    try:
        tree = ast.parse(main_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == aux_function_name:
                return True, f"Main code defines its own '{aux_function_name}' function (detected via AST)"
    except SyntaxError:
        # If AST parsing fails, rely on regex result only
        pass
    
    return False, f"Main code does not define '{aux_function_name}' function internally"


def check_collaboration_violation(main_code, aux_function_name="aux"):
    """
    Comprehensive check for collaboration violations.
    A collaboration violation occurs when:
    1. Main function defines its own aux function instead of using the externally provided one
    2. Main function doesn't use the external aux function at all (optional, for future use)

    Args:
        main_code: The code containing the main function (completion2)
        aux_function_name: Name of the aux function to check for (default: "aux")

    Returns:
        Tuple[bool, str, dict]: (violation_detected, message, details)
            - violation_detected: True if any collaboration violation detected
            - message: Human-readable description
            - details: Dictionary with violation details
    """
    details = {
        "defines_own_aux": False,
        "uses_external_aux": False,
        "violation_type": None,
    }

    # Check if main defines its own aux
    defines_aux, aux_def_msg = check_main_defines_aux_internally(main_code, aux_function_name)
    details["defines_own_aux"] = defines_aux

    if defines_aux:
        details["violation_type"] = "self_defined_aux"
        return True, f"COLLABORATION VIOLATION: {aux_def_msg}", details

    # Check if main uses aux function (good behavior)
    uses_aux = check_aux_function_usage(main_code, aux_function_name)
    details["uses_external_aux"] = uses_aux

    if not uses_aux:
        # Not using aux at all is less severe than defining own aux
        # This may be intentional for some problems
        details["violation_type"] = "no_aux_usage"
        return False, f"Main function does not use '{aux_function_name}' function", details

    return False, "Collaboration pattern is correct: main uses external aux function", details


def extract_function_signature_from_prompt(prompt, entry_point):
    """
    Extract the function signature (name and parameters) from a prompt.
    
    Looks for function definition patterns like:
    - def entry_point(param1, param2, ...):
    - def entry_point(param1: type, param2: type) -> return_type:
    
    Args:
        prompt: The prompt text containing the function definition
        entry_point: The name of the function to find
    
    Returns:
        Tuple[str, str]: (function_name, params_string)
            - function_name: The extracted function name (should match entry_point)
            - params_string: The parameters as a string (e.g., "x, n: int")
            Returns (None, None) if not found.
    """
    if not prompt or not entry_point:
        return None, None
    
    # Pattern to match function definition with the given entry_point
    # Handles type hints and return type annotations
    pattern = rf"def\s+({re.escape(entry_point)})\s*\(([^)]*)\)"
    
    match = re.search(pattern, prompt)
    if match:
        func_name = match.group(1)
        params_str = match.group(2).strip()
        return func_name, params_str
    
    return None, None


def prepend_function_def(code, func_name, params=""):
    """
    Prepend function definition if the code doesn't start with 'def func_name'.
    
    This handles cases where the model generates just the function body
    without the 'def' statement, since training prompts may end with the
    function signature.
    
    Args:
        code: The generated code (may or may not start with 'def')
        func_name: The expected function name
        params: The function parameters string (e.g., "x, n: int")
    
    Returns:
        str: Code with function definition prepended if necessary
    """
    if not code or not isinstance(code, str):
        return code
    
    code = code.strip()
    
    # Check if code already starts with a function definition
    # Pattern matches: def func_name( or def func_name (
    def_pattern = rf"^\s*def\s+{re.escape(func_name)}\s*\("
    if re.match(def_pattern, code):
        # Already has the function definition
        return code
    
    # Check if code starts with any function definition
    any_def_pattern = r"^\s*def\s+\w+\s*\("
    if re.match(any_def_pattern, code):
        # Has a different function definition, return as-is
        return code
    
    # Check if the target function definition exists anywhere in the code
    # This handles cases where model generates extra text before the function
    func_def_in_code = re.search(rf"def\s+{re.escape(func_name)}\s*\(", code)
    if func_def_in_code:
        # Extract from the function definition onwards
        code = code[func_def_in_code.start():]
        return code
    
    # Check if code contains <predicted_aux> tags (agent chaining format)
    # In this case, don't wrap it - let extraction handle it
    if "<predicted_aux>" in code or "</predicted_aux>" in code:
        return code
    
    # Check if code starts with natural language (not valid Python)
    # Don't wrap natural language text as it will break parsing
    first_line = code.split('\n')[0].strip() if code else ""
    if first_line and not _is_likely_python_code(first_line):
        # Try to find any 'def func_name' in the code and extract from there
        any_func_match = re.search(rf"def\s+\w+\s*\(", code)
        if any_func_match:
            code = code[any_func_match.start():]
            return code
        # No function found, return as-is (don't wrap non-code)
        return code
    
    # Code doesn't start with a function definition, prepend it
    # Build the function signature
    if params:
        func_def = f"def {func_name}({params}):\n"
    else:
        func_def = f"def {func_name}():\n"
    
    # Indent the code body if it's not already indented
    lines = code.split("\n")
    indented_lines = []
    for line in lines:
        if line.strip():  # Non-empty line
            if not line.startswith("    ") and not line.startswith("\t"):
                line = "    " + line
        indented_lines.append(line)
    
    indented_code = "\n".join(indented_lines)
    
    return func_def + indented_code


def _is_likely_python_code(line: str) -> bool:
    """
    Check if a line looks like Python code rather than natural language.
    
    Returns True if the line looks like valid Python code.
    """
    if not line:
        return False
    
    # Python keywords and patterns that indicate code
    code_indicators = [
        r"^\s*def\s+",       # function definition
        r"^\s*class\s+",     # class definition
        r"^\s*return\s*",    # return statement
        r"^\s*import\s+",    # import statement
        r"^\s*from\s+\w+",   # from import
        r"^\s*if\s+",        # if statement
        r"^\s*for\s+",       # for loop
        r"^\s*while\s+",     # while loop
        r"^\s*try\s*:",      # try block
        r"^\s*except",       # except block
        r"^\s*with\s+",      # with statement
        r"^\s*#",            # comment
        r"^\s*\w+\s*=",      # assignment
        r"^\s*\w+\s*\(",     # function call
        r"^\s*\[",           # list
        r"^\s*\{",           # dict
        r"^\s*\(",           # tuple/expression
        r"^\s*'''",          # docstring
        r'^\s*"""',          # docstring
    ]
    
    for pattern in code_indicators:
        if re.match(pattern, line):
            return True
    
    # Natural language indicators
    natural_lang_indicators = [
        r"^[A-Z][a-z]+\s+[A-Z]?[a-z]+",  # "Predicted Aux Function"
        r"^Here\s+is",
        r"^This\s+",
        r"^The\s+",
        r"^Create\s+",
        r"^Write\s+",
        r"^Implement\s+",
        r"^Function\s*:",
        r"^Output\s*:",
    ]
    
    for pattern in natural_lang_indicators:
        if re.match(pattern, line, re.IGNORECASE):
            return False
    
    return True  # Default to assuming it's code


# =============================================================================
# BigCodeBench Sandboxing Utilities (adapted from official BigCodeBench code)
# =============================================================================

TIMEOUT_LIMIT = 10.0  # Shorter timeout for training (vs 240 in official)

# Status codes
_SUCCESS = 0
_FAILED = 1
_TIMEOUT = 2
_UNKNOWN = 3

PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"

_mapping = {_SUCCESS: PASS, _FAILED: FAIL, _TIMEOUT: TIMEOUT, _UNKNOWN: None}


@contextlib.contextmanager
def swallow_subprocess_output():
    """Context manager to swallow stdout and stderr for subprocesses."""
    original_popen = subprocess.Popen
    original_run = subprocess.run

    def _popen_patch(*args, **kwargs):
        if 'capture_output' in kwargs and kwargs['capture_output']:
            kwargs.pop('stdout', None)
            kwargs.pop('stderr', None)
        else:
            kwargs.setdefault('stdout', subprocess.PIPE)
            kwargs.setdefault('stderr', subprocess.PIPE)
        return original_popen(*args, **kwargs)

    def _run_patch(*args, **kwargs):
        if 'capture_output' in kwargs and kwargs['capture_output']:
            kwargs.pop('stdout', None)
            kwargs.pop('stderr', None)
        else:
            kwargs.setdefault('stdout', subprocess.PIPE)
            kwargs.setdefault('stderr', subprocess.PIPE)
        return original_run(*args, **kwargs)

    subprocess.Popen = _popen_patch
    subprocess.run = _run_patch
    try:
        yield
    finally:
        subprocess.Popen = original_popen
        subprocess.run = original_run


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from."""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        return False


class redirect_stdin(contextlib._RedirectStream):
    _stream = "stdin"


@contextlib.contextmanager
def swallow_io():
    """Capture and suppress all I/O."""
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                with swallow_subprocess_output():
                    yield


@contextlib.contextmanager
def time_limit(seconds: float):
    """Context manager for setting a time limit."""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def chdir(root):
    """Context manager to change directory temporarily."""
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def create_tempdir():
    """Create and use a temporary directory."""
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


@contextlib.contextmanager
def safe_environment():
    """Create a safe execution environment by intercepting dangerous system calls."""
    # Save original functions
    original_kill = os.kill
    original_killpg = os.killpg
    original_system = os.system
    original_subprocess_call = subprocess.call
    original_subprocess_check_output = subprocess.check_output
    original_subprocess_run = subprocess.run
    original_subprocess_popen = subprocess.Popen
    original_os_popen = os.popen
    original_os_execv = os.execv
    original_os_execvp = os.execvp
    original_os_execvpe = os.execvpe

    current_pid = os.getpid()
    current_pgid = os.getpgid(current_pid)
    manager = multiprocessing.Manager()
    child_pids = manager.list()

    def safe_kill(pid, sig):
        try:
            if pid == current_pid or pid in child_pids:
                original_kill(pid, sig)
        except ProcessLookupError:
            pass

    def safe_killpg(pgid, sig):
        if pgid == current_pgid or pgid in {os.getpgid(pid) for pid in child_pids}:
            original_killpg(pgid, sig)

    def safe_system(command):
        if 'kill' in command or 'killall' in command:
            return 0
        return original_system(command)

    def safe_subprocess_call(command, *args, **kwargs):
        if 'kill' in str(command) or 'killall' in str(command):
            return 0
        return original_subprocess_call(command, *args, **kwargs)

    def safe_subprocess_check_output(command, *args, **kwargs):
        if 'ps' in str(command):
            return b""
        return original_subprocess_check_output(command, *args, **kwargs)

    def safe_subprocess_run(*args, **kwargs):
        if args and ('kill' in str(args[0]) or 'killall' in str(args[0])):
            return subprocess.CompletedProcess(args, 0, b'', b'')
        return original_subprocess_run(*args, **kwargs)

    class SafePopen(subprocess.Popen):
        def __init__(self, *args, **kwargs):
            kwargs['preexec_fn'] = os.setsid
            super().__init__(*args, **kwargs)
            child_pids.append(self.pid)

        def communicate(self, *args, **kwargs):
            try:
                return super().communicate(*args, **kwargs)
            except subprocess.TimeoutExpired:
                return None, None

        def kill(self):
            safe_kill(self.pid, signal.SIGTERM)

        def terminate(self):
            safe_kill(self.pid, signal.SIGTERM)

    def safe_os_popen(command):
        if 'kill' in command or 'killall' in command:
            return os.popen('echo Intercepted')
        return original_os_popen(command)

    def safe_exec(*args, **kwargs):
        pass  # Block exec calls

    # Override risky functions
    os.kill = safe_kill
    os.killpg = safe_killpg
    os.system = safe_system
    subprocess.call = safe_subprocess_call
    subprocess.check_output = safe_subprocess_check_output
    subprocess.run = safe_subprocess_run
    subprocess.Popen = SafePopen
    os.popen = safe_os_popen
    os.execv = safe_exec
    os.execvp = safe_exec
    os.execvpe = safe_exec

    try:
        yield
    finally:
        # Cleanup child processes
        for pid in child_pids:
            try:
                os.kill(pid, signal.SIGTERM)
                for _ in range(10):
                    time.sleep(0.1)
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        break
                else:
                    os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, Exception):
                pass

        # Restore original functions
        os.kill = original_kill
        os.killpg = original_killpg
        os.system = original_system
        subprocess.call = original_subprocess_call
        subprocess.check_output = original_subprocess_check_output
        subprocess.run = original_subprocess_run
        subprocess.Popen = original_subprocess_popen
        os.popen = original_os_popen
        os.execv = original_os_execv
        os.execvp = original_os_execvp
        os.execvpe = original_os_execvpe


def reliability_guard(max_as_limit=30*1024, max_data_limit=30*1024, max_stack_limit=10):
    """
    Set resource limits to prevent destructive operations.
    WARNING: This is NOT a security sandbox.
    """
    os.environ['TZ'] = 'UTC'
    try:
        time.tzset()
    except Exception:
        pass
    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

    if max_as_limit and max_data_limit and max_stack_limit:
        try:
            import resource
            
            max_as_limit_bytes = max_as_limit * 1024 * 1024
            max_data_limit_bytes = max_data_limit * 1024 * 1024
            max_stack_limit_bytes = max_stack_limit * 1024 * 1024

            resource.setrlimit(resource.RLIMIT_AS, (max_as_limit_bytes, max_as_limit_bytes))
            resource.setrlimit(resource.RLIMIT_DATA, (max_data_limit_bytes, max_data_limit_bytes))
            if platform.uname().system != "Darwin":
                resource.setrlimit(resource.RLIMIT_STACK, (max_stack_limit_bytes, max_stack_limit_bytes))
        except Exception:
            pass  # Resource limits may not be available on all systems

    faulthandler.disable()

    import builtins
    builtins.exit = None
    builtins.quit = None

    # Close matplotlib figures if available
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
    except ImportError:
        pass


def unsafe_execute_bigcodebench(
    entry_point: str,
    code: str,
    test_code: str,
    timeout: float,
    max_as_limit: float,
    max_data_limit: float,
    max_stack_limit: float,
    stat,  # Value
    details,  # Manager dict
):
    """
    Execute code in a sandboxed environment.
    This function runs in a separate process.
    """
    with safe_environment(), create_tempdir():
        import os
        import shutil
        import builtins

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir_func = os.chdir

        reliability_guard(max_as_limit, max_data_limit, max_stack_limit)

        module_name = "__test__"
        new_module = types.ModuleType(module_name)
        new_module.__dict__.update({
            '__builtins__': builtins,
            '__file__': f"{module_name}.py",
            '__package__': None,
            '__doc__': None,
            'sys': sys,
            'os': os,
            'environ': os.environ,
        })

        try:
            full_code = code + "\n" + test_code

            with swallow_io():
                exec(compile(full_code, f"{module_name}.py", 'exec'), new_module.__dict__)
                sys.modules[module_name] = new_module
                TestCases = getattr(new_module, 'TestCases')
                loader = unittest.TestLoader()
                suite = loader.loadTestsFromTestCase(TestCases)
                test_result = unittest.TestResult()
                with time_limit(timeout):
                    suite.run(test_result)

            issues = test_result.failures + test_result.errors
            for test, trace in issues:
                test_name = test.id().split(".")[-1]
                details[test_name] = trace[:2000]  # Limit trace length (increased for debugging)
            
            details["_total"] = test_result.testsRun
            details["_passed"] = test_result.testsRun - len(test_result.failures) - len(test_result.errors)
            stat.value = _SUCCESS

        except TimeoutException:
            details["ALL"] = "Timeout"
            stat.value = _TIMEOUT
        except ImportError as e:
            details["ALL"] = f"ImportError: {str(e)[:1000]}"
            stat.value = _FAILED
        except BaseException as e:
            details["ALL"] = str(e)[:2000]
            stat.value = _FAILED

        # Restore for cleanup
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir_func


def untrusted_check_bigcodebench(
    code: str,
    test_code: str,
    entry_point: str,
    timeout: float = 5.0,
    max_as_limit: float = 30*1024,
    max_data_limit: float = 30*1024,
    max_stack_limit: float = 10,
) -> Tuple[str, Dict[str, Any], int, int]:
    """
    Run BigCodeBench tests in an isolated process with sandboxing.
    
    Returns:
        Tuple of (status, details_dict, passed_tests, total_tests)
    """
    timeout = max(timeout, 1.0)
    
    # Shared memory objects
    stat = Value("i", _UNKNOWN)
    manager = Manager()
    details = manager.dict()

    p = multiprocessing.Process(
        target=unsafe_execute_bigcodebench,
        args=(
            entry_point,
            code,
            test_code,
            timeout,
            max_as_limit,
            max_data_limit,
            max_stack_limit,
            stat,
            details,
        ),
    )
    p.start()
    p.join(timeout=timeout + 1)
    
    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)

    status = _mapping[stat.value]
    details_dict = dict(details)

    if not status:
        status = TIMEOUT

    if status == PASS and details_dict:
        # Check if there were any failures in details
        if any(k not in ("_total", "_passed") for k in details_dict if not k.startswith("_")):
            status = FAIL

    passed = details_dict.get("_passed", 0)
    total = details_dict.get("_total", 1)

    return status, details_dict, passed, total


def run_bigcodebench_tests(combined_code, test_code, entry_point, timeout=5):
    """
    Run BigCodeBench unittest-based tests on the combined code.
    Uses multiprocessing-based sandboxing for isolation.
    
    Args:
        combined_code: The aux + main function code
        test_code: The unittest test code from BigCodeBench
        entry_point: The function name being tested
        timeout: Timeout in seconds (default 5s for training speed)
        
    Returns:
        tuple: (passed_tests, total_tests, error_message)
    """
    # Use the sandboxed execution
    status, details, passed, total = untrusted_check_bigcodebench(
        code=combined_code,
        test_code=test_code,
        entry_point=entry_point,
        timeout=timeout,
    )
    
    # Build error message from details
    error_msg = None
    if status != PASS:
        error_parts = []
        for key, value in details.items():
            if not key.startswith("_"):  # Skip internal keys
                error_parts.append(f"{key}: {value[:500]}")
        error_msg = "\n".join(error_parts[:5]) if error_parts else status
    
    return passed, total, error_msg


def extract_imports_from_code_prompt(code_prompt):
    """
    Extract import statements AND constant definitions from BigCodeBench code_prompt.
    The code_prompt contains imports, constants, and function signature.
    
    This extracts everything before the main function definition, including:
    - import statements
    - from ... import statements
    - constant definitions (e.g., TARGET_JSON_FILE = 'downloaded.json')
    """
    if not code_prompt:
        return ""
    
    preamble_lines = []
    lines = code_prompt.split("\n")
    
    for line in lines:
        stripped = line.strip()
        # Stop at the main function definition
        if stripped.startswith("def "):
            break
        # Include import statements
        if stripped.startswith("import ") or stripped.startswith("from "):
            preamble_lines.append(line)
        # Include constant definitions (UPPER_CASE = value pattern)
        elif re.match(r'^[A-Z_][A-Z0-9_]*\s*=', stripped):
            preamble_lines.append(line)
        # Include other assignments that might be needed (e.g., lowercase constants)
        elif '=' in stripped and not stripped.startswith('#') and not stripped.startswith('def '):
            # Only include simple assignments at module level (not indented)
            if not line.startswith(' ') and not line.startswith('\t'):
                preamble_lines.append(line)
    
    return "\n".join(preamble_lines)
