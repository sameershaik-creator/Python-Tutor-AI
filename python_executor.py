import subprocess
import sys
import os
import tempfile

def execute_python_code(code_str, stdin_str=""):
    """Execute Python code safely and return result dict."""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code_str)
        temp_file = f.name
    
    try:
        result = subprocess.run(
            [sys.executable, temp_file],
            input=stdin_str,
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        if result.returncode == 0:
            return {
                "success": True,
                "output": result.stdout,
            }
        else:
            error_type = "RuntimeError"
            if "SyntaxError" in result.stderr:
                error_type = "SyntaxError"
            elif "NameError" in result.stderr:
                error_type = "NameError"
            elif "TypeError" in result.stderr:
                error_type = "TypeError"
            elif "IndexError" in result.stderr:
                error_type = "IndexError"
            elif "ValueError" in result.stderr:
                error_type = "ValueError"
            elif "ZeroDivisionError" in result.stderr:
                error_type = "ZeroDivisionError"
                
            return {
                "success": False,
                "error_type": error_type,
                "error_message": result.stderr.strip(),
            }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error_type": "TimeoutError",
            "error_message": "Code execution timed out after 10 seconds.",
        }
    finally:
        try:
            os.unlink(temp_file)
        except:
            pass