"""
Security and sandboxing for RLM code execution.

Provides restricted execution environment to limit potential damage from
generated code.
"""

import sys
import io
import math
import multiprocessing
import queue
import signal
import platform
import threading
from contextlib import contextmanager
from typing import Dict, Any, Set, Tuple


# ---------------------------------------------------------------------------
# Thread-aware output capture and timeouts
#
# Child RLMs may execute REPL code concurrently in worker threads, so stdout is
# captured per thread instead of by swapping sys.stdout for the whole process.
# SIGALRM timeouts only work in the main thread; they are paused while a code
# block waits on LLM calls so the limit measures local computation only.
# ---------------------------------------------------------------------------

_capture_state = threading.local()
_proxy_lock = threading.Lock()
_proxy_users = 0
_alarm = {"armed": False}


class _ThreadAwareStdout(io.TextIOBase):
    """sys.stdout replacement that routes writes to the current thread's capture buffer."""

    def __init__(self, fallback):
        self._fallback = fallback

    def _target(self):
        return getattr(_capture_state, "buffer", None) or self._fallback

    def write(self, text):
        return self._target().write(text)

    def flush(self):
        return self._target().flush()

    def isatty(self):
        return False


def _install_stdout_proxy():
    global _proxy_users
    with _proxy_lock:
        if _proxy_users == 0 and not isinstance(sys.stdout, _ThreadAwareStdout):
            sys.stdout = _ThreadAwareStdout(sys.stdout)
        _proxy_users += 1


def _uninstall_stdout_proxy():
    global _proxy_users
    with _proxy_lock:
        _proxy_users -= 1
        if _proxy_users == 0 and isinstance(sys.stdout, _ThreadAwareStdout):
            sys.stdout = sys.stdout._fallback


def _in_main_thread() -> bool:
    return threading.current_thread() is threading.main_thread()


@contextmanager
def suspend_timeout():
    """Pause the active REPL execution timeout (if any) for the duration of the block.

    Wrap blocking LLM calls made from inside executed code so that slow API
    responses or nested RLM runs do not count against the execution time limit.
    """
    if not (_in_main_thread() and hasattr(signal, "SIGALRM") and _alarm["armed"]):
        yield
        return
    remaining = signal.alarm(0)
    _alarm["armed"] = False
    try:
        yield
    finally:
        signal.alarm(max(1, remaining))
        _alarm["armed"] = True



def exec_captured(code: str, globals_dict: Dict[str, Any]) -> Tuple[bool, str, str]:
    """Execute code capturing this thread's stdout, without safety checks or timeout."""
    output_capture = io.StringIO()
    _install_stdout_proxy()
    previous_buffer = getattr(_capture_state, "buffer", None)
    _capture_state.buffer = output_capture
    try:
        exec(code, globals_dict)
        return True, output_capture.getvalue(), ""
    except Exception as e:
        return False, "", str(e)
    finally:
        _capture_state.buffer = previous_buffer
        _uninstall_stdout_proxy()
        output_capture.close()

class RestrictedGlobals:
    """
    Creates a restricted globals dictionary for safer exec() execution.

    Blocks access to dangerous modules and functions while allowing
    safe operations needed for RLM functionality.
    """

    # Dangerous builtins to exclude
    DANGEROUS_BUILTINS = {
        '__import__',
        'eval',
        'compile',
        'open',  # Can be added back via safe wrapper
        'input',
        'execfile',
        'reload',
        'breakpoint',
    }

    # Dangerous modules to block
    DANGEROUS_MODULES = {
        'os',
        'sys',
        'subprocess',
        'socket',
        'urllib',
        'requests',
        'http',
        'ftplib',
        'smtplib',
        'pickle',
        'shelve',
        'dbm',
        '__builtin__',
        '__builtins__',
    }

    # Safe modules that can be imported
    SAFE_MODULES = {
        're',
        'json',
        'math',
        'statistics',
        'collections',
        'itertools',
        'functools',
        'datetime',
        'time',
        'random',
        'string',
        'textwrap',
        'concurrent',  # Required for map_reduce(parallel=True)
    }

    @classmethod
    def _safe_import(cls, name, globals=None, locals=None, fromlist=(), level=0):
        """
        Safe import wrapper that only allows whitelisted modules.

        Args:
            name: Module name to import
            globals: Globals dict (unused but required by import signature)
            locals: Locals dict (unused but required by import signature)
            fromlist: List of names to import from module
            level: Relative import level

        Returns:
            Imported module

        Raises:
            ImportError: If module is not in the safe list
        """
        # Get the base module name (before any dots)
        base_module = name.split('.')[0]

        # Check if module is in safe list
        if base_module not in cls.SAFE_MODULES:
            raise ImportError(f"Import of module '{name}' is not allowed. Safe modules: {', '.join(sorted(cls.SAFE_MODULES))}")

        # Use the real __import__ to load the module
        import builtins
        return builtins.__import__(name, globals, locals, fromlist, level)

    @classmethod
    def create_safe_globals(cls, custom_globals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a safe globals dictionary with restricted builtins.

        Args:
            custom_globals: Custom globals to include (e.g., llm_query, context)

        Returns:
            Restricted globals dictionary
        """
        # Get builtins - __builtins__ can be either a dict or a module
        import builtins as builtins_module

        # Start with a copy of safe builtins
        safe_builtins = {
            name: getattr(builtins_module, name)
            for name in dir(builtins_module)
            if name not in cls.DANGEROUS_BUILTINS
        }

        # Add safe import wrapper
        safe_builtins['__import__'] = cls._safe_import

        # Create base safe globals
        safe_globals = {
            '__builtins__': safe_builtins,
            '__name__': '__rlm__',
            '__doc__': None,
        }

        # Add custom globals (like llm_query, context, helpers)
        safe_globals.update(custom_globals)

        return safe_globals


def _subprocess_execute(code: str, globals_dict: Dict[str, Any], result_queue: multiprocessing.Queue, max_output_size: int):
    """
    Helper function to execute code in a subprocess.

    Args:
        code: Code to execute
        globals_dict: Globals dictionary for execution
        result_queue: Queue to put results
        max_output_size: Maximum output size
    """
    output_capture = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = output_capture

    try:
        exec(code, globals_dict)
        output = output_capture.getvalue()

        # Truncate if needed
        if len(output) > max_output_size:
            output = output[:max_output_size] + "\n[Output truncated...]"

        result_queue.put(("success", output, ""))
    except Exception as e:
        result_queue.put(("error", "", str(e)))
    finally:
        sys.stdout = original_stdout
        output_capture.close()


class TimeoutException(Exception):
    """Raised when execution timeout is reached."""
    pass


class ExecutionMonitor:
    """
    Monitors code execution for safety violations and resource limits.
    Uses multiprocessing for proper isolation and enforceable timeouts.
    """

    def __init__(
        self,
        max_execution_time: float = 30.0,
        max_output_size: int = 100_000,
        forbidden_patterns: Set[str] = None,
        use_multiprocessing: bool = True
    ):
        """
        Initialize execution monitor.

        Args:
            max_execution_time: Maximum execution time in seconds
            max_output_size: Maximum output size in characters
            forbidden_patterns: Set of forbidden string patterns in code
            use_multiprocessing: If True, use multiprocessing for isolation (recommended)
        """
        self.max_execution_time = max_execution_time
        self.max_output_size = max_output_size
        self.use_multiprocessing = use_multiprocessing
        # Note: We no longer block '__import__' or 'compile(' since we provide safe wrappers
        # We block specific dangerous module imports, eval/exec, and introspection bypasses
        self.forbidden_patterns = forbidden_patterns or {
            # Direct dangerous imports
            'import os',
            'import sys',
            'import subprocess',
            'import socket',
            'import urllib',
            'import requests',
            'import pickle',
            'import shelve',
            'import shutil',
            'import tempfile',
            # Dangerous functions
            'eval(',
            'exec(',
            # Object introspection bypasses (gadget chains)
            # These are the most dangerous for escaping the sandbox
            '__subclasses__',  # Used in gadget chains to find dangerous classes
            '__globals__',     # Can access global namespace
            '__builtins__',    # Can access builtin functions
            '__code__',        # Can access function code objects
            # Attribute manipulation that can bypass protections
            '__getattribute__',
            '__setattr__',
            '__delattr__',
        }

        # Additional regex-based checks for complex bypasses
        import re
        self.bypass_patterns = [
            # Gadget chain starts - common pattern for escaping sandboxes
            re.compile(r'\(\s*\)\s*\.__class__'),  # ().__class__ gadget chain start
            re.compile(r'\[\s*\]\s*\.__class__'),  # [].__class__ gadget chain start
            re.compile(r'\{\s*\}\s*\.__class__'),  # {}.__class__ gadget chain start
            # Dunder method chaining (e.g., .__class__.__base__.__subclasses__())
            re.compile(r'__\w+__\s*\.\s*__\w+__\s*\.\s*__\w+__'),  # Triple dunder chain
        ]

    def check_code_safety(self, code: str) -> tuple[bool, str]:
        """
        Check if code contains forbidden patterns or bypass attempts.

        Uses both string matching and regex patterns to detect:
        - Direct dangerous imports and functions
        - Object introspection (gadget chains)
        - String concatenation bypasses
        - Encoding/decoding bypasses

        Args:
            code: Code string to check

        Returns:
            Tuple of (is_safe, error_message)
        """
        code_lower = code.lower()

        # Check string-based forbidden patterns
        for pattern in self.forbidden_patterns:
            if pattern.lower() in code_lower:
                return False, f"Forbidden pattern detected: {pattern}"

        # Check regex-based bypass patterns
        for regex in self.bypass_patterns:
            if regex.search(code):
                return False, f"Potential bypass pattern detected: {regex.pattern}"

        return True, ""

    def _timeout_handler(self, signum, frame):
        """Signal handler for timeout."""
        raise TimeoutException(f"Execution exceeded {self.max_execution_time} seconds")

    def capture_execution(self, code: str, globals_dict: Dict[str, Any]) -> tuple[bool, str, str]:
        """
        Execute code with output capture, safety monitoring, and timeout enforcement.

        On Unix/Linux/Mac, uses signal.alarm() for hard timeout enforcement.
        On Windows, timeout is not enforced (limitation of platform).

        Args:
            code: Code to execute
            globals_dict: Globals dictionary for execution

        Returns:
            Tuple of (success, output, error_message)
        """
        # Check code safety first
        is_safe, error_msg = self.check_code_safety(code)
        if not is_safe:
            return False, "", error_msg

        # Capture stdout for this thread only
        output_capture = io.StringIO()
        _install_stdout_proxy()
        previous_buffer = getattr(_capture_state, "buffer", None)
        _capture_state.buffer = output_capture

        # Setup timeout (Unix main thread only; nested runs re-arm their own)
        use_alarm = (platform.system() in ('Linux', 'Darwin') and hasattr(signal, 'SIGALRM')
                     and _in_main_thread() and not _alarm["armed"])
        old_handler = None

        try:
            if use_alarm:
                old_handler = signal.signal(signal.SIGALRM, self._timeout_handler)
                signal.alarm(max(1, math.ceil(self.max_execution_time)))
                _alarm["armed"] = True

            # Execute code
            exec(code, globals_dict)

            # Get captured output
            output = output_capture.getvalue()

            # Check output size limit
            if len(output) > self.max_output_size:
                output = output[:self.max_output_size] + "\n[Output truncated...]"

            return True, output, ""

        except TimeoutException as e:
            return False, "", f"Timeout: {str(e)}"

        except Exception as e:
            return False, "", str(e)

        finally:
            if use_alarm:
                signal.alarm(0)
                _alarm["armed"] = False
                signal.signal(signal.SIGALRM, old_handler or signal.SIG_DFL)
            _capture_state.buffer = previous_buffer
            _uninstall_stdout_proxy()
            output_capture.close()


class SafeFileAccess:
    """
    Provides safe file access wrapper for RLM.

    Restricts file operations to specific directories and file types.
    """

    def __init__(self, allowed_dirs: Set[str] = None, allowed_extensions: Set[str] = None):
        """
        Initialize safe file access.

        Args:
            allowed_dirs: Set of allowed directory paths
            allowed_extensions: Set of allowed file extensions
        """
        self.allowed_dirs = allowed_dirs or set()
        self.allowed_extensions = allowed_extensions or {'.txt', '.json', '.csv', '.md'}

    def safe_read(self, filepath: str) -> str:
        """
        Safely read a file with restrictions.

        Args:
            filepath: Path to file

        Returns:
            File contents

        Raises:
            PermissionError: If file access is not allowed
        """
        import os

        # Normalize path
        abs_path = os.path.abspath(filepath)

        # Check directory restriction
        if self.allowed_dirs:
            allowed = any(abs_path.startswith(d) for d in self.allowed_dirs)
            if not allowed:
                raise PermissionError(f"Access to {abs_path} is not allowed")

        # Check extension restriction
        _, ext = os.path.splitext(abs_path)
        if ext.lower() not in self.allowed_extensions:
            raise PermissionError(f"File type {ext} is not allowed")

        # Read file
        with open(abs_path, 'r', encoding='utf-8') as f:
            return f.read()

    def safe_write(self, filepath: str, content: str):
        """
        Safely write to a file with restrictions.

        Args:
            filepath: Path to file
            content: Content to write

        Raises:
            PermissionError: If file access is not allowed
        """
        import os

        # Normalize path
        abs_path = os.path.abspath(filepath)

        # Check directory restriction
        if self.allowed_dirs:
            allowed = any(abs_path.startswith(d) for d in self.allowed_dirs)
            if not allowed:
                raise PermissionError(f"Access to {abs_path} is not allowed")

        # Check extension restriction
        _, ext = os.path.splitext(abs_path)
        if ext.lower() not in self.allowed_extensions:
            raise PermissionError(f"File type {ext} is not allowed")

        # Write file
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(content)
