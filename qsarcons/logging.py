import sys
import os
import logging
import threading


class FailedMolecule:
    """A sentinel for a SMILES string that could not be parsed or processed anywhere in the pipeline."""

    def __init__(self, smiles, message="failed"):
        self.smiles = smiles
        self.message = message

    def __repr__(self):
        return f"{self.smiles} -> {self.message}"


class OutputSuppressor:
    """
    Completely suppress ALL output:
    - Python prints
    - logging
    - C/C++ libraries writing to stdout/stderr (CatBoost, XGBoost, etc.)
    Thread-safe and nestable.
    """

    _lock = threading.Lock()
    _active = 0

    def __enter__(self):
        with OutputSuppressor._lock:
            if OutputSuppressor._active == 0:
                # Save original file descriptors
                self._orig_stdout_fd = os.dup(1)
                self._orig_stderr_fd = os.dup(2)

                # Open null file
                self._devnull = os.open(os.devnull, os.O_WRONLY)

                # Redirect Python-level stdio
                self._orig_stdout = sys.stdout
                self._orig_stderr = sys.stderr
                sys.stdout = open(os.devnull, "w")
                sys.stderr = open(os.devnull, "w")

                # Redirect C-level stdout/stderr
                os.dup2(self._devnull, 1)
                os.dup2(self._devnull, 2)

                # Disable logging
                logging.disable(logging.CRITICAL)

            OutputSuppressor._active += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        with OutputSuppressor._lock:
            OutputSuppressor._active -= 1
            if OutputSuppressor._active == 0:
                # Restore file descriptors
                os.dup2(self._orig_stdout_fd, 1)
                os.dup2(self._orig_stderr_fd, 2)

                # Close temp files
                os.close(self._devnull)
                os.close(self._orig_stdout_fd)
                os.close(self._orig_stderr_fd)

                # Restore Python-level stdio
                sys.stdout.close()
                sys.stderr.close()
                sys.stdout = self._orig_stdout
                sys.stderr = self._orig_stderr

                # Re-enable logging
                logging.disable(logging.NOTSET)
