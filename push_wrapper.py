#!/usr/bin/env python
"""Wrapper to run openenv with proper UTF-8 encoding."""
import subprocess
import sys
import os

# Set environment variables for UTF-8
env = os.environ.copy()
env['PYTHONIOENCODING'] = 'utf-8'
env['PYTHONLEGACYWINDOWSSTDIO'] = '0'
env['PYTHONUNBUFFERED'] = '1'

# Run openenv as a module with UTF-8 handling
result = subprocess.run(
    [sys.executable, '-m', 'openenv.cli', 'push', '--repo-id', 'gankit-aiml/my_env'],
    env=env,
    encoding='utf-8',
    errors='replace'
)
sys.exit(result.returncode)
