from swe_env import BaseSWEEnv
from tools import tool # Assuming 'tool' decorator is available from a 'tools' module
import shlex # For robust quoting of arguments that might contain spaces

import json
from pathlib import Path

# The path to commands.json relative to tool.py
commands_json_path = Path("commands.json")

class SWEEnv(BaseSWEEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_commands()

    # --- LangGraph Tools ---

    @tool
    def open(self, path: str, line_number: int = 0):
        """Opens the file at the given path in the editor. \
        If line_number is provided and is greater than 0, the window will be moved to include that line.
        
        Args:
            path: str
                The path to the file to open.
            line_number: int = 0
                The line number to move the window to (if not provided or 0, the window will start at the top/center of the file).
        
        """
        # Bash 'open' command expects file path and optional line number directly
        if line_number > 0:
            bash_command = f"open {shlex.quote(path)} {line_number}"
        else:
            bash_command = f"open {shlex.quote(path)}"
        
        return self.run_bash_command(bash_command)

    @tool
    def goto(self, line_number: int):
        """Moves the window to show <line_number>.
        
        Args:
            line_number: int
                The line number to move the window to.
        
        """
        # Bash 'goto' command expects a single line number
        bash_command = f"goto {line_number}"
        return self.run_bash_command(bash_command)

    @tool
    def scroll_down(self):
        """Moves the window down {WINDOW} lines."""
        # Bash 'scroll_down' command takes no arguments
        bash_command = "scroll_down"
        return self.run_bash_command(bash_command)

    @tool
    def scroll_up(self):
        """Moves the window up {WINDOW} lines."""
        # Bash 'scroll_up' command takes no arguments
        bash_command = "scroll_up"
        return self.run_bash_command(bash_command)

    @tool
    def create(self, filename: str):
        """Creates and opens a new file with the given name.
        
        Args:
            filename: str
                The name of the file to create.
        
        """
        # Bash 'create' command expects the filename
        bash_command = f"create {shlex.quote(filename)}"
        return self.run_bash_command(bash_command)

    @tool
    def submit(self):
        """Submits your current code and terminates the session."""
        # Bash 'submit' command takes no arguments
        bash_command = "submit"
        return self.run_bash_command(bash_command)

    @tool
    def search_dir(self, search_term: str, directory: str = None):
        """Searches for search_term in all files in dir. \
        If dir is not provided, searches in the current directory.
        
        Args:
            search_term: str
                The term to search for.
            directory: str = None
                The directory to search in (if not provided, searches in the current directory).
        
        """
        # Bash 'search_dir' command expects quoted search_term and optional directory
        if directory:
            bash_command = f"search_dir {shlex.quote(search_term)} {shlex.quote(directory)}"
        else:
            bash_command = f"search_dir {shlex.quote(search_term)}"
        return self.run_bash_command(bash_command)

    @tool
    def search_file(self, search_term: str, file: str = None):
        """Searches for search_term in file. \
        If file is not provided, searches in the current open file.
        
        Args:
            search_term: str
                The term to search for.
            file: str = None
                The file to search in (if not provided, searches in the current open file).
        
        """
        # Bash 'search_file' command expects quoted search_term and optional file
        if file:
            bash_command = f"search_file {shlex.quote(search_term)} {shlex.quote(file)}"
        else:
            bash_command = f"search_file {shlex.quote(search_term)}"
        return self.run_bash_command(bash_command)

    @tool
    def find_file(self, file_name: str, directory: str = None):
        """Finds all files with the given name in dir. \
        If dir is not provided, searches in the current directory.
        
        Args:
            file_name: str
                The name of the file to search for.
            directory: str = None
                The directory to search in (if not provided, searches in the current directory).
        
        """
        # Bash 'find_file' command expects quoted file_name and optional directory
        if directory:
            bash_command = f"find_file {shlex.quote(file_name)} {shlex.quote(directory)}"
        else:
            bash_command = f"find_file {shlex.quote(file_name)}"
        return self.run_bash_command(bash_command)

    @tool
    def edit(self, to_replace: str, new_content: str):
        """Replaces occurrence of $<to_replace> with $<new_content> in the currently open file.
        
        Args:
            to_replace: str
                The text to be replaced in the file.
            new_content: str
                The new text to replace with.
        
        """
        # The 'edit' command explicitly uses the $'...' syntax for literal strings
        # This is crucial for handling newlines and other special characters correctly within Bash.
        bash_command = f"edit ${shlex.quote(to_replace)} ${shlex.quote(new_content)}"
        return self.run_bash_command(bash_command)