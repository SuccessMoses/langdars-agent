from .base_env import BaseSWEEnv
from ..utils.commands import COMMANDS, ENV_VARIABLES
import shlex # For robust quoting of arguments that might contain spaces

import json
from pathlib import Path

# The path to commands.json relative to tool.py
commands_json_path = Path("commands.json")

def tool(func):
    """Dummy decorator"""
    return func

COMMANDS = [cmd for cmd in COMMANDS if not "execute" in cmd["name"]]

class SWEEnv(BaseSWEEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_commands(COMMANDS)
        self._setup_initial_env_variables(ENV_VARIABLES)

    def _setup_initial_env_variables(self, env_vars: dict) -> None:
        """
        Exports environment variables into the container's shell session.
        """
        for key, value in env_vars.items():
            if isinstance(value, str):
                self.communicate(f"export {key}='{value}'")
            elif isinstance(value, (int, float)):
                self.communicate(f"export {key}={value}")
            elif isinstance(value, tuple) and not value:
                 self.communicate(f"export {key}=()")

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
        
        return self._communicate(bash_command)

    @tool
    def goto(self, line_number: int):
        """Moves the window to show <line_number>.
        
        Args:
            line_number: int
                The line number to move the window to.
        
        """
        # Bash 'goto' command expects a single line number
        bash_command = f"goto {line_number}"
        return self._communicate(bash_command)

    @tool
    def scroll_down(self):
        """Moves the window down {WINDOW} lines."""
        # Bash 'scroll_down' command takes no arguments
        bash_command = "scroll_down"
        return self._communicate(bash_command)

    @tool
    def scroll_up(self):
        """Moves the window up {WINDOW} lines."""
        # Bash 'scroll_up' command takes no arguments
        bash_command = "scroll_up"
        return self._communicate(bash_command)

    @tool
    def create(self, filename: str):
        """Creates and opens a new file with the given name.
        
        Args:
            filename: str
                The name of the file to create.
        
        """
        # Bash 'create' command expects the filename
        bash_command = f"create {shlex.quote(filename)}"
        return self._communicate(bash_command)

    @tool
    def submit(self):
        """Submits your current code and terminates the session."""
        # Bash 'submit' command takes no arguments
        bash_command = "submit"
        return self._communicate(bash_command)

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
        return self._communicate(bash_command)

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
        return self._communicate(bash_command)

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
        return self._communicate(bash_command)

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
        return self._communicate(bash_command)
    
    @tool
    def undo_edit(self, file_path: str = None):
        """Reverts the last edit made to the specified file.
        If no file is provided, reverts the last edit on the currently open file.

        Args:
            file_path: str = None
                The path to the file to undo the last edit for.
                (Optional: if not provided, undoes the last edit on the currently open file).

        """
        # The 'undo_edit' bash command (which is a Python script) can optionally
        # take a file_path. If provided, it needs to be properly quoted.
        if file_path:
            bash_command = f"undo_edit {shlex.quote(file_path)}"
        else:
            # If no file_path is provided, call the command without an argument.
            # The bash script will then check the CURRENT_FILE environment variable.
            bash_command = "undo_edit"
        return self._communicate(bash_command)    
    
    
    @tool
    def insert(self, line_number: int, content: str):
        """Inserts <content> at the given <line_number> in the currently open file.

        Args:
            line_number: int
                The line number where the content should be inserted.
            content: str
                The content to insert at the specified line number.

        """
        # The 'insert' bash command (which is a Python script) expects a line number
        # and then the content. The content needs to be properly quoted for the shell.
        bash_command = f"insert {line_number} {shlex.quote(content)}"
        return self._communicate(bash_command)

    @tool
    def append(self, content: str):
        """Appends <content> to the end of the currently open file.

        Args:
            content: str
                The content to append to the end of the file.

        """
        # The 'append' bash command (which is a Python script) expects the content.
        # The content needs to be properly quoted for the shell.
        bash_command = f"append {shlex.quote(content)}"
        return self._communicate(bash_command)