open = """
open() {
    if [ -z "$1" ]; then
        echo "Usage: open <file>"
        return
    fi

    if [ -n "$2" ]; then
        if ! [[ $2 =~ ^[0-9]+$ ]]; then
            echo "Usage: open <file> [<line_number>]"
            echo "Error: <line_number> must be a number"
            return
        fi

        local max_line=$(awk 'END {print NR}' $1)
        if [ $2 -gt $max_line ]; then
            echo "Warning: <line_number> ($2) is greater than the number of lines in the file ($max_line)"
            echo "Warning: Setting <line_number> to $max_line"
            local line_number=$(jq -n "$max_line")
        elif [ $2 -lt 1 ]; then
            echo "Warning: <line_number> ($2) is less than 1"
            echo "Warning: Setting <line_number> to 1"
            local line_number=$(jq -n "1")
        else
            local OFFSET=$(jq -n "$WINDOW/6" | jq 'floor')
            local line_number=$(jq -n "[$2 + $WINDOW/2 - $OFFSET, 1] | max | floor")
        fi
    else
        local line_number=$(jq -n "$WINDOW/2")
    fi

    if [ -f "$1" ]; then
        export CURRENT_FILE=$(realpath $1)
        export CURRENT_LINE=$line_number
        _constrain_line
        _print
    elif [ -d "$1" ]; then
        echo "Error: $1 is a directory. You can only open files. Use cd or ls to navigate directories."
    else
        echo "File $1 not found"
    fi
}
"""

goto = """
goto() {
    if [ $# -gt 1 ]; then
        echo "goto allows only one line number at a time."
        return
    fi

    if [ -z "$CURRENT_FILE" ]; then
        echo "No file open. Use the open command first."
        return
    fi

    if [ -z "$1" ]; then
        echo "Usage: goto <line>"
        return
    fi

    if ! [[ $1 =~ ^[0-9]+$ ]]; then
        echo "Usage: goto <line>"
        echo "Error: <line> must be a number"
        return
    fi

    local max_line=$(awk 'END {print NR}' $CURRENT_FILE)
    if [ $1 -gt $max_line ]; then
        echo "Error: <line> must be less than or equal to $max_line"
        return
    fi

    local OFFSET=$(jq -n "$WINDOW/6" | jq 'floor')
    export CURRENT_LINE=$(jq -n "[$1 + $WINDOW/2 - $OFFSET, 1] | max | floor")
    _constrain_line
    _print
}
"""

scroll_down = """scroll_down() {
    if [ -z "$CURRENT_FILE" ]; then
        echo "No file open. Use the open command first."
        return
    fi

    export CURRENT_LINE=$(jq -n "$CURRENT_LINE + $WINDOW - $OVERLAP")
    _constrain_line
    _print
}
"""

scroll_up = """scroll_up() {
    if [ -z "$CURRENT_FILE" ]; then
        echo "No file open. Use the open command first."
        return
    fi

    export CURRENT_LINE=$(jq -n "$CURRENT_LINE - $WINDOW + $OVERLAP")
    _constrain_line
    _print
}
"""

create = """create() {
    if [ -z "$1" ]; then
        echo "Usage: create <filename>"
        return
    fi

    # Check if the file already exists
    if [ -e "$1" ]; then
        echo "Error: File '$1' already exists."
        open "$1"
        return
    fi

    # Create the file with an empty new line
    printf "\\n" > "$1"

    # Use the existing open command to open the created file
    open "$1"
}
"""

submit = """submit() {
    cd $ROOT

    # Check if the patch file exists and is non-empty
    if [ -s "/root/test.patch" ]; then
        # Apply the patch in reverse
        git apply -R < "/root/test.patch"
    fi

    git add -A
    git diff --cached > model.patch

    echo "<<SUBMISSION||"
    cat model.patch
    echo "||SUBMISSION>>"
}
"""

search_dir = """search_dir() {
    if [ $# -eq 1 ]; then
        local search_term="$1"
        local dir="./"
    elif [ $# -eq 2 ]; then
        local search_term="$1"
        if [ -d "$2" ]; then
            local dir="$2"
        else
            echo "Directory $2 not found"
            return
        fi
    else
        echo "Usage: search_dir <search_term> [<dir>]"
        return
    fi

    dir=$(realpath "$dir")
    local matches=$(find "$dir" -type f ! -path '*/.*' -exec grep -nIH -- "$search_term" {} + | cut -d: -f1 | sort | uniq -c)

    # If no matches, return
    if [ -z "$matches" ]; then
        echo "No matches found for \"$search_term\" in $dir"
        return
    fi

    # Calculate total number of matches
    local num_matches=$(echo "$matches" | awk '{sum+=$1} END {print sum}')

    # Calculate total number of files matched
    local num_files=$(echo "$matches" | wc -l | awk '{$1=$1; print $0}')

    # If num_files is > 100, print an error
    if [ $num_files -gt 100 ]; then
        echo "More than $num_files files matched for \"$search_term\" in $dir. Please narrow your search."
        return
    fi

    echo "Found $num_matches matches for \"$search_term\" in $dir:"
    echo "$matches" | awk '{$2=$2; gsub(/^\\.+\\/+/, "./", $2); print $2 " ("$1" matches)"}'
    echo "End of matches for \"$search_term\" in $dir"
}
"""

search_file = """search_file() {
    # Check if the first argument is provided
    if [ -z "$1" ]; then
        echo "Usage: search_file <search_term> [<file>]"
        return
    fi

    # Check if the second argument is provided
    if [ -n "$2" ]; then
        # Check if the provided argument is a valid file
        if [ -f "$2" ]; then
            local file="$2"  # Set file if valid
        else
            echo "Usage: search_file <search_term> [<file>]"
            echo "Error: File name $2 not found. Please provide a valid file name."
            return  # Exit if the file is not valid
        fi
    else
        # Check if a file is open
        if [ -z "$CURRENT_FILE" ]; then
            echo "No file open. Use the open command first."
            return  # Exit if no file is open
        fi
        local file="$CURRENT_FILE"  # Set file to the current open file
    fi

    local search_term="$1"
    file=$(realpath "$file")

    # Use grep to directly get the desired formatted output
    local matches=$(grep -nH -- "$search_term" "$file")

    # Check if no matches were found
    if [ -z "$matches" ]; then
        echo "No matches found for \"$search_term\" in $file"
        return
    fi

    # Calculate total number of matches
    local num_matches=$(echo "$matches" | wc -l | awk '{$1=$1; print $0}')

    # Calculate total number of lines matched
    local num_lines=$(echo "$matches" | cut -d: -f1 | sort | uniq | wc -l | awk '{$1=$1; print $0}')

    # If num_lines is > 100, print an error
    if [ $num_lines -gt 100 ]; then
        echo "More than $num_lines lines matched for \"$search_term\" in $file. Please narrow your search."
        return
    fi

    # Print the total number of matches and the matches themselves
    echo "Found $num_matches matches for \"$search_term\" in $file:"
    echo "$matches" | cut -d: -f1-2 | sort -u -t: -k2,2n | while IFS=: read -r filename line_number; do
        echo "$line_number:$(sed -n \"${line_number}p\" \"$file\")"
    done
    echo "End of matches for \"$search_term\" in $file"
}
"""

find_file = """find_file() {
    if [ $# -eq 1 ]; then
        local file_name="$1"
        local dir="./"
    elif [ $# -eq 2 ]; then
        local file_name="$1"
        if [ -d "$2" ]; then
            local dir="$2"
        else
            echo "Directory $2 not found"
            return
        fi
    else
        echo "Usage: find_file <file_name> [<dir>]"
        return
    fi

    dir=$(realpath "$dir")
    local matches=$(find "$dir" -type f -name "$file_name")

    # if no matches, return
    if [ -z "$matches" ]; then
        echo "No matches found for \"$file_name\" in $dir"
        return
    fi

    # Calculate total number of matches
    local num_matches=$(echo "$matches" | wc -l | awk '{$1=$1; print $0}')

    echo "Found $num_matches matches for \"$file_name\" in $dir:"
    echo "$matches" | awk '{print $0}'
}
"""

edit = """#!/root/miniconda3/envs/aider/bin/python

# @yaml
# signature: edit $<to_replace> $<new_content>
# docstring: Replaces occurrence of $<to_replace> with $<new_content> in the currently open file.
# arguments:
#   to_replace:
#       type: string
#       description: The text to be replaced in the file.
#       required: true
#   new_content:
#       type: string
#       description: The new text to replace with.
#       required: true

import os
import sys
import shutil
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from _agent_skills import edit_file_by_replace

# Suppress any future warnings if necessary
warnings.simplefilter("ignore", category=FutureWarning)

# Configuration
BACKUP_DIR = '/root/tmp/file_edit_backups'
BACKUP_HISTORY_FILE = os.path.join(BACKUP_DIR, 'backup_history.txt')

def create_backup(file_path):
    \"\"\"Create a backup of the file before editing.\"\"\"
    try:
        # Create backup directory if it doesn't exist
        Path(BACKUP_DIR).mkdir(parents=True, exist_ok=True)
        
        # Create backup history file if it doesn't exist
        if not os.path.exists(BACKUP_HISTORY_FILE):
            Path(BACKUP_HISTORY_FILE).touch()
            
        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"{Path(file_path).stem}_{timestamp}{Path(file_path).suffix}"
        backup_path = os.path.join(BACKUP_DIR, backup_filename)
        
        # Create backup
        shutil.copy2(file_path, backup_path)
        
        # Record backup in history file
        with open(BACKUP_HISTORY_FILE, 'a') as f:
            f.write(f"{backup_path}::{file_path}\\n")
            
    except Exception as e:
        print(f"Warning: Failed to create backup: {e}", file=sys.stderr)

def main():
    # Check if CURRENT_FILE environment variable is set
    current_file = os.environ.get('CURRENT_FILE')
    if not current_file:
        print('No file open. Use the `open` command first.')
        sys.exit(1)

    os.environ['ENABLE_AUTO_LINT'] = 'true'

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Edit a file by replacing specific content based on diffs.'
    )
    parser.add_argument('to_replace', type=str, help='The text to be replaced in the file.')
    parser.add_argument('new_content', type=str, help='The new text to replace with.')
    args = parser.parse_args()

    to_replace = args.to_replace
    new_content = args.new_content

    # Validate arguments
    if not to_replace:
        print("Error: 'to_replace' must not be empty.")
        print("Usage: edit $<to_replace> $<new_content>")
        sys.exit(1)
    if to_replace == new_content:
        print("Error: 'to_replace' and 'new_content' must be different.")
        print("Usage: edit $<to_replace> $<new_content>")
        sys.exit(1)

    # Create backup before editing
    create_backup(current_file)

    # Call the edit function
    try:
        edit_file_by_replace(current_file, to_replace, new_content)
    except Exception as e:
        print(f"Error editing file: {e}", file=sys.stderr)
        print("Usage: edit $<to_replace> $<new_content>")
        sys.exit(1)

if __name__ == '__main__':
    main()
"""

undo_edit = """#!/root/miniconda3/envs/aider/bin/python

# @yaml
# signature: undo_edit [file_path]
# docstring: Reverts the last edit made to the specified file. If no file is provided, reverts the last edit on the currently open file.
# arguments:
#   file_path:
#     type: string
#     description: The path to the file to undo the last edit for.
#     required: false

import os
import sys
import warnings
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple

# Suppress any future warnings if necessary
warnings.simplefilter("ignore", category=FutureWarning)

# Configuration
BACKUP_DIR = '/root/tmp/file_edit_backups'
BACKUP_HISTORY_FILE = os.path.join(BACKUP_DIR, 'backup_history.txt')

class BackupManager:
    @staticmethod
    def get_file_backups(file_path: str) -> List[Tuple[str, str]]:
        \"\"\"Get all backups for a specific file.\"\"\"
        if not os.path.exists(BACKUP_HISTORY_FILE):
            return []

        backups = []
        with open(BACKUP_HISTORY_FILE, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        backup_path, orig_path = line.strip().split("::")
                        if orig_path == file_path and os.path.exists(backup_path):
                            backups.append((backup_path, orig_path))
                    except ValueError:
                        continue
        return backups

    @staticmethod
    def restore_backup(backup_path: str, target_file: str) -> bool:
        \"\"\"Restore a file from its backup.\"\"\"
        try:
            if not os.path.exists(backup_path):
                return False
            shutil.copy2(backup_path, target_file)
            return True
        except Exception:
            return False

    @staticmethod
    def update_history(entries: List[str]) -> None:
        \"\"\"Update the backup history file.\"\"\"
        with open(BACKUP_HISTORY_FILE, 'w') as f:
            f.writelines(entries)

    @staticmethod
    def cleanup_old_backups(file_path: str, keep_last: int = 5) -> None:
        \"\"\"Remove old backups keeping only the specified number of recent ones.\"\"\"
        backups = BackupManager.get_file_backups(file_path)
        if len(backups) <= keep_last:
            return

        # Remove older backups
        for backup_path, _ in backups[:-keep_last]:
            try:
                os.remove(backup_path)
            except OSError:
                pass

def undo_last_edit(file_path: str) -> bool:
    \"\"\"
    Undo the last edit for a specific file.
    Returns True if successful, False otherwise.
    \"\"\"
    try:
        # Get all backups for the file
        backups = BackupManager.get_file_backups(file_path)
        if not backups:
            print(f"No edits have been made to the file: {file_path}")
            return False

        # Get the most recent backup
        last_backup, original_file = backups[-1]

        # Verify files exist
        if not os.path.exists(last_backup):
            print(f"Backup file not found: {last_backup}")
            return False

        if not os.path.exists(original_file):
            print(f"Original file not found: {original_file}")
            return False

        # Restore from backup
        if not BackupManager.restore_backup(last_backup, original_file):
            print("Failed to restore from backup")
            return False

        # Update backup history
        with open(BACKUP_HISTORY_FILE, 'r') as f:
            all_entries = f.readlines()

        entries_to_keep = [
            entry for entry in all_entries 
            if entry.strip() != f"{last_backup}::{original_file}"
        ]

        BackupManager.update_history(entries_to_keep)

        # Cleanup old backups
        BackupManager.cleanup_old_backups(file_path)

        print(f"Successfully restored {file_path} to previous version")
        return True

    except Exception as e:
        print(f"Error during undo operation: {e}", file=sys.stderr)
        return False

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Reverts the last edit made to the specified file.'
    )
    parser.add_argument('file_path', nargs='?', default=None, 
                       help='The path to the file to undo the last edit for.')
    args = parser.parse_args()

    # Determine the file to undo
    file_path = args.file_path
    if not file_path:
        file_path = os.environ.get('CURRENT_FILE')
        if not file_path:
            print('No file specified and no file open. Use the `open` command first or specify a file.')
            sys.exit(1)

    # Attempt to undo last edit
    if not undo_last_edit(file_path):
        sys.exit(1)

if __name__ == '__main__':
    main()
"""

insert = """#!/root/miniconda3/envs/aider/bin/python

# @yaml
# signature: insert <line_number> $<content>
# docstring: Inserts $<content> at the given <line_number> in the currently open file.
# arguments:
#   line_number:
#       type: int
#       description: The line number where the content should be inserted.
#       required: true
#   content:
#       type: string
#       description: The content to insert at the specified line number.
#       required: true

import os
import re
import sys
import shutil
import argparse
import warnings
from pathlib import Path
from datetime import datetime

# Suppress any future warnings if necessary
warnings.simplefilter("ignore", category=FutureWarning)

from _agent_skills import insert_content_at_line

# Configuration
BACKUP_DIR = '/root/tmp/file_edit_backups'
BACKUP_HISTORY_FILE = os.path.join(BACKUP_DIR, 'backup_history.txt')

def create_backup(file_path):
    \"\"\"Create a backup of the file before editing.\"\"\"
    try:
        # Create backup directory if it doesn't exist
        Path(BACKUP_DIR).mkdir(parents=True, exist_ok=True)
        
        # Create backup history file if it doesn't exist
        if not os.path.exists(BACKUP_HISTORY_FILE):
            Path(BACKUP_HISTORY_FILE).touch()
            
        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f\"{Path(file_path).stem}_{timestamp}{Path(file_path).suffix}\"
        backup_path = os.path.join(BACKUP_DIR, backup_filename)
        
        # Create backup
        shutil.copy2(file_path, backup_path)
        
        # Record backup in history file
        with open(BACKUP_HISTORY_FILE, 'a') as f:
            f.write(f\"{backup_path}::{file_path}\\n\")
            
    except Exception as e:
        print(f\"Warning: Failed to create backup: {e}\", file=sys.stderr)

def main():
    # Check if CURRENT_FILE environment variable is set
    current_file = os.environ.get('CURRENT_FILE')
    if not current_file:
        print('No file open. Use the `open` command first.')
        sys.exit(1)

    # Set ENABLE_AUTO_LINT environment variable
    os.environ['ENABLE_AUTO_LINT'] = 'true'

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Inserts $<content> at the given <line_number> in the currently open file.'
    )
    parser.add_argument('line_number', type=int, help='The line number where the content should be inserted.')
    parser.add_argument('content', type=str, help='The content to insert at the specified line number.')
    args = parser.parse_args()

    line_number = args.line_number
    content = args.content

    # Validate arguments
    if line_number <= 0:
        print(\"Error: 'line_number' must be a valid integer.\")
        print(\"Usage: insert <line_number> $<content>\")
        sys.exit(1)
    if not content:
        print(\"Error: 'content' must not be empty.\")
        print(\"Usage: insert <line_number> $<content>\")
        sys.exit(1)

    # Create backup before editing
    create_backup(current_file)

    # Call the insert function
    try:
        insert_content_at_line(current_file, line_number, content)
    except Exception as e:
        print(f\"Error inserting content: {e}\", file=sys.stderr)
        print(\"Usage: insert <line_number> $<content>\")
        sys.exit(1)

if __name__ == '__main__':
    main()
"""

append = """#!/root/miniconda3/envs/aider/bin/python

# @yaml
# signature: append $<content>
# docstring: Appends $<content> to the end of the currently open file.
# arguments:
#   content:
#       type: string
#       description: The content to append to the end of the file.
#       required: true

import os
import sys
import shutil
import argparse
import warnings
from pathlib import Path
from datetime import datetime

# Suppress any future warnings if necessary
warnings.simplefilter("ignore", category=FutureWarning)

from _agent_skills import append_file

# Configuration
BACKUP_DIR = '/root/tmp/file_edit_backups'
BACKUP_HISTORY_FILE = os.path.join(BACKUP_DIR, 'backup_history.txt')

def create_backup(file_path):
    \"\"\"Create a backup of the file before editing.\"\"\"
    try:
        # Create backup directory if it doesn't exist
        Path(BACKUP_DIR).mkdir(parents=True, exist_ok=True)
        
        # Create backup history file if it doesn't exist
        if not os.path.exists(BACKUP_HISTORY_FILE):
            Path(BACKUP_HISTORY_FILE).touch()
            
        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"{Path(file_path).stem}_{timestamp}{Path(file_path).suffix}"
        backup_path = os.path.join(BACKUP_DIR, backup_filename)
        
        # Create backup
        shutil.copy2(file_path, backup_path)
        
        # Record backup in history file
        with open(BACKUP_HISTORY_FILE, 'a') as f:
            f.write(f"{backup_path}::{file_path}\\n")
            
    except Exception as e:
        print(f"Warning: Failed to create backup: {e}", file=sys.stderr)

def main():
    # Check if CURRENT_FILE environment variable is set
    current_file = os.environ.get('CURRENT_FILE')
    if not current_file:
        print('No file open. Use the `open` command first.')
        sys.exit(1)

    # Set ENABLE_AUTO_LINT environment variable
    os.environ['ENABLE_AUTO_LINT'] = 'true'

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Appends $<content> to the end of the currently open file.'
    )
    parser.add_argument('content', type=str, help='The content to append to the end of the file.')
    args = parser.parse_args()

    content = args.content

    # Validate content
    if not content:
        print("Error: 'content' must not be empty.")
        print("Usage: append $<content>")
        sys.exit(1)

    # Create backup before editing
    create_backup(current_file)

    # Call the append function
    try:
        append_file(current_file, content)
    except Exception as e:
        print(f"Error appending content: {e}", file=sys.stderr)
        print("Usage: append $<content>")
        sys.exit(1)

if __name__ == '__main__':
    main()
"""

execute_ipython = """#!/usr/bin/env python3

# @yaml
# signature: execute_ipython $<code>
# docstring: Executes Python code in a persistent cell, returning its output. Variables persist between executions.
# arguments:
#   code:
#       type: string
#       description: Python code to execute in the cell.
#       required: true

import sys
import os
import argparse
import socket
import pickle

# Path to the Unix domain socket
SOCKET_FILE = '/tmp/python_cell_socket'

def start_server_process():
    \"\"\"Start the server process if it's not already running.\"\"\"
    import subprocess
    import time
    server_script = 'python_cell_server'

    try:
        # Start the server script directly, assuming it's in the PATH
        subprocess.Popen([server_script])
    except FileNotFoundError:
        print(f"Error: Server script '{server_script}' not found in PATH.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error starting server process: {e}", file=sys.stderr)
        sys.exit(1)

    # Wait a moment for the server to start
    time.sleep(0.5)

def send_code_to_server(code: str):
    \"\"\"Send code to the server and get the output.\"\"\"
    # Ensure the server is running
    if not os.path.exists(SOCKET_FILE):
        start_server_process()

    # Connect to the server
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.connect(SOCKET_FILE)
    except socket.error as e:
        print(f"Error connecting to server: {e}", file=sys.stderr)
        sys.exit(1)

    # Send the length of the code
    code_bytes = code.encode('utf-8')
    client.sendall(len(code_bytes).to_bytes(8, byteorder='big'))
    # Send the actual code
    client.sendall(code_bytes)

    # Receive the length of the response
    response_length_bytes = client.recv(8)
    if not response_length_bytes:
        print("No response from server.", file=sys.stderr)
        sys.exit(1)
    response_length = int.from_bytes(response_length_bytes, byteorder='big')
    # Receive the actual response
    response_bytes = b''
    while len(response_bytes) < response_length:
        chunk = client.recv(response_length - len(response_bytes))
        if not chunk:
            break
        response_bytes += chunk
    response = pickle.loads(response_bytes)
    client.close()
    return response

def format_output(output: str, errors: str) -> None:
    \"\"\"Format and print the output from the cell execution.\"\"\"
    if output.strip():
        print("Output:")
        print("-" * 50)
        print(output.rstrip())
        
    if errors.strip():
        print("\\nErrors:", file=sys.stderr)
        print("-" * 50, file=sys.stderr)
        print(errors.rstrip(), file=sys.stderr)

def main():
    # Set up argument parser to take code as a command-line argument
    parser = argparse.ArgumentParser(
        description="Executes Python code in a persistent cell."
    )
    parser.add_argument(
        "code", 
        type=str, 
        nargs='?', 
        help="The Python code to execute in the cell. If not provided, reads from stdin."
    )
    args = parser.parse_args()
    
    if args.code:
        code = args.code
    else:
        # Read code from stdin
        code = sys.stdin.read()
    
    # Check if the provided code is empty or only contains whitespace
    if not code.strip():
        print("Error: 'code' must not be empty.")
        print("Usage: execute_ipython $<code>")
        sys.exit(1)
    
    # Send the code to the server and get the response
    response = send_code_to_server(code)
    output = response.get('output', '')
    errors = response.get('errors', '')

    # Format and display the output
    format_output(output, errors)
    
    # Exit with error code if there were any errors
    if errors.strip():
        sys.exit(1)

if __name__ == '__main__':
    main()
"""

execute_server = """#!/usr/bin/env python3

# @yaml
# signature: execute_server <command>
# docstring: To run long-lived processes such as server or daemon. It runs the command in the background and provides a log of the output.
# arguments:
#   command:
#       type: string
#       description: Bash command to execute in the shell.
#       required: true
# special_commands:
#   - get_logs: Retrieves the last 100 lines of the server log.
#   - stop: Stops the background Bash server process.

import sys
import os
import argparse
import socket
import pickle
import errno

SOCKET_FILE = '/tmp/bash_command_socket'

def start_server_process():
    \"\"\"Start the Bash server process if it's not already running.\"\"\"
    import subprocess
    import time
    server_script = 'bash_server'

    try:
        subprocess.Popen([server_script])
    except FileNotFoundError:
        print(f\"Error: Server script '{server_script}' not found in PATH.\", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f\"Error starting server process: {e}\", file=sys.stderr)
        sys.exit(1)

    # Wait for the server to start
    timeout = 5  # seconds
    start_time = time.time()
    while not os.path.exists(SOCKET_FILE):
        if time.time() - start_time > timeout:
            print(\"Error: Server did not start within expected time.\", file=sys.stderr)
            sys.exit(1)
        time.sleep(0.1)

def send_command_to_server(command: str):
    \"\"\"Send a command to the server and get the output.\"\"\"
    if not os.path.exists(SOCKET_FILE):
        start_server_process()

    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.connect(SOCKET_FILE)
    except socket.error as e:
        if e.errno in (errno.ENOENT, errno.ECONNREFUSED):
            print(\"Server not running, starting server...\")
            start_server_process()
            try:
                client.connect(SOCKET_FILE)
            except socket.error as e:
                print(f\"Error connecting to server after restart: {e}\", file=sys.stderr)
                sys.exit(1)
        else:
            print(f\"Error connecting to server: {e}\", file=sys.stderr)
            sys.exit(1)

    try:
        # Send the command
        command_bytes = command.encode('utf-8')
        client.sendall(len(command_bytes).to_bytes(8, byteorder='big'))
        client.sendall(command_bytes)

        # Receive the response length
        response_length_bytes = client.recv(8)
        if not response_length_bytes:
            print(\"No response from server.\", file=sys.stderr)
            sys.exit(1)
        response_length = int.from_bytes(response_length_bytes, byteorder='big')

        # Read the response
        response_bytes = b''
        while len(response_bytes) < response_length:
            chunk = client.recv(min(response_length - len(response_bytes), 4096))
            if not chunk:
                break
            response_bytes += chunk
        response = pickle.loads(response_bytes)
        client.close()
        return response
    except Exception as e:
        print(f\"Error during communication with server: {e}\", file=sys.stderr)
        client.close()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description=\"Executes Bash commands through a persistent server.\"
    )
    parser.add_argument(
        \"command\",
        type=str,
        help=\"The Bash command to execute, or 'stop'/'get_logs' for special commands.\"
    )
    args = parser.parse_args()

    # Send the command to the server
    response = send_command_to_server(args.command)
    output = response.get('output', '')
    errors = response.get('errors', '')

    if output.strip():
        print(\"Output:\")
        print(\"-\" * 50)
        print(output.rstrip())

    if errors.strip():
        print(\"\\nErrors:\", file=sys.stderr)
        print(\"-\" * 50, file=sys.stderr)
        print(errors.rstrip(), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
"""

print_ = """_print() {
    local total_lines=$(awk 'END {print NR}' $CURRENT_FILE)
    echo "[File: $(realpath $CURRENT_FILE) ($total_lines lines total)]"

    lines_above=$(jq -n "$CURRENT_LINE - $WINDOW/2" | jq '[0, .] | max | floor')
    lines_below=$(jq -n "$total_lines - $CURRENT_LINE - $WINDOW/2" | jq '[0, .] | max | round')

    if [ $lines_above -gt 0 ]; then
        echo "($lines_above more lines above)"
    fi

    cat $CURRENT_FILE | grep -n $ | head -n $(jq -n "[$CURRENT_LINE + $WINDOW/2, $WINDOW/2] | max | floor") | tail -n $(jq -n "$WINDOW")

    if [ $lines_below -gt 0 ]; then
        echo "($lines_below more lines below)"
    fi
}
"""

constrain_line = """_constrain_line() {
    if [ -z "$CURRENT_FILE" ]
    then
        echo "No file open. Use the open command first."
        return
    fi

    local max_line=$(awk 'END {print NR}' $CURRENT_FILE)
    local half_window=$(jq -n "$WINDOW/2" | jq 'floor')

    export CURRENT_LINE=$(jq -n "[$CURRENT_LINE, $max_line - $half_window] | min")
    export CURRENT_LINE=$(jq -n "[$CURRENT_LINE, $half_window] | max")
}
"""


COMMANDS = [
  {
    "contents": open,
    "name": "open",
    "type": "source_file"
  },
  {
    "contents": goto,
    "name": "goto",
    "type": "source_file"
  },
  {
    "contents": scroll_down,
    "name": "scroll_down",
    "type": "source_file"
  },
  {
    "contents": scroll_up,
    "name": "scroll_up",
    "type": "source_file"
  },
  {
    "contents": create,
    "name": "create",
    "type": "source_file"
  },
  {
    "contents": submit,
    "name": "submit",
    "type": "source_file"
  },
  {
    "contents": search_dir,
    "name": "search_dir",
    "type": "source_file"
  },
  {
    "contents": search_file,
    "name": "search_file",
    "type": "source_file"
  },
  {
    "contents": find_file,
    "name": "find_file",
    "type": "source_file"
  },
  {
    "contents": edit,
    "name": "edit",
    "type": "script"
  },
  {
    "contents": undo_edit,
    "name": "undo_edit",
    "type": "script"
  },
  {
    "contents": insert,
    "name": "insert",
    "type": "script"
  },
  {
    "contents": append,
    "name": "append",
    "type": "script"
  },
  {
    "contents": execute_ipython,
    "name": "execute_ipython",
    "type": "script"
  },
  {
    "contents": execute_server,
    "name": "execute_server",
    "type": "script"
  },
  {
    "contents": print_,
    "name": "_print",
    "type": "utility"
  },
  {
    "contents": constrain_line,
    "name": "_constrain_line",
    "type": "utility"
  }
]