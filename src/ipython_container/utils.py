from pathlib import Path
import subprocess
import os

import traceback

from .container import IPythonContainer, UMain

from ..utils.log import get_logger
logger = get_logger("env_utils")

def udocker_init():

    if not os.path.exists("/home/user"):
        print("Setting up udocker for the first time...")
        subprocess.run(["pip", "install", "udocker"], capture_output=True, text=True, check=True)
        subprocess.run(["udocker", "--allow-root", "install"], capture_output=True, text=True, check=True)
        subprocess.run(["useradd", "-m", "user"], capture_output=True, text=True, check=True)
        print("udocker setup complete.")

    print(f'Docker-in-Colab 1.1.0\n')
    print(f'Usage:     udocker("--help")')
    print(f'Examples:  https://github.com/indigo-dc/udocker?tab=readme-ov-file#examples')


    def execute(command: str) -> dict:
        user_prompt = "\033[1;32muser@pc\033[0m"
        print(f"{user_prompt}$ udocker {command}")

        try:
            full_cmd = ["su", "-", "user", "-c", f"udocker {command}"]

            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=300
            )

            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except FileNotFoundError:
            error_msg = f"Error: Command '{full_cmd[0]}' not found. Ensure 'su' and 'udocker' are in PATH."
            print(error_msg)
            return {"stdout": "", "stderr": error_msg, "returncode": 127}
        except subprocess.TimeoutExpired:
            error_msg = f"Error: Command '{command}' timed out after 300 seconds."
            print(error_msg)
            return {"stdout": "", "stderr": error_msg, "returncode": 124}
        except Exception as e:
            error_msg = f"An unexpected error occurred during udocker command execution: {e}"
            print(error_msg)
            return {"stdout": "", "stderr": error_msg, "returncode": 1}

    return execute


####################################################
#           udocker command executor
####################################################
udocker_command_executor = udocker_init()


def image_exists(image_name: str) -> bool:
    """
    Checks if a udocker image exists in the local repository by inspecting
    the output of 'udocker images'.

    Args:
        image_name: The name of the image to check (e.g., 'busybox', 'ubuntu:latest').

    Returns:
        True if the image is found in the local udocker repository, False otherwise.
    """
    if not image_name:
        return False

    print(f"Checking if udocker image '{image_name}' exists locally...")
    
    # Run 'udocker images' to list all local images
    result = udocker_command_executor("images")
    
    if result["returncode"] != 0:
        print(f"WARNING: 'udocker images' command failed with error: {result['stderr'].strip()}")
        return False # Cannot reliably determine image existence if the command itself fails

    images_output = result["stdout"]
    return image_name in images_output


# This function copies string contents to a file inside the container
def copy_file_to_container(container: IPythonContainer, contents: str, container_path: str) -> None:
    """
    Copies a given string into a udocker container at a specified path.

    Args:
        container: Our custom udocker.Container object.
        contents: The string to copy into the container.
        container_path: The absolute path inside the container where the string should be copied to.
                        This will be the final path of the file inside the container.
    """
    try:
        # Use the Container's method to copy this temporary file into the container
        copy_result = container.write_file_to_container(contents, container_path)
        
    except Exception as e:
        logger.error(f"An error occurred in copy_file_to_container (string content): {e}")
        logger.error(traceback.format_exc()) # Log the full traceback
        raise # Re-raise to ensure calling code knows it failed


# This function copies any file or directory from host to container
def copy_anything_to_container(container: IPythonContainer, host_path: str, container_path: str) -> None:
    """
    Copies a file or directory from the host to a udocker container at a specified path.

    Args:
        container: Our custom udocker.Container object.
        host_path: The absolute path on the host to the file or directory to copy.
        container_path: The absolute destination path inside the container.
                        If host_path is a file, this is the file's final destination.
                        If host_path is a directory, this is the directory's final destination.
    """
    if not Path(host_path).exists():
        msg = f"Path {host_path} does not exist on host, cannot copy it to container."
        logger.error(msg)
        raise FileNotFoundError(msg)
    
    # Delegate to the Container's general copy method
    copy_result = container.copy_host_path_to_container(host_path, container_path)