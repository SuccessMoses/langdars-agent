import shutil
import os
import subprocess
import shlex
import re
from pathlib import Path

import tempfile
import traceback

from ..utils.log import get_logger
logger = get_logger("env_utils")


def copy_dir(dir, final_parent_dir):
    full_path = os.path.join(final_parent_dir, os.path.basename(dir))
    print(f"full_path: {full_path}")
    if os.path.exists(full_path):
        print(f"Removing existing directory: {full_path}")
        shutil.rmtree(full_path)
    shutil.copytree(dir, full_path)

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

class _Container:
    """
    Represents a udocker container and provides methods to interact with it.
    Assumes 'udocker_command_executor' is defined in the global scope.
    """
    def __init__(self, image: str, name: str = None):
        self.image = image
        self.name = name if name else image.replace("/", "_").replace(":", "_").replace("-", "_") + "_udocker_cont"

        if 'udocker_command_executor' not in globals() or not callable(globals()['udocker_command_executor']):
            raise RuntimeError(
                "udocker_command_executor is not defined or not callable. "
                "Please run the `udocker_command_executor = udocker_init()` cell first."
            )
        self.executor = globals()['udocker_command_executor']

        self.mounts = {}

        print(f"Container Init: Pulling image '{self.image}'...")
        pull_result = self.executor(f"pull {self.image}")
        if pull_result["returncode"] != 0:
            print(f"WARNING: Image pull failed with exit code {pull_result['returncode']}. Stderr: {pull_result['stderr'].strip()}")

        print(f"Container Init: Creating container instance '{self.name}' from '{self.image}'...")
        create_result = self.executor(f"create --name={self.name} {self.image}")
        if create_result["returncode"] != 0:
            print(f"ERROR: Container creation failed with exit code {create_result['returncode']}. Stderr: {create_result['stderr'].strip()}")
            raise RuntimeError(f"Failed to create container '{self.name}'.")

        print(f"Container '{self.name}' initialized.")

    def add_mount(self, key: str, mount_str: str) -> None:
        if not mount_str.startswith("/"):
            print(f"WARNING: Mount string '{mount_str}' does not start with '/'. "
                  f"Ensure it's an absolute host path.")
        self.mounts[key] = mount_str
        print(f"Mount '{key}' added: {mount_str}")

    def run(self, cmd: str, mount_keys: list = None, adhoc_mounts: list = None, interactive: bool = False) -> dict:
        all_mount_strings = []

        if mount_keys:
            for key in mount_keys:
                if key in self.mounts:
                    all_mount_strings.append(self.mounts[key])
                else:
                    print(f"WARNING: Mount key '{key}' not found in stored mounts. Skipping.")
        if adhoc_mounts:
            for mount_str in adhoc_mounts:
                if not mount_str.startswith("/"):
                    print(f"WARNING: Ad-hoc mount path '{mount_str}' does not start with '/'. Ensure it's an absolute path.")
                all_mount_strings.append(mount_str)

        mount_args = ""
        if all_mount_strings:
            mount_args = " " + " ".join([f"-v {m}" for m in all_mount_strings])

        interactive_arg = "-it" if interactive else ""

        # Safely wrap the command in double quotes for bash -c
        bash_wrapped_cmd = f'/bin/bash -c "{cmd}"'

        full_udocker_cmd = f"run {interactive_arg}{mount_args} {self.name} {bash_wrapped_cmd}"

        print(f"Container Run: Executing '{cmd}' in '{self.name}'...")

        return self.executor(full_udocker_cmd)

    def remove(self) -> None:
        print(f"Container Remove: Removing container instance '{self.name}'...")
        remove_result = self.executor(f"rm {self.name}")
        if remove_result["returncode"] != 0:
            print(f"WARNING: Container removal failed with exit code {remove_result['returncode']}. Stderr: {remove_result['stderr'].strip()}")
        print(f"Container '{self.name}' removed.")
                
class Container(_Container):

    def _get_container_host_root_path(self) -> str:
        """
        Retrieves the absolute path of the container's root filesystem on the host
        by using 'udocker inspect -p'. This is the most reliable method.
        """
        inspect_cmd = f"inspect -p {self.name}"
        print(f"Container Inspect: Running '{inspect_cmd}' to get host root path...")
        inspect_result = self.executor(inspect_cmd)

        if inspect_result["returncode"] != 0:
            error_msg = f"Failed to get host root path for container '{self.name}': {inspect_result['stderr'].strip()}"
            print(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg)

        # The 'udocker inspect -p' command directly outputs the path.
        # We just need to strip any leading/trailing whitespace.
        container_root_path = inspect_result["stdout"].strip()

        if not container_root_path:
            raise RuntimeError(f"udocker inspect -p returned an empty path for container '{self.name}'.")

        print(f"Successfully retrieved container host root path: {container_root_path}")
        return container_root_path
        
    def copy_host_path_to_container(self, host_src_path: str, container_full_dest_path: str) -> dict:
        """
        Copies a file or directory from the host to a specific path inside this udocker container's
        root filesystem by directly accessing the container's host-side directory.

        Args:
            host_src_path: The absolute path on the host to the file or directory to copy.
            container_full_dest_path: The absolute destination path inside the container.
                                     If host_src_path is a file, this is the file's final destination.
                                     If host_src_path is a directory, this is the directory's final destination
                                     (e.g., if copying 'my_dir' to '/app', result is '/app/my_dir').

        Returns:
            A dictionary with 'stdout', 'stderr', and 'returncode' of the operation.
        """
        try:
            if not os.path.isabs(host_src_path):
                raise ValueError(f"host_src_path must be an absolute path: '{host_src_path}'")
            if not os.path.isabs(container_full_dest_path):
                raise ValueError(f"container_full_dest_path must be an absolute path (start with /) inside the container: '{container_full_dest_path}'")

            container_host_root = self._get_container_host_root_path()

            # Determine the actual final target path on the host where the item will reside
            copied_item_name = os.path.basename(host_src_path) # copied_item_name is "my_data"
            final_parent_dir_on_host = os.path.join(container_host_root, container_full_dest_path.lstrip(os.sep))
            os.makedirs(os.path.dirname(final_parent_dir_on_host), exist_ok=True)

            if os.path.isfile(host_src_path):
                shutil.copyfile(host_src_path, os.path.join(final_parent_dir_on_host, copied_item_name))
            elif os.path.isdir(host_src_path):
                copy_dir(host_src_path, final_parent_dir_on_host)
            else:
                raise ValueError(f"host_src_path '{host_src_path}' is neither a file nor a directory.")

            print(f"Successfully copied '{host_src_path}' to '{final_parent_dir_on_host}' (inside udocker container '{self.name}')")
            return {
                "stdout": f"Successfully copied '{copied_item_name}' to '{os.path.join(container_full_dest_path, copied_item_name).replace(os.sep + os.sep, os.sep)}' in container '{self.name}'.",
                "stderr": "",
                "returncode": 0
            }
        except Exception as e:
            error_msg = f"Error copying '{host_src_path}' to udocker container '{self.name}': {e}"
            # logger.error(error_msg)
            # logger.error(traceback.format_exc()) # Log the full traceback
            return {"stdout": "", "stderr": error_msg, "returncode": 1}
        


# This function copies string contents to a file inside the container
def copy_file_to_container(container: Container, contents: str, container_path: str) -> None:
    """
    Copies a given string into a udocker container at a specified path.

    Args:
        container: Our custom udocker.Container object.
        contents: The string to copy into the container.
        container_path: The absolute path inside the container where the string should be copied to.
                        This will be the final path of the file inside the container.
    """
    temp_file_name = None
    try:
        # Create a temporary file on the host with the string contents
        with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as temp_file:
            temp_file_name = temp_file.name
            temp_file.write(contents)

        # Use the Container's method to copy this temporary file into the container
        copy_result = container.copy_host_path_to_container(temp_file_name, container_path)
        
        # Check the result of the copy operation
        if copy_result["returncode"] != 0:
            logger.error(f"Failed to copy string content to container (udocker): {copy_result['stderr']}")
            raise RuntimeError(f"Failed to copy string content to container: {copy_result['stderr']}")

    except Exception as e:
        logger.error(f"An error occurred in copy_file_to_container (string content): {e}")
        logger.error(traceback.format_exc()) # Log the full traceback
        raise # Re-raise to ensure calling code knows it failed
    finally:
        # Cleanup: Remove the temporary file if it was created
        if temp_file_name and os.path.exists(temp_file_name):
            os.remove(temp_file_name)


# This function copies any file or directory from host to container
def copy_anything_to_container(container: Container, host_path: str, container_path: str) -> None:
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

    if copy_result["returncode"] != 0:
        msg = f"Error copying '{host_path}' to container at '{container_path}': {copy_result['stderr']}"
        logger.error(msg)
        raise RuntimeError(msg)