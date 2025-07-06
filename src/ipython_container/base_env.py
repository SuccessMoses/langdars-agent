from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import random
import re
import shlex
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import yaml
from ghapi.all import GhApi
from git import Repo
from simple_parsing.helpers.serialization.serializable import FrozenSerializable
from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
from swebench.harness.test_spec.python import get_environment_yml, get_requirements

from .utils import (
    copy_anything_to_container,
    copy_file_to_container,
    image_exists,
)
from .container import IPythonContainer

from .. import REPO_ROOT
from ..utils.utils import (
    InvalidGithubURL,
    format_trajectory_markdown, # fixme
    get_gh_issue_data, #fixme
    get_instances,
    parse_gh_issue_url, # fixme
)
from ..utils.config import keys_config
from ..utils.log import default_logger, get_logger

from udocker.config import Config

LONG_TIMEOUT = float(keys_config.get("SWE_AGENT_ENV_LONG_TIMEOUT", 500))
AGENT_ACTION_TIMEOUT = float(keys_config.get("SWE_AGENT_ACTION_TIMEOUT", 120))
PATH_TO_REQS = "/root/requirements.txt"
PATH_TO_ENV_YML = "/root/environment.yml"


@dataclass(frozen=True)
class EnvironmentArguments(FrozenSerializable):
    """Configure data sources and setup instructions for the environment in which we solve the tasks."""

    # Source of issue statement/problem statement. To run over a batch of issues: Path to a data file
    # (`json`, `jsonl`) or directory. To run over single issue: github issue url or path to markdown file
    # with problem statement or problem statement as text prefixed with `text://`.
    data_path: str
    # Name of the docker image to use for the environment. Defaults to sweagent/swe-agent:latest
    image_name: str = "sweagent/swe-agent:latest"
    # When running over SWE-bench issues: Specify the split to use.
    split: str = "dev"
    # Specify a branch name or a commit hash to checkout before running the task.
    # Only used when running over a single problem statement/issue.
    base_commit: str | None = None
    # Use a persistent container with this name. After every task, the container will be paused, but not removed.
    # This is useful for speedup when running multiple tasks from the same repositories in a row, as the repositories
    # will have already been cloned and the conda environments will have been installed.
    container_name: str | None = None
    # Try to install the environment before running the task.
    install_environment: bool = True
    # No effect, kept for backwards compatibility.
    timeout: int | None = None
    # Enable environment logger.
    verbose: bool = False
    # Do not use attempt to use a repository mirror from https://github.com/swe-bench.
    no_mirror: bool = False
    # Cache task images to speed up task initialization. This means that the environment will be saved as a
    # docker image for every repository, base commit, and setup combination. This uses quite a bit of disk space
    # but speeds up task initialization significantly when running over multiple issues from the same repository
    # (or using different models for the same issues).
    cache_task_images: bool = False
    # Custom environment setup. Currently only used when data_path points to a single issue.
    # This needs to be either a string pointing to a yaml file (with yaml, yml file extension)
    # or a shell script (with sh extension).
    # See https://princeton-nlp.github.io/SWE-agent/usage/cl_tutorial#environment-setup
    environment_setup: str | None = None
    # Only used when running on single issue. Path to local repository or github repository.
    repo_path: str = ""
    persistent_volume: str | None = None

    def __post_init__(self):
        if self.timeout is not None:
            default_logger.warning("The 'timeout' argument is deprecated and has no effect.")
        if self.cache_task_images and self.container_name:
            msg = (
                "Not allowed to use persistent container with caching task images "
                "(probably doesn't make sense and takes excessive space)."
            )
            raise ValueError(msg)
        if self.container_name is not None and self.container_name.strip() == "":
            msg = "Set container_name to None if you don't want to use a persistent container."
            raise ValueError(msg)


class EnvHook:
    """Hook to be used in `SWEEnv`.

    Subclass this class, add functionality and add it with `SWEEEnv.add_hook(hook)`.
    This allows to inject custom functionality at different stages of the environment
    lifecycle, in particular to connect SWE-agent to a new interface (like a GUI).
    """

    def on_init(self) -> None:
        """Gets called when the hook is added"""

    def on_copy_repo_started(self, *, repo_type: str, repo_path: str) -> None:
        """Gets called when the repository is being cloned to the container

        Args:
            repo_type: Type of repository. Either 'local' or 'github'
            repo_path: Path to the repository
        """

    def on_install_env_started(self) -> None:
        """Called when we start installing the environment"""

    def on_close(self):
        """Called when the environment is closed"""


class BaseSWEEnv(gym.Env):
    """Gym environment for SWE-bench. This class should handle all communication with the docker container."""

    name = "swe_main"
    # This prefix will be prepended to the image name when caching task images
    cached_image_prefix = "swe-agent-task-env-"

    def __init__(self, args: EnvironmentArguments):
        super().__init__()
        t0 = time.perf_counter()
        self.args = args
        self.base_commit: str | None = None
        self.communicate_output: str | None = None
        self.container_name: str | None = args.container_name
        self.install_environment = args.install_environment
        self.logger = get_logger("SWEEnv")
        self.persistent = args.container_name is not None
        self.returncode: None | int = None
        if not self.args.verbose:
            # fixme: This creates problems if we have multiple instances of this class
            self.logger.disabled = True

        #: The commit hash of the swe-agent repository
        self.commit_sha = None
        try:
            repo = Repo(REPO_ROOT, search_parent_directories=True)
            self.commit_sha = repo.head.object.hexsha
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.logger.exception("Failed to get commit hash for this repo: %s", str(e))

        self._github_token: str = keys_config.get("GITHUB_TOKEN", "")  # type: ignore

        # Load Task Instances
        self.data_path = self.args.data_path
        self.data = get_instances(
            self.data_path,
            self.args.base_commit,
            self.args.split,
            token=self._github_token,
            repo_path=self.args.repo_path,
        )
        #: Instance we're currently processing. Gets set in self.reset.
        self.record: dict[str, Any] | None = None
        self.logger.info(f"💽 Loaded dataset from {self.data_path}")

        # Establish connection with execution container
        self.image_name = args.image_name
        self.container_obj: IPythonContainer | None = None
        self._reset_container()

        self.idx = 0
        self.clean_multi_line_functions = lambda x: x
        self.hooks: list[EnvHook] = []

        self.logger.debug("Environment initialization took %.2f seconds", time.perf_counter() - t0)

    def _get_cached_task_image_name(self) -> str:
        assert self.record is not None
        inputs: list[str] = [
            self.record["repo"],
            self.record["base_commit"],
            self.args.environment_setup or "no_setup",
        ]
        tag = hashlib.sha256("".join(inputs).encode()).hexdigest()[:50]
        return f"{self.cached_image_prefix}{tag}"

    def add_hook(self, hook: EnvHook):
        """Add `EnvHook` to the environment.

        This allows to inject custom functionality at different stages of the environment
        lifecycle, in particular to connect SWE-agent to a new interface (like a GUI).
        """
        hook.on_init()
        self.hooks.append(hook)

    @property
    def _repo_name(self) -> str:
        """Name of the local copy of the repository"""
        assert self.record is not None
        return self.record["repo"].replace("/", "__")

    def _copy_repo(self) -> str:
        """Clone/copy repository/codebase in container

        Returns:
            folder name of clone
        """
        assert self.container_obj is not None
        assert self.record is not None  # mypy
        for hook in self.hooks:
            hook.on_copy_repo_started(repo_type=self.record["repo_type"], repo_path=self.record["repo"])
        if self.record["repo_type"] == "local":
            copy_anything_to_container(
                self.container_obj,
                self.record["repo"].removeprefix("local://"),
                "/" + self._repo_name,
            )
            self.communicate_with_handling(
                input=f"chown -R root:root {self._repo_name}",
                error_msg="Failed to change permissions on copied repository",
            )
            return self._repo_name
        assert self.record["repo_type"] == "github"
        token_prefix = ""
        if self._github_token:
            token_prefix = f"{self._github_token}@"
        # fixme: This if statement is brittle and should probably be replaced with better logic
        if not self.args.no_mirror and self.record["problem_statement_source"] == "swe-bench":
            self.logger.info(f"{self._repo_name} not found in container, cloning...")
            clone_url = f"https://{token_prefix}github.com/swe-bench/{self._repo_name}.git"
        else:
            self.logger.info("Trying to clone from non-mirror...")
            clone_url = f"https://{token_prefix}github.com/{self.record['repo']}.git"
        clone_method = keys_config.get("SWE_AGENT_CLONE_METHOD", default="shallow", choices=["shallow", "full"])
        if len(self.data) > 1 or self.persistent:
            msg = "Falling back to full cloning method due to multiple instances or persistent container"
            clone_method = "full"
            self.logger.debug(msg)
        if clone_method == "full":
            self.communicate_with_handling(
                input=f"git clone {clone_url} {self._repo_name}",
                error_msg="Failed to clone repository from conservative method",
                timeout_duration=LONG_TIMEOUT,
            )
        else:
            base_commit = self.record["base_commit"]
            self.communicate_with_handling(
                input="&&".join(
                    (
                        f"mkdir {self._repo_name}",
                        f"cd {self._repo_name}",
                        "git init",
                        f"git remote add origin {clone_url}",
                        f"git fetch --depth 1 origin {base_commit}",
                        "git checkout FETCH_HEAD",
                        "cd ..",
                    )
                ),
                error_msg="Failed to clone repository with fast method",
                timeout_duration=LONG_TIMEOUT,
            )
        return self._repo_name

    def reset(self, index: int | None = None, apply_test_patch: bool = False) -> tuple[str | None, dict]:
        """
        Function to reset container between each task instance.

        * Clones instance's repository
        * Cleans repository of prior modifications
        * Resets environment variables
        * Check out base commit

        Args:
            index: index of task instance to reset to

        Returns:
            observation: output from container
            info: additional information (e.g. debugging information)
        """
        info = {}
        info["commit_sha"] = self.commit_sha

        # Get task instance
        self.idx = index if index is not None else self.idx
        self.record = self.data[self.idx]
        self.idx += 1

        # Set query, gold command
        self.base_commit = self.record["base_commit"]
        self.query = self.record["problem_statement"]
        self.reward = None

        ### Reset Container ###

        if self.args.cache_task_images:
            cached_image = self._get_cached_task_image_name()
            if image_exists(cached_image):
                self.logger.info(f"Restore environment from cached image {cached_image}")
                self.close()  # stop current container
                self._init_container(cached_image=cached_image)
                self.communicate("export $(xargs </.env)")
                envs = self.communicate("env")
                self.logger.debug(f"Environment variables restored from the image:\n{envs}\n")
                if apply_test_patch:
                    self._apply_test_patch()
                return None, info
            else:
                self.logger.info(f"Cached image {cached_image} not found, rebuilding task environment...")

        # Clone repository if not already cloned
        self.communicate(input="cd /")
        folders = self.communicate(input="ls").split("\n")
        if self._repo_name not in folders:
            self._copy_repo()

        # Clean repository of any modifications + Checkout base commit
        for cmd in [
            "echo -n > /root/files_to_edit.txt",
            f"cd {self._repo_name}",
            "export ROOT=$(pwd -P)",
            "git status",
            "git restore .",
            f"git reset --hard {self.base_commit}",
            "git clean -fdxq",
        ]:
            self.communicate_with_handling(
                input=cmd,
                error_msg="Failed to clean repository",
            )

        assert self.container_obj is not None # Ensure container_obj is initialized
        self.container_obj.copy_file_to_container(
            host_src_path=os.path.join(os.getcwd(), "src", "environment", "utils_codegraph.py"),
            container_dest_path="/root/utils_codegraph.py"
        )
        self.container_obj.copy_file_to_container(
            host_src_path=os.path.join(os.getcwd(), "src", "environment", "construct_graph.py"),
            container_dest_path="/root/construct_graph.py"
        )
        # Move graph retrieval script to the container
        self.container_obj.copy_file_to_container(
            host_src_path=os.path.join(os.getcwd(), "src", "environment", "retrieve_graph.py"),
            container_dest_path="/root/retrieve_graph.py"
        )

        # Reset environment variables
        for cmd in [
            'export CURRENT_FILE=""',
            "export CURRENT_LINE=0",
            "export SEARCH_RESULTS=()",
            "export SEARCH_FILES=()",
            "export SEARCH_INDEX=0",
        ]:
            self.communicate_with_handling(
                input=cmd,
                error_msg="Failed to reset environment variables",
            )

        # Set up environment
        self.communicate_with_handling(
            "source /root/miniconda3/etc/profile.d/conda.sh",
            error_msg="Failed to source conda",
        )

        system = self.communicate("uname -s").strip().lower()
        arch = self.communicate("uname -m").strip().lower()
        if system == "linux" and arch == "x86_64":
            self.communicate_with_handling(
                "apt update; apt install build-essential -y",
                error_msg="Failed to install build-essential",
                timeout_duration=LONG_TIMEOUT,
            )

        # Call install environment helper function if specified
        if self.install_environment:
            self.install_env()
        # Install mypy for linting purposes
        self.communicate_with_handling("pip install flake8", error_msg="Failed to install flake8 (lint library)")

        if self.args.cache_task_images:
            envs = self.communicate("env")
            self.logger.debug(f"Environment variables to save:\n{envs}\n")
            self.communicate("env >> /.env")
            assert self.container_obj is not None  # mypy
            self.container_obj.commit(cached_image)
            self.logger.info(f"Container with environment {self.container_obj.id} cached as image {cached_image}")

        if apply_test_patch:
            self._apply_test_patch()

        # Initialize code graph
        self.initialize_code_graph()

        # Write any metadata to info if necessary
        return None, info

    def initialize_code_graph(self):
        """
        Initializes code graph for the current task instance
        """
        self.communicate(input="cd /")
        # hardcode environment setting for code graph script
        self.logger.info('Setting up environment for code graph...')
        output_logs = self.communicate_with_handling('pip install networkx grep_ast diskcache tqdm pygments dataclasses && pip list && which pip', 
                                                     error_msg="Failed to install some packages --- ",
                                                     timeout_duration=LONG_TIMEOUT)
        # resolve the issue of incompatible version for tree-sitter
        self.communicate_with_handling('pip install tree-sitter==0.20.4', error_msg="Failed to install downgrade tree-sitter.\n")
        self.communicate_with_handling('chmod +x /root/construct_graph.py', error_msg="Failed to make construct graph file executable.\n")
        self.logger.info('Constructing code graph...')

        base_path = "/root/persistent_data" if self.args.persistent_volume else ""
        code_graph_path = f"{base_path}/{self.record['instance_id']}"
        self.logger.info(f'Code graph path: {code_graph_path}')

        response = self.communicate_with_handling(
            input=f"/root/construct_graph.py --repo_dir {self._repo_name} --output_dir {code_graph_path}",
            error_msg="Failed to initialize code graph\n",
            timeout_duration=LONG_TIMEOUT,
        )
        self.logger.info(f"Code graph initialized:\n {response}")
        self.communicate_with_handling(
            input=f"cd {self._repo_name}",
            error_msg="Failed to change directory to main\n",
        )

    def _apply_test_patch(self):
        """
        Apply test patch for oracle setting
        """
        assert self.record is not None
        path_to_patch = "test.patch"
        with open(path_to_patch, "w") as f:
            f.write(self.record["test_patch"])
        subprocess.run(
            f"docker cp {path_to_patch} {self.container_name}:/root/test.patch",
            shell=True,
            check=False,
        )
        self.communicate_with_handling(
            input="git apply /root/test.patch",
            error_msg="Failed to apply test patch correctly",
        )
        os.remove(path_to_patch)

    def correct_edit_action(self, action):
        pattern = r'edit\s+(\d+):(\d+)(?:\n|$)(.*?)(?:(?:\nend_of_edit)|$)'
        match = re.search(pattern, action, re.DOTALL)
        if match:
            start_line = match.group(1)
            end_line = match.group(2)
            replacement_text = match.group(3).strip()
            replacement_text = re.sub(r'\s*end_of_edit\s*$', '', replacement_text, flags=re.IGNORECASE)
            if not replacement_text:
                replacement_text = ""
            new_action = f'edit {start_line}:{end_line} << end_of_edit\n{replacement_text}\nend_of_edit\n'
            return new_action
        return action

    def close(self) -> None:
        """
        Handle environment shutdown for udocker containers.
        This method will manage the lifecycle of the udocker container.
        """
        self.logger.info("Beginning environment shutdown...")
        
        if self.container_obj is None:
            # If container_obj is None, it means it was never properly initialized or already removed.
            self.logger.info("Container object is already None, no further action needed for container.")
            pass # Nothing to close/remove
        elif self.persistent:
            self.logger.info(f"Persistent udocker container '{self.container_obj.name}' will remain on disk for reuse.")
        else:
            # For non-persistent containers, remove them to clean up resources.
            try:
                self.container_obj.remove() # Call our custom Container's remove method
                self.logger.info(f"Non-persistent udocker container '{self.container_obj.name}' removed successfully.")
            except Exception as e:
                # Catch any errors during removal and log them, but don't prevent hook calls.
                self.logger.warning(f"Failed to remove udocker container '{self.container_obj.name}': {e}", exc_info=True)
        
        self.container_obj = None

        for hook in self.hooks:
            hook.on_close()
        
        self.logger.info("Environment shutdown complete.")

# MARK: Helper functions #
    def _reset_container(self) -> None:
        if self.container_obj is not None:
            try:
                self.container_obj.remove()
            except:
                self.logger.warning("Failed to terminate container", exc_info=True)
            else:
                self.logger.debug("Terminated container")
        self._init_container()
        self._init_scripts()

    def reset_container(self) -> None:
        self.close()
        self.container = None
        self.container_obj = None
        self._reset_container()

    @staticmethod
    def _get_container_name(image_name: str) -> str:
        """Return name of container"""
        process_id = str(os.getpid())
        current_time = str(datetime.datetime.now())
        unique_string = current_time + process_id
        hash_object = hashlib.sha256(unique_string.encode())
        image_name_sanitized = image_name.replace("/", "-")
        image_name_sanitized = image_name_sanitized.replace(":", "-")
        return f"{image_name_sanitized}-{hash_object.hexdigest()[:10]}"


    def _init_container(self, cached_image: str | None = None) -> None:
        """
        Handles container initialization for a udocker environment.
        Leverages the custom 'Container' class to manage udocker instances.

        Args:
            cached_image: If provided, uses this image name instead of the default self.image_name.
        """
        # Determine the image name to use
        image_to_use = self.image_name
        if cached_image is not None:
            image_to_use = cached_image
            self.logger.info(f"Using cached image: {image_to_use}")

        udocker_instance_name = None
        if self.persistent:
            # If persistent, we must have a pre-defined container_name
            # This assumes self.container_name is already set when persistent is True
            assert self.container_name is not None, "Persistent container requires a predefined name."
            udocker_instance_name = self.container_name
        else:
            udocker_instance_name = self._get_container_name(image_to_use)

        # Define the volume mounts
        # The 'Container' class expects a list of mount strings
        initial_mounts = []
        if self.args.persistent_volume:
            # Assuming the persistent volume always maps to /root/persistent_data in the container
            initial_mounts.append(f"{self.args.persistent_volume}:/root/persistent_data")

        try:
            self.container_obj = IPythonContainer(image=image_to_use, name=udocker_instance_name)
            # assign container to Config,conf for udocker API
            Config.conf['container'] = self.container_obj
        except Exception as e:
            # Catch any errors during Container class instantiation (pull/create)
            msg = f"Failed to initialize udocker container '{udocker_instance_name}' from image '{image_to_use}': {e}"
            self.logger.error(msg)
            raise RuntimeError(msg) from e

        # # Add the persistent volume mount to the container object's stored mounts
        # # This ensures it's available by key for future 'run' commands
        # if self.args.persistent_volume:
        #     self.container_obj.add_mount("persistent_data_volume", initial_mounts[0])
        # FIXME: IPythonContainer does not have add_mount() method

        self.logger.info("🌱 Environment Initialized")

    def _init_scripts(self):
        """
        Initialize custom commands within container
        """
        self.communicate_with_handling(
            "source /root/.bashrc",
            error_msg="Failed to source .bashrc",
        )
        self.communicate_with_handling(
            "mkdir -p /root/commands",
            error_msg="Failed to create commands directory",
        )
        self.communicate_with_handling(
            "touch /root/commands/__init__.py",
            error_msg="Failed to create __init__.py",
        )
        self.communicate_with_handling(
            "export PATH=$PATH:/root/commands",
            error_msg="Failed to add commands directory to PATH",
        )

    def _communicate(
        self,
        input_command: str, 
        timeout_duration: int | float = 25, 
        mount_keys: list = None
    ) -> str:
        """
        Runs command in udocker container and returns output.
        Simplified for custom udocker.Container class.

        Args:
            input_command: command to run in container
            timeout_duration: (Currently ignored in this simplified version, as subprocess handles completion)
                            Could be passed to subprocess.run(timeout=...) if fine-grained timeout needed.

        Returns:
            The combined stdout and stderr of the command.
        """
        assert self.container_obj is not None, "Container object is not initialized."

        # Use self.container_obj.run() directly
        # This will call our modified executor which returns stdout, stderr, returncode
        try:
            self.returncode = self.container_obj.run(input_command)

            # Check for potential errors based on returncode
            if self.returncode != 0:
                self.logger.warning(
                    f"Command '{input_command}' failed with exit code {self.returncode}. "
                )
        except Exception as e:
            self.logger.error(f"Error executing command '{input_command}' in udocker container: {e}")
            raise RuntimeError(f"Failed to execute command in udocker container: {e}") from e


    def _check_syntax(self, input: str) -> tuple[str, bool]:
        """
        Check syntax of command.

        Returns:
            output: Output of the command
            success: whether the exit code was 0
        """
        output = self._communicate(f"/bin/bash -n <<'EOF'\n{input}\nEOF\n")
        return output, self.returncode == 0

    def communicate(self, input: str, timeout_duration: int | float = 25, *, set_last_action: bool = False) -> str:
        """
        Sends input to container and returns output

        Args:
            input: input to send to container
            timeout_duration: duration to wait for output
            set_last_action: whether to set the LAST_ACTION environment variable

        Returns:
            output: output from container
        """
        if input.strip() != "exit":
            self.logger.log(logging.TRACE, "Input:\n%s", input)  # type: ignore
            output, valid = self._check_syntax(input)
            if not valid:
                return output  # shows syntax errors
            self._communicate(
                input,
                timeout_duration=timeout_duration,
            )
            if set_last_action:
                # Cannot merge this with last command, because of multiline command
                # handling.
                last_action_string = shlex.quote(input.strip())
                input = f"export LAST_ACTION={last_action_string}"
                self._communicate(input, timeout_duration=60)
            return output
        else:
            self.container.terminate()
            self.returncode = 0
            return ""

    def communicate_with_handling(self, input: str, error_msg: str, timeout_duration: int | float = 25) -> str:
        """
        Wrapper for communicate function that raises error if return code is non-zero

        Args:
            input: input to send to container
            error_msg: error message to raise if return code is non-zero
            timeout_duration: duration to wait for output

        Returns:
            output: output from container
        """
        logs = self.communicate(input, timeout_duration=timeout_duration)
        if self.returncode != 0:
            self.logger.error(f"{error_msg}: {logs}")
            self.close()
            msg = f"{error_msg}: {logs}"
            raise RuntimeError(msg)
        return logs

    def get_submission(self, output: str) -> str | None:
        """
        Function for extracting diff patch submission at the end of an episode.

        Args:
            output: `submit` observation

        Returns:
            submission: diff patch submission
        """
        pattern = r"\<\<SUBMISSION\|\|(.*)\|\|SUBMISSION\>\>"
        match = re.search(pattern, output, re.DOTALL)
        if match is None:
            return None
        return match.group(1)

    def run_shell_script(self, script_path: Path, *, location: str) -> None:
        """Run custom script supplied by user at `script_path`

        Args:
            script_path: path to script file
            location: location of script file 'host' or 'container'
        """
        if location == "host":
            return self._run_shell_script_host(script_path)
        elif location == "container":
            raise NotImplementedError
        msg = f"Invalid 'location': {location}"
        raise ValueError(msg)

    def _run_shell_script_host(self, script_path: Path) -> None:
        """Run shell script file (located on host) in container"""
        if not script_path.is_file():
            msg = f"Script not found at {script_path}"
            raise FileNotFoundError(msg)
        shell_commands = Path(script_path).read_text().splitlines(keepends=True)
        for i, cmd in enumerate(shell_commands):
            self.communicate_with_handling(
                cmd,
                error_msg=f"Failed to execute line {i}.",
                timeout_duration=LONG_TIMEOUT,
            )

    def _get_install_configs(self) -> dict | None:
        """Return config for environment setup"""
        assert self.record is not None  # mypy
        if (
            self.record["problem_statement_source"] != "swe-bench" or self.record["repo_type"] == "local"
        ) and self.args.environment_setup is None:
            self.logger.warning(
                "install_environment is set to True, but the data path is a GitHub URL "
                "without an environment config file (environment_config key/flag). "
                "Skipping conda environment installation.",
            )
            return None
        if self.args.environment_setup is not None:
            assert isinstance(self.args.environment_setup, (str, os.PathLike))
            if Path(self.args.environment_setup).suffix in [".yml", ".yaml"]:
                try:
                    return yaml.safe_load(Path(self.args.environment_setup).read_text())
                except Exception as e:
                    msg = "Environment config file needs to be a yaml file"
                    raise ValueError(msg) from e
            elif Path(self.args.environment_setup).suffix == ".sh":
                return {
                    "shell_script_path": self.args.environment_setup,
                }
            else:
                msg = "Environment config file needs to be a yaml file or shell script"
                raise ValueError(msg)
        else:
            try:
                return MAP_REPO_VERSION_TO_SPECS[self.record["repo"]][str(self.record["version"])]
            except KeyError as e:
                msg = (
                    "Tried to look up install configs in swe-bench, but failed. "
                    "You can set a custom environment config with the environment_config key/flag."
                )
                raise ValueError(msg) from e

    def _conda_environment_exists(self, env_name: str) -> bool:
        env_check = self.communicate(f"conda env list | grep {env_name}", timeout_duration=LONG_TIMEOUT)
        return env_check.strip() != ""

    def install_env(self) -> None:
        """
        Creates conda environment and installs third party dependencies to allow code execution
        """
        t0 = time.perf_counter()
        for hook in self.hooks:
            hook.on_install_env_started()
        install_configs = self._get_install_configs()
        if not install_configs:
            return
        if "shell_script_path" in install_configs:
            assert len(install_configs) == 1
            self.run_shell_script(Path(install_configs["shell_script_path"]), location="host")
            return
        assert self.record is not None  # mypy
        # Create environment if does not exist yet
        env_name = f"{self._repo_name}__{self.record['version']}"
        if not self._conda_environment_exists(env_name):
            self.logger.info(f"{env_name} conda env not found, creating...")
            packages = install_configs.get("packages", "")
            if packages == "requirements.txt":
                # Create conda environment
                self.communicate_with_handling(
                    f"conda create -n {env_name} python={install_configs['python']} -y",
                    error_msg="Failed to create conda environment",
                    timeout_duration=LONG_TIMEOUT,
                )
                self.logger.debug("Created conda environment")
                # Write reqs to requirements.txt in docker container
                content_reqs = get_requirements(self.record)
                copy_file_to_container(self.container_obj, content_reqs, PATH_TO_REQS)
                # Create conda environment + install reqs
                self.communicate_with_handling(
                    f"conda activate {env_name}",
                    error_msg="Failed to activate conda environment",
                )
                self.communicate_with_handling(
                    f"pip install -r {PATH_TO_REQS}",
                    error_msg="Failed to install requirements.txt",
                    timeout_duration=LONG_TIMEOUT,
                )
                self.logger.debug("Installed requirements from requirements.txt")
                self.communicate(f"rm {PATH_TO_REQS}")
            elif packages == "environment.yml":
                # Write environment.yml to file
                content_env_yml = get_environment_yml(self.record, env_name)
                # Hotfix for
                if not install_configs.get("no_use_env"):
                    content_env_yml += f'\n  - python={install_configs["python"]}\n'
                copy_file_to_container(self.container_obj, content_env_yml, PATH_TO_ENV_YML)
                if install_configs.get("no_use_env"):
                    # Create conda environment
                    self.communicate_with_handling(
                        f"conda create -c conda-forge -n {env_name} python={install_configs['python']} -y",
                        error_msg="Failed to create conda environment",
                        timeout_duration=LONG_TIMEOUT,
                    )
                    self.logger.debug("Created conda environment")
                    # Install packages
                    self.communicate_with_handling(
                        f"conda env update -f {PATH_TO_ENV_YML}",
                        error_msg="Failed to install environment.yml",
                        timeout_duration=LONG_TIMEOUT,
                    )
                    self.logger.debug("Installed packages from environment.yml")
                else:
                    # Create environment + install packages
                    self.communicate_with_handling(
                        f"conda env create --file {PATH_TO_ENV_YML}",
                        error_msg="Failed to create conda environment with environment.yml",
                        timeout_duration=LONG_TIMEOUT,
                    )
                    self.logger.debug("Created conda environment with environment.yml")
                self.communicate(f"rm {PATH_TO_ENV_YML}")
            else:
                python_env = f"python{install_configs['python']}"
                if self._conda_environment_exists(python_env):
                    self.communicate_with_handling(
                        f"conda create --name {env_name} --clone {python_env}",
                        error_msg="Failed to clone conda environment",
                        timeout_duration=LONG_TIMEOUT,
                    )
                    self.logger.debug("Cloned python conda environment")
                else:
                    self.logger.debug(f"Could not find {python_env}, creating new environment")
                    self.communicate_with_handling(
                        f"conda create -n {env_name} python={install_configs['python']} -y",
                        error_msg="Failed to create conda environment",
                        timeout_duration=LONG_TIMEOUT,
                    )
                self.communicate_with_handling(
                    f"conda activate {env_name}",
                    error_msg="Failed to activate conda environment",
                )
                if packages.strip():
                    self.communicate_with_handling(
                        f"conda install {packages} -y",
                        error_msg="Failed to install packages",
                        timeout_duration=LONG_TIMEOUT,
                    )
                    self.logger.debug("Installed conda packages")
            # Install extra pip packages if specified
            if install_configs.get("pip_packages"):
                self.communicate_with_handling(
                    f"source activate {env_name} && pip install {' '.join(install_configs['pip_packages'])}",
                    error_msg="Failed to install pip packages",
                    timeout_duration=LONG_TIMEOUT,
                )
                self.logger.debug("Installed extra pip dependencies")

        # Activate environment
        self.communicate_with_handling(f"conda activate {env_name}", error_msg="Failed to activate conda environment")

        # Install repo at base commit
        if install_configs.get("pre_install"):
            self.logger.info("Running pre-install commands...")
            for pre_install_cmd in install_configs["pre_install"]:
                self.communicate_with_handling(
                    pre_install_cmd,
                    error_msg="Pre-install commands failed to execute successfully",
                    timeout_duration=LONG_TIMEOUT,
                )
            self.logger.debug("Ran pre-install commands")
        self.logger.info(f"Installing {self._repo_name} at base commit...")
        if install_configs.get("install"):
            install_cmd = install_configs["install"]
            self.communicate_with_handling(
                install_cmd,
                error_msg="Install command failed to execute successfully",
                timeout_duration=LONG_TIMEOUT,
            )
            self.logger.debug("Ran install command")
        if install_configs.get("post_install"):
            self.logger.info("Running post-install commands...")
            for post_install_cmd in install_configs["post_install"]:
                self.communicate_with_handling(
                    post_install_cmd,
                    error_msg="Post-install commands failed to execute successfully",
                )
            self.logger.debug("Ran post-install commands")

        self.logger.info("Installation step took %.2f seconds", time.perf_counter() - t0)

    def add_commands(self, commands: list[dict]) -> None:
        """
        Adds custom commands to container
        """
        for command in commands:
            name = command["name"]
            contents = command["contents"]
            copy_file_to_container(self.container_obj, contents, f"/root/commands/{name}")
            if command["type"] == "source_file":
                self.communicate_with_handling(
                    f"source /root/commands/{name}",
                    error_msg=(
                        f"Failed to source {name}. If you meant to make a script,"
                        " start the file with a shebang (e.g. #!/usr/bin/env python)."
                    ),
                )
            elif command["type"] == "script":
                self.communicate_with_handling(
                    f"chmod +x /root/commands/{name}",
                    error_msg=f"Failed to chmod {name}",
                )
            elif command["type"] == "utility":
                # nothing to do for utility scripts
                pass
            else:
                msg = f"Invalid command type: {command['type']}"
                raise ValueError(msg)

    # def open_pr(self, *, trajectory, _dry_run: bool = False) -> None:
    #     """Create PR to repository

    #     Args:
    #         trajectory: Trajectory of actions taken by the agent
    #         _dry_run: Whether to actually push anything or just simulate it
    #     """
    #     self.logger.info("Opening PR")
    #     # Adding random string suffix to avoid name conflicts if we had a previously failed run
    #     issue_url = self.args.data_path
    #     try:
    #         issue = get_gh_issue_data(issue_url, token=self._github_token)
    #     except InvalidGithubURL as e:
    #         msg = "Data path must be a github issue URL if --open_pr is set."
    #         raise ValueError(msg) from e
    #     branch_name = f"swe-agent-fix-#{issue.number}-" + str(random.random())[2:10]

    #     self.communicate_with_handling(
    #         input="rm -f model.patch",
    #         error_msg="Failed to remove model patch",
    #         timeout_duration=60,
    #     )
    #     self.communicate_with_handling(
    #         input=f"git checkout -b {branch_name}",
    #         error_msg="Failed to switch to new branch",
    #         timeout_duration=60,
    #     )
    #     self.communicate_with_handling(
    #         input="git add .",
    #         error_msg="Failed to add commits",
    #         timeout_duration=60,
    #     )
    #     dry_run_flag = "--allow-empty" if _dry_run else ""
    #     commit_msg = [
    #         shlex.quote("Fix: {issue.title}"),
    #         shlex.quote("Closes #{issue.number}"),
    #     ]
    #     self.communicate_with_handling(
    #         input=f"git commit -m {commit_msg[0]} -m  {commit_msg[1]} {dry_run_flag}",
    #         error_msg="Failed to commit changes",
    #         timeout_duration=60,
    #     )

    #     owner, repo, _ = parse_gh_issue_url(issue_url)
    #     # If `--repo_path` was specified with a different github URL, then the record will contain
    #     # the forking user
    #     assert self.record is not None
    #     if self.record["repo_type"] != "github":
    #         # We already validated that `--data_path` is a github issue URL
    #         # so this is the only case where we can reach here
    #         msg = "--repo_path must point to a github URL if --open_pr is set"
    #         raise ValueError(msg)
    #     forker, _ = self.record["repo"].split("/")
    #     head = branch_name
    #     remote = "origin"
    #     if forker != owner:
    #         head = f"{forker}:{branch_name}"
    #         token_prefix = ""
    #         if self._github_token:
    #             token_prefix = f"{self._github_token}@"
    #         fork_url = f"https://{token_prefix}github.com/{forker}/{repo}.git"
    #         self.logger.debug(f"Using fork: {fork_url}")
    #         self.communicate_with_handling(
    #             input=f"git remote add fork {fork_url}",
    #             error_msg="Failed to create new git remote",
    #             timeout_duration=60,
    #         )
    #         remote = "fork"
    #     dry_run_prefix = "echo " if _dry_run else ""
    #     self.communicate_with_handling(
    #         input=f"{dry_run_prefix} git push {remote} {branch_name}",
    #         error_msg=(
    #             "Failed to push branch to remote. Please check your token and permissions. "
    #             "You might want to push to a fork with the push_gh_repo_url option."
    #         ),
    #         timeout_duration=60,
    #     )
    #     body = (
    #         f"This is a PR opened by AI tool [SWE Agent](https://github.com/princeton-nlp/SWE-agent/) "
    #         f"to close [#{issue.number}]({issue_url}) ({issue.title}).\n\nCloses #{issue.number}."
    #     )
    #     body += "\n\n" + format_trajectory_markdown(trajectory)
    #     api = GhApi(token=self._github_token)
    #     if not _dry_run:
    #         pr_info = api.pulls.create(
    #             owner=owner,
    #             repo=repo,
    #             title=f"SWE-agent[bot] PR to fix: {issue.title}",
    #             head=head,
    #             base="main",
    #             body=body,
    #             draft=True,
    #         )
    #         self.logger.info(
    #             f"🎉 PR created as a draft at {pr_info.html_url}. Please review it carefully, push "
    #             "any required changes onto the branch and then click "
    #             "'Ready for Review' to bring it to the attention of the maintainers.",
    #         )