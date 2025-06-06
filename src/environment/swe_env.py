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
from swebench.harness.utils import get_environment_yml, get_requirements

import docker
import docker.errors
import docker.models.containers
from sweagent import REPO_ROOT
from sweagent.environment.utils import (
    PROCESS_DONE_MARKER_END,
    PROCESS_DONE_MARKER_START,
    InvalidGithubURL,
    copy_anything_to_container,
    copy_file_to_container,
    format_trajectory_markdown,
    get_container,
    get_gh_issue_data,
    get_instances,
    image_exists,
    parse_gh_issue_url,
    read_with_timeout,
    read_with_timeout_experimental,
)
from sweagent.utils.config import keys_config
from sweagent.utils.log import default_logger, get_logger

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
        self.container_obj: docker.models.containers.Container | None = None
        self.container: subprocess.Popen | None = None
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

        # Move construct_codegraph.py to the container
        subprocess.run(
            f"docker cp {os.getcwd()}/sweagent/environment/utils_codegraph.py {self.container_name}:/root/utils_codegraph.py",
            shell=True,
        )
        subprocess.run(
            f"docker cp {os.getcwd()}/sweagent/environment/construct_graph.py {self.container_name}:/root/construct_graph.py",
            shell=True,
        )
        # Move graph retrieval script to the container
        subprocess.run(
            f"docker cp {os.getcwd()}/sweagent/environment/retrieve_graph.py {self.container_name}:/root/retrieve_graph.py",
            shell=True,
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

    def close(self) -> None:
        for hook in self.hooks:
            hook.on_close()

    def run_bash_command(self, command: str, timeout_duration: int | float = 120) -> tuple[str, int | None]:
        """
        Executes a bash command (single or multi-line) in the environment's container.

        This method consolidates the core logic for sending a command to the container,
        handling timeouts, capturing output, and managing common execution errors
        like BrokenPipeError or general RuntimeErrors.

        Args:
            command: The bash command string to execute. Can be single or multi-line.
            timeout_duration: The maximum time (in seconds) to wait for the command to complete.

        Returns:
            A tuple containing:
                - str: The output received from the command's execution in the container.
                - int | None: The return code of the executed command, or None if it couldn't be determined.
        """
        self.logger.log(logging.TRACE, "Executing bash command:\n%s", command)

        output = ""
        return_code = None

        try:
            # Add a newline if not already present, to ensure execution
            cmd_to_send = command if command.endswith("\n") else command + "\n"

            # Use the end-marker approach for robustness in getting exit code
            # This is similar to _communicate_experimental
            command_suffix = (
                f'EXITSTATUS="$?"; sleep 0.01; echo {PROCESS_DONE_MARKER_START}$EXITSTATUS{PROCESS_DONE_MARKER_END}\n'
            )
            final_cmd = cmd_to_send + command_suffix

            os.write(self.container.stdin.fileno(), final_cmd.encode())
            time.sleep(0.03) # A small sleep to ensure write propagates
            self.container.stdin.flush()

            buffer, exit_code_str = read_with_timeout_experimental(self.container, timeout_duration)
            output = buffer

            if exit_code_str == "$EXITSTATUS":
                # Fallback for when the exit code isn't properly captured
                output = (
                    "Unknown error occurred when running the command. Please double check syntax "
                    "and that you're not running an interactive command."
                )
                self.logger.warning("Couldn't get real exit code. Setting it to 999 for this command.")
                return_code = 999
            elif exit_code_str and exit_code_str.isdigit():
                return_code = int(exit_code_str)
            else:
                # This should ideally not happen with the end-marker, but for safety
                self.logger.error(f"Failed to parse exit code. Raw output:\n---\n{output}\n---")
                return_code = -1 # Indicate an unknown or unparseable exit code

        except TimeoutError:
            self.logger.warning("Command execution timed out. Attempting to interrupt container processes.")
            output += "\nEXECUTION TIMED OUT. Attempted to interrupt running processes."
            try:
                self.interrupt() # Use the existing interrupt mechanism
                output += "\nInterrupt signal sent successfully."
            except RuntimeError as e:
                output += f"\nINTERRUPT FAILED: {e}. Consider restarting the container."
                self.logger.error(f"Failed to interrupt container after timeout: {e}")
            return_code = -2 # Custom code for timeout with attempted interrupt
        except BrokenPipeError:
            self.logger.error("Broken pipe error during command execution. Container communication might be compromised.")
            output += "\nBROKEN PIPE ERROR. Container communication issue detected."
            return_code = -3 # Custom code for broken pipe
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred during command execution: {e}")
            output += f"\nUNEXPECTED EXECUTION ERROR: {e}"
            return_code = -4 # Custom code for general unexpected error

        self.logger.log(logging.TRACE, "Command output:\n%s\nReturn Code: %s", output, return_code)
        return output, return_code

    # MARK: Helper functions #

    def _reset_container(self) -> None:
        self._init_scripts()

    def reset_container(self) -> None:
        self._reset_container()

    # @staticmethod
    # def _get_container_name(image_name: str) -> str:
    #     return f"{image_name_sanitized}-{hash_object.hexdigest()[:10]}"

    def _init_container(self, cached_image: str | None = None) -> None:
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

    def get_pids(self, all_pids: bool = False) -> list[str]:
        """
        Gets list of processes running inside docker container

        Args:
            all_pids: whether to return all pids, or whether to exclude ps
                and parent PIDs

        Returns:
            list of PIDs
        """
        pids = self.container_obj.exec_run("ps -eo pid,comm --no-headers").output.decode().split("\n")
        pids = [x.split() for x in pids if x]
        if not all_pids:
            pids = [x for x in pids if x[1] != "ps" and x[0] not in self.parent_pids]
        return pids

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

    def interrupt(self) -> None:
        """
        Send interrupt signal to container and exhaust stdout buffer with a communicate call
        """
        assert self.container is not None
        assert self.container_obj is not None
        pids = self.get_pids()
        for pid, cmd in pids:
            if pid not in self.parent_pids and cmd != "ps":
                self.container_obj.exec_run(f"kill -9 {pid}")
        try:
            _ = read_with_timeout(self.container, self.get_pids, 20)
        except TimeoutError:
            pass
        try:
            output = self.communicate(input="echo 'interrupted'", timeout_duration=60)
            assert output.strip().endswith("interrupted"), "container health check failed"
        except TimeoutError:
            msg = "Failed to interrupt container"
            raise RuntimeError(msg)

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