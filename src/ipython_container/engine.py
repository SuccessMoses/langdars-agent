import sys
import os
import subprocess

import threading
import time
from queue import Queue, Empty

from udocker.msg import Msg
from udocker.config import Config
from udocker.container.structure import ContainerStructure
from udocker.engine.base import ExecutionEngineCommon as _ExecutionEngineCommon
from udocker.helper.hostinfo import HostInfo
from udocker.utils.uprocess import Uprocess
from udocker.utils.fileutil import FileUtil
from udocker.utils.uvolume import Uvolume


class ExecutionEngineCommon(_ExecutionEngineCommon):

    def _run_env_set(self):
        """Environment variables to set"""
        self.opt["env"].appendif("HOME=" + self.opt["home"])
        self.opt["env"].append("USER=" + self.opt["user"])
        self.opt["env"].append("LOGNAME=" + self.opt["user"])
        self.opt["env"].append("USERNAME=" + self.opt["user"])

        if str(self.opt["uid"]) == "0":
            self.opt["env"].append(r"PS1=%s# " % self.container_id[:8])
        else:
            self.opt["env"].append(r"PS1=%s\$ " % self.container_id[:8])

        self.opt["env"].append("SHLVL=0")
        self.opt["env"].append("container_ruser=" + HostInfo().username())
        self.opt["env"].append("container_root=" + self.container_root)
        self.opt["env"].append("container_uuid=" + self.container_id)
        self.opt["env"].append("container_execmode=" + "P1")
        cont_name = self.container_names
        # if Python 3
        if sys.version_info[0] >= 3:
            names = str(cont_name).translate(str.maketrans('', '', " '\"[]"))
        else:
            names = str(cont_name).translate(None, " '\"[]")

        self.opt["env"].append("container_names=" + names)


class BackgroundShell:
    def __init__(self, init_cmd=['/bin/bash']):
        self.process = None
        self.output_queue = Queue()
        self.output_thread = None
        self.running = False
        self.init_cmd = init_cmd

    def start(self, env=None):
        """Start the background shell process"""
        if self.running:
            Msg().out("Shell is already running")
            return

        # Start subprocess with bash (which supports 'source' command)
        self.process = subprocess.Popen(
            self.init_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )

        self.running = True

        # Start thread to read output
        self.output_thread = threading.Thread(target=self._read_output, daemon=True)
        self.output_thread.start()

        Msg().out("Background shell started")

    def _read_output(self):
        """Read output from subprocess in background thread"""
        while self.running and self.process.poll() is None:
            try:
                line = self.process.stdout.readline()
                if line:
                    self.output_queue.put(line.rstrip())
            except Exception as e:
                self.output_queue.put(f"Error reading output: {e}")
                break

    def send_command(self, command):
        """Send a command to the shell"""
        if not self.running or self.process is None:
            Msg().out("Shell is not running. Call start() first.")
            return

        try:
            if isinstance(command, list):
                command_str = ' '.join(str(item) for item in command)
            else:
                command_str = str(command)

            self.process.stdin.write(command_str + '\n')
            self.process.stdin.flush()
            Msg().out(f"Sent: {command_str}", l=4)
        except Exception as e:
            Msg().err(f"Error sending command: {e}")

    def get_output(self, timeout=1):
        """Get output from the shell (non-blocking)"""
        output_lines = []
        end_time = time.time() + timeout
        
        done = False
        while time.time() < end_time:
            try:
                line = self.output_queue.get(timeout=0.1)
                output_lines.append(line)
                
                # Check if we got the exit code marker - command is done!
                if line.startswith("__EXIT_CODE__"):
                    done = True
                    break
                    
            except Empty:
                time.sleep(0.1)
                    
        Msg().out("\n".join(output_lines[:-1]), l=4)
        if not done:
            Msg().err(f"shell.get_output timed out after {timeout} sec")
            return 66 # timeout error
        else:
            return {
                "returncode": int(output_lines[-1].replace("__EXIT_CODE__", "")),
                "output": "\n".join(output_lines[:-1]),
            }

    def send_and_get_output(self, command, timeout=2):
        """Send command and wait for output"""
        self.send_command(command)
        time.sleep(0.1)  # Give command time to execute
        return self.get_output(timeout)

    def stop(self):
        """Stop the background shell"""
        if self.process:
            self.running = False
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            Msg().out("Background shell stopped")

    def __del__(self):
        self.stop()

#FIXME: ChRootEngine, does not have volume mount and _setup()
class ChRootEngine(ExecutionEngineCommon):
    """Docker container execution engine using chroot
    Inherits from ContainerEngine class
    """
    
    def __init__(self, localrepo, container_id):
        self.shell = None
        
        # Set container_root BEFORE calling _setup
        container_structure = ContainerStructure(localrepo, container_id)
        container_dir, _ = container_structure.get_container_attr()
        print(f"container_dir: {container_dir}")
        self.container_root = container_dir + "/ROOT"
        
        # Now call _setup after container_root is defined
        self._setup(container_id)

    def _setup(self, container_id):
        # setup execution
        # if not self._run_init(container_id):
        #     return 2
        
        # build the actual command for chroot
        print(f"container_root: {self.container_root}")
        _cmd = [
            "sudo", "chroot", self.container_root,
            "/bin/bash"  # Remove "-c" since we want interactive shell
        ]
        
        # Initialize BackgroundShell with the chroot command
        self.shell = BackgroundShell(init_cmd=_cmd)
        
        # cleanup the environment
        # self._run_env_cleanup_dict()
        try:
            self.shell.start()                        
        except Exception as e:
            print(f"Error starting chroot shell: {e}")
            if self.shell:
                self.shell.stop()
    
    def run(self, cmd):
        if self.shell:
            return self.shell.send_and_get_output(cmd, timeout=Config.conf['cmd_timeout'])
        else:
            raise Exception("Shell not initialized")
        
    def cleanup(self):
        """Clean up resources"""
        if self.shell:
            self.shell.stop()
            self.shell = None


class PRootEngine(ExecutionEngineCommon):
    """Docker container execution engine using PRoot
    Provides a chroot like environment to run containers.
    Uses PRoot both as chroot alternative and as emulator of
    the root identity and privileges.
    Inherits from ContainerEngine class
    """

    def __init__(self):
        super(PRootEngine, self).__init__(None, None)
        self.executable = None                   # PRoot
        self.proot_noseccomp = False             # No seccomp mode
        self.proot_newseccomp = False            # New seccomp mode
        self._kernel = HostInfo().oskernel()     # Emulate kernel
        self.shell = None

    def select_proot(self):
        """Set proot executable and related variables"""
        self.executable = Config.conf['use_proot_executable']
        if not self.executable:
            self.executable = FileUtil("proot").find_exec()

        arch = HostInfo().arch()
        if self.executable == "UDOCKER" or not self.executable:
            self.executable = ""
            if HostInfo().oskernel_isgreater([4, 8, 0]):
                image_list = ["proot-%s-4_8_0" % (arch), "proot-%s" % (arch), "proot"]
            else:
                image_list = ["proot-%s" % (arch), "proot"]
            f_util = FileUtil(self.localrepo.bindir)
            self.executable = f_util.find_file_in_dir(image_list)

        if not os.path.exists(self.executable):
            Msg().err("Error: proot executable not found")
            Msg().out("Info: Host architecture might not be supported by",
                      "this execution mode:", arch,
                      "\n      specify path to proot with environment",
                      "UDOCKER_USE_PROOT_EXECUTABLE",
                      "\n      or choose other execution mode with: udocker",
                      "setup --execmode=<mode>", l=Msg.INF)
            sys.exit(1)

        if Config.conf['proot_noseccomp'] is not None:
            self.proot_noseccomp = Config.conf['proot_noseccomp']
        # FIXME: prrot_noseccomp should depend on host architecture
        # if self.exec_mode.get_mode() == "P2":
        #     self.proot_noseccomp = True
        if self._is_seccomp_patched(self.executable):
            self.proot_newseccomp = True

    def _is_seccomp_patched(self, executable):
        """Check if kernel has ptrace/seccomp fixes added
           on 4.8.0.
           Only required for kernels below 4.8.0 to check
           if the patch has been backported e.g CentOS 7
        """
        if "PROOT_NEW_SECCOMP" in os.environ:
            return True
        if ("PROOT_NO_SECCOMP" in os.environ or
                self.proot_noseccomp or
                HostInfo().oskernel_isgreater([4, 8, 0])):
            return False

        host_file = self.container_dir + "/osenv.json"
        host_info = self._get_saved_osenv(host_file)
        if host_info:
            if "PROOT_NEW_SECCOMP" in host_info:
                return True
            return False
        out = Uprocess().get_output([executable, "-r", "/",
                                     executable, "--help"])
        if not out:
            os.environ["PROOT_NEW_SECCOMP"] = "1"
            out = Uprocess().get_output([executable, "-r", "/",
                                         executable, "--help"])
            del os.environ["PROOT_NEW_SECCOMP"]
            if out:
                self._save_osenv(host_file, dict([("PROOT_NEW_SECCOMP", 1), ]))
                return True
        self._save_osenv(host_file)
        return False

    def _set_uid_map(self):
        """Set the uid_map string for container run command"""
        if self.opt["uid"] == "0":
            uid_map_list = ["-0", ]
        else:
            uid_map_list = \
                ["-i", self.opt["uid"] + ":" + self.opt["gid"], ]
        return uid_map_list

    def _create_mountpoint(self, host_path, cont_path, dirs_only=False):
        """Override create mountpoint"""
        return True

    def _get_volume_bindings(self):
        """Get the volume bindings string for container run command"""
        proot_vol_list = []
        for vol in self.opt["vol"]:
            proot_vol_list.extend(["-b", "%s:%s" % Uvolume(vol).split()])
        return proot_vol_list

    def _get_network_map(self):
        """Get mapping of TCP/IP ports"""
        proot_netmap_list = []
        for (cont_port, host_port) in list(self._get_portsmap().items()):
            proot_netmap_list.extend(["-p", "%d:%d" % (cont_port, host_port)])
        if self.opt["netcoop"] and self._has_option("--netcoop"):
            proot_netmap_list.extend(["-n", ])
        if proot_netmap_list and self._has_option("--port"):
            return proot_netmap_list
        return []

    def _get_qemu_string(self):
        """Get the qemu string for container run command if emulation needed"""
        qemu_filename = self._get_qemu()
        return ["-q", qemu_filename] if qemu_filename else []

    def run(self, container_id):
        """Execute a Docker container using PRoot. This is the main method
        invoked to run the a container with PRoot.
          * argument: container_id or name
          * options:  many via self.opt see the help
        """
        if self.shell is None:
            self._setup(container_id)
        _cmd = self.opt['cmd'][0]
        # exit code is necessary to signal the end of execution in the shell
        cmd = f'{_cmd}; echo \"__EXIT_CODE__$?\"' 
        return self.shell.send_and_get_output(cmd, timeout=Config.conf['cmd_timeout'])

    def _check_executable(self):
        """Overwrite super._check_executable to be compatible."""
        return "dummy"

    def _setup(self, container_id):
        # setup execution
        if not self._run_init(container_id):
            return 2

        self.select_proot()

        # seccomp and ptrace behavior change on 4.8.0 onwards
        if self.proot_noseccomp or os.getenv("PROOT_NO_SECCOMP"):
            self.opt["env"].append("PROOT_NO_SECCOMP=1")

        if self.proot_newseccomp or os.getenv("PROOT_NEW_SECCOMP"):
            self.opt["env"].append("PROOT_NEW_SECCOMP=1")

        if not HostInfo().oskernel_isgreater([3, 0, 0]):
            self._kernel = "6.0.0"

        if self.opt["kernel"]:
            self._kernel = self.opt["kernel"]

        # set environment variables
        self._run_env_set()

        if Msg.level >= Msg.DBG:
            proot_verbose = ["-v", "9", ]
        else:
            proot_verbose = []

        if (Config.conf['proot_link2symlink'] and
                self._has_option("--link2symlink")):
            proot_link2symlink = ["--link2symlink", ]
        else:
            proot_link2symlink = []

        if (Config.conf['proot_killonexit'] and
                self._has_option("--kill-on-exit")):
            proot_kill_on_exit = ["--kill-on-exit", ]
        else:
            proot_kill_on_exit = []

        # build the actual command
        cmd_l = self._set_cpu_affinity()
        cmd_l.append(self.executable)
        cmd_l.extend(proot_verbose)
        cmd_l.extend(proot_kill_on_exit)
        cmd_l.extend(proot_link2symlink)
        cmd_l.extend(self._get_qemu_string())
        cmd_l.extend(self._get_volume_bindings())
        cmd_l.extend(self._set_uid_map())
        cmd_l.extend(["-k", self._kernel, ])
        cmd_l.extend(self._get_network_map())
        cmd_l.extend(["-r", self.container_root, ])

        if self.opt["cwd"]:  # set current working directory
            cmd_l.extend(["-w", self.opt["cwd"], ])
        cmd_l.extend(['/bin/bash'])
        Msg().out("CMD =", cmd_l, l=Msg.VER)

        # cleanup the environment
        self._run_env_cleanup_dict()

        # execute
        # self._run_banner(self.opt["cmd"][0])
        self.shell = BackgroundShell(init_cmd=cmd_l)

        # Start the shell
        self.shell.start(env=os.environ.update(self.opt["env"].dict()))
        time.sleep(2) # Give proot and bash a moment to start