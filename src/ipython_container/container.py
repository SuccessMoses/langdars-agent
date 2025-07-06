import os
import sys
import shutil

from udocker.cli import UdockerCLI as _UdockerCLI
from udocker.msg import Msg
from udocker.umain import UMain as _UMain

from udocker.msg import Msg
from udocker.cmdparser import CmdParser
from udocker.config import Config
from udocker.container.localrepo import LocalRepository
from udocker.container.structure import ContainerStructure

from udocker.engine.base import ExecutionEngineCommon

Config.conf["cmd_timeout"] = 60
Config.conf["container"] = "__NOT_SET__"

class UdockerCLI(_UdockerCLI):

    def do_run(self, cmdp):
        """
        run: execute a container
        run [options] <container-id-or-name>
        run [options] <repo/image:tag>
        --rm                       :delete container upon exit
        --workdir=/home/userXX     :working directory set to /home/userXX
        --user=userXX              :run as userXX
        --user=root                :run as root
        --volume=/data:/mnt        :mount host directory /data in /mnt
        --novol=/proc              :remove /proc from list of volumes to mount
        --env="MYTAG=xxx"          :set environment variable
        --env-file=<file>          :read environment variables from file
        --hostauth                 :get user account and group from host
        --containerauth            :use container passwd and group directly
        --nosysdirs                :do not bind the host /proc /sys /run /dev
        --nometa                   :ignore container metadata
        --dri                      :bind directories relevant for dri graphics
        --hostenv                  :pass the host environment to the container
        --cpuset-cpus=<1,2,3-4>    :CPUs in which to allow execution
        --name=<container-name>    :set or change the name of the container
        --bindhome                 :bind the home directory into the container
        --kernel=<kernel-id>       :simulate this Linux kernel version
        --device=/dev/xxx          :pass device to container (R1 mode only)
        --location=<container-dir> :use container outside the repository
        --nobanner                 :don't print a startup banner
        --entrypoint               :override the container metadata entrypoint
        --platform=os/arch         :pull image for OS and architecture
        --pull=<when>              :when to pull (missing|never|always|reuse)
        --httpproxy=<proxy>        :use http proxy, see udocker pull --help

        Only available in Rn execution modes:
        --device=/dev/xxx          :pass device to container (R1 mode only)

        Only available in Pn execution modes:
        --publish=<hport:cport>    :map container TCP/IP <cport> into <hport>
        --publish-all              :bind and connect to random ports

        run <container-id-or-name> executes an existing container, previously
        created from an image by using: create <repo/image:tag>

        run <repo/image:tag> always creates a new container from the image.
        If needed the image is pulled. This is slow and may waste storage.
        Using run --name=<container-name> --pull=reuse allows to use existing
        container and only pull/create if the <container-name> does not exist.
        """
        self._get_run_options(cmdp)
        container_or_image = cmdp.get("P1")
        Config.conf['location'] = cmdp.get("--location=")
        delete = cmdp.get("--rm")
        name = cmdp.get("--name=")
        pull = cmdp.get("--pull=")
        cmdp.get("--pull")          # if invoked without option
        cmdp.get("--index=")        # used in do_pull()
        cmdp.get("--registry=")     # used in do_pull()
        cmdp.get("--httpproxy=")    # used in do_pull()

        if cmdp.missing_options():  # syntax error
            return self.STATUS_ERROR

        container_id = ""
        if Config.conf['location']:
            pass
        elif not container_or_image:
            Msg().err("Error: must specify container_id or image:tag")
            return self.STATUS_ERROR
        else:
            if pull == "reuse" and name:
                container_id = self.localrepo.get_container_id(name)
            if not container_id:
                container_id = self.localrepo.get_container_id(container_or_image)
            if not container_id:
                (imagerepo, tag) = self._check_imagespec(container_or_image)
                if (imagerepo and
                        self.localrepo.cd_imagerepo(imagerepo, tag)):
                    container_id = self._create(imagerepo + ":" + tag)
                if pull != "never" and (not container_id or pull == "always"):
                    self.do_pull(cmdp)
                    if self.localrepo.cd_imagerepo(imagerepo, tag):
                        container_id = self._create(imagerepo + ":" + tag)
                    if not container_id:
                        Msg().err("Error: image or container not available")
                        return self.STATUS_ERROR
            if name and container_id:
                if not self.localrepo.set_container_name(container_id, name):
                    if pull != "reuse":
                        Msg().err("Error: invalid container name")
                        return self.STATUS_ERROR
                    
        if Config.conf["container"] == "__NOT_SET__": #FIXME: use not isinstance()
            msg  = "Create an instance `container` and set `Config['container'] = container`."
            raise ContainerError(msg)
        exec_engine = Config.conf['container'].engine
        if not exec_engine:
            Msg().err("Error: no execution engine for this execmode")
            return self.STATUS_ERROR

        self._get_run_options(cmdp, exec_engine)
        exit_status = exec_engine.run(exec_engine.opt["cmd"])
        if delete and not self.localrepo.isprotected_container(container_id):
            self.localrepo.del_container(container_id)

        return exit_status


class UMain(_UMain):

    def _prepare_exec(self):
        """Prepare configuration, parse and execute the command line"""
        self.cmdp = CmdParser()
        self.cmdp.parse(self.argv)
        allow_root = self.cmdp.get("--allow-root", "GEN_OPT")
        if not (os.geteuid() or allow_root):
            Msg().err("Error: do not run as root !")
            sys.exit(self.STATUS_ERROR)

        if self.cmdp.get("--config=", "GEN_OPT"):
            conf_file = self.cmdp.get("--config=", "GEN_OPT")
            Config().getconf(conf_file)
        else:
            Config().getconf()

        if (self.cmdp.get("--debug", "GEN_OPT") or
                self.cmdp.get("-D", "GEN_OPT")):
            Config.conf['verbose_level'] = Msg.DBG
        elif (self.cmdp.get("--quiet", "GEN_OPT") or
              self.cmdp.get("-q", "GEN_OPT")):
            Config.conf['verbose_level'] = Msg.MSG
        Msg().setlevel(Config.conf['verbose_level'])

        if self.cmdp.get("--insecure", "GEN_OPT"):
            Config.conf['http_insecure'] = True

        topdir = self.cmdp.get("--repo=", "GEN_OPT")
        if topdir:  # override repo root tree
            Config.conf['topdir'] = topdir
        self.local = LocalRepository()
        if not self.local.is_repo():
            if topdir:
                Msg().err("Error: invalid udocker repository:", topdir)
                sys.exit(self.STATUS_ERROR)
            else:
                Msg().out("Info: creating repo: " + Config.conf['topdir'],
                          l=Msg.INF)
                self.local.create_repo()

        self.cli = UdockerCLI(self.local)


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


import subprocess
import threading
import time
import os
from queue import Queue, Empty

class BackgroundShell:
    def __init__(self, init_cmd=['/bin/bash']):
        self.process = None
        self.output_queue = Queue()
        self.output_thread = None
        self.running = False
        self.init_cmd = init_cmd
        
    def start(self):
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
            universal_newlines=True
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
                    
        Msg().out("\n".join(output_lines[:-1]))
        if not done:
            Msg().err(f"shell.get_output timed out after {timeout} sec")
            return 66 # timeout error
        else:
            return int(output_lines[-1].replace("__EXIT_CODE__", ""))        

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


class ContainerError(Exception):
    pass

class _IPythonContainer:

    def __init__(self, image, name):
        self.image = image
        self.name = name
        self.created = False
        self._setup()

        # pulling image
        res = self._pull(image)
        if res:
            raise ContainerError(f"Error pulling image {image}, return code: {res}")

        # creating container
        res = self._create(name)
        if res:
            raise ContainerError(f"Error creating container --name={name}, return code: {res}")
        else:
            self.created = True
            self._setup_engine()

    def _setup(self):
        if not os.path.exists("/home/user"):
            print("Setting up container manager for the first time...")
            subprocess.run(["pip", "install", "udocker"], capture_output=True, text=True, check=True)
            subprocess.run(["udocker", "--allow-root", "install"], capture_output=True, text=True, check=True)
            subprocess.run(["useradd", "-m", "user"], capture_output=True, text=True, check=True)
            print("Setup complete.")

    def _pull(self, image):
        return UMain(['udocker', '--allow-root', 'pull', image]).execute()

    def _create(self, name):
        return UMain(['udocker', '--allow-root', 'create', '--name=' + name, self.image]).execute()

    def _setup_engine(self):
        self.engine = ChRootEngine(LocalRepository(), self.name)

    def run(self, cmd):
        return UMain([
            'udocker', 
            '--allow-root', 
            'run', 
            self.name, 
            f'{cmd}; echo \"__EXIT_CODE__$?\"'
        ]).execute()
    
    def check_syntax(self, cmd):
        Msg().out(f"checking syntax: '{cmd}'")
        # Wrap with bash -n for syntax check
        syntax_check_cmd = f"/bin/bash -n <<'EOF'\n{cmd}\nEOF"

        # Run syntax check
        # 0 means syntax OK, non-zero means syntax error
        res = UMain([
            'udocker',
            '--allow-root',
            'run',
            self.name,
            syntax_check_cmd
        ]).execute()
        if res == 0:
            Msg().out("Syntax is valid")
        else:
            Msg().out("Syntax is not valid")

    def remove(self):
        print(f"Removing container {self.name}...")
        res = UMain(['udocker', '--allow-root', 'rm', self.name]).execute()
        if res:
            raise ContainerError(f"Error removing container {self.name}, return code: {res}")
        print(f"Container {self.name} removed.")

    def __del__(self):
        if self.created:
            self.remove()



def copy_dir(dir, final_parent_dir):
    full_path = os.path.join(final_parent_dir, os.path.basename(dir))
    print(f"full_path: {full_path}")
    if os.path.exists(full_path):
        print(f"Removing existing directory: {full_path}")
        shutil.rmtree(full_path)
    shutil.copytree(dir, full_path)


class IPythonContainer(_IPythonContainer):

    def write_file_to_container(self, contents: str, container_dest_path: str) -> dict:
        """
        Writes a string directly to a file inside the container at the specified absolute path.

        Args:
            contents: The string content to write.
            container_dest_path: The absolute file path inside the container where the string should be written.

        Returns:
            A dictionary with 'stdout', 'stderr', and 'returncode' of the operation.
        """
        try:
            if not os.path.isabs(container_dest_path):
                raise ValueError(f"container_dest_path must be an absolute path: '{container_dest_path}'")

            container_host_root = self.engine.container_root
            host_file_path = os.path.join(container_host_root, container_dest_path.lstrip(os.sep))

            # Ensure the parent directory exists
            os.makedirs(os.path.dirname(host_file_path), exist_ok=True)

            # Write contents directly to the file
            with open(host_file_path, "w", encoding="utf-8") as f:
                f.write(contents)

            Msg().out(f"Successfully wrote contents to '{host_file_path}' in container '{self.name}'")
            return 0
        except Exception as e:
            error_msg = f"Error writing to '{container_dest_path}' in udocker container '{self.name}': {e}"
            raise ContainerError(error_msg)

        
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

            container_host_root = self.engine.container_root

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

            Msg().out(f"Successfully copied '{host_src_path}' to '{final_parent_dir_on_host}' (inside udocker container '{self.name}')")
            return 0
        except Exception as e:
            error_msg = f"Error copying '{host_src_path}' to udocker container '{self.name}': {e}"
            raise ContainerError(error_msg)
