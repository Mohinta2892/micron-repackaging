import configargparse
import os
import logging
from subprocess import check_call

logger = logging.getLogger(__name__)

p = configargparse.ArgParser(default_config_files=['~/.pysub'])

p.add('-p', required=True,
      help="The command to run" +
      " e.g. ``python train.py``.")

p.add('-w', '--working_directory', required=False,
      help="The working directory for <command>," +
           "defaults to current directory",
      default=".")

p.add('-s', '--singularity', required=False,
      help="Optional singularity image to use to" +
           "execute <command>. The singularity" +
           "container will have some common local" +
           "directories mounted. See ``~/.deploy``.",
      default="")

p.add('-d', '--mount_dirs', required=False,
      help="Directories to mount in container.",
      default="")


def run_singularity(command,
                    singularity_image,
                    working_dir=".",
                    mount_dirs=[],
                    execute=False,
                    expand=True):

    if not singularity_image:
        raise ValueError("No singularity image provided.")

    if not os.path.exists(singularity_image):
        raise ValueError("Singularity image {}".format(singularity_image) +
                         " does not exist.")

    run_command = ['singularity exec']
    run_command += ['-B {}'.format(mount) for mount in mount_dirs
                    if mount != "None"]
    run_command += ['-W {}'.format(working_dir),
                    '--nv',
                    singularity_image,
                    command]

    os.environ["NV_GPU"] = str(os.environ.get("CUDA_VISIBLE_DEVICES"))

    if not execute:
        if not expand:
            return run_command
        else:
            return ' '.join(run_command)
    else:
        run_command = ' '.join(run_command)
        check_call(run_command,
                   shell=True)


if __name__ == "__main__":
    options = p.parse_known_args()[0]

    command = options.p
    working_dir = options.working_directory
    singularity_image = options.singularity
    mount_dirs = list(options.mount_dirs.split(","))
    execute = True

    run_singularity(command,
                    singularity_image,
                    working_dir,
                    mount_dirs,
                    execute)
