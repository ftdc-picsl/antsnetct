import os
import shutil
import subprocess
import traceback

# Controls verbosity of subcommands
_verbose = False

# Sets the verbosity of subcommand output
def set_verbose(verbose):
    """ Set the verbosity level for system commands

    Args:
        verbose (bool): If True, enable verbose mode. The command to be run will be printed along with its terminal output.
    """
    global _verbose
    _verbose = verbose


# Catches pipeline errors from helper functions
class PipelineError(Exception):
    """Exception raised when helper functions encounter an error"""
    pass

# Uses subprocess.run to run a command, and prints the command and output if verbose is set
#
# Example:
#   result = run_command(['c3d', my_image, '-swapdim', output_orientation, '-o', reoriented_image])
#
# Input: a list of command line arguments
#
# Returns a dictionary with keys 'cmd_str', 'stderr', 'stdout'
#
# Raises PipelineError if the command returns a non-zero exit code
#
def run_command(cmd):
    """Runs a command and returns the output

    Verbosity is controlled by the set_verbose(verbose) function.

    Args:
        cmd (list): A list of command line arguments, as used in subprocess.run

    Raises:
        PipelineError: If the command returns a non-zero exit code. If an error is raised, a stack trace is printed. The full
        command stdout and stderr are also printed, unless verbose is set (in which case they are already printed).

    Returns:
        dict: a dictionary with keys 'cmd_str', 'stderr', 'stdout'.
    """
    # Just to be clear we use the global var set by the main function
    global _verbose

    if (_verbose):
        print(f"--- Running {cmd[0]} ---")
        print(" ".join(cmd))

    result = subprocess.run(cmd, check = False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if (_verbose):
        print("--- command stdout ---")
        print(result.stdout)
        print("--- command stderr ---")
        print(result.stderr)
        print(f"--- end {cmd[0]} ---")

    if result.returncode != 0:
        print(f"Error running command: {' '.join(cmd)}")
        traceback.print_stack()
        if not _verbose: # print output if not already printed
            print('command stdout:\n' + result.stdout)
            print('command stderr:\n' + result.stderr)
            raise PipelineError(f"Error running command: {' '.join(cmd)}")

    return { 'cmd_str': ' '.join(cmd), 'stderr': result.stderr, 'stdout': result.stdout }

def get_nifti_file_prefix(image_file):
    """Get the prefix of a NIFTI image, without directory or extensions

    Will remove .nii or .nii.gz extensions

    Parameters:
    ----------
    image_file (str):
        The image file.

    Returns:
    -------
    str: The prefix of the image file

    Example:
    --------
    get_image_file_prefix('/path/to/my_image.nii.gz') returns 'my_image'

    """

    if image_file.endswith('.nii.gz'):
        return os.path.basename(image_file)[:-7]
    elif image_file.endswith('.nii'):
        return os.path.basename(image_file)[:-4]
    else:
        raise ValueError(f"Image file {image_file} does not end in .nii or .nii.gz")


def copy_file(source, destination):
    """Copy a file from src to dst. User write permission will be added to the destination file if not already present.

    The copy is done by shutil and will work with regular files or symbolic links to regular files.

    User write permission will be added to the destination file if not already present. This prevents errors from being
    unable to edit or overwrite the destination file if it was copied from a read-only directory.

    Parameters:
    ----------
    src (str):
        The source file.
    dst (str):
        The destination file.

    Returns:
    -------
    None

    Example:
    --------
    copy_file('/path/to/my_image.nii.gz', '/path/to/destination/my_image.nii.gz')
    """
    if not os.path.exists(source):
        raise FileNotFoundError(f"The source file {source} does not exist.")
    if not (os.path.isfile(source) or (os.path.islink(source) and os.path.isfile(os.readlink(source)))):
        raise Exception(f"The source path {source} is not a file.")

    try:
        shutil.copy(source, destination)
    except Exception as e:
        raise Exception(f"Failed to copy file {source} to {destination}: {e}")

    if not os.path.exists(destination):
        raise Exception(f"Copy failed, file {destination} does not exist.")

    current_mode = os.stat(destination).st_mode
    if not current_mode & 0o200:  # Check if user write bit is not set
        new_mode = current_mode | 0o200  # Add user write permission
        try:
            os.chmod(destination, new_mode)
        except Exception as e:
            raise Exception(f"Failed to set user write permission on file {destination}: {e}")


