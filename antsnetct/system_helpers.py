import os
import subprocess
import traceback

# Controls verbosity of subcommands
__verbose__ = False

# Sets the verbosity of subcommand output
def set_verbose(verbose):
    """ Set the verbosity level for system commands

    Args:
        verbose (bool): If True, enable verbose mode. The command to be run will be printed along with its terminal output.
    """
    global __verbose__
    __verbose__ = verbose


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
    global __verbose__

    if (__verbose__):
        print(f"--- Running {cmd[0]} ---")
        print(" ".join(cmd))

    result = subprocess.run(cmd, check = False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if (__verbose__):
        print("--- command stdout ---")
        print(result.stdout)
        print("--- command stderr ---")
        print(result.stderr)
        print(f"--- end {cmd[0]} ---")

    if result.returncode != 0:
        print(f"Error running command: {' '.join(cmd)}")
        traceback.print_stack()
        if not __verbose__: # print output if not already printed
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

