import logging
import os
import shutil
import subprocess
import tempfile

import tensorflow as tf


# Controls verbosity of subcommands
_verbose = False

# Sets the verbosity of subcommand output
def set_verbose(verbose):
    """ Set the verbosity level for system commands

    Parameters
    ----------
    verbose : bool
        If True, enable verbose mode. The command to be run will be printed along with its terminal output.
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

    Parameters
    ----------
    cmd : list
        A list of command line arguments, as used in subprocess.run

    Raises
    ------
    PipelineError :
        Raised if the command returns a non-zero exit code. The full command stdout and stderr are also printed, unless verbose
        is set (in which case they are already printed).

    Returns
    -------
    dict : with keys 'cmd_str', 'stderr', 'stdout'.
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

    if result.returncode != 0:
        if not _verbose: # print output if not already printed
            print("--- command stdout ---")
            print(result.stdout)
            print("--- command stderr ---")
            print(result.stderr)

        print(f"--- end {cmd[0]} (exited with error) ---")
        raise PipelineError(f"Error running command: {' '.join(cmd)}")

    if (_verbose):
        print(f"--- end {cmd[0]} ---")

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
    src : str
        The source file.
    dst : str
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


def get_temp_file(work_dir, prefix=None, suffix=None):
    """Get a unique tempfile in work_dir

    The file will not be automatically deleted. It should be used to ensure unique file names
    within working directories.

    Temp files are named '{prefix}_{unique_id}{suffix}'. This is done to preserve readability of temp files for debugging
    purposes.

    You can also use this function to define a temporary file prefix, eg

    tmp_file_prefix = get_temp_file(work_dir, prefix='my_function_tmp')

    will create the file '{tmp_file_prefix}', allowing processes to write files starting with tmp_file_prefix without
    conflicting with other processes.

    Parameters
    ----------
    work_dir : str
        The directory in which to create the temporary file. This must exist.
    prefix : str, optional
        The prefix of the temporary file.
    suffix : str, optional
        The suffix of the temporary file. The suffix should start with "." if it denotes an extension.

    Returns
    -------
    str : The path to the temporary file, f"{work_dir}/{prefix}_{unique_id}{suffix}".
    """
    if not os.path.exists(work_dir):
        raise FileNotFoundError(f"Working directory {work_dir} does not exist.")

    if prefix is None:
        formatted_prefix = '_'
    else:
        formatted_prefix = prefix + '_'

    fd, tmp_name = tempfile.mkstemp(prefix=formatted_prefix, suffix=suffix, dir=work_dir)
    # don't leave the file open
    os.close(fd)

    return tmp_name


def get_temp_dir(work_dir, prefix=None):
    """Get a unique temp dir under work_dir

    Temp dirs are named '{prefix}_{unique_id}'. This is done to preserve readability of temp files for debugging purposes.

    Parameters
    ----------
    work_dir : str
        The directory in which to create the temporary directory. This must exist.
    prefix : str, optional
        The prefix of the temporary file.

    Returns
    -------
    str : The path to the temporary file, "{work_dir}/{prefix}_{unique_id}".
    """
    if not os.path.exists(work_dir):
        raise FileNotFoundError(f"Working directory {work_dir} does not exist.")

    if prefix is None:
        formatted_prefix = '_'
    else:
        formatted_prefix = prefix + '_'

    tmp_dir = tempfile.mkdtemp(prefix=formatted_prefix, dir=work_dir)
    return tmp_dir


def set_num_threads(num_threads=0):
    """Set the number of threads to use.

    The number of threads to use in ANTs commands is set by the environment variable ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS.

    The number of threads in ANTsPyNet is set to 1 because extra threads use much more memory, and these commands are already
    relatively fast.

    The number of threads can be set explicitly, or automatically. If num_threads is 0, the number of threads is set
    automatically to min(system_cores, 8).

    Parameters:
    ----------
    num_threads : int, optional
        The number of threads to use. If 0, the number of threads is set automatically.
    """
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    if num_threads > 0:
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(num_threads)
    else:
        num_threads = min(os.cpu_count(), 8)
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(num_threads)


def get_num_threads():
    """Get the number of threads currently set in the environment.

    To avoid using all cores and degrading system performance, this function will raise an error if the environment variable
    ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS is not set, or is not a positive integer.

    Returns:
    -------
    int : The value of ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS in the current environment.

    Raises:
    ------
    KeyError : If ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS is not set in the environment.
    RuntimeError : If ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS is not a positive integer.
    """
    if 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS' in os.environ:
        try:
            num_threads = int(os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'])
            return num_threads
        except:
            raise RuntimeError('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS must be a positive integer, not '
                               f"'{os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']}'")
    else:
        raise KeyError("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS is not set in the environment.")