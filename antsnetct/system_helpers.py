import subprocess
import traceback

# Controls verbosity of subcommands
__verbose__ = False

# Sets the verbosity of subcommand output
def set_verbose(verbose):
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
