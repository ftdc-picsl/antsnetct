import antshelpers

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback


# Get a dictionary for the GeneratedBy field for the BIDS dataset_description.json
# This is used to record the software used to generate the dataset
# The environment variables DOCKER_IMAGE_TAG and DOCKER_IMAGE_VERSION are used if set
#
# Container type is assumed to be "docker" unless the variable SINGULARITY_CONTAINER
# is defined
def get_generated_by(existing_generated_by=None):

    import copy

    generated_by = []

    if existing_generated_by is not None:
        generated_by = copy.deepcopy(existing_generated_by)
        for gb in existing_generated_by:
            if gb['Name'] == 'T1wPreprocessing' and gb['Container']['Tag'] == os.environ.get('DOCKER_IMAGE_TAG'):
                # Don't overwrite existing generated_by if it's already set to this pipeline
                return generated_by

    container_type = 'docker'

    if 'SINGULARITY_CONTAINER' in os.environ:
        container_type = 'singularity'

    gen_dict = {'Name': 'T1wPreprocessing',
                'Version': os.environ.get('DOCKER_IMAGE_VERSION', 'unknown'),
                'CodeURL': os.environ.get('GIT_REMOTE', 'unknown'),
                'Container': {'Type': container_type, 'Tag': os.environ.get('DOCKER_IMAGE_TAG', 'unknown')}
                }

    generated_by.append(gen_dict)
    return generated_by

# Helps with CLI help formatting
class RawDefaultsHelpFormatter(
    argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    pass


def longitudinal_analysis(args):

    # Handle args with argparse
    parser = parser = argparse.ArgumentParser(formatter_class=RawDefaultsHelpFormatter,
                                    add_help = False,
                                    description='''Longitudinal cortical thickness analysis with ANTsPyNet.

    Input can either be by participant or by session. By participant:

        '--participant 01'
        '--participant-list subjects.txt' where the text file contains a list of participants, one per line.

    The analysis level for longitudinal data is the subject.

    ''')

    required_parser = parser.add_argument_group('required_parser arguments')
    required_parser.add_argument("--input-dataset", help="Input BIDS dataset dir, containing the source images", type=str,
                          required_parser=True)
    required_parser.add_argument("--cross-sectional-dataset", help="BIDS derivatives dataset dir, containing the "
                                 "cross-sectional analysis", type=str, required_parser=True)
    required_parser.add_argument("--output-dataset", help="Output BIDS dataset dir", type=str, required_parser=True)
    optional_parser = parser.add_argument_group('optional_parser arguments')
    optional_parser.add_argument("-h", "--help", action="help", help="show this help message and exit")
    optional_parser.add_argument("--device", help="GPU device to use, or 'cpu' to use CPU. Note CPU mode is many times slower",
                          type=str, default='0')
    optional_parser.add_argument("--participant", "--participant-list", help="Participant to process, or a text file "
                                 "containing a list of participants", type=str)
    optional_parser.add_argument("--no-reset-origin", help="Don't reset image and mask origin to mask centroid",
                                 action='store_true')
    optional_parser.add_argument("--keep-workdir", help="Copy working directory to output, for debugging purposes. Either "
                                 "'never', 'on_error', or 'always'.", type=str, default='on_error')
    optional_parser.add_argument("--verbose", help="Verbose output", action='store_true')

    args = parser.parse_args()

    verbos = args.verbose

    # If the only arg is "--longitudinal", print help and exit
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

