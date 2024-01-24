import antshelpers

import argparse
import os
import sys

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
    optional_parser.add_argument("--participant", "--participant-list", help="Participant to process, or a text file containing a "
                          "list of participants", type=str)
    optional_parser.add_argument("--no-reset-origin", help="Don't reset image and mask origin to mask centroid", action='store_true')
    optional_parser.add_argument("--keep-workdir", help="Copy working directory to output, for debugging purposes. Either 'never', 'on_error', or 'always'.", type=str, default='on_error')
    optional_parser.add_argument("--verbose", help="Verbose output", action='store_true')

    args = parser.parse_args()

    verbos = args.verbose

    # If the only arg is "--longitudinal", print help and exit
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)



def cross_sectional_analysis(args):

    parser = parser = argparse.ArgumentParser(formatter_class=RawDefaultsHelpFormatter,
                                    add_help = False,
                                    description='''Cortical thickness analysis with ANTsPyNet.

    Input can either be by participant or by session. By participant:

        '--participant 01'
        '--participant-list subjects.txt' where the text file contains a list of participants, one per line.

   All available sessions will be processed for each participant. To process selected sessions:

        '--session 01,MR1'
        '--sesion-list sessions.txt' where the text file contains a list of 'subject,session', one per line.

    Output is to a BIDS derivative dataset.

    If the output dataset does not exist, it will be created.

    ''')
    required_parser = parser.add_argument_group('Required arguments')
    required_parser.add_argument("--input-dataset", help="Input BIDS dataset dir, containing the source images", type=str,
                          required_parser=True)
    required_parser.add_argument("--output-dataset", help="Output BIDS dataset dir", type=str, required_parser=True)
    optional_parser = parser.add_argument_group('Optional arguments')
    optional_parser.add_argument("-h", "--help", action="help", help="show this help message and exit")
    optional_parser.add_argument("--participant", "--participant-list", help="Participant to process, or a text file "
                                 "containing a list of participants", type=str)
    optional_parser.add_argument("--session", "--session-list", help="Session to process, in the format 'participant,session "
                                 "or a text file containing a list of participants and sessions.", type=str)
    optional_parser.add_argument("--keep-workdir", help="Copy working directory to output, for debugging purposes. Either 'never', 'on_error', or 'always'.", type=str, default='on_error')
    optional_parser.add_argument("--verbose", help="Verbose output", action='store_true')

    atlas_parser = parser.add_argument_group('Atlas arguments')
    atlas_parser.add_argument("--atlas", help="Atlas to use for registration, or 'none' to disable this step", type=str, default='MNI152NLin2009cAsym')
    atlas_parser.add_argument("--atlas-res", help="Resolution of the atlas, eg 01, 02, etc. Note this is a templateflow index "
                              "and not a physical spacing", type=str, default='01')
    atlas_parser.add_argument("--atlas-reg-quick", help="Do quick registration to the atlas", action='store_true')
    atlas_parser.add_argument("--atlas-labels", help="Labels to transfer to session space", action='store_true')
    atlas_parser.add_argument("--atlas-cortical-labels", help="Cortical labels to transfer to session space",
                              action='store_true')

    segmentation_parser = parser.add_argument_group('Segmentation arguments')
    segmentation_parser.add_argument("--segmentation-method", help="Segmentation method to use. Either 'antspynet' or "
                                     "'atropos'. If atropos, we will use the output of antspynet segmentation as priors for "
                                     "segmentation and bias correction.", type=str, default='atropos')
    segmentation_parser.add_argument("--atropos-iterations", help="Number of iterations for atropos", type=int, default=5)
    segmentation_parser.add_argument("--atropos-prior-weight", help="Prior weight for Atropos", type=float, default=0.25)
    segmentation_parser.add_argument("--use-existing-masks", help="Use existing brain masks to define the domain of the final "
                                     "segmentation.", action='store_true')

    thickness_parser = parser.add_argument_group('Thickness arguments')
    thickness_parser.add_argument("--thickness-iterations", help="Number of iterations for cortical thickness estimation. "
                                  "Set to 0 to skip thickness calculation", type=int, default=45)

    args = parser.parse_args()

    verbose = args.verbose