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

# Controls verbosity of subcommands
__verbose__ = False


# Helps with CLI help formatting
class RawDefaultsHelpFormatter(
    argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    pass

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

    The analysis level for cross-sectional data is the session. Output is to a BIDS derivative dataset.

    Pre-computed brain masks and segmentations can optionally be used for processing. These should be in a BIDS derivative
    dataset with sidecars describing the input T1w image as the source image.

    --- Brain masking ---

    The script will check for a brain mask from the brain mask dataset, if defined. If not, a brain mask may be found in the
    input dataset. If no brain mask is available, one will be generated using anyspynet.


    --- Segmentation ---

    The script will check for a segmentation from the segmentation dataset, if defined. If a segmentation is available,
    it can either be used directly or as priors for iterative segmentation and bias correction with N4 and Atropos.

    If no segmentation is available, priors are generated with anyspynet and used for segmentation and bias correction.


    --- Atlas registration ---

    The bias-corrected image is registered to the atlas. The transforms are saved along with template information. The
    template should follow templateflow conventions, and be named 'tpl-atlasName[_otherEntities]_desc-brain_T1w.nii.gz`. The
    template should also define a sidecar file `template_description.json` that provides an "Identifier" for the template.
    If there are multiple template resolutions, there should be a key "res", detailing the shape and spacing of the template.
    See templateflow's "tpl-MNI152NLin2009cAsym" for an example.


    --- Processing steps ---

    1. Brain masking. If a brain mask is available, it is used. Otherwise, one is generated with anyspynet.

    2. Segmentation and bias correction. If a segmentation is available, it is used according to the user instructions.
    Otherwise, one is generated with anyspynet. Segmentation may be refined with Atropos. The T1w image is simultaneously
    segmented and bias-corrected.

    3. Cortical thickness estimation. If the number of iterations is greater than 0, cortical thickness is estimated.

    4. Atlas registration. If an atlas is defined, the T1w image is registered to the atlas.

    5. Atlas label transfer. If an atlas is defined, labels are transferred to the T1w image.

    '''

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
    optional_parser.add_argument("--keep-workdir", help="Copy working directory to output, for debugging purposes. Either "
                                 "'never', 'on_error', or 'always'.", type=str, default='on_error')
    optional_parser.add_argument("--verbose", help="Verbose output", action='store_true')

    atlas_parser = parser.add_argument_group('Atlas arguments')
    atlas_parser.add_argument("--atlas", help="Atlas to use for registration, or 'none' to disable this step", type=str,
                              default='MNI152NLin2009cAsym')
    atlas_parser.add_argument("--atlas-res", help="Resolution of the atlas, eg 01, 02, etc. Note this is a templateflow index "
                              "and not a physical spacing", type=str, default='01')
    atlas_parser.add_argument("--atlas-reg-quick", help="Do quick registration to the atlas", action='store_true')

    brain_mask_parser = parser.add_argument_group('Brain mask arguments')
    brain_mask_parser.add_argument("--brain-mask-dataset", help="Dataset containing brain masks. Masks from here will be used "
                                   "in preference to those in the input dataset.", type=str, default=None)
    brain_mask_parser.add_argument("--brain-mask-method", help="Brain masking method to use with antspynet. Only used if no "
                                   "pre-existing mask is found. Options are 't1', 't1nobrainer', 't1combined'",
                                   type=str, default='t1')
    segmentation_parser = parser.add_argument_group('Segmentation arguments')
    segmentation_parser.add_argument("--segmentation-method", help="Segmentation method to use. Either 'atropos' or "
                                     "'none'. If atropos, probseg images from the segmentation dataset, if defined, will be "
                                     "used as priors for segmentation and bias correction. If no segmentation dataset is "
                                     "provided, a segmentation will be generated by antspynet.", type=str, default='atropos')
    segmentation_parser.add_argument("--segmentation-dataset", help="Dataset containing segmentations. This dataset can be "
                                     "used for priors, or as a replacement for the built-in segmentation routines.",
                                     type=str, default=None)
    segmentation_parser.add_argument("--atropos-iterations", help="Number of iterations for atropos", type=int, default=5)
    segmentation_parser.add_argument("--atropos-prior-weight", help="Prior weight for Atropos", type=float, default=0.25)

    thickness_parser = parser.add_argument_group('Thickness arguments')
    thickness_parser.add_argument("--thickness-iterations", help="Number of iterations for cortical thickness estimation. "
                                  "Set to 0 to skip thickness calculation", type=int, default=45)

    args = parser.parse_args()

    verbose = args.verbose

    # If the only arg is "--cross-sectional", print help and exit
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # Check for valid inputs
'