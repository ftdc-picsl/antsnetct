from . import ants_helpers
from . import bids_helpers
from . import preprocessing
from . import system_helpers

from .system_helpers import PipelineError

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import traceback

logger = logging.getLogger(__name__)

# Helps with CLI help formatting
class RawDefaultsHelpFormatter(
    argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    pass


def longitudinal_analysis():

    # Handle args with argparse
    parser = argparse.ArgumentParser(formatter_class=RawDefaultsHelpFormatter, add_help = False,
                                     description='''Longitudinal cortical thickness analysis with ANTsPyNet.

    The analysis level for longitudinal data is the subject. By default, all T1w images for a participant will be
    processed longitudinally. To process a specific subset of images for a participant, use the --participant-images option.

    ''')

    required_parser = parser.add_argument_group('required_parser arguments')
    required_parser.add_argument("--cross-sectional-dataset", help="BIDS derivatives dataset dir, containing the "
                                 "cross-sectional analysis", type=str, required=True)
    required_parser.add_argument("--output-dataset", help="Output BIDS dataset dir", type=str, required=True)
    required_parser.add_argument("--participant", "--participant-list", help="Participant to process", type=str)

    template_parser = parser.add_argument_group('Template arguments')
    template_parser.add_argument("--template-name", help="Template to use for registration, or 'none' to disable this step.",
                                 type=str, default='MNI152NLin2009cAsym')
    template_parser.add_argument("--template-res", help="Resolution of the template, eg '01', '02', etc. Note this is a "
                                 "templateflow index and not a physical spacing. If the selected template does not define "
                                 "multiple resolutions, this is ignored.", type=str, default='01')
    template_parser.add_argument("--template-cohort", help="Template cohort, only needed for templates that define multiple "
                                 "cohorts", type=str, default=None)
    template_parser.add_argument("--template-reg-quick", help="Do quick registration to the template", action='store_true')

    sst_parser = parser.add_argument_group('Single Subject Template arguments')
    sst_parser.add_argument("--sst-transform", help="SST transform, rigid or SyN", default='rigid')
    sst_parser.add_argument("--sst-iterations", help="Number of iterations for SST registration", type=int, default=4)
    sst_parser.add_argument("--sst-brain-extracted-weight", help="Relative weighting of brain-extracted images in SST "
                            "construction. 0.0 means only use whole-head images, 1.0 means only use brain-extracted images.",
                            type=float, default=0.5)
    sst_parser.add_argument("--sst-segmentation-method", help="Segmentation method to use on the SST. Either "
                            "'atropos' or 'antspynet'. If atropos, antspynet posteriors are used as priors.",
                            type=str, default='atropos')
    sst_parser.add_argument("--sst-atropos-prior-weight", help="Number of iterations of atropos-n4", type=int, default=3)

    segmentation_parser = parser.add_argument_group('Segmentation arguments for session processing')
    segmentation_parser.add_argument("--atropos-n4-iterations", help="Number of iterations of atropos-n4",
                                     type=int, default=3)
    segmentation_parser.add_argument("--atropos-prior-weight", help="Prior weight for Atropos", type=float, default=0.5)

    thickness_parser = parser.add_argument_group('Thickness arguments for session processing')
    thickness_parser.add_argument("--thickness-iterations", help="Number of iterations for cortical thickness estimation. "
                                  "Set to 0 to skip thickness calculation", type=int, default=45)

    template_parser = parser.add_argument_group('Group template arguments')
    template_parser.add_argument("--template-name", help="Template to use for registration, or 'none' to disable this step.",
                                 type=str, default='MNI152NLin2009cAsym')
    template_parser.add_argument("--template-res", help="Resolution of the template, eg '01', '02', etc. Note this is a "
                                 "templateflow index and not a physical spacing. If the selected template does not define "
                                 "multiple resolutions, this is ignored.", type=str, default='01')
    template_parser.add_argument("--template-cohort", help="Template cohort, only needed for templates that define multiple "
                                 "cohorts", type=str, default=None)
    template_parser.add_argument("--template-reg-quick", help="Do quick registration to the template", action='store_true')

    optional_parser = parser.add_argument_group('optional_parser arguments')
    optional_parser.add_argument("-h", "--help", action="help", help="show this help message and exit")
    optional_parser.add_argument("--participant-images", help="Text file containing a list of participant images to process "
                                 "relative to the cross-sectional dataset. If not provided, all images for the participant "
                                 "will be processed.", type=str, default=None)
    optional_parser.add_argument("--keep-workdir", help="Copy working directory to output, for debugging purposes. Either "
                                 "'never', 'on_error', or 'always'.", type=str, default='on_error')
    optional_parser.add_argument("--verbose", help="Verbose output", action='store_true')

    args = parser.parse_args()

    system_helpers.set_verbose(args.verbose)

    # If the only arg is "--longitudinal", print help and exit
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    if args.participant is None:
        raise ValueError('Participant must be defined')

    cx_ds = args.cross_sectional_dataset

    input_t1w_bids = ()

    if args.participant_images is not None:
        with open(args.participant_images, 'r') as f:
            t1w_relpaths = [line.strip() for line in f]
            for relpath in t1w_relpaths:
                if not os.path.exists(os.path.join(cx_ds, relpath)):
                    raise ValueError(f"Image {relpath} not found in cross-sectional dataset")
                input_t1w_bids.append(bids_helpers.BIDSImage(os.path.join(cx_ds, relpath)))
    else:
        input_t1w_bids = bids_helpers.find_participant_images(cx_ds, args.participant, 'anat', 'desc_biascorr-T1w')

    # Check that the output dataset exists, and if not, create
    # Update dataset_description.json if needed

    # Check cross-sectional and output datasets are not the same - will cause too much confusion
    if os.realpath(args.cross_sectional_dataset) == os.realpath(args.output_dataset):
        raise PipelineError("Cross-sectional and output datasets cannot be the same")

    with tempfile.TemporaryDirectory(suffix=f"antsnetct_longitudinal_{args.participant}.tmpdir") as working_dir:
        try:

            # Output structure

            # output_dataset/dataset_description.json
            # output_dataset/sub-123456
            # output_dataset/sub-123456/anat - contains SST and transforms to group template
            # output_dataset/sub-123456/ses-MR1/anat - contains session-specific files
            # output_dataset/sub-123456/ses-MR2/anat - contains session-specific files

            # SST input images - filenames, not BIDSImage objects
            # head images
            sst_input_t1w_denoised_normalized_images = list()
            # brain_images
            sst_input_t1w_denoised_normalized_brains = list()
            sst_input_t1w_masks = list()

            template_weights = [1.0 - args.sst_brain_extracted_weight, args.sst_brain_extracted_weight]

            for t1w in input_t1w_bids:
                # get the T1w image and mask, and reset their origins to the mask centroid

                input_t1w_denoised_image = t1w.get_path()

                seg = t1w.get_derivative_prefix() + '_seg-antsnetct_dseg.nii.gz'

                normalized = ants_helpers.normalize_intensity(input_t1w_denoised_image, seg, working_dir)

                input_t1w_mask = os.path.join(cx_ds, t1w.get_derivative_prefix() + 'desc-brain_mask.nii.gz')

                origin_fix = preprocessing.reset_origin_by_centroid(normalized, input_t1w_mask, working_dir)

                sst_input_t1w_denoised_normalized_images.append(origin_fix)

                mask_origin_fix = preprocessing.reset_origin_by_centroid(input_t1w_mask, input_t1w_mask, working_dir)

                sst_input_t1w_masks.append(mask_origin_fix)

                brain_origin_fix = ants_helpers.apply_mask(origin_fix, mask_origin_fix, working_dir)

                sst_input_t1w_denoised_normalized_brains.append(brain_origin_fix)

            # First round is rigid
            sst_output_rigid = ants_helpers.build_sst(
                [sst_input_t1w_denoised_normalized_images, sst_input_t1w_denoised_normalized_brains], working_dir,
                reg_transform='Rigid[0.1]', reg_iterations='20x20x40x0', template_iterations=3)

            # Second round is SyN
            sst_output = ants_helpers.build_sst(
                [sst_input_t1w_denoised_normalized_images, sst_input_t1w_denoised_normalized_brains], working_dir,
                reg_transform='SyN[0.2, 3, 0]', initial_templates=sst_output_rigid['template_images'],
                reg_iterations='20x30x40x10', reg_metric_weights=template_weights, template_iterations=5)

            # Warp all the masks to the SST space
            sst_input_t1w_masks = list()

            for idx, mask in enumerate(sst_input_t1w_masks):
                sst_input_t1w_masks.append(ants_helpers.apply_transforms(sst_output['template'], mask,
                                                                         sst_output['template_transforms'][idx],
                                                                         working_dir, interpolation = 'Linear'))

            # Combine the masks
            unified_mask_sst = ants_helpers.combine_masks(sst_input_t1w_masks, working_dir, thresh = 0.1)

            # Warp the unified mask back to the session spaces

            sst_input_t1w_denoised_unified_mask_brains = ()
            unified_masks_session_space = ()

            for idx, mask in enumerate(sst_input_t1w_masks):
                unified_masks_session_space.append(
                    ants_helpers.apply_transforms(sst_input_t1w_denoised_normalized_images[idx], sst_output['template'],
                                                    unified_mask_sst, sst_output['template_inverse_transforms'][idx],
                                                    working_dir, interpolation = 'GenericLabel')
                    )
                sst_input_t1w_denoised_unified_mask_brains.append(
                    ants_helpers.apply_mask(sst_input_t1w_denoised_normalized_images[idx],
                                            unified_masks_session_space[idx], working_dir)
                )

            # Write SST to output directory
            sst_bids = bids_helpers.image_to_bids(sst, args.output_dataset,
                                                  os.path.join('sub-' + args.participant, 'anat', 'sub-' + args.participant +
                                                               '_desc-SST_T1w.nii.gz'))


            # Segment the SST

            # align the SST to the group template

            # for each session
                # Warp priors to the session space
                # Segment the session
                # Compute thickness
                # Warp thickness to SST space
                # Compute jacobian in SST space if applicable
                # Warp thickness to group space

        except Exception as e:
            logger.error(f"Caught {type(e)} during processing of {str(t1w_bids)}")
            # Print stack trace
            traceback.print_exc()
            debug_workdir = os.path.join(output_dataset, t1w_bids.get_derivative_rel_path_prefix() + "_workdir")
            if args.keep_workdir.lower() != 'never':
                logger.info("Saving working directory to " + debug_workdir)
                shutil.copytree(working_dir, debug_workdir)
