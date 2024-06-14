from . import ants_helpers
from . import bids_helpers
from . import cross_sectional_pipeline
from . import system_helpers

from .system_helpers import PipelineError

import argparse
import json
import logging
import os
import re
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
    sst_parser.add_argument("--sst-iterations", help="Number of iterations for SST template building", type=int, default=4)
    sst_parser.add_argument("--sst-brain-extracted-weight", help="Relative weighting of brain-extracted images in SST "
                            "construction. 0.0 means only use whole-head images, 1.0 means only use brain-extracted images.",
                            type=float, default=0.5)
    template_parser.add_argument("--sst-reg-quick", help="Do quick registration to the SST", action='store_true')
    sst_parser.add_argument("--sst-segmentation-method", help="Segmentation method to use on the SST. Either "
                            "'atropos' or 'antspynet'. If atropos, antspynet posteriors are used as priors.",
                            type=str, default='atropos')
    sst_parser.add_argument("--sst-atropos-prior-weight", help="Prior weight in the SST segmenation. A higher value "
                            "gives more weight to the antsnet priors", type=int, default=0.5)

    segmentation_parser = parser.add_argument_group('Segmentation arguments for session processing')
    segmentation_parser.add_argument("--atropos-n4-iterations", help="Number of iterations of atropos-n4",
                                     type=int, default=3)
    segmentation_parser.add_argument("--atropos-prior-weight", help="Prior weight for Atropos in the session space",
                                     type=float, default=0.5)

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

    group_template = None
    group_template_brain_mask = None

    # setup templateflow, check template can be found
    if args.template_name.lower() != 'none':
        if not 'TEMPLATEFLOW_HOME' in os.environ or not os.path.exists(os.environ.get('TEMPLATEFLOW_HOME')):
            raise PipelineError(f"templateflow directory not found at " +
                                f"TEMPLATEFLOW_HOME={os.environ.get('TEMPLATEFLOW_HOME')}")

        group_template = bids_helpers.TemplateImage(args.template_name, suffix='T1w', description=None,
                                              resolution=args.template_res, cohort=args.template_cohort)

        group_template_brain_mask = bids_helpers.TemplateImage(args.template_name, suffix='mask', description='brain',
                                                         resolution=args.template_res, cohort=args.template_cohort)

    system_helpers.set_verbose(args.verbose)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    cx_dataset = args.cross_sectional_dataset

    output_dataset = args.output_dataset

    if (cx_dataset == output_dataset):
        raise ValueError('Input and output datasets cannot be the same')

    cx_dataset_description = None

    if os.path.exists(os.path.join(cx_dataset, 'dataset_description.json')):
        with open(os.path.join(cx_dataset, 'dataset_description.json'), 'r') as f:
            cx_dataset_description = json.load(f)
    else:
        raise ValueError('Cross-sectional dataset does not contain a dataset_description.json file')

    logger.info("Cross-sectional dataset path: " + cx_dataset)
    logger.info("Cross-sectional dataset name: " + cx_dataset_description['Name'])

    # Create the output dataset and add this container to the GeneratedBy, if needed
    bids_helpers.update_output_dataset(output_dataset, cx_dataset_description['Name'] + '_antsnetct')

    with open(os.path.join(output_dataset, 'dataset_description.json'), 'r') as f:
        output_dataset_description = json.load(f)

    logger.info("Output dataset path: " + output_dataset)
    logger.info("Output dataset name: " + output_dataset_description['Name'])

    # preprocessed images to be processed longitudinally
    cx_preproc_t1w_bids = list()

    # denoised, bias-corrected images to be used for SST construction and registration
    cx_biascorr_t1w_bids = list()

    # brain masks for the preprocessed images
    cx_brain_mask_bids = list()

    if args.participant_images is not None:
        with open(args.participant_images, 'r') as f:
            t1w_relpaths = [line.strip() for line in f]
            for relpath in t1w_relpaths:
                if not os.path.exists(os.path.join(cx_ds, relpath)):
                    raise ValueError(f"Image {relpath} not found in cross-sectional dataset")
                cx_preproc_t1w_bids.append(bids_helpers.BIDSImage(cx_ds, relpath))
                cx_biascorr_t1w_bids.append(bids_helpers.BIDSImage(cx_ds, relpath.replace('desc-preproc_T1w',
                                                                                          'desc-biascorr_T1w')))
                cx_brain_mask_bids.append(bids_helpers.BIDSImage(cx_ds, relpath.replace('desc-preproc_T1w', 'desc-brain_mask')))
    else:
        cx_preproc_t1w_bids = bids_helpers.find_participant_images(cx_ds, args.participant, 'anat', 'desc-preproc_T1w')
        for idx in range(len(cx_preproc_t1w_bids)):
             cx_biascorr_t1w_bids.append(bids_helpers.BIDSImage(cx_ds, relpath.replace('desc-preproc_T1w',
                                                                                       'desc-biascorr_T1w')))
             cx_brain_mask_bids.append(bids_helpers.BIDSImage(cx_ds, relpath.replace('desc-preproc_T1w', 'desc-brain_mask')))

    num_sessions = len(cx_preproc_t1w_bids)

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

            # Write preprocessed T1w images to output dataset - this creates output session directories
            long_preproc_t1w_bids = list()

            for idx in range(num_sessions):
                long_preproc_t1w_bids.append(bids_helpers.copy_image(cx_preproc_t1w_bids[idx], args.output_dataset))

            sst_reg_metric = 'CC'
            sst_reg_metric_params=[3]
            sst_reg_iterations = '20x20x40x10'

            if args.sst_reg_quick:
                sst_reg_metric = 'MI'
                sst_reg_metric_params = [32]
                sst_reg_iterations = '20x30x40x0'

            # SST construction
            sst_preproc_input = get_preproc_sst_input(cx_biascorr_t1w_bids, working_dir)

            template_weights = [1.0 - args.sst_brain_extracted_weight, args.sst_brain_extracted_weight]

            sst_output_rigid = ants_helpers.build_sst(sst_preproc_input, working_dir, initial_templates=None,
                                                      reg_transform='Rigid[ 0.1 ]', reg_iterations=sst_reg_iterations,
                                                      reg_metric=sst_reg_metric, reg_metric_params=sst_reg_metric_params,
                                                      reg_metric_weights=template_weights, template_iterations=3)

            sst_output = ants_helpers.build_sst(sst_preproc_input, working_dir, initial_templates=sst_output_rigid,
                                                reg_transform='SyN[ 0.2, 3, 0.5 ]', reg_iterations=sst_reg_iterations,
                                                reg_metric=sst_reg_metric, reg_metric_params=sst_reg_metric_params,
                                                reg_metric_weights=template_weights, template_iterations=3)

            session_sst_transforms = list()

            # Register all subjects to SST
            for idx in range(num_sessions):
                moving_head = cx_biascorr_t1w_bids[idx].get_path()
                moving_brain = ants_helpers.apply_mask(moving_head, cx_brain_mask_bids[idx].get_path(), working_dir)
                moving = [moving_head, moving_brain]
                session_sst_transforms.append(
                    ants_helpers.multivariate_pairwise_registration(sst_output['template_images'], moving, working_dir,
                                                      transform='SyN[0.2, 3, 0.5]', iterations=sst_reg_iterations,
                                                      metric=sst_reg_metric, metric_params=sst_reg_metric_params,
                                                      be_metric_weight=template_weights, apply_transforms=False)
                )

            # Write SST to output dataset under sub-<label>/anat
            sst_bids = bids_helpers.image_to_bids(sst_output['template_images'][0], args.output_dataset,
                                                  os.path.join('sub-' + args.participant, 'anat', 'sub-' + args.participant +
                                                               '_desc-sst_T1w.nii.gz'))

            # Masks in SST space
            sst_t1w_masks = list()

            for idx in range(num_sessions):
                sst_t1w_masks.append(
                    ants_helpers.apply_transforms(sst_output['template_images'][0], cx_brain_mask_bids[idx].get_path(),
                                                  session_sst_transforms[idx]['forward_transform'], working_dir,
                                                  interpolation='GenericLabel')
                )

            # Combine the masks
            unified_mask_sst = ants_helpers.combine_masks(sst_t1w_masks, working_dir, thresh = 0.1)

            # Save this in the output dataset
            unified_mask_sst_bids = bids_helpers.image_to_bids(unified_mask_sst, args.output_dataset,
                                                               os.path.join('sub-' + args.participant, 'anat',
                                                                            'sub-' + args.participant +
                                                                            '_desc-sstbrain_mask.nii.gz'))

            sst_brain_bids = bids_helpers.image_to_bids(
                ants_helpers.apply_mask(sst_output['template_images'][0], unified_mask_sst, working_dir),
                args.output_dataset, os.path.join('sub-' + args.participant, 'anat', 'sub-' + args.participant +
                                                                    '_desc-sstbrain_T1w.nii.gz'))

            # Warp the unified mask back to the session spaces
            long_brain_mask_bids = list()

            for idx in range(num_sessions):
                session_mask = ants_helpers.apply_transforms(cx_biascorr_t1w_bids[idx].get_path(), unified_mask_sst,
                                              session_sst_transforms[idx]['inverse_transform'], working_dir,
                                              interpolation='GenericLabel')
                long_brain_mask_bids.append(
                    bids_helpers.image_to_bids(session_mask, args.output_dataset,
                                               long_preproc_t1w_bids.get_derivative_rel_path_prefix() +
                                               '_desc-brain_mask.nii.gz')
                    )

            # Segment the SST
            sst_seg = cross_sectional_pipeline.segment_and_bias_correct(sst_bids, unified_mask_sst_bids, working_dir,
                             segmentation_method=args.sst_segmentation_method, atropos_n4_iterations=1,
                             atropos_prior_weight=0.25, denoise=True, n4_spline_spacing=180,
                             n4_convergence='[ 0,1e-7 ]', n4_shrink_factor=3)

            # align the SST to the group template
            sst_to_group_template_reg = cross_sectional_pipeline.template_brain_registration(
                group_template, group_template_brain_mask, sst_brain_bids, args.template_reg_quick, working_dir)

            # for each session
            for idx in range(num_sessions):
                t1w_bids = cx_preproc_t1w_bids[idx]
                brain_mask_bids = cx_brain_mask_bids[idx]
                # Warp priors to the session space
                t1w_priors = list()

                for idx in range(6):
                    t1w_priors.append(
                        ants_helpers.apply_transforms(t1w_bids.get_path(), sst_seg['segmentation_posteriors'][idx],
                                                      session_sst_transforms[idx]['inverse_transform'], working_dir)
                    )
                # Segment the session
                seg_n4 = cross_sectional_pipeline.segment_and_bias_correct(t1w_bids, brain_mask_bids, working_dir, denoise=True,
                                                  segmentation_priors=t1w_priors, segmentation_method='atropos',
                                                  atropos_n4_iterations=args.atropos_n4_iterations,
                                                  atropos_prior_weight=args.atropos_prior_weight)
                # Compute thickness
                thickness = cross_sectional_pipeline.compute_cortical_thickness(seg_n4, working_dir,
                                                                                iterations=args.thickness_iterations)
                # Warp thickness to SST space
                # Compute jacobian in SST space if applicable
                # Warp thickness to group space
        except Exception as e:
            logger.error(f"Caught {type(e)} during processing of {args.participant}")
            # Print stack trace
            traceback.print_exc()
            debug_workdir = os.path.join(args.output_dataset, f"sub-{args.participant}", f"sub-{args.participant}_workdir")
            if args.keep_workdir.lower() != 'never':
                logger.info("Saving working directory to " + debug_workdir)
                shutil.copytree(working_dir, debug_workdir)


def get_preproc_sst_input(cx_biascorr_t1w_bids, work_dir):
    """Preprocess the input images for SST construction.

    The preprocessing steps are:
        1. Normalize intensity such that the mean WM intensity is the same across images
        2. Reset the origin of the images to the centroid of the brain mask. This prevents unwanted shifts
           in the SST position.

    Parameters:
    ----------
    cx_biascorr_t1w_bids (list of BIDSImage):
        List of BIDSImage objects for the input images. These should be the bias-corrected T1w images.
    work_dir (str):
        Working directory

    Returns:
    -------
    list:
        List containing a list of head images and brain images, for SST construction.
    """
    sst_input_t1w_denoised_normalized_heads = list()
    sst_input_t1w_denoised_normalized_brains = list()

    for t1w in cx_biascorr_t1w_bids:
        # get the T1w image and mask, and reset their origins to the mask centroid
        input_t1w_denoised_image = t1w.get_path()

        seg = t1w.get_derivative_path_prefix() + '_seg-antsnetct_dseg.nii.gz'

        normalized = ants_helpers.normalize_intensity(input_t1w_denoised_image, seg, work_dir)

        input_t1w_mask = t1w.get_derivative_path_prefix() + 'desc-brain_mask.nii.gz'

        origin_fix = preprocessing.reset_origin_by_centroid(normalized, input_t1w_mask, work_dir)

        sst_input_t1w_denoised_normalized_heads.append(origin_fix)

        mask_origin_fix = preprocessing.reset_origin_by_centroid(input_t1w_mask, input_t1w_mask, work_dir)

        brain_origin_fix = ants_helpers.apply_mask(origin_fix, mask_origin_fix, work_dir)

        sst_input_t1w_denoised_normalized_brains.append(brain_origin_fix)

    sst_input_combined = [sst_input_t1w_denoised_normalized_heads, sst_input_t1w_denoised_normalized_brains]

    return sst_input_combined




def build_sst(sst_input, work_dir, initial_templates=None, template_iterations=4, reg_transform='Rigid[0.1]', reg_metric='CC',
              reg_metric_params=[3], be_metric_weight=0.5, reg_iterations='40x40x40x0', reg_shrink_factors='4x3x2x1', reg_smoothing_sigmas='3x2x1x0vox'):
    """Build the SST for a set of cross-sectional BIDS images.

    Parameters:
    ----------
    cx_biascorr_t1w_bids (list of BIDSImage):
        List of BIDSImage objects for the input images. These should be cross-sectionally processed T1w images.
    work_dir (str):
        Working directory
    initial_templates (list of str):
        Initial templates to use for the SST. If None, the first image in cx_t1w_bids is the initial template.
    template_iterations (int):
        Number of iterations for template building.
    reg_iterations (str):
        Iterations for pairwise registration in the template building process.
    be_metric_weight (float):
        Relative weight of brain-extracted images in the SST construction. 0.0 means only use whole-head images, 1.0 means
        only use brain-extracted images. Default is 0.5.
    reg_shrink_factors (str):
        Shrink factors for registration.
    reg_smoothing_sigmas (str):
        Smoothing sigmas for registration.
    reg_transform (str):
        Registration transform to use.

    Returns:
    -------
    sst_output (dict):
        Dictionary containing the SST images. The first template is the whole-head T1w, the second is brain-extracted.
    """
    template_weights = [1.0 - be_metric_weight, be_metric_weight]

    sst_output = ants_helpers.build_sst(sst_input, work_dir, initial_templates=initial_templates,
                                        reg_transform=reg_transform, reg_iterations=reg_iterations, reg_metric=reg_metric,
                                        reg_metric_params=reg_metric_params, reg_metric_weights=template_weights,
                                        reg_shrink_factors=reg_shrink_factors, reg_smoothing_sigmas=reg_smoothing_sigmas,
                                        template_iterations=template_iterations)

    return sst_output

