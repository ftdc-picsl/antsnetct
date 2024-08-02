from . import ants_helpers
from . import bids_helpers
from . import cross_sectional_pipeline
from . import preprocessing
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
    required_parser.add_argument("--participant", help="Participant to process", type=str)

    subject_parser = parser.add_argument_group('Subject arguments')
    subject_parser.add_argument("--participant-images", help="Text file containing a list of participant images to process "
                                 "relative to the cross-sectional dataset. If not provided, all images for the participant "
                                 "will be processed.", type=str, default=None)

    sst_parser = parser.add_argument_group('Single Subject Template arguments')
    sst_parser.add_argument("--sst-transform", help="SST transform, rigid affine or syn", type=str, default='rigid')
    sst_parser.add_argument("--sst-iterations", help="Number of iterations for SST template building", type=int, default=4)
    sst_parser.add_argument("--sst-brain-extracted-weight", help="Relative weighting of brain-extracted images in SST "
                            "construction. 0.0 means only use whole-head images, 1.0 means only use brain-extracted images.",
                            type=float, default=0.5)
    sst_parser.add_argument("--sst-reg-quick", help="Do quick registration to the SST", action='store_true')
    sst_parser.add_argument("--sst-segmentation-method", help="Segmentation method to use on the SST. Either "
                            "'antspynet_atropos' (antspynet priors, then atropos) or 'antspynet' (no atropos) or "
                            "'cx_atropos' (average cross-sectional priors, then atropos), or 'cx' "
                            "(average cross-sectional priors).", type=str, choices =
                                ['antspynet_atropos', 'antspynet', 'cx_atropos', 'cx'], default='antspynet_atropos')
    sst_parser.add_argument("--sst-atropos-prior-weight", help="Prior weight in the SST segmenation. A higher value "
                            "gives more weight to the priors", type=float, default=0.25)

    segmentation_parser = parser.add_argument_group('Segmentation arguments for session processing')
    segmentation_parser.add_argument("--atropos-n4-iterations", help="Number of iterations of atropos-n4",
                                     type=int, default=3)
    segmentation_parser.add_argument("--atropos-prior-weight", help="Prior weight for Atropos in the session space",
                                     type=float, default=0.5)
    segmentation_parser.add_argument("--prior-smoothing-sigma", help="Sigma for smoothing the priors before session "
                                     "segmentation, in voxels. Experimental", type=float, default=0)
    segmentation_parser.add_argument("--prior-csf-gamma", help="Gamma value for CSF prior. Experimental",
                                     type=float, default=0)

    thickness_parser = parser.add_argument_group('Thickness arguments for session processing')
    thickness_parser.add_argument("--thickness-iterations", help="Number of iterations for cortical thickness estimation.",
                                  type=int, default=45)

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
    optional_parser.add_argument("--keep-workdir", help="Copy working directory to output, for debugging purposes. Either "
                                 "'never', 'on_error', or 'always'.", type=str, default='on_error')
    optional_parser.add_argument("--verbose", help="Verbose output", action='store_true')

    args = parser.parse_args()

    logger.info("Parsed args: " + str(args))

    system_helpers.set_verbose(args.verbose)

    # If no args, print help and exit
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

    if os.path.exists(os.path.join(output_dataset, f"sub-{args.participant}")):
        raise ValueError(f"Output exists for participant {args.participant}")

    cx_dataset_description = None

    if os.path.exists(os.path.join(cx_dataset, 'dataset_description.json')):
        with open(os.path.join(cx_dataset, 'dataset_description.json'), 'r') as f:
            cx_dataset_description = json.load(f)
    else:
        raise ValueError('Cross-sectional dataset does not contain a dataset_description.json file')

    logger.info("Cross-sectional dataset path: " + cx_dataset)
    logger.info("Cross-sectional dataset name: " + cx_dataset_description['Name'])

    # Create the output dataset and add this container to the GeneratedBy, if needed
    bids_helpers.update_output_dataset(output_dataset, cx_dataset_description['Name'] + '_longitudinal')

    with open(os.path.join(output_dataset, 'dataset_description.json'), 'r') as f:
        output_dataset_description = json.load(f)

    logger.info("Output dataset path: " + output_dataset)
    logger.info("Output dataset name: " + output_dataset_description['Name'])

    # Note - it's important that the session images be in a consistent order

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
                if not os.path.exists(os.path.join(cx_dataset, relpath)):
                    raise ValueError(f"Image {relpath} not found in cross-sectional dataset")
                cx_preproc_t1w_bids.append(bids_helpers.BIDSImage(cx_dataset, relpath))
                cx_biascorr_t1w_bids.append(bids_helpers.BIDSImage(cx_dataset, relpath.replace('desc-preproc_T1w',
                                                                                          'desc-biascorr_T1w')))
                cx_brain_mask_bids.append(bids_helpers.BIDSImage(cx_dataset, relpath.replace('desc-preproc_T1w',
                                                                                             'desc-brain_mask')))
        logger.info(f"Using selected participant images: {[im.get_uri() for im in cx_preproc_t1w_bids]}")
    else:
        cx_preproc_t1w_bids = bids_helpers.find_participant_images(cx_dataset, args.participant, 'anat', 'desc-preproc_T1w')
        for idx in range(len(cx_preproc_t1w_bids)):
             cx_biascorr_t1w_bids.append(bids_helpers.BIDSImage(cx_dataset, relpath.replace('desc-preproc_T1w',
                                                                                            'desc-biascorr_T1w')))
             cx_brain_mask_bids.append(bids_helpers.BIDSImage(cx_dataset, relpath.replace('desc-preproc_T1w',
                                                                                          'desc-brain_mask')))
        logger.info(f"Using all available participant images: {cx_preproc_t1w_bids}")

    num_sessions = len(cx_preproc_t1w_bids)

    logger.info(f"Found {num_sessions} sessions for participant {args.participant}")

    # Check that the output dataset exists, and if not, create
    # Update dataset_description.json if needed

    # Check cross-sectional and output datasets are not the same - will cause too much confusion
    if os.path.realpath(args.cross_sectional_dataset) == os.path.realpath(args.output_dataset):
        raise PipelineError("Cross-sectional and longitudinal output datasets cannot be the same")

    with tempfile.TemporaryDirectory(suffix=f"antsnetct_longitudinal_{args.participant}.tmpdir") as working_dir:
        try:

            # Output structure

            # output_dataset/dataset_description.json
            # output_dataset/sub-123456
            # output_dataset/sub-123456/anat - contains SST and transforms to group template
            # output_dataset/sub-123456/ses-MR1/anat - contains session-specific files
            # output_dataset/sub-123456/ses-MR2/anat - contains session-specific files

            # These are just copies of the cross-sectional preprocessed data. They are copied to the output dataset
            # for reference. They aren't used for the SST construction or registration.
            long_preproc_t1w_bids = list()

            for idx in range(num_sessions):
                # Write preprocessed T1w images to output dataset - this creates output session directories
                long_preproc_t1w_bids.append(cx_preproc_t1w_bids[idx].copy_image(args.output_dataset))

            # for antsMultivariateTemplateConstruction2.sh
            sst_build_metric = 'CC[3]'
            sst_build_iterations = '20x20x20x50x10'
            sst_build_shrink_factors = '6x4x3x2x1'
            sst_build_smoothing_sigmas = '4x2x1x1x0vox'

            # session pairwise reg to SST after SST construction
            sess_reg_metric = 'CC'
            sess_reg_metric_param_str='3'
            sess_reg_iterations = '20x20x20x40x10'
            sess_reg_shrink_factors = '6x4x3x2x1'
            sess_reg_smoothing_sigmas = '4x3x2x1x0vox'

            if args.sst_reg_quick:
                # for antsMultivariateTemplateConstruction2.sh
                sst_build_metric = 'MI'
                sst_build_iterations = '20x20x20x40x0'
                sst_build_shrink_factors = '6x4x3x2x1'
                sst_build_smoothing_sigmas = '4x2x1x1x0vox'

                # session pairwise reg to SST after SST construction
                sess_reg_metric = 'MI'
                sess_reg_metric_param_str='32'
                sess_reg_iterations = '20x20x20x20x0'
                sess_reg_shrink_factors = '6x4x3x2x1'
                sess_reg_smoothing_sigmas = '4x3x2x1x0vox'

            if args.sst_transform.lower() == 'rigid':
                # CC is very slow, not worth it for linear transforms
                sst_build_metric = 'MI'
                sess_reg_metric = 'MI'
                sess_reg_metric_param_str='32'
                sst_build_transform = 'Rigid[0.1]'
            elif args.sst_transform.lower() == 'affine':
                # CC is very slow, not worth it for linear transforms
                sst_build_metric = 'MI'
                sess_reg_metric = 'MI'
                sess_reg_metric_param_str='32'
                sst_build_transform = 'Affine[0.1]'
            elif args.sst_transform.lower() == 'syn':
                sst_build_transform = 'SyN[0.2,3,0.5]'
            else:
                raise ValueError(f"Unknown SST transform {sst_build_transform}")

            sess_reg_transform = sst_build_transform

            # SST construction
            logger.info("Preprocessing structural images for SST")
            sst_preproc_input = preprocess_sst_input(cx_biascorr_t1w_bids, working_dir)

            template_weights = [1.0 - args.sst_brain_extracted_weight, args.sst_brain_extracted_weight]

            logger.info("Building SST")

            # First do a rigid registration to produce an initial template
            sst_output_rigid = ants_helpers.build_sst(sst_preproc_input, working_dir, initial_templates=None,
                                                      reg_transform='Rigid[0.1]', reg_iterations=sst_build_iterations,
                                                      reg_metric='MI', reg_metric_weights=template_weights,
                                                      reg_shrink_factors=sst_build_shrink_factors,
                                                      reg_smoothing_sigmas=sst_build_smoothing_sigmas,
                                                      template_iterations=2)

            sst_output = ants_helpers.build_sst(sst_preproc_input, working_dir, initial_templates=sst_output_rigid,
                                                reg_transform=sst_build_transform, reg_iterations=sst_build_iterations,
                                                reg_metric=sst_build_metric, reg_metric_weights=template_weights,
                                                reg_shrink_factors=sst_build_shrink_factors,
                                                reg_smoothing_sigmas=sst_build_smoothing_sigmas,
                                                template_iterations=args.sst_iterations)

            # Write SST to output dataset under sub-<label>/anat

            sst_sources = [cx_biascorr_t1w_bids[idx].get_uri(relative=False) for idx in range(num_sessions)]

            sst_metadata = { 'Sources' : sst_sources, 'SkullStripped': False }

            sst_bids = bids_helpers.image_to_bids(sst_output[0], args.output_dataset,
                                                  os.path.join('sub-' + args.participant, 'anat', 'sub-' + args.participant +
                                                               '_T1w.nii.gz'), metadata=sst_metadata)

            session_sst_transforms = list()

            # Register all subjects to SST

            logger.info("Registering all sessions to SST")
            for idx in range(num_sessions):
                logger.info(f"Registering session {idx + 1} to SST")
                moving_head = cx_biascorr_t1w_bids[idx].get_path()
                moving_brain = ants_helpers.apply_mask(moving_head, cx_brain_mask_bids[idx].get_path(), working_dir)
                moving = [moving_head, moving_brain]
                session_sst_transforms.append(
                    ants_helpers.multivariate_pairwise_registration(sst_output, moving, working_dir,
                                                      transform=sess_reg_transform, iterations=sess_reg_iterations,
                                                      metric=sess_reg_metric, metric_param_str=sess_reg_metric_param_str,
                                                      metric_weights=template_weights, shrink_factors=sess_reg_shrink_factors,
                                                      smoothing_sigmas=sess_reg_smoothing_sigmas, apply_transforms=False)
                )
                forward_transform_path = long_preproc_t1w_bids[idx].get_derivative_path_prefix() + \
                    f"_from-T1w_to-sst_mode-image_xfm.h5"
                system_helpers.copy_file(session_sst_transforms[idx]['forward_transform'], forward_transform_path)

                inverse_transform_path = long_preproc_t1w_bids[idx].get_derivative_path_prefix() + \
                    f"_from-sst_to-T1w_mode-image_xfm.h5"
                system_helpers.copy_file(session_sst_transforms[idx]['inverse_transform'], inverse_transform_path)

            # Masks in SST space
            logger.info("Creating SST brain mask")

            sst_t1w_masks = list()

            for idx in range(num_sessions):
                sst_t1w_masks.append(
                    ants_helpers.apply_transforms(sst_output[0], cx_brain_mask_bids[idx].get_path(),
                                                  session_sst_transforms[idx]['forward_transform'], working_dir,
                                                  interpolation='GenericLabel')
                )

            # Combine the masks
            unified_mask_sst = ants_helpers.combine_masks(sst_t1w_masks, working_dir, thresh = 0.1)

            # Save this in the output dataset
            unified_mask_sst_bids = bids_helpers.image_to_bids(unified_mask_sst, args.output_dataset,
                                                               os.path.join('sub-' + args.participant, 'anat',
                                                                            'sub-' + args.participant +
                                                                            '_desc-brain_mask.nii.gz'))

            sst_brain_metadata = { 'Sources': [sst_bids.get_uri(), unified_mask_sst_bids.get_uri()], 'SkullStripped': True }
            sst_brain_bids = bids_helpers.image_to_bids(
                ants_helpers.apply_mask(sst_output[0], unified_mask_sst, working_dir),
                args.output_dataset, os.path.join('sub-' + args.participant, 'anat', 'sub-' + args.participant +
                                                                    '_desc-brain_T1w.nii.gz'), metadata=sst_brain_metadata)

            # Warp the unified mask back to the session spaces
            long_brain_mask_bids = list()

            for idx in range(num_sessions):
                session_mask = ants_helpers.apply_transforms(cx_biascorr_t1w_bids[idx].get_path(), unified_mask_sst,
                                              session_sst_transforms[idx]['inverse_transform'], working_dir,
                                              interpolation='GenericLabel')
                long_brain_mask_bids.append(
                    bids_helpers.image_to_bids(session_mask, args.output_dataset,
                                               long_preproc_t1w_bids[idx].get_derivative_rel_path_prefix() +
                                               '_desc-brain_mask.nii.gz')
                    )

            # Segment the SST

            sst_prior_seg_probabilities = None

            if args.sst_segmentation_method.startswith('antspynet'):
                logger.info("Segmenting SST with deep_atropos")
                sst_prior_seg_probabilities = \
                    get_antsnet_sst_segmentation_priors(sst_bids, working_dir, prior_smoothing_sigma=args.prior_smoothing_sigma,
                                                        prior_csf_gamma=args.prior_csf_gamma)
            else:
                logger.info("Segmenting SST with cross-sectional priors")
                sst_prior_seg_probabilities = \
                    get_cx_sst_segmentation_priors(sst_bids, cx_preproc_t1w_bids, session_sst_transforms, working_dir,
                                                   prior_smoothing_sigma=args.prior_smoothing_sigma,
                                                   prior_csf_gamma=args.prior_csf_gamma)

            # 'none' if we are not segmenting the SST, and just pasing the priors on
            # or 'atropos' if we are using the priors to segment the SST, and passing those posteriors for session processing
            sst_segmentation_method = 'atropos' if args.sst_segmentation_method.endswith('atropos') else 'none'

            logger.info("Segmenting SST")

            sst_seg = segment_sst(sst_bids, unified_mask_sst_bids, sst_prior_seg_probabilities, working_dir,
                                  segmentation_method=sst_segmentation_method,
                                  atropos_prior_weight=args.sst_atropos_prior_weight)

            # align the SST to the group template - note this is a univariate registration, similar to what is done
            # in the cross-sectional pipeline
            sst_to_group_template_reg = None

            if group_template is not None:
                logger.info("Registering SST to group template {group_template.get_name()}")
                sst_to_group_template_reg = cross_sectional_pipeline.template_brain_registration(
                    group_template, group_template_brain_mask, sst_brain_bids, args.template_reg_quick, working_dir
                    )

            # for each session, warp priors, segment, compute thickness, and warp to SST and group template spaces
            logger.info("Processing sessions using SST priors")
            for idx in range(num_sessions):
                logger.info(f"Processing session {idx + 1} of {num_sessions}: " + \
                    f"{cx_preproc_t1w_bids[idx].get_uri(relative=False)}")

                brain_mask_bids = long_brain_mask_bids[idx]
                # Warp priors to the session space
                t1w_priors = list()

                for seg_class in range(6):
                    t1w_priors.append(
                        ants_helpers.apply_transforms(long_preproc_t1w_bids[idx].get_path(),
                                                      sst_seg['posteriors'][seg_class].get_path(),
                                                      session_sst_transforms[idx]['inverse_transform'], working_dir)
                    )

                if args.prior_smoothing_sigma > 0:
                    for idx in range(6):
                        t1w_priors[idx] = ants_helpers.smooth_image(t1w_priors[idx], args.prior_smoothing_sigma, working_dir)

                if args.prior_csf_gamma > 0:
                    t1w_priors[0] = ants_helpers.gamma_image(t1w_priors[0], args.prior_csf_gamma, working_dir)

                # Segment the session
                logger.info(f"Segmenting session {idx + 1}")
                seg_n4 = cross_sectional_pipeline.segment_and_bias_correct(
                    long_preproc_t1w_bids[idx], brain_mask_bids, t1w_priors, working_dir, denoise=True,
                    segmentation_method='atropos', atropos_n4_iterations=args.atropos_n4_iterations,
                    atropos_prior_weight=args.atropos_prior_weight)
                # Compute thickness
                logger.info(f"Cortical thickness for session {idx + 1}")
                thickness = cross_sectional_pipeline.cortical_thickness(seg_n4, working_dir,
                                                                        thickness_iterations=args.thickness_iterations)
                # Derivatives in SST space: head, brain, thickness, jacobian, GM probability
                # Derivatives in group template space: brain, thickness, GM probability
                logger.info(f"Computing template space derivatives for session {idx + 1}")
                sst_derivatives = None

                if group_template is not None:
                    sst_derivatives = template_space_derivatives(
                        sst_bids, session_sst_transforms[idx]['forward_transform'],
                        seg_n4, thickness, working_dir, group_template=group_template,
                        sst_group_template_transform=sst_to_group_template_reg['forward_transform'])
                else:
                    sst_derivatives = template_space_derivatives(sst_bids, session_sst_transforms[idx]['forward_transform'],
                                                                 seg_n4, thickness, working_dir)

                logger.info(f"Finished processing session {idx + 1} of {num_sessions}")

            logger.info(f"Finished processing {args.participant}")

            if args.keep_workdir.lower() == 'always':
                debug_workdir = os.path.join(args.output_dataset, f"sub-{args.participant}", f"sub-{args.participant}_workdir")
                logger.info(f"Saving working directory {working_dir} to {debug_workdir}")
                shutil.copytree(working_dir, debug_workdir)

        except Exception as e:
            logger.error(f"Caught {type(e)} during processing of {args.participant}")
            # Print stack trace
            traceback.print_exc()
            debug_workdir = os.path.join(args.output_dataset, f"sub-{args.participant}", f"sub-{args.participant}_workdir")
            if args.keep_workdir.lower() != 'never':
                logger.info(f"Saving working directory {working_dir} to {debug_workdir}")
                shutil.copytree(working_dir, debug_workdir)


def preprocess_sst_input(cx_biascorr_t1w_bids, work_dir):
    """Preprocess the input images for SST construction.

    The preprocessing steps are:
        1. Normalize intensity such that the mean WM intensity is the same across images
        2. Reset the origin of the images to the centroid of the brain mask. This prevents unwanted shifts
           in the SST position.

    Parameters:
    ----------
    cx_biascorr_t1w_bids : list of BIDSImage
        List of BIDSImage objects for the input images. These should be the bias-corrected T1w images from cross-sectional
        processing. This function will look for brain masks in the same directory with the suffix 'desc-brain_mask.nii.gz'.

    work_dir : str
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

        input_t1w_mask = t1w.get_derivative_path_prefix() + '_desc-brain_mask.nii.gz'

        origin_fix = preprocessing.reset_origin_by_centroid(normalized, input_t1w_mask, work_dir)

        sst_input_t1w_denoised_normalized_heads.append(origin_fix)

        mask_origin_fix = preprocessing.reset_origin_by_centroid(input_t1w_mask, input_t1w_mask, work_dir)

        brain_origin_fix = ants_helpers.apply_mask(origin_fix, mask_origin_fix, work_dir)

        sst_input_t1w_denoised_normalized_brains.append(brain_origin_fix)

    sst_input_combined = [sst_input_t1w_denoised_normalized_heads, sst_input_t1w_denoised_normalized_brains]

    return sst_input_combined

def get_antsnet_sst_segmentation_priors(sst_bids, work_dir, prior_smoothing_sigma=0, prior_csf_gamma=0):
    """Get segmentation priors for the SST from deep_atropos

    Parameters:
    -----------
    sst_bids : BIDSImage
        SST image
    work_dir : str
        Working directory
    prior_smoothing_sigma : float, optional
        Sigma for smoothing the priors, in voxels. Default is 0.
    prior_csf_gamma : float, optional
        Gamma value for the CSF prior. Default is 0 (no correction).

    Returns:
    --------
    list
        List of segmentation priors, in the order CSF, CGM, WM, SGM, BS, CBM

    """
    deep_atropos = ants_helpers.deep_atropos(sst_bids.get_path(), work_dir)

    posteriors = deep_atropos['posteriors']

    atropos_prior_images = list()

    if prior_smoothing_sigma > 0:
        logger.info(f"Smoothing priors with sigma {prior_smoothing_sigma}")
        atropos_prior_images = [ants_helpers.smooth_image(prior, prior_smoothing_sigma, work_dir)
                                for prior in deep_atropos['posteriors']]
    else:
        atropos_prior_images = posteriors

    if prior_csf_gamma > 0:
        logger.info(f"Gamma correcting CSF prior with gamma {prior_csf_gamma}")
        atropos_prior_images[0] = ants_helpers.gamma_image(atropos_prior_images[0], prior_csf_gamma, work_dir)

    return atropos_prior_images

def get_cx_sst_segmentation_priors(sst_bids, cx_t1w_preproc_bids, cx_sst_transforms, work_dir, prior_smoothing_sigma=0,
                                   prior_csf_gamma=0):
    """Get segmentation priors for the SST by averaging the cross-sectional segmentations

    Parameters:
    -----------
    sst_bids : BIDSImage
        SST image
    cx_t1w_preproc_bids : list
        List of cross-sectionally processed T1w images, these are used to find priors
    cx_sst_transforms : list
        List of transforms from the cross-sectional images to the SST, in the same order as cx_t1w_preproc_bids
    work_dir : str
        Working directory
    prior_smoothing_sigma : float, optional
        Sigma for smoothing the priors, in voxels. Default is 0.
    prior_csf_gamma : float, optional
        Gamma value for the CSF prior. Default is 0 (no correction).

    Returns:
    --------
    list
        List of segmentation priors, in the order CSF, CGM, WM, SGM, BS, CBM

    """
    sst_priors = list()

    for post_label in ['CSF', 'CGM', 'WM', 'SGM', 'BS', 'CBM']:
        label_posteriors = list()
        for idx in range(len(cx_t1w_preproc_bids)):
            session_posterior = cx_t1w_preproc_bids[idx].get_derivative_path_prefix() + '_seg-antsnetct_label-' + \
                                post_label + '_probseg.nii.gz'
            label_posteriors.append(ants_helpers.apply_transforms(sst_bids.get_path(), session_posterior,
                                                                  cx_sst_transforms[idx]['forward_transform'],
                                                                  work_dir, interpolation='Linear'))
        sst_priors.append(ants_helpers.average_images(label_posteriors, work_dir))

    if prior_smoothing_sigma > 0:
        logger.info(f"Smoothing priors with sigma {prior_smoothing_sigma}")
        sst_priors = [ants_helpers.smooth_image(prior, prior_smoothing_sigma, work_dir) for prior in sst_priors]

    if prior_csf_gamma > 0:
        logger.info(f"Gamma correcting CSF prior with gamma {prior_csf_gamma}")
        sst_priors[0] = ants_helpers.gamma_image(sst_priors[0], prior_csf_gamma, work_dir)

    return sst_priors


def segment_sst(sst_bids, sst_brain_mask_bids, segmentation_priors, work_dir, segmentation_method='atropos',
                atropos_prior_weight=0.25):
    """Segment the SST

    If the segmentation_method is 'none', the prior segmentation, generated with deep_atropos) is copied to the output.

    If the segmentation_method is 'atropos', the priors are used to to refine the segmentation with Atropos.

    Parameters:
    -----------
    sst_bids : BIDSImage
        T1w SST object. Output is in the space of this image.
    sst_brain_mask_bids : BIDSImage
        Brain mask for the SST.
    segmentation_priors :  list
        List of files containing segmentation probabilities, in their antsct order: CSF, CGM, WM, SGM, BS, CBM. These are used
        as priors for segmentation and bias correction.
    work_dir : str
        Path to the working directory.
    segmentation method : str, optional
        Method to use for segmentation. Default is 'atropos', meaning the priors are used for segmentation with Atropos.
    atropos_prior_weight : float, optional
        Prior weight for Atropos. Default is 0.25. Minimum useful value is 0.2, below this the priors are not very well
        constrained and you will see priors that overlap in intensity (like SGM and CBM) appear in the wrong places.

    Returns:
    --------
    dict
        Dictionary of segmentation images (as BIDSImage objects) with keys:
        'segmentation_image' - the segmentation image
        'posteriors' - list of segmentation posteriors
    """

    # Output images, to be converted to BIDS before returning
    seg_output = {}

    if segmentation_method.lower() == 'atropos':

        atropos_prior_images = segmentation_priors

        logger.info("Running Atropos")

        seg_output = ants_helpers.atropos_segmentation(sst_bids.get_path(), sst_brain_mask_bids.get_path(), work_dir,
                                                       prior_probabilities=atropos_prior_images,
                                                       prior_weight=atropos_prior_weight)

        # remap the segmentation posteriors to BIDS labels
        seg_output['segmentation_image'] = ants_helpers.posteriors_to_segmentation(seg_output['posteriors'], work_dir)
    elif segmentation_method.lower() == 'none':
        logger.info("Segmentation method is none, generating final segmentation directly from priors")
        posteriors_masked = [ants_helpers.apply_mask(posterior, sst_brain_mask_bids.get_path(), work_dir)
                                for posterior in segmentation_priors]

        seg_output['segmentation_image'] = ants_helpers.posteriors_to_segmentation(posteriors_masked, work_dir)
        seg_output['posteriors'] = posteriors_masked

    else:
        raise ValueError('Unknown segmentation method: ' + segmentation_method)

    # Copy the segmentation outputs to the output dataset
    seg_output_bids = {}

    seg_output_bids['posteriors'] = list()

    seg_posterior_labels = ['CSF', 'CGM', 'WM', 'SGM', 'BS', 'CBM']

    for idx, seg_label in enumerate(seg_posterior_labels):
        seg_output_bids['posteriors'].append(
            bids_helpers.image_to_bids(seg_output['posteriors'][idx], sst_bids.get_ds_path(),
                                       sst_bids.get_derivative_rel_path_prefix() +
                                       f"_seg-antsnetct_label-{seg_label}_probseg.nii.gz",
                                       metadata={'Sources': [sst_bids.get_uri(), sst_brain_mask_bids.get_uri()]}))

    seg_sources = [sst_bids.get_uri(), sst_brain_mask_bids.get_uri()]

    if segmentation_method.lower() == 'none':
        # If not running Atropos, the posteriors are used to generate the segmentation
        seg_sources.extend([prob.get_uri() for prob in seg_output_bids['posteriors']])

    seg_output_bids['segmentation_image'] = \
        bids_helpers.image_to_bids(seg_output['segmentation_image'], sst_bids.get_ds_path(),
                                   sst_bids.get_derivative_rel_path_prefix() + '_seg-antsnetct_dseg.nii.gz',
                                   metadata={'Sources': seg_sources})

    return seg_output_bids


def template_space_derivatives(sst, session_sst_transform, seg_n4, thickness, work_dir, group_template=None,
                               sst_group_template_transform=None):
    """Warp images to the SST and group template spaces.

    The registration parameters assume the session to SST and SST to group template warps are the forward transforms.

    Output images are written to the session directory in the output dataset.

    Parameters:
    ----------
    sst : BIDSImage
        SST image.
    session_sst_reg : str
        Transform from session space to SST space.
    seg_n4 : dict
        Segmentation results for the session.
    thickness : dict
        Cortical thickness results for the session.
    work_dir : sry
        Working directory.
    group_template : TemplateImage
        Group template image.
    sst_group_template_transform : dict
        Transform from SST to group template space.

    Returns:
    -------
    dict:
        Dictionary containing the warped images, for the SST and optionally group template spaces.
    """
    session_ref_image_bids = seg_n4['bias_corrected_t1w_brain']

    sst_space_rel_output_prefix = session_ref_image_bids.get_derivative_rel_path_prefix() + '_space-sst'

    # SST space derivatives

    thickness_sst_space = ants_helpers.apply_transforms(sst.get_path(), thickness.get_path(), session_sst_transform, work_dir)

    sst_space_bids = {}

    sst_space_bids['thickness'] = bids_helpers.image_to_bids(thickness_sst_space, session_ref_image_bids.get_ds_path(),
                                                             sst_space_rel_output_prefix + '_desc-thickness.nii.gz')

    # Make the jacobian log determinant image in the template space
    jacobian_sst_space = ants_helpers.get_log_jacobian_determinant(sst.get_path(), session_sst_transform, work_dir)

    if jacobian_sst_space is not None:
        sst_space_bids['jacobian'] = bids_helpers.image_to_bids(jacobian_sst_space, session_ref_image_bids.get_ds_path(),
                                                                sst_space_rel_output_prefix + '_desc-logjacobian.nii.gz')

    # gray matter probability
    gm_prob_sst_space = ants_helpers.apply_transforms(sst.get_path(), seg_n4['posteriors'][1].get_path(), session_sst_transform,
                                                      work_dir)

    sst_space_bids['gmp'] = bids_helpers.image_to_bids(gm_prob_sst_space, session_ref_image_bids.get_ds_path(),
                                                       sst_space_rel_output_prefix + '_label-GM_probseg.nii.gz')

    # Denoised / bias-corrected head image
    bias_corrected_head_sst_space = ants_helpers.apply_transforms(sst.get_path(), seg_n4['bias_corrected_t1w'].get_path(),
                                                                  session_sst_transform, work_dir)

    sst_space_bids['t1w'] = bids_helpers.image_to_bids(bias_corrected_head_sst_space,
                                                       session_ref_image_bids.get_ds_path(),
                                                       sst_space_rel_output_prefix + '_desc-biascorr_T1w.nii.gz')

    # bias-corrected brain image
    bias_corrected_brain_sst_space = ants_helpers.apply_transforms(sst.get_path(),
                                                                   seg_n4['bias_corrected_t1w_brain'].get_path(),
                                                                   session_sst_transform, work_dir)

    sst_space_bids['t1w_brain'] = bids_helpers.image_to_bids(bias_corrected_brain_sst_space,
                                                             session_ref_image_bids.get_ds_path(),
                                                             sst_space_rel_output_prefix + '_desc-biascorrbrain_T1w.nii.gz')

    group_space_bids = {}

    # Warp to group template space through SST
    if group_template is not None:
        session_group_template_transforms = [ sst_group_template_transform, session_sst_transform]

        group_space_rel_output_prefix = session_ref_image_bids.get_derivative_rel_path_prefix() + '_' + \
            group_template.get_derivative_space_string()

        thickness_group_space = ants_helpers.apply_transforms(group_template.get_path(), thickness.get_path(),
                                                              session_group_template_transforms, work_dir)


        group_space_bids['thickness'] = bids_helpers.image_to_bids(thickness_group_space, session_ref_image_bids.get_ds_path(),
                                                                   group_space_rel_output_prefix + '_desc-thickness.nii.gz')

        # gray matter probability
        gm_prob_group_space = ants_helpers.apply_transforms(group_template.get_path(), seg_n4['posteriors'][1].get_path(),
                                                            session_group_template_transforms, work_dir)

        group_space_bids['gmp'] = bids_helpers.image_to_bids(gm_prob_group_space, session_ref_image_bids.get_ds_path(),
                                                             group_space_rel_output_prefix + '_label-GM_probseg.nii.gz')

        # bias-corrected brain image
        bias_corrected_brain_group_space = ants_helpers.apply_transforms(group_template.get_path(),
                                                                         seg_n4['bias_corrected_t1w_brain'].get_path(),
                                                                         session_group_template_transforms, work_dir)

        group_space_bids['t1w_brain'] = bids_helpers.image_to_bids(bias_corrected_brain_group_space,
                                                                   session_ref_image_bids.get_ds_path(),
                                                                   group_space_rel_output_prefix + \
                                                                       '_desc-biascorrbrain_T1w.nii.gz')

    template_space_bids = { 'sst': sst_space_bids, 'group_template': group_space_bids }

    return template_space_bids

