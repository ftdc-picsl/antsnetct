from . import ants_helpers
from . import bids_helpers
from . import preprocessing
from . import system_helpers

from .system_helpers import PipelineError

import argparse
import copy
import glob
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

def cross_sectional_analysis():

    parser = argparse.ArgumentParser(formatter_class=RawDefaultsHelpFormatter, add_help = False,
                                     description='''Cortical thickness analysis with ANTsPyNet.

    Input is by participant and session,

        '--participant 01 --session MR1'

    or just by participant,

        '--participant 01'

    which will process all sessions in sequence. A BIDS filter can be used to select a subset of T1w images to process. Session
    labels on the command line override any session keys in the filter file.

    Output is to a BIDS derivative dataset.

    If the output dataset does not exist, it will be created.

    The analysis level for cross-sectional data is the session. Output is to a BIDS derivative dataset.

    Pre-computed brain masks and segmentations can optionally be used for processing. These should be in a BIDS derivative
    dataset with sidecars describing the input T1w image as the source image.

    If a brain mask dataset or segmentation dataset are specified at run time, there must be masks or segmentations for the
    input data, or an error will be raised. This is to prevent inconsistent processing.

    --- BIDS options ---

    By default, the pybids BIDS validator is run on the input dataset. This can be disabled with '--skip-bids-validation'. BIDS
    validation looks for "raw BIDS" data only, and must be skipped if your input dataset is a derivative dataset.

    The BIDS validator will simply ignore data that does not conform to the BIDS standard. To get useful diagnostic information
    on validation failures, run a standalone BIDS validator on the input dataset.

    Users can select a subset of T1w images to process with a BIDS filter file. Following the *prep pipelines, the filters
    should contain a dict "t1w" with keys and values to filter on. The default filter for t1w images is

        "t1w": {
            "datatype": "anat",
            "suffix": "T1w"
        }

    User-defined filters override this, so they should include the default filter keys.


    --- Preprocessing ---

    All input images are conformed to LPI orientation with c3d.


    --- Brain masking ---

    The script will check for a brain mask from the brain mask dataset, if defined. If not, a brain mask may be found in the
    input dataset. If no brain mask is available, one will be generated using anyspynet.


    --- Segmentation ---

    Posteriors from an existing segmentation may be used as priors for Atropos. Posteriors should have the entity
    'label-<structure>' where structure includes CSF, CGM, WM, SGM, BS, CBM.


    --- Template registration ---

    The bias-corrected brain image is registered to the template, if provided.

    All templates are accessed through templateflow. Set the environment variable TEMPLATEFLOW_HOME to point to a specific
    templateflow directory. Any templateflow template can be used as long as it has a T1w image and a brain mask.


    --- Processing steps ---

    1. Preprocessing. The T1w image is conformed to LPI orientation and optionally trimmed to remove neck coverage.

    2. Brain masking. If a brain mask is available, it is used. Otherwise, one is generated with anyspynet.

    3. Segmentation and bias correction. If a segmentation is available, it is used according to the user instructions.
    Otherwise, one is generated with anyspynet. Segmentation may be refined with Atropos. The T1w image is simultaneously
    segmented and bias-corrected.

    4. Cortical thickness estimation.

    5. Atlas registration. If an atlas is defined, the T1w image is registered to the atlas.

    If the '--longitudinal-preproc' option is used, steps 4 and 5 are skipped. This avoids having to compute thickness and
    template registration twice for longitudinal data.


    --- Debugging / development / testing options ---

    To keep all output including the working directory, use '--keep-workdir always'. This will copy the working directory to
    the output dataset. The default is to save the working directory only if an error is encountered.

    The most time-consuming part of the pipeline is the cortical thickness estimation and the template registration. To get
    output faster for testing, use `--template-reg-quick` and `--thickness-iterations 10`.


    --- Controlling multi-threading ---

    Set the environment variable ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS to control the number of threads used by ITK processes
    and other general multi-threaded processes. Set `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=0` to allow automatic calculation,
    in which case the number of threads is set to the number of available CPUs, or 8, whichever is smaller.

    For tensorflow, set the environment variables TF_NUM_INTEROP_THREADS and TF_NUM_INTRAOP_THREADS to control the number of
    threads in ANTsPyNet. The default is 1 for both.

    ''')

    required_parser = parser.add_argument_group("Required arguments")
    required_parser.add_argument("--input-dataset", help="Input BIDS dataset dir, containing the source images", type=str,
                          required=True)
    required_parser.add_argument("--participant", help="Participant to process", type=str, required=True)
    required_parser.add_argument("--output-dataset", help="Output BIDS dataset dir", type=str, required=True)

    optional_parser = parser.add_argument_group("General optional arguments")
    optional_parser.add_argument("-h", "--help", action="help", help="show this help message and exit")
    optional_parser.add_argument("--bids-filter-file", help="BIDS filter file", type=str, default=None)
    optional_parser.add_argument("--longitudinal-preproc", help="Skip thickness and template registration steps. Use this for "
                                 "data that is only analyzed longitudinally", action='store_true')
    optional_parser.add_argument("--keep-workdir", help="Copy working directory to output, for debugging purposes. Either "
                                 "'never', 'on_error', or 'always'.", type=str, default='on_error')
    optional_parser.add_argument("--session", help="Session to process. Using this overrides any session filter in the BIDS "
                                 "filter file.", type=str, default=None)
    optional_parser.add_argument("--skip-bids-validation", help="Skip BIDS validation", action='store_true')
    optional_parser.add_argument("--verbose", help="Verbose output from subcommands", action='store_true')

    neck_trim_parser = parser.add_argument_group("Pre-processing arguments")
    neck_trim_parser.add_argument("--no-neck-trim", help="Disable neck trimming from the T1w image", dest="do_neck_trim",
                                  action='store_false')
    neck_trim_parser.add_argument("--pad-mm", help="Padding in mm to add to the image, after neck trimming (if enabled) but "
                                  "before further processing", type=int, default=10)
    template_parser = parser.add_argument_group("Template arguments")
    template_parser.add_argument("--template-name", help="Template to use for registration, or 'none' to disable this step.",
                                 type=str, default='MNI152NLin2009cAsym')
    template_parser.add_argument("--template-res", help="Resolution of the template, eg '01', '02', etc. Note this is a "
                                 "templateflow index and not a physical spacing. If the selected template does not define "
                                 "multiple resolutions, this is ignored.", type=str, default='01')
    template_parser.add_argument("--template-cohort", help="Template cohort, only needed for templates that define multiple "
                                 "cohorts", type=str, default=None)
    template_parser.add_argument("--template-reg-quick", help="Do quick registration to the template", action='store_true')

    brain_mask_parser = parser.add_argument_group("Brain mask arguments")
    brain_mask_parser.add_argument("--brain-mask-dataset", help="Dataset containing brain masks. Masks from here will be used "
                                   "in preference to those in the input dataset.", type=str, default=None)
    brain_mask_parser.add_argument("--brain-mask-modality", help="Brain masking modality option to use with antspynet. Only "
                                   "used if no pre-existing mask is found. Options are 't1threetissue', 't1', 't1nobrainer', "
                                   "'t1combined'", type=str, default='t1threetissue')

    segmentation_parser = parser.add_argument_group("Segmentation arguments. The default segmentation method is ANTsPyNet "
                                                    "deep_atropos. The output from deep_atropos or from external priors can "
                                                    "optionally be used as priors for antsAtroposN4.sh.")
    segmentation_parser.add_argument("--segmentation-dataset", help="Dataset containing individual segmentations.", type=str,
                                     default=None)
    segmentation_parser.add_argument('--segmentation-template-name', help="Template to use for warping segmentation priors to "
                                     "the session space. This is a legacy option to provide priors similar to those used in "
                                     "antsCorticalThickness.sh.", type=str, default=None)
    segmentation_parser.add_argument('--segmentation-template-res', help="Resolution of the segmentation template, eg '01', "
                                     "'02', etc. Note this is a templateflow index and not a physical spacing. If the selected "
                                     "template does not define multiple resolutions, this is ignored.", type=str, default='01')
    segmentation_parser.add_argument("--do-ants-atropos-n4", help="Run antsAtroposN4.sh. If this is specified, the "
                                     "segmentation defined by the segmentation dataset, atlas, or deep_atropos is used as a "
                                     "prior.", action='store_true')
    segmentation_parser.add_argument("--legacy-deep-atropos", help="Use the legacy deep_atropos network. "
                                     "Has no effect if deep_atropos is not called", action='store_true')
    segmentation_parser.add_argument("--atropos-n4-iterations", help="Number of iterations of atropos-n4",
                                     type=int, default=2)
    segmentation_parser.add_argument("--atropos-prior-weight", help="Prior weight for Atropos", type=float, default=0.25)
    segmentation_parser.add_argument("--prior-smoothing-sigma", help="Sigma for smoothing the priors, in voxels. Experimental",
                                     type=float, default=0)
    segmentation_parser.add_argument("--csf-prior-gamma", help="Gamma correction for the CSF prior. Defaults to 0 for "
                                     "no correction. Experimental", type=float,
                                     default=0)

    thickness_parser = parser.add_argument_group("Thickness arguments")
    thickness_parser.add_argument("--thickness-iterations", help="Number of iterations for cortical thickness estimation.",
                                  type=int, default=45)

    args = parser.parse_args()

    logger.info("Parsed args: " + str(args))

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    template = None
    template_brain_mask = None

    # setup templateflow, check template can be found
    if args.template_name.lower() != 'none':
        if not 'TEMPLATEFLOW_HOME' in os.environ or not os.path.exists(os.environ.get('TEMPLATEFLOW_HOME')):
            raise PipelineError(f"templateflow directory not found at " +
                                f"TEMPLATEFLOW_HOME={os.environ.get('TEMPLATEFLOW_HOME')}")

        template = bids_helpers.TemplateImage(args.template_name, suffix='T1w', description=None,
                                              resolution=args.template_res, cohort=args.template_cohort)

        template_brain_mask = bids_helpers.TemplateImage(args.template_name, suffix='mask', description='brain',
                                                         resolution=args.template_res, cohort=args.template_cohort)

    system_helpers.set_verbose(args.verbose)

    input_dataset = args.input_dataset
    output_dataset = args.output_dataset

    if (os.path.realpath(input_dataset) == os.path.realpath(output_dataset)):
        raise ValueError('Input and output datasets cannot be the same')

    input_dataset_description = None

    if os.path.exists(os.path.join(input_dataset, 'dataset_description.json')):
        with open(os.path.join(input_dataset, 'dataset_description.json'), 'r') as f:
            input_dataset_description = json.load(f)
    else:
        raise ValueError('Input dataset does not contain a dataset_description.json file')


    if args.participant is None:
        raise ValueError('Participant must be defined')

    validate_bids = not args.skip_bids_validation

    # Check segmentation options make sense
    if args.segmentation_template_name != None and args.segmentation_dataset != None:
        raise ValueError('Only one of segmentation-dataset and segmentation-template-name can be defined')

    # Using a segmentation template implies we want to do antsAtroposN4
    if args.segmentation_template_name != None:
        args.do_ants_atropos_n4 = True

    logger.info("Input dataset path: " + input_dataset)
    logger.info("Input dataset name: " + input_dataset_description['Name'])

    # Create the output dataset and add metadata as needed
    output_dataset_link_paths = [os.path.abspath(input_dataset)]
    if args.brain_mask_dataset is not None:
        output_dataset_link_paths.append(os.path.abspath(args.brain_mask_dataset))
    if args.segmentation_dataset is not None:
        output_dataset_link_paths.append(os.path.abspath(args.segmentation_dataset))

    bids_helpers.update_output_dataset(output_dataset, input_dataset_description['Name'] + '_antsnetct',
                                       output_dataset_link_paths)

    with open(os.path.join(output_dataset, 'dataset_description.json'), 'r') as f:
        output_dataset_description = json.load(f)

    logger.info("Output dataset path: " + output_dataset)
    logger.info("Output dataset name: " + output_dataset_description['Name'])

    # get all the T1ws, optionally with a filter to select a subset
    bids_t1w_filter = bids_helpers.get_modality_filter_query('t1w', args.bids_filter_file)

    if args.session is not None:
        # Add session to the filter
        bids_t1w_filter['session'] = args.session

    logger.info("Searching for T1w images with filter: " + str(bids_t1w_filter))

    with tempfile.TemporaryDirectory(suffix=f"antsnetct_bids_{args.participant}.tmpdir") as bids_wd:
        input_t1w_bids = bids_helpers.find_participant_images(input_dataset, args.participant, bids_wd, validate=validate_bids,
                                                              **bids_t1w_filter)

    if input_t1w_bids is None or len(input_t1w_bids) == 0:
        logger.error(f"No T1w images found for participant {args.participant}. If your input dataset is derivative, use "
                     "--skip-bids-validation to disable BIDS validation.")
        return

    for t1w_bids in input_t1w_bids:
        logger.info("Found T1w image: " + t1w_bids.get_uri(relative=False))

    for t1w_bids in input_t1w_bids:
        with tempfile.TemporaryDirectory(
                suffix=f"antsnetct_{system_helpers.get_nifti_file_prefix(t1w_bids.get_path())}.tmpdir") as working_dir:
            try:

                logger.info("Processing T1w image: " + t1w_bids.get_uri(relative=False))

                if bool(glob.glob(os.path.join(output_dataset, t1w_bids.get_derivative_rel_path_prefix() + '*'))):
                    logger.warning(f"Skipping {str(t1w_bids)}, output already exists. Clean up files matching " +
                                   f"{t1w_bids.get_derivative_rel_path_prefix()} to re-process")
                    continue

                # Preprocessing
                # Conform to LPI orientation
                logger.info("Preprocessing T1w image")
                preproc_t1w_bids = preprocess_t1w(t1w_bids, output_dataset, working_dir, orient='LPI',
                                                  trim_neck=args.do_neck_trim, pad=args.pad_mm)

                # Find a brain mask using the first available in order of preference:
                # 1. Brain mask dataset
                # 2. Brain mask in input dataset
                # 3. Generate a brain mask with antspynet
                logger.info("Brain mask T1w")
                brain_mask_bids = get_brain_mask(t1w_bids, preproc_t1w_bids, working_dir, args.brain_mask_dataset,
                                                 args.brain_mask_modality)

                # With mask defined, we can now do segmentation and bias correction
                logger.info("Segmentation and bias correction")

                # Get segmentation priors. If a segmentation dataset is defined, it is an error to not find a segmentation.
                # If no pre-existing segmentation is found, priors are generated with anyspynet.
                # If the user asked for deep_atropos segmentation, then there can be no existing segmentation, and deep_atropos
                # is called to generate the segmentation priors.
                if args.segmentation_template_name is not None:
                    # Old way, this is slow
                    seg_template = bids_helpers.TemplateImage(args.segmentation_template_name, suffix='T1w', description=None,
                                              resolution=args.segmentation_template_res)
                    seg_template_brain_mask = bids_helpers.TemplateImage(args.segmentation_template_name, suffix='mask',
                                                                         description='brain',
                                                                         resolution=args.segmentation_template_res)

                    seg_priors = get_template_segmentation_priors(preproc_t1w_bids, brain_mask_bids, seg_template,
                                                                  seg_template_brain_mask, working_dir,
                                                                  quick_reg=args.template_reg_quick)
                else:
                    # Uses the segmentation dataset to find segmentation priors, or generates them with deep_atropos
                    seg_priors = get_segmentation_priors(t1w_bids, preproc_t1w_bids, working_dir, args.segmentation_dataset,
                                                            args.csf_prior_gamma, args.prior_smoothing_sigma,
                                                            use_legacy_deep_atropos = args.legacy_deep_atropos)

                seg_n4 = segment_and_bias_correct(preproc_t1w_bids, brain_mask_bids, seg_priors['prior_seg_probabilities'],
                                                  working_dir, prior_metadata=seg_priors['prior_metadata'],
                                                  do_atropos_n4=args.do_ants_atropos_n4,
                                                  atropos_n4_iterations=args.atropos_n4_iterations,
                                                  atropos_prior_weight=args.atropos_prior_weight)

                if args.longitudinal_preproc:
                    logger.info("Skipping thickness and template registration steps")
                    logger.info(f"Finished processing {t1w_bids.get_uri(relative=False)}")
                    continue
                logger.info("Computing cortical thickness")
                thickness = cortical_thickness(seg_n4, working_dir, args.thickness_iterations)

                # If an atlas is defined, register the T1w image to the atlas
                if args.template_name.lower() != 'none':
                    logger.info("Registering to template")
                    template_reg = template_brain_registration(template, template_brain_mask,
                                                               seg_n4['bias_corrected_t1w_brain'], args.template_reg_quick,
                                                               working_dir)
                    # Make template space derivatives: thickness, jacobian, GM probability, t1w brain
                    logger.info("Creating template space derivatives")
                    template_space_derivatives(template, template_reg, seg_n4, thickness, working_dir)

                # Make QC plots
                logger.info("Creating QC plots")
                make_qc_plots(seg_n4['bias_corrected_t1w'], brain_mask_bids, seg_n4['segmentation_image'], thickness,
                              working_dir)

                if args.keep_workdir.lower() == 'always':
                    logger.info("Keeping working directory: " + working_dir)
                    shutil.copytree(working_dir, os.path.join(output_dataset, preproc_t1w_bids.get_derivative_rel_path_prefix()
                                                              + "_workdir"))

                logger.info(f"Finished processing {t1w_bids.get_uri(relative=False)}")

            except Exception as e:
                logger.error(f"Caught {type(e)} during processing of {str(t1w_bids)}")
                # Print stack trace
                traceback.print_exc()
                debug_workdir = os.path.join(output_dataset, t1w_bids.get_derivative_rel_path_prefix() + "_workdir")
                if args.keep_workdir.lower() != 'never':
                    logger.info("Saving working directory to " + debug_workdir)
                    shutil.copytree(working_dir, debug_workdir)


def preprocess_t1w(t1w_bids, output_dataset, work_dir, orient='LPI', trim_neck=True, pad=10):
    """Preprocess a T1w image, including orientation, neck trimming, and padding.

    Parameters:
    -----------
    t1w_bids : BIDSImage
        BIDSImage object for the T1w image.
    output_dataset : str
        Path to the output dataset.
    work_dir : str
        Path to the working directory.
    orient : str, optional
        Orientation to conform the preprocessed image. Default is 'LPI'.
    trim_neck : bool, optional
        Whether to trim the neck from the image. Default is True.
    pad : float, optional
        Padding in mm to add to the image (after neck trimming, if enabled). Default is 10.

    Returns:
    --------
    BIDSImage
        The preprocessed T1w image in the output dataset.

    """
    preproc_t1w = preprocessing.conform_image_orientation(t1w_bids.get_path(), orient, work_dir)

    if trim_neck:
        logger.info("Trimming neck")
        preproc_t1w = preprocessing.trim_neck(preproc_t1w, work_dir)

    preproc_t1w = preprocessing.pad_image(preproc_t1w, work_dir, pad_mm=pad)

    # Copy the preprocessed T1w to the output
    preproc_t1w_bids = bids_helpers.image_to_bids(preproc_t1w, output_dataset, t1w_bids.get_derivative_rel_path_prefix() +
                                                  '_desc-preproc_T1w.nii.gz',
                                                  metadata={'Sources': [t1w_bids.get_uri(relative=False)],
                                                            'SkullStripped': False})

    return preproc_t1w_bids


def get_brain_mask(t1w_bids, t1w_bids_preproc, work_dir, brain_mask_dataset=None, brain_mask_method='t1'):
    """Get a brain mask for a T1w image.

    Copies or defines a brain mask using the first available in order of preference:
        1. Brain mask dataset
        2. Brain mask in input dataset
        3. Generate a brain mask with antspynet

    Parameters:
    -----------
    t1w_bids : BIDSImage
        BIDSImage object for the T1w image. This is used to search for existing masks.
    t1w_bids_preproc: BIDSImage
        BIDSImage object for the preprocessed T1w image in the output dataset. The mask, if found,
        will be resliced to this space.
    work_dir : str
        Path to the working directory.
    brain_mask_dataset : str, optional
        Path to the brain mask dataset. If provided, it is an error to not find a brain mask.
    brain_mask_method : str, optional
        Method to generate the brain mask with antspynet. Default is 't1'.

    Returns:
    --------
    BIDSImage
        The brain mask in the output dataset.

    Raises:
    -------
    ValueError
        If the brain mask dataset is not None and does not contain a brain mask for the specified T1w image.
    """
    brain_mask_bids = None
    brain_mask_path = None
    brain_mask_metadata = None
    brain_mask_reslice = None

    found_brain_mask = False

    # If a mask dataset is defined, it is an error to not find a mask
    if brain_mask_dataset is not None:
        brain_mask_bids = bids_helpers.find_brain_mask(brain_mask_dataset, t1w_bids)
        if brain_mask_bids is None:
            raise ValueError('Brain mask dataset does not contain a brain mask for ' + str(t1w_bids))
        # Found a brain mask
        logger.info("Using brain mask: " + str(brain_mask_bids))
        found_brain_mask = True
        brain_mask_path = brain_mask_bids.get_path()
        brain_mask_metadata = brain_mask_bids.get_metadata()

    # If no mask dataset is defined, try to find a mask in the input dataset
    if not found_brain_mask:
        brain_mask_bids = bids_helpers.find_brain_mask(t1w_bids.get_ds_path(), t1w_bids)
        if brain_mask_bids is not None:
            # Found a brain mask in input dataset
            logger.info("Using brain mask: " + str(brain_mask_bids))
            found_brain_mask = True
            brain_mask_path = brain_mask_bids.get_path()
            brain_mask_metadata = brain_mask_bids.get_metadata()

    if found_brain_mask:
        brain_mask_metadata['Sources'] = [t1w_bids_preproc.get_uri(), brain_mask_bids.get_uri(relative=False)]
    else:
        logger.info("No brain mask found, generating one with antspynet")
        brain_mask_path = ants_helpers.deep_brain_extraction(t1w_bids_preproc.get_path(), work_dir, brain_mask_method)
        brain_mask_metadata = {'Type': 'Brain', 'Sources': [t1w_bids_preproc.get_uri()],
                               'BrainMaskMethod': f"antspynet {brain_mask_method}"}
        brain_mask_reslice = brain_mask_path # mask is already in the preprocessed space

    if brain_mask_reslice is None:
        # Relice an existing mask into the preprocessed space
        brain_mask_reslice = ants_helpers.reslice_to_reference(t1w_bids_preproc.get_path(), brain_mask_path, work_dir)
        # relative path for the new bids image

    brain_mask_output_rel_path = t1w_bids_preproc.get_derivative_rel_path_prefix() + '_desc-brain_mask.nii.gz'

    return bids_helpers.image_to_bids(brain_mask_reslice, t1w_bids_preproc.get_ds_path(), brain_mask_output_rel_path,
                                      metadata=brain_mask_metadata)


def get_template_segmentation_priors(t1w_bids_preproc, t1w_brain_mask_bids, template, template_brain_mask, work_dir,
                                     quick_reg=False):
    """Get segmentation priors for a T1w image using the template, as in the old antsCorticalThickness.sh pipeline.

    This function registers the brain-extracted image to a group template, and then warps the priors from the template to the
    session space. This can be used to provide priors similar to the old antsCorticalThickness.sh pipeline. It is much slower
    than using deep_atropos.

    Parameters:
    -----------
    t1w_bids_preproc : BIDSImage
        T1w image object, should be the preprocessed T1w image in the output dataset.
    t1w_brain_mask_bids : BIDSImage
        Brain mask for the preprocessed T1w image.
    template : bids_helpers.TemplateImage
        Template to use for registration and priors. The template must have priors defined for the six classes used in antsct.
    template_brain_mask : bids_helpers.TemplateImage
        Brain mask for the template.
    work_dir : str
        Path to the working directory.
    quick_reg : bool, optional
        If true, use a quick registration to the template. Default is False.

    Returns:
    --------
    dict with keys:
        prior_seg_probabilities : list
            List of segmentation posterior images, generated in or resliced to the space of the t1w_bids_preproc image.
        prior_metadata : dict
            Metadata for the segmentation priors. This includes the sources of the priors, and any processing steps applied.

    """
    logger.info("Generating segmentation priors from template")

    template_brain = ants_helpers.apply_mask(template.get_path(), template_brain_mask.get_path(), work_dir)

    t1w_brain = ants_helpers.apply_mask(t1w_bids_preproc.get_path(), t1w_brain_mask_bids.get_path(), work_dir)

    t1w_brain_n4 = ants_helpers.n4_bias_correction(t1w_brain, t1w_brain_mask_bids.get_path(), work_dir)

    logger.info("Registering T1w brain to template")

    reg_mask = ants_helpers.binary_morphology(template_brain_mask.get_path(), work_dir, operation='dilate', radius=12)

    reg = _pairwise_brain_registration(template_brain, t1w_brain_n4, quick_reg, work_dir, fixed_mask=reg_mask)

    # Warp the priors from the template to the session space
    seg_class_labels = ['CSF', 'CGM', 'WM', 'SGM', 'BS', 'CBM']

    priors_session_space = list()

    prior_sources = list()

    prior_metadata = { 'PriorGenerationMethod' : 'atlas' }

    for seg_class in seg_class_labels:
        try:
            prior_template = bids_helpers.TemplateImage(name=template.get_name(), resolution=template.get_resolution(),
                                                        cohort=template.get_cohort(), suffix='probseg', label=seg_class)
        except Exception:
            # templateflow doesn't throw anything more specific if it doesn't find the image
            # templateflow does not use BIDS common derived labels, so we have to catch this
            if (seg_class == 'SGM'):
                logger.warning(f"Template {template.get_name()} does not have a segmentation for SGM, trying SCGM")
                prior_template = bids_helpers.TemplateImage(name=template.get_name(), resolution=template.get_resolution(),
                                                            cohort=template.get_cohort(), suffix='probseg', label='SCGM')
            elif (seg_class == 'CGM'):
                logger.warning(f"Template {template.get_name()} does not have a segmentation for CGM, trying GM")
                prior_template = bids_helpers.TemplateImage(name=template.get_name(), resolution=template.get_resolution(),
                                                            cohort=template.get_cohort(), suffix='probseg', label='GM')

        prior_sources.append(prior_template.get_uri())
        priors_session_space.append(ants_helpers.apply_transforms(t1w_brain, prior_template.get_path(),
                                                                  reg['inverse_transform'], work_dir,
                                                                  interpolation='Gaussian'))

    prior_metadata['Sources'] = prior_sources

    prior_dict = { 'prior_seg_probabilities' : priors_session_space, 'prior_metadata' : prior_metadata }

    return prior_dict


def get_segmentation_priors(t1w_bids, t1w_bids_preproc, work_dir, segmentation_dataset=None,
                            antsnet_csf_prior_gamma=0, anstnet_prior_smoothing_sigma=0, use_legacy_deep_atropos=False):
    """Get segmentation priors for a T1w image

    If segmentatation_dataset is not None, it is searched for segmentation probabilities based on the T1w image. It is an error
    if the dataset does not contain these files for the T1w image. The segmentation posteriors must be defined for the six
    classes used in antsct: CSF, CGM, WM, SGM, BS, CBM. If a pre-existing segmentation is found, it can either be used directly
    or used as priors for iterative segmentation and bias correction with N4 and Atropos.

    If no pre-existing segmentation is found, priors are generated with anyspynet, and either copied directly or used for
    segmentation and bias correction with N4 and Atropos.

    Parameters:
    -----------
    t1w_bids : BIDSImage
        T1w image from the input dataset. This is used to search for segmentation posteriors in the segmentation dataset.
    t1w_bids_preproc : BIDSImage
        T1w image object, should be the preprocessed T1w image in the output dataset.
    work_dir : str
        Path to the working directory.
    segmentation_dataset : str, optional
        Path to the segmentation dataset for priors. If provided, it is an error to not find a segmentation. If None,
        segmentation priors are generated with antspynet.
    antsnet_csf_prior_gamma : float, optional
        Gamma correction for the antspynet CSF prior. Applied before smoothing. Default is 0, meaning no correction.
    anstnet_prior_smoothing_sigma : float, optional
        Sigma for smoothing the antspynet priors, in voxels. Default is 0, meaning no smoothing.
    use_legacy_deep_atropos : bool, optional
        If true, use the legacy deep_atropos network. Default is False.

    Returns:
    --------
    dict with keys:
        prior_seg_probabilities : list
            List of segmentation posterior images, generated in or resliced to the space of the t1w_bids_preproc image.
        prior_metadata : dict
            Metadata for the segmentation priors. This includes the sources of the priors, and any processing steps applied.

    Raises:
    -------
    PipelineError
        If the segmentation dataset is not None and does not contain segmentation posteriors for the specified T1w image.
    """
    prior_seg_probabilities = None

    prior_metadata = dict()

    # If a segmentation dataset is defined, it is an error to not find a segmentation
    if segmentation_dataset is not None:
        prior_seg_probabilities_bids = \
            bids_helpers.find_segmentation_probability_images(segmentation_dataset, t1w_bids)
        if prior_seg_probabilities_bids is None:
            raise PipelineError('Segmentation dataset does not contain a segmentation for ' + t1w_bids.get_uri())
        # Double check we have all the posteriors
        if len(prior_seg_probabilities_bids) != 6:
            raise PipelineError('Segmentation dataset does not contain all six posteriors for ' +
                             t1w_bids.get_uri())
        logger.info("Using segmentation priors:\n" +
                    '\n'.join([ f"  {prior_seg_probabilities_bids[i].get_uri()}" for i in range(6) ]))
        # reslice the posteriors to the preprocessed space
        prior_seg_probabilities = [ants_helpers.reslice_to_reference(t1w_bids_preproc.get_path(), prob.get_path(), work_dir)
                                for prob in prior_seg_probabilities_bids]
        # Add external priors as sources for the segmentation
        prior_metadata['Sources'] = [prob.get_uri(relative=False) for prob in prior_seg_probabilities_bids]
        prior_metadata['PriorGenerationMethod'] = 'external'

    else:
        # If no segmentation is found, generate one with antspynet
        logger.info("No segmentation priors found, generating with antspynet")
        antsnet_seg = ants_helpers.deep_atropos(t1w_bids_preproc.get_path(), work_dir,
                                                use_legacy_network=use_legacy_deep_atropos)
        prior_seg_probabilities = antsnet_seg['posteriors']
        # don't add sources because these priors are not saved
        prior_metadata['PriorGenerationMethod'] = 'deep_atropos'

        if antsnet_csf_prior_gamma > 0:
            logger.info(f"Gamma correction on CSF prior with gamma {antsnet_csf_prior_gamma}")
            prior_seg_probabilities[0] = ants_helpers.gamma_correction(prior_seg_probabilities[0], antsnet_csf_prior_gamma,
                                                                          work_dir)
            prior_metadata['PriorCSFGamma'] = antsnet_csf_prior_gamma

        if anstnet_prior_smoothing_sigma > 0:
            logger.info(f"Smoothing priors with sigma {anstnet_prior_smoothing_sigma}")
            priors_smooth = [ants_helpers.smooth_image(prior, anstnet_prior_smoothing_sigma, work_dir)
                             for prior in prior_seg_probabilities]
            prior_seg_probabilities = priors_smooth
            prior_metadata['PriorSmoothingSigma'] = anstnet_prior_smoothing_sigma

    prior_dict = { 'prior_seg_probabilities' : prior_seg_probabilities, 'prior_metadata' : prior_metadata }
    return prior_dict


def segment_and_bias_correct(t1w_bids_preproc, brain_mask_bids, segmentation_priors, work_dir, prior_metadata=None,
                             do_atropos_n4=True, atropos_n4_iterations=2, atropos_prior_weight=0.25, denoise=True,
                             n4_spline_spacing=180, n4_convergence='[ 50x50x50x50,1e-7 ]', n4_shrink_factor=3):
    """Segment and bias correct a T1w image

    If do_atropos_n4 is True, the priors are used to iteratively refine the bias correction and segmentation
    using `antsAtroposN4.sh`. Otherwise, the prior segmentation (whether from another dataset, or generated with deep_atropos)
    is copied, and the T1w image is bias corrected with N4.

    Parameters:
    -----------
    t1w_bids_preproc : BIDSImage
        T1w image object, should be the preprocessed T1w image in the output dataset. Output is in the space of this image.
    brain_mask_bids : BIDSImage
        Brain mask for the preprocessed T1w image.
    segmentation_priors : list
        List of files containing segmentation probabilities, in their antsct order: CSF, CGM, WM, SGM, BS, CBM. These are used
        as priors for segmentation and bias correction. Note these are just a list of file names, not BIDSImage objects.
    prior_metadata : dict, optional
        Metadata for the segmentation priors. This includes the sources of the priors, and any processing steps applied.
    work_dir : str
        Path to the working directory.
    do_atropos_n4 : bool, optional
        If true, use the ANTs antsAtroposN4.sh script to defined the segmentation and bias correction. If false, the priors
        are used as the posteriors for the segmentation, and the T1w image is bias-corrected using the priors.
    atropos_n4_iterations : int, optional
        Number of iterations for antsAtroposN4.sh.
    atropos_prior_weight : float, optional
        Prior weight for Atropos. Minimum useful value is 0.2, below this the priors are not very well constrained and you will
        see labels that overlap in intensity (like SGM and CBM) appear in the wrong places.

    Returns:
    --------
    dict
        Dictionary of segmentation images (as BIDSImage objects) with keys:
        'bias_corrected_t1w' - bias-corrected T1w image
        'segmentation_image' - the segmentation image
        'posteriors' - list of segmentation posteriors

    """
    # dict to be populated by ants_atropos_n4 or by using the priors directly
    seg_output = {}

    if do_atropos_n4:
        logger.info("Running Atropos with N4 bias correction")
        # Run antsAtroposN4.sh, using the priors for segmentation and bias correction
        seg_output = ants_helpers.segment_and_bias_correct(t1w_bids_preproc.get_path(), brain_mask_bids.get_path(),
                                                           segmentation_priors, work_dir, iterations=atropos_n4_iterations,
                                                           prior_weight=atropos_prior_weight, denoise=denoise,
                                                           n4_spline_spacing=n4_spline_spacing, n4_convergence=n4_convergence,
                                                           n4_shrink_factor=n4_shrink_factor)

        # remap the segmentation posteriors to BIDS labels
        seg_output['segmentation_image'] = ants_helpers.posteriors_to_segmentation(seg_output['posteriors'], work_dir)
    else:
        logger.info("Generating final segmentation directly from priors")
        posteriors_masked = [ants_helpers.apply_mask(posterior, brain_mask_bids.get_path(), work_dir)
                                for posterior in segmentation_priors]

        # remap the prior posteriors to BIDS labels
        seg_output['segmentation_image'] = ants_helpers.posteriors_to_segmentation(posteriors_masked, work_dir)
        seg_output['posteriors'] = posteriors_masked

        # Remove outliers from the image
        logger.info("Removing outliers from the T1w image")
        t1w_winsorized = ants_helpers.winsorize_intensity(t1w_bids_preproc.get_path(), brain_mask_bids.get_path(), work_dir)

        # Denoise and then bias correct the T1w image
        if denoise:
            logger.info("Denoising T1w image")
            denoised_t1w_image = ants_helpers.denoise_image(t1w_winsorized, work_dir)
        else:
            denoised_t1w_image = t1w_winsorized

        logger.info("Bias correcting T1w image")
        denoised_n4_t1w = ants_helpers.n4_bias_correction(denoised_t1w_image, brain_mask_bids.get_path(), work_dir,
                                                          segmentation_posteriors=posteriors_masked,
                                                          n4_spline_spacing=n4_spline_spacing, n4_convergence=n4_convergence,
                                                          n4_shrink_factor=n4_shrink_factor)
        normalized_n4_t1w = ants_helpers.normalize_intensity(denoised_n4_t1w, seg_output['segmentation_image'], work_dir,
                                                             label=2)

        seg_output['bias_corrected_anatomical_images'] = [normalized_n4_t1w]


    # Copy the segmentation outputs to the output dataset
    seg_output_bids = {}

    seg_output_bids['posteriors'] = list()

    seg_posterior_labels = ['CSF', 'CGM', 'WM', 'SGM', 'BS', 'CBM']

    seg_metadata = dict()

    if prior_metadata is not None:
        seg_metadata = copy.deepcopy(prior_metadata)

    if 'Sources' not in seg_metadata:
        seg_metadata['Sources'] = list()

    seg_metadata['Sources'].extend([t1w_bids_preproc.get_uri(), brain_mask_bids.get_uri()])
    seg_metadata['SegmentationMethod'] = 'antsAtroposN4' if do_atropos_n4 else 'prior_maxprob'

    for idx, seg_label in enumerate(seg_posterior_labels):
        seg_output_bids['posteriors'].append(
            bids_helpers.image_to_bids(seg_output['posteriors'][idx], t1w_bids_preproc.get_ds_path(),
                                       t1w_bids_preproc.get_derivative_rel_path_prefix() +
                                       f"_seg-antsnetct_label-{seg_label}_probseg.nii.gz",
                                       metadata=seg_metadata))

    seg_output_bids['segmentation_image'] = \
        bids_helpers.image_to_bids(seg_output['segmentation_image'], t1w_bids_preproc.get_ds_path(),
                                   t1w_bids_preproc.get_derivative_rel_path_prefix() + '_seg-antsnetct_dseg.nii.gz',
                                   metadata=seg_metadata)

    bias_corrected_t1w = bids_helpers.image_to_bids(seg_output['bias_corrected_anatomical_images'][0],
                                                                       t1w_bids_preproc.get_ds_path(),
                                                                       t1w_bids_preproc.get_derivative_rel_path_prefix() +
                                                                       "_desc-biascorr_T1w.nii.gz",
                                                                       metadata = {'Sources': [t1w_bids_preproc.get_uri()],
                                                                                   'SkullStripped': False})

    seg_output_bids['bias_corrected_t1w'] = bias_corrected_t1w

    # Write a brain-masked T1w image to the output - this will also be used for template registration
    masked_t1w_image = ants_helpers.apply_mask(bias_corrected_t1w.get_path(), brain_mask_bids.get_path(), work_dir)
    masked_t1w_metadata = {'SkullStripped': True, 'Sources': [bias_corrected_t1w.get_uri(), brain_mask_bids.get_uri()]}
    seg_output_bids['bias_corrected_t1w_brain'] = \
        bids_helpers.image_to_bids(masked_t1w_image, bias_corrected_t1w.get_ds_path(),
                                   bias_corrected_t1w.get_derivative_rel_path_prefix() +
                                   '_desc-biascorrbrain_T1w.nii.gz', metadata=masked_t1w_metadata)

    return seg_output_bids


def cortical_thickness(seg_n4, work_dir, thickness_iterations=45):
    """Calculate cortical thickness

    Parameters:
    -----------
    seg_n4_bids (dict):
        Dictionary of segmentation images (as BIDSImage objects) as returned from ants_helpers.ants_atropos_n4.
        The thickness image will be written to the same dataset as the segmentation image.
    work_dir (str):
        Path to the working directory.
    thickness_iterations (int, optional):
        Number of iterations for cortical thickness estimation. Default is 45.

    Returns:
    --------
    BIDSImage
        The cortical thickness image in the output dataset.

    """
    posterior_files = [posterior.get_path() for posterior in seg_n4['posteriors']]

    logger.info("Calculating cortical thickness on segmentation image: " + seg_n4['segmentation_image'].get_uri(relative=False))
    logger.info("Calculating cortical thickness with " + str(thickness_iterations) + " iterations")

    thickness = ants_helpers.cortical_thickness(seg_n4['segmentation_image'].get_path(), posterior_files, work_dir,
                                                kk_its=thickness_iterations)

    thick_sources = [seg_n4['segmentation_image'].get_uri()]
    thick_sources.extend([posterior.get_uri() for posterior in seg_n4['posteriors']])

    thickness_metadata = {'Sources': thick_sources}

    thickness_bids = bids_helpers.image_to_bids(thickness, seg_n4['segmentation_image'].get_ds_path(),
                                                seg_n4['segmentation_image'].get_derivative_rel_path_prefix() +
                                                '_desc-thickness.nii.gz', metadata=thickness_metadata)

    return thickness_bids


def template_brain_registration(template, template_brain_mask, t1w_brain_image, quick_reg, work_dir, use_reg_mask=True):
    """Register a brain-extracted moving image to a template

    Parameters:
    -----------
    template : bids_helpers.TemplateImage
        Template image. This will be brain-masked and used as the fixed image.
    template_brain_mask : bids_helpers.TemplateImage
        Brain mask for the template. This will be used to mask the template image.
    t1w_brain_image : BIDSImage
        Moving T1w brain image to register to the template.
    quick_reg : bool
        Do quick registration to the template.
    work_dir : str
        Path to the working directory.
    use_reg_mask : bool, optional
        Use dilated brain mask in the template space to speed up registration.

    Output is to the same dataset as the t1w_brain_image.

    Returns:
    --------
    dict
        A dictionary with keys:
            'forward_transform' - path to the forward transform in the output dataset
            'inverse_transform' - path to the inverse transform in the output dataset
    """

    fixed_image = ants_helpers.apply_mask(template.get_path(), template_brain_mask.get_path(), work_dir)

    fixed_mask = None

    if use_reg_mask:
        fixed_mask = ants_helpers.binary_morphology(template_brain_mask.get_path(), work_dir, operation='dilate', radius=12)

    if quick_reg:
        logger.info("Quick registration to template " + template.get_name())
    else:
        logger.info("Registration to template " + template.get_name())

    template_reg = _pairwise_brain_registration(fixed_image, t1w_brain_image.get_path(), quick_reg, work_dir,
                                                fixed_mask=fixed_mask)

    template_reg_bids = {}

    # Note warps do not contain res- because they may be used to produce output at any res- in the same space
    forward_transform_path = t1w_brain_image.get_derivative_path_prefix() + \
        f"_from-T1w_to-{template.get_name()}_mode-image_xfm.h5"

    system_helpers.copy_file(template_reg['forward_transform'], forward_transform_path)

    template_reg_bids['forward_transform'] = forward_transform_path

    inverse_transform_path = t1w_brain_image.get_derivative_path_prefix() + \
        f"_from-{template.get_name()}_to-T1w_mode-image_xfm.h5"

    system_helpers.copy_file(template_reg['inverse_transform'], inverse_transform_path)

    template_reg_bids['inverse_transform'] = inverse_transform_path

    return template_reg_bids


def _pairwise_brain_registration(fixed, moving, quick_reg, work_dir, fixed_mask=None, moving_mask=None):
    """Register two brain images. Similar to template_brain_registration, but does not produce BIDS information

    Parameters:
    -----------
    fixed : str
        Path to the fixed image.
    moving : str
        Path to the moving image.
    quick_reg : bool
        Do quick registration.
    work_dir : str
        Path to the working directory.
    fixed_mask : str, optional
        Path to the fixed registration mask.
    moving_mask : str, optional
        Path to the moving registration mask.

    Returns:
    --------
    dict
        A dictionary with keys:
            'forward_transform' - path to the forward transform in the output dataset
            'inverse_transform' - path to the inverse transform in the output dataset
    """
    if quick_reg:
        template_reg = ants_helpers.univariate_template_registration(fixed, moving, work_dir,
                                                                     metric='Mattes', metric_param_str='32',
                                                                     fixed_mask=fixed_mask, moving_mask=moving_mask,
                                                                     transform='SyN[0.25,3,0]', iterations='40x40x60x40x0',
                                                                     shrink_factors='6x5x4x2x1',
                                                                     smoothing_sigmas='4x3x2x1x0vox', apply_transforms=False)
    else:
        template_reg = ants_helpers.univariate_template_registration(fixed, moving, work_dir,
                                                                     metric='CC', metric_param_str='2', fixed_mask=fixed_mask,
                                                                     moving_mask=moving_mask, transform='SyN[0.2,3,0]',
                                                                     iterations='30x30x70x70x20', shrink_factors='8x6x4x2x1',
                                                                     smoothing_sigmas='4x3x2x1x0vox', apply_transforms=False)

    return template_reg


def template_space_derivatives(template, template_reg, seg_n4, thickness, work_dir):
    """Make template space derivatives: thickness, jacobian, GM probability

    Parameters:
    -----------
    template : bids_helpers.TemplateImage
        Reference space for derivatives
    template_reg : dict
        Dictionary of template registration outputs as returned from template_brain_registration
    seg_n4 : dict
        Dictionary of segmentation images as returned from segment_and_bias_correct
    thickness : BIDSImage
        Cortical thickness image as returned from cortical_thickness

    Returns:
    --------
    dict
        Dictionary of template space derivatives as BIDSImage objects with keys:
            'thickness' - cortical thickness image
            'jacobian' - jacobian log determinant image
            'gmp' - grey matter probability image
            't1w_brain' - bias-corrected brain image
    """
    source_image_bids = seg_n4['bias_corrected_t1w_brain']

    transform = template_reg['forward_transform']

    # Make the thickness image in the template space
    thickness_template_space = ants_helpers.apply_transforms(template.get_path(), thickness.get_path(), transform, work_dir)

    template_space_bids = {}

    template_space_bids['thickness'] = bids_helpers.image_to_bids(thickness_template_space, source_image_bids.get_ds_path(),
                                                                  source_image_bids.get_derivative_rel_path_prefix() + '_' +
                                                                  template.get_derivative_space_string() +
                                                                  '_desc-thickness.nii.gz')

    # Make the jacobian log determinant image in the template space
    jacobian_template_space = ants_helpers.get_log_jacobian_determinant(template.get_path(), transform, work_dir)

    template_space_bids['jacobian'] = bids_helpers.image_to_bids(jacobian_template_space, source_image_bids.get_ds_path(),
                                                                 source_image_bids.get_derivative_rel_path_prefix() + '_' +
                                                                 template.get_derivative_space_string() +
                                                                 '_desc-logjacobian.nii.gz')

    # gray matter probability
    gm_prob_template_space = ants_helpers.apply_transforms(template.get_path(), seg_n4['posteriors'][1].get_path(), transform,
                                                           work_dir)

    template_space_bids['gmp'] = bids_helpers.image_to_bids(gm_prob_template_space, source_image_bids.get_ds_path(),
                                                            source_image_bids.get_derivative_rel_path_prefix() + '_' +
                                                            template.get_derivative_space_string() +
                                                            '_label-CGM_probseg.nii.gz')

    # bias-corrected brain image
    bias_corrected_brain_template_space = ants_helpers.apply_transforms(template.get_path(),
                                                                        seg_n4['bias_corrected_t1w_brain'].get_path(),
                                                                        transform, work_dir)

    template_space_bids['t1w_brain'] = bids_helpers.image_to_bids(bias_corrected_brain_template_space,
                                                                  source_image_bids.get_ds_path(),
                                                                  source_image_bids.get_derivative_rel_path_prefix() + '_' +
                                                                  template.get_derivative_space_string() +
                                                                  '_desc-biascorrbrain_T1w.nii.gz')

    return template_space_bids


def make_qc_plots(t1w_bids, mask_bids, seg_bids, thick_bids, work_dir):
    """Generate tiled QC plots for a T1w image with segmentation and cortical thickness overlays

    Output is written as a derivative of the t1w_bids image.

    Parameters:
    -----------
    t1w_bids : BIDSImage
        T1w image object, should be the preprocessed T1w image in the output dataset.
    mask_bids : BIDSImage
        Brain mask for the preprocessed T1w image.
    seg_bids : BIDSImage
        Segmentation image in the output dataset.
    thick_bids : BIDSImage
        Cortical thickness image in the output dataset.
    work_dir : str
        Path to the working directory.
    """
    # Resample everything to 1mm so the PNG plots have a roughly consistent number of slices
    scalar_image = ants_helpers.resample_image_by_spacing(t1w_bids.get_path(), [1, 1, 1], work_dir)

    mask_image = ants_helpers.resample_image_by_spacing(mask_bids.get_path(), [1, 1, 1], work_dir,
                                                       interpolation='NearestNeighbor')
    seg_image = ants_helpers.resample_image_by_spacing(seg_bids.get_path(), [1, 1, 1], work_dir,
                                                       interpolation='NearestNeighbor')
    thick_image = ants_helpers.resample_image_by_spacing(thick_bids.get_path(), [1, 1, 1], work_dir)

    # winsorize a bit more aggressively to boost brightness of the brain
    scalar_image = ants_helpers.winsorize_intensity(scalar_image, mask_image, work_dir, lower_percentile=0.0,
                                                    upper_iqr_scale=1.5)

    seg_rgb = ants_helpers.convert_scalar_image_to_rgb(seg_image, work_dir, colormap='antsct')

    thick_rgb = ants_helpers.convert_scalar_image_to_rgb(thick_image, work_dir, colormap='hot', min_value=0, max_value=8)

    # Make a tiled image with the T1w, mask, segmentation, and cortical thickness
    tiled_seg_ax = ants_helpers.create_tiled_mosaic(scalar_image, mask_image, work_dir, overlay=seg_rgb, overlay_alpha=0.25,
                                                    axis=2, pad=('mask+5'), slice_spec=(3,'mask','mask'))
    tiled_seg_cor = ants_helpers.create_tiled_mosaic(scalar_image, mask_image, work_dir, overlay=seg_rgb, overlay_alpha=0.25,
                                                    axis=1, pad=('mask+5'), slice_spec=(3,'mask','mask'))
    system_helpers.copy_file(tiled_seg_ax, t1w_bids.get_derivative_path_prefix() + '_desc-segax_qc.png')
    system_helpers.copy_file(tiled_seg_cor, t1w_bids.get_derivative_path_prefix() + '_desc-segcor_qc.png')

    tiled_thick_ax = ants_helpers.create_tiled_mosaic(scalar_image, mask_image, work_dir, overlay=thick_rgb,
                                                      overlay_alpha=0.25, axis=2, pad=('mask+5'), slice_spec=(3,'mask','mask'))
    tiled_thick_cor = ants_helpers.create_tiled_mosaic(scalar_image, mask_image, work_dir, overlay=thick_rgb,
                                                       overlay_alpha=0.25, axis=1, pad=('mask+5'), slice_spec=(3,'mask','mask'))

    system_helpers.copy_file(tiled_thick_ax, t1w_bids.get_derivative_path_prefix() + '_desc-thickax_qc.png')
    system_helpers.copy_file(tiled_thick_cor, t1w_bids.get_derivative_path_prefix() + '_desc-thickcor_qc.png')


def compute_qc_stats(t1w_bids, mask_bids, seg_bids, thick_bids, work_dir):
    """Compute QC statistics for a T1w image, with segmentation and cortical thickness data

    Makes TSV file with some QC statistics for the T1w image, segmentation, and cortical thickness.

    Parameters:
    -----------
    t1w_bids : BIDSImage
        T1w image object, should be the preprocessed T1w image in the output dataset.
    mask_bids : BIDSImage
        Brain mask for the preprocessed T1w image.
    seg_bids : BIDSImage
        Segmentation image in the output dataset.
    thick_bids : BIDSImage
        Cortical thickness image in the output dataset.
    work_dir : str
        Path to the working directory.
    """
    # Read in the images
    t1w_image = ants_helpers.read_image(t1w_bids.get_path())
    mask_image = ants_helpers.read_image(mask_bids.get_path())
    seg_image = ants_helpers.read_image(seg_bids.get_path())
    thick_image = ants_helpers.read_image(thick_bids.get_path())

    # Compute some basic stats
    t1w_mean = t1w_image.mean()
    t1w_std = t1w_image.std()
    mask_vol = mask_image.sum() * np.prod(mask_image.spacing)
    seg_vols = [seg_image[seg_image == i].sum() * np.prod(seg_image.spacing) for i in range(1, 7)]
    thick_mean = thick_image.mean()
    thick_std = thick_image.std()

    # Write the stats to a TSV file
    with open(t1w_bids.get_derivative_path_prefix() + '_desc-qc.tsv', 'w') as f:
        f.write('metric\tvalue\n')
        f.write(f't1w_mean\t{t1w_mean}\n')
        f.write(f't1w_std\t{t1w_std}\n')
        f.write(f'brain_volume\t{mask_vol}\n')
        for i, vol in enumerate(seg_vols):
            f.write(f'segmentation_volume_{i+1}\t{vol}\n')
        f.write(f'thickness_mean\t{thick_mean}\n')
        f.write(f'thickness_std\t{thick_std}\n')


