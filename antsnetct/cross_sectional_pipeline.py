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

def cross_sectional_analysis():

    parser = argparse.ArgumentParser(formatter_class=RawDefaultsHelpFormatter, add_help = False,
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

    If a brain mask dataset or segmentation dataset are specified at run time, there must be masks or segmentations for the
    input data, or an error will be raised. This is to prevent inconsistent processing.


    --- Preprocessing ---

    All input images are conformed to LPI orientation with c3d.


    --- Brain masking ---

    The script will check for a brain mask from the brain mask dataset, if defined. If not, a brain mask may be found in the
    input dataset. If not, an implicit mask can be derived from segmentation input. If no brain mask is available, one will be
    generated using anyspynet.


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

    4. Cortical thickness estimation. If the number of iterations is greater than 0, cortical thickness is estimated.

    5. Atlas registration. If an atlas is defined, the T1w image is registered to the atlas.

    ''')

    required_parser = parser.add_argument_group('Required arguments')
    required_parser.add_argument("--input-dataset", help="Input BIDS dataset dir, containing the source images", type=str,
                          required=True)
    required_parser.add_argument("--participant", help="Participant to process", type=str, required=True)
    required_parser.add_argument("--session", help="Session to process", type=str, required=True)
    required_parser.add_argument("--output-dataset", help="Output BIDS dataset dir", type=str, required=True)

    optional_parser = parser.add_argument_group('General optional arguments')
    optional_parser.add_argument("-h", "--help", action="help", help="show this help message and exit")
    optional_parser.add_argument("--keep-workdir", help="Copy working directory to output, for debugging purposes. Either "
                                 "'never', 'on_error', or 'always'.", type=str, default='on_error')
    optional_parser.add_argument("--verbose", help="Verbose output from subcommands", action='store_true')

    neck_trim_parser = parser.add_argument_group('Pre-processing arguments')
    neck_trim_parser.add_argument("--no-neck-trim", help="Disable neck trimming from the T1w image", dest="do_neck_trim",
                                  action='store_false')
    neck_trim_parser.add_argument("--pad-mm", help="Padding in mm to add to the image before processing, this is mostly "
                                  "useful for registration with the skull on", type=int, default=10)

    template_parser = parser.add_argument_group('Template arguments')
    template_parser.add_argument("--template-name", help="Template to use for registration, or 'none' to disable this step.",
                                 type=str, default='MNI152NLin2009cAsym')
    template_parser.add_argument("--template-res", help="Resolution of the template, eg '01', '02', etc. Note this is a "
                                 "templateflow index and not a physical spacing. If the selected template does not define "
                                 "multiple resolutions, this is ignored.", type=str, default='01')
    template_parser.add_argument("--template-cohort", help="Template cohort, only needed for templates that define multiple "
                                 "cohorts", type=str, default=None)
    template_parser.add_argument("--template-reg-quick", help="Do quick registration to the template", action='store_true')

    brain_mask_parser = parser.add_argument_group('Brain mask arguments')
    brain_mask_parser.add_argument("--brain-mask-dataset", help="Dataset containing brain masks. Masks from here will be used "
                                   "in preference to those in the input dataset.", type=str, default=None)
    brain_mask_parser.add_argument("--brain-mask-modality", help="Brain masking modality option to use with antspynet. Only "
                                   "used if no pre-existing mask is found. Options are 't1', 't1nobrainer', 't1combined'",
                                   type=str, default='t1')
    segmentation_parser = parser.add_argument_group('Segmentation arguments')
    segmentation_parser.add_argument("--segmentation-dataset", help="Dataset containing segmentations. This dataset can be "
                                     "used for priors, or as a replacement for the built-in segmentation routines.",
                                     type=str, default=None)
    segmentation_parser.add_argument("--segmentation-method", help="Segmentation method to use. Either 'atropos' or "
                                     "'none'. If atropos, probseg images from the segmentation dataset, if defined, or from "
                                     "deep_atropos will be used as priors for segmentation and bias correction. If no "
                                     "segmentation dataset is provided, a segmentation will be generated by antspynet.",
                                     type=str, default='atropos')
    segmentation_parser.add_argument("--atropos-n4-iterations", help="Number of iterations for atropos-n4",
                                     type=int, default=5)
    segmentation_parser.add_argument("--atropos-prior-weight", help="Prior weight for Atropos", type=float, default=0.25)

    thickness_parser = parser.add_argument_group('Thickness arguments')
    thickness_parser.add_argument("--thickness-iterations", help="Number of iterations for cortical thickness estimation. "
                                  "Set to 0 to skip thickness calculation", type=int, default=45)

    args = parser.parse_args()

    # setup templateflow
    if args.template_name.lower() != 'none':
        if not 'TEMPLATEFLOW_HOME' in os.environ or not os.path.exists(os.environ.get('TEMPLATEFLOW_HOME')):
            raise PipelineError(f"templateflow directory not found at " +
                                f"TEMPLATEFLOW_HOME={os.environ.get('TEMPLATEFLOW_HOME')}")

    system_helpers.set_verbose(args.verbose)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    input_dataset = args.input_dataset
    output_dataset = args.output_dataset

    if (input_dataset == output_dataset):
        raise ValueError('Input and output datasets cannot be the same')

    input_dataset_description = None

    if os.path.exists(os.path.join(input_dataset, 'dataset_description.json')):
        with open(os.path.join(input_dataset, 'dataset_description.json'), 'r') as f:
            input_dataset_description = json.load(f)
    else:
        raise ValueError('Input dataset does not contain a dataset_description.json file')

    # There might be multiple T1ws, if so we process them all
    input_t1w_images = None

    if args.participant is None:
        raise ValueError('Participant must be defined')
    if args.session is None:
        raise ValueError('Session must be defined')

    logger.info("Input dataset path: " + input_dataset)
    logger.info("Input dataset name: " + input_dataset_description['Name'])

    # Create the output dataset and add this container to the GeneratedBy, if needed
    bids_helpers.update_output_dataset(output_dataset, input_dataset_description['Name'] + '_antsnetct')

    with open(os.path.join(output_dataset, 'dataset_description.json'), 'r') as f:
        output_dataset_description = json.load(f)

    logger.info("Output dataset path: " + output_dataset)
    logger.info("Output dataset name: " + output_dataset_description['Name'])

    # Returns a list of T1w images and URIs
    input_t1w_bids = bids_helpers.find_images(input_dataset, args.participant, args.session, 'anat', 'T1w')

    if input_t1w_bids is None or len(input_t1w_bids) == 0:
        logger.error(f"No T1w images found for participant {args.participant}, session {args.session}")
        return

    for t1w_bids in input_t1w_bids:

        with tempfile.TemporaryDirectory(
                suffix=f"antsnetct_{system_helpers.get_nifti_file_prefix(t1w_bids.get_path())}.tmpdir") as working_dir:
            try:

                logger.info("Processing T1w image: " + t1w_bids.get_uri())

                # check completeness
                if os.path.exists(os.path.join(output_dataset, t1w_bids.get_derivative_rel_path_prefix() +
                                               '_seg-antsnetct_desc-thickness.nii.gz')):
                    logger.info(f"Skipping {str(t1w_bids)}, cortical thickness already exists")
                    continue

                # Preprocessing
                # Conform to LPI orientation
                preproc_t1w_bids = preprocess_t1w(t1w_bids, output_dataset, working_dir, orient='LPI',
                                                  trim_neck=args.do_neck_trim, pad=args.pad_mm)

                # Find a brain mask using the first available in order of preference:
                # 1. Brain mask dataset
                # 2. Brain mask in input dataset
                # 3. Generate a brain mask with antspynet
                brain_mask_bids = get_brain_mask(t1w_bids, preproc_t1w_bids, working_dir, args.brain_mask_dataset,
                                                args.brain_mask_modality)

                # With mask defined, we can now do segmentation and bias correction
                # Segmentation is either pre-defined, or generated with antspynet. Either an external segmentation or an
                # antspynet segmentation can be used as priors for iterative segmentation and bias correction with N4 and
                # Atropos. If Atropos is not used, the T1w is bias-corrected separately with N4.
                seg_n4 = segment_and_bias_correct(t1w_bids, preproc_t1w_bids, brain_mask_bids, working_dir,
                                                  args.segmentation_dataset, args.segmentation_method,
                                                  args.atropos_n4_iterations)

                # If thickness is requested, calculate it
                if args.thickness_iterations > 0:
                    thickness = cortical_thickness(seg_n4, working_dir, args.thickness_iterations)

                # If an atlas is defined, register the T1w image to the atlas
                if args.template_name.lower() != 'none':

                    template = bids_helpers.TemplateImage(args.template_name, suffix='T1w', description=None,
                                                          resolution=args.template_res, cohort=args.template_cohort)

                    template_brain_mask = bids_helpers.TemplateImage(args.template_name, suffix='mask', description='brain',
                                                                     resolution=args.template_res, cohort=args.template_cohort)

                    template_reg = template_brain_registration(template, template_brain_mask,
                                                               seg_n4['bias_corrected_t1w_brain'], args.template_reg_quick,
                                                               working_dir)
                    # Make template space derivatives: thickness, jacobian, GM probability, t1w brain
                    template_space_derivatives(template, template_reg, seg_n4, thickness, working_dir)

                if args.keep_workdir.lower() == 'always':
                    logger.info("Keeping working directory: " + working_dir)
                    shutil.copytree(working_dir, os.path.join(output_dataset, preproc_t1w_bids.get_derivative_rel_path_prefix()
                                                              + "_workdir"))

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
        preproc_t1w = preprocessing.trim_neck(preproc_t1w, work_dir)

    preproc_t1w = preprocessing.pad_image(preproc_t1w, work_dir, pad_mm=pad)

    # Copy the preprocessed T1w to the output
    preproc_t1w_bids = bids_helpers.image_to_bids(preproc_t1w, output_dataset, t1w_bids.get_derivative_rel_path_prefix() +
                                                  '_desc-preproc_T1w.nii.gz',
                                                  metadata={'Sources': [t1w_bids.get_uri()], 'SkullStripped': False})

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

    brain_mask_path = None
    brain_mask_metadata = None
    brain_mask_reslice = None

    found_brain_mask = False

    # If a mask dataset is defined, it is an error to not find a mask
    if brain_mask_dataset is not None:
        brain_mask = bids_helpers.find_brain_mask(brain_mask_dataset, t1w_bids)
        if brain_mask is None:
            raise ValueError('Brain mask dataset does not contain a brain mask for ' + str(t1w_bids))
        # Found a brain mask
        logger.info("Using brain mask: " + str(brain_mask))
        found_brain_mask = True
        brain_mask_path = brain_mask.get_path()
        brain_mask_metadata = brain_mask.get_metadata()

    # If no mask dataset is defined, try to find a mask in the input dataset
    if not found_brain_mask:
        brain_mask = bids_helpers.find_brain_mask(t1w_bids.get_ds_path(), t1w_bids)
        if brain_mask is not None:
            # Found a brain mask in input dataset
            logger.info("Using brain mask: " + brain_mask)
            found_brain_mask = True
            brain_mask_path = brain_mask.get_path()
            brain_mask_metadata = brain_mask.get_metadata()

    if not found_brain_mask:
        logger.info("No brain mask found, generating one with antspynet")
        brain_mask_path = ants_helpers.deep_brain_extraction(t1w_bids_preproc.get_path(), work_dir, brain_mask_method)
        brain_mask_metadata = {'Type': 'Brain', 'Sources': [t1w_bids_preproc.get_uri()]}
        brain_mask_reslice = brain_mask_path # mask is already in the preprocessed space

    if brain_mask_reslice is None:
        # Relice an existing mask into the preprocessed space
        brain_mask_reslice = ants_helpers.reslice_to_reference(t1w_bids_preproc.get_path(), brain_mask_path, work_dir)

    brain_mask_reslice_rel_path = t1w_bids_preproc.get_derivative_rel_path_prefix() + '_desc-brain_mask.nii.gz'

    brain_mask_metadata['Sources'] = [t1w_bids_preproc.get_uri()]

    return bids_helpers.image_to_bids(brain_mask_reslice, t1w_bids_preproc.get_ds_path(), brain_mask_reslice_rel_path,
                                      metadata=brain_mask_metadata)


def segment_and_bias_correct(t1w_bids, t1w_bids_preproc, brain_mask_bids, work_dir, segmentation_dataset=None,
                             segmentation_method='atropos', atropos_n4_iterations=5):
    """Segment and bias correct a T1w image

    If segmentatation_dataset is not None, it is searched for segmenation probabilities based on the T1w image. It is an error
    if the dataset does not contain these files for the T1w image. The segmentation posteriors must be defined for the six
    classes used in antsct: CSF, CGM, WM, SGM, BS, CBM. If a pre-existing segmentation is found, it can either be used directly
    or used as priors for iterative segmentation and bias correction with N4 and Atropos.

    If no pre-existing segmentation is found, priors are generated with anyspynet, and either copied directly or used for
    segmentation and bias correction with N4 and Atropos.

    If the segmentation_method is 'none', the prior segmentation (whether from another dataset, or generated with deep_atropos)
    is copied, and the T1w image is bias corrected with N4.

    If the segmentation_method is 'atropos', the priors are used to iteratively refine the bias correction and segmentation
    using `antsAtroposN4.sh`.

    All priors are masked by the provided brain_mask_image before running Atropos. If not running Atropos, the segmentations
    may be defined in the domain of some other mask, but will be masked by the brain mask before being copied to the output.

    Parameters:
    -----------
    t1w_bids : BIDSImage
        T1w image from the input dataset. This is used to search for segmentation posteriors in the segmentation dataset.
    t1w_bids_preproc : BIDSImage
        T1w image object, should be the preprocessed T1w image in the output dataset. Output is in the space of this image.
    brain_mask_bids : BIDSImage
        Brain mask for the preprocessed T1w image.
    work_dir : str
        Path to the working directory.
    segmentation_dataset : str, optional
        Path to the segmentation dataset for priors.
    segmentation method : str, optional
        Method to use for segmentation. Default is 'atropos', meaning the priors are used for iterative segmentation and bias
        correction with antsAtroposN4.sh. If 'none', the priors (either from the segmentation dataset or ANTsPyNet) are used
        as the posteriors, and the T1w image is bias corrected with N4.
    atropos_n4_iterations : int, optional
        Number of iterations for antsAtroposN4.sh. Default is 5.

    Returns:
    --------
    dict
        Dictionary of segmentation images (as BIDSImage objects) with keys:
        'bias_corrected_t1w' - bias-corrected T1w image
        'segmentation_image' - the segmentation image
        'segmentation_posteriors' - list of segmentation posteriors

    Raises:
    -------
    ValueError
        If the brain mask dataset is not None and does not contain a brain mask for the specified T1w image.
    """
    prior_seg_probabilities = None

    # If a segmentation dataset is defined, it is an error to not find a segmentation
    if segmentation_dataset is not None:
        prior_seg_probabilities_bids = \
            bids_helpers.find_segmentation_probability_images(segmentation_dataset, t1w_bids)
        if prior_seg_probabilities_bids is None:
            raise ValueError('Segmentation dataset does not contain a segmentation for ' + t1w_bids.get_uri())
        # Double check we have all the posteriors
        if len(prior_seg_probabilities_bids) != 6:
            raise ValueError('Segmentation dataset does not contain all six posteriors for ' +
                             t1w_bids.get_uri())
        logger.info("Using segmentation priors:\n" +
                    '\n'.join([ f"  {prior_seg_probabilities_bids[i].get_uri()}" for i in range(6) ]))
        # reslice the posteriors to the preprocessed space
        prior_seg_probabilities = [ants_helpers.reslice_to_reference(t1w_bids_preproc.get_path(), prob.get_path(), work_dir)
                                for prob in prior_seg_probabilities_bids]
    else:
        # If no segmentation is found, generate one with antspynet
        logger.info("No segmentation found, generating one with antspynet")
        antsnet_seg = ants_helpers.deep_atropos(t1w_bids_preproc.get_path(), work_dir)
        prior_seg_probabilities = antsnet_seg['posteriors']

    # dict to be populated by ants_atropos_n4 or by using the priors directly
    seg_output = {}

    if segmentation_method.lower() == 'atropos':
        logger.info("Running Atropos with N4 bias correction")
        # Run antsAtroposN4.sh, using the priors for segmentation and bias correction
        seg_output = ants_helpers.ants_atropos_n4(t1w_bids_preproc.get_path(), brain_mask_bids.get_path(),
                                                         prior_seg_probabilities, work_dir, iterations=atropos_n4_iterations)
        seg_output['segmentation_image'] = ants_helpers.posteriors_to_segmentation(seg_output['posteriors'], work_dir)
    elif segmentation_method.lower() == 'none':
        logger.info("Segmentation method is none, generating final segmentation directly from priors")
        posteriors_masked = [ants_helpers.apply_mask(posterior, brain_mask_bids.get_path(), work_dir)
                                for posterior in prior_seg_probabilities]

        # Copy the prior segmentation
        segmentation_image = ants_helpers.posteriors_to_segmentation(posteriors_masked, work_dir)

        seg_output['segmentation_image'] = segmentation_image
        seg_output['posteriors'] = posteriors_masked

        # Add the bias corrected image
        seg_output['bias_corrected_anatomical_images'] = [
            ants_helpers.n4_bias_correction(t1w_bids_preproc.get_path(), brain_mask_bids.get_path(), posteriors_masked,
                                            work_dir)]
    else:
        raise ValueError('Unknown segmentation method: ' + segmentation_method)

    # Copy the segmentation outputs to the output dataset
    seg_output_bids = {}

    seg_output_bids['posteriors'] = list()

    seg_posterior_labels = ['CSF', 'CGM', 'WM', 'SGM', 'BS', 'CBM']

    for idx, seg_label in enumerate(seg_posterior_labels):
        seg_output_bids['posteriors'].append(
            bids_helpers.image_to_bids(seg_output['posteriors'][idx], t1w_bids_preproc.get_ds_path(),
                                       t1w_bids_preproc.get_derivative_rel_path_prefix() +
                                       f"_seg-antsnetct_label-{seg_posterior_labels[idx]}_probseg.nii.gz",
                                       metadata={'Sources': [t1w_bids_preproc.get_uri(), brain_mask_bids.get_uri()]})
        )

    seg_sources = [t1w_bids_preproc.get_uri(), brain_mask_bids.get_uri()]
    seg_sources.extend([prob.get_uri() for prob in seg_output_bids['posteriors']])

    seg_output_bids['segmentation_image'] = \
        bids_helpers.image_to_bids(seg_output['segmentation_image'], t1w_bids_preproc.get_ds_path(),
                                   t1w_bids_preproc.get_derivative_rel_path_prefix() + '_seg-antsnetct_dseg.nii.gz',
                                   metadata={'Sources': seg_sources})

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
    seg_n4_bids : dict
        Dictionary of segmentation images (as BIDSImage objects) as returned from ants_helpers.ant_atropos_n4.
        The thickness image will be written to the same dataset as the segmentation image.
    work_dir : str
        Path to the working directory.
    thickness_iterations : int, optional
        Number of iterations for cortical thickness estimation. Default is 45.
    """
    posterior_files = [posterior.get_path() for posterior in seg_n4['posteriors']]

    logger.info("Calculating cortical thickness on segmentation image: " + seg_n4['segmentation_image'].get_uri())
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


def template_brain_registration(template, template_brain_mask, t1w_brain_image, quick_reg, work_dir):
    """Register a brain-extracted moving image to a template

    Parameters:
    -----------
    template : TemplateImage
        Template image. This will be brain-masked and used as the fixed image.
    template_brain_mask : TemplateImage
        Brain mask for the template. This will be used to mask the template image.
    t1w_brain_image : BIDSImage
        Moving T1w brain image to register to the template.
    quick_reg : bool
        Do quick registration to the template.
    work_dir : str
        Path to the working directory.

    Output is to the same dataset as the t1w_brain_image.

    Returns:
    --------
    dict
        A dictionary with keys:
            'forward_transform' - path to the forward transform in the output dataset
            'inverse_transform' - path to the inverse transform in the output dataset
    """

    fixed_image = ants_helpers.apply_mask(template.get_path(), template_brain_mask.get_path(), work_dir)

    if quick_reg:
        logger.info("Quick registration to template " + template.get_name())
        template_reg = ants_helpers.anatomical_template_registration(fixed_image, t1w_brain_image.get_path(), work_dir,
                                                                     metric='Mattes', metric_params=[1, 32],
                                                                     transform='SyN[0.25,3,0]', iterations='30x70x30x5',
                                                                     shrink_factors='6x4x2x1', smoothing_sigmas='3x2x1x0vox',
                                                                     apply_transforms=False)
    else:
        logger.info("Registration to template " + template.get_name())
        template_reg = ants_helpers.anatomical_template_registration(fixed_image, t1w_brain_image.get_path(), work_dir,
                                                                     metric='CC', metric_params=[1, 4],
                                                                     transform='SyN[0.2,3,0]', iterations='30x70x70x20',
                                                                     shrink_factors='6x4x2x1', smoothing_sigmas='3x2x1x0vox',
                                                                     apply_transforms=False)

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


def template_space_derivatives(template, template_reg, seg_n4, thickness, work_dir):
    """Make template space derivatives: thickness, jacobian, GM probability

    Parameters:
    -----------
    template : TemplateImage
        Template image to use as reference image
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
    logger.info("Creating template space derivatives")

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