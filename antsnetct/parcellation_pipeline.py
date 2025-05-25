from . import ants_helpers
from . import bids_helpers
from . import cross_sectional_pipeline
from . import preprocessing
from . import system_helpers

from .system_helpers import PipelineError

from ants import image_read as ants_image_read

import argparse
import json
import logging
import numpy as np
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


def parcellation_pipeline():

    # Handle args with argparse
    parser = argparse.ArgumentParser(formatter_class=RawDefaultsHelpFormatter, add_help = False,
                                     description='''Parcellation and regional statistics with ANTsPyNet.

    The analysis level for parcellation is the subject. By default, all T1w images for a participant will be
    processed To process a specific subset of images for a participant, use the --participant-images option or
    a BIDS filter file.

    --- BIDS options ---

    Users can select a subset of T1w images to process with a BIDS filter file. Following the *prep pipelines, the filters
    should contain a dict "t1w" with keys and values to filter on. The default filter for t1w images is

        "t1w": {
            "datatype": "anat",
            "desc": "preproc",
            "suffix": "T1w"
        }

    User-defined filters override this, so they should include the default filter keys.

    Required inputs are cross-sectional or longitudinal antsnetct datasets. If atlas-based parcellation is required, a
    label definition configuration file is also required.


    ANTsPyNet parcellation schemes
    ------------------------------

    The default parcellations are

        DKT31 - cortical parcellation with 31 labels per hemisphere (https://mindboggle.readthedocs.io/en/latest/labels.html).

        Harvard-Oxford Subcortical - Harvard-Oxford subcortical atlas with 48 labels
        (https://github.com/HOA-2/SubcorticalParcellations).

    Optional parcellations are

        Cerebellum - cerebellar segmentation into 3 classes (CSF, GM, WM) and lobular parcellation
        (https://www.cobralab.ca/cerebellum-lobules).




    Configuration of atlas parcellation
    -----------------------------------

    Atlas parcellation works by warping an atlas label image to the native space of the session T1w or SST image. We use atlas
    to mean a set of discrete labels, such as the Schaefer atlas. Each atlas resided in a template space - for example, the
    Schaefer labels may be in the space of tpl-MNI152NLin2009cAsym. There must be existing transforms that connect the atlas
    template to the input space. This means that either the session has a transform to the source template, or the session has a
    transform to another template which in turn has a transform to the source template.

    For example, if the session has a transform to tpl-MNI152NLin2009cAsym, a parcellation in tpl-MNI152NLin6Asym can be used,
    because tpl-MNI152NLin2009cAsym has a transform from tpl-MNI152NLin6Asym. But a parcellation in tpl-OASIS30ANTs cannot be
    used, because tpl-MNI152NLin2009cAsym does not have a transform to tpl-OASIS30ANTs.

    The atlas parcellation configuration file is a JSON file that contains a named dict for each atlas parcellation. For
    example,

    {
      "schaefer100x7": {
        "label_definition": "schaefer100x7.json",
        "label_image": "schaefer100x7.nii.gz",
        "template_name": "MNI152NLin2009cAsym",
        "sample_thickness": true,
        "restrict_to_cortex": true,
        "propagate_to_cortex": true
      },
      brainCOLORSubcortical: {
        "label_definition": "brainCOLORSubcortical.json",
        "label_image": "tpl-ADNINormalAgingANTs_res-01_atlas-BrainColor_desc-subcortical_dseg.nii.gz",
        "template_name": "ADNINormalAgingANTs",
        "sample_thickness": false
      }
    }

    Optional keys define the masking and propagation options. All labels are restricted to the mask defined by the antsnetct
    brain segmentation, excluding CSF. If "restrict_to_cortex" is true, the parcellation will be restricted to the cortical GM
    label.  If "propagate_to_cortex" is true, the parcellation will be propagated to the cortical mask of the target
    image - this fills in cortical voxels that are missed due to registration error.

    if sample_thickness is true, the parcellation will be used to compute statistics on cortical thickness, if thickness exists
    for the session.


    Cortical propagation into hippocampus and amygdala
    --------------------------------------------------

    The ants deep_atropos pipeline includes hippocampus and amygdala in the cortical mask. This becomes a problem when
    propagating cortical labels, which do not generally include these structures. To avoid propagation of cortical labels into
    these structures, the hippocampus and amygdala are temporarily filled with labels from the subcortical Harvard-Oxford
    segmentation. Therefore, the `--hoa` option is required if label propagation is to be done.
    ''')

    required_parser = parser.add_argument_group('Required arguments')
    required_parser.add_argument("--longitudinal", help="Do longitudinal analysis", action='store_true', required=True)
    required_parser.add_argument("--input-dataset", "--input-dataset", help="BIDS derivatives dataset dir, "
                                 "containing the antsnetct output", type=str, required=True)
    required_parser.add_argument("--participant", help="Participant to process", type=str)

    optional_parser = parser.add_argument_group('General optional arguments')
    optional_parser.add_argument("-h", "--help", action="help", help="show this help message and exit")
    optional_parser.add_argument("--keep-workdir", help="Copy working directory to output, for debugging purposes. Either "
                                 "'never', 'on_error', or 'always'.", type=str, default='on_error')
    optional_parser.add_argument("--verbose", help="Verbose output", action='store_true')

    subject_parser = parser.add_argument_group('Input filtering arguments')
    subject_parser.add_argument("--bids-filter-file", help="BIDS filter to apply to the input dataset", type=str, default=None)
    subject_parser.add_argument("--participant-images", help="Text file containing a list of participant images to process "
                                 "relative to the input dataset. If not provided, all images for the participant "
                                 "will be processed.", type=str, default=None)
    subject_parser.add_argument("--parcellate-sst", help="Parcellate SST images, if they exist (implies --longitudinal)",
                                action='store_true')

    label_parser = parser.add_argument_group('AntsXNet parcellation options')
    label_parser.add_argument("--dkt31", help="Do DKT31 parcellation", action=argparse.BooleanOptionalAction, default=True)
    label_parser.add_argument("--hoa", help="Do Harvard-Oxford atlas parcellation. If you are using atlas labels, this may be "
                              "required", action=argparse.BooleanOptionalAction, default=True)
    label_parser.add_argument("--cerebellum", help="Do cerebellum parcellation", action=argparse.BooleanOptionalAction,
                              default=False)


    template_parser = parser.add_argument_group('Atlas-based labeling arguments')
    template_parser.add_argument("--template-label-config", help="JSON file containing the label definitions for the group "
                                 "template", type=str, default=None)

    if len(sys.argv) == 1:
        parser.print_usage()
        print(f"\nRun {os.path.basename(sys.argv[0])} --help for more information")
        sys.exit(1)

    args = parser.parse_args()

    logger.info("Parsed args: " + str(args))

    system_helpers.set_verbose(args.verbose)

    if args.participant is None:
        raise ValueError('Participant must be defined')

    atlas_label_config = None

    # Only need group template if we are doing atlas-based labeling
    if args.template_label_config is not None:
        if not 'TEMPLATEFLOW_HOME' in os.environ or not os.path.exists(os.environ.get('TEMPLATEFLOW_HOME')):
            raise PipelineError(f"templateflow directory not found at " +
                                f"TEMPLATEFLOW_HOME={os.environ.get('TEMPLATEFLOW_HOME')}")
        atlas_label_config = json.load(f)

    output_dataset = args.output_dataset

    input_dataset = args.input_dataset

    if (input_dataset == output_dataset):
        raise ValueError('Input and output datasets cannot be the same')

    if os.path.exists(os.path.join(output_dataset, f"sub-{args.participant}")):
        raise ValueError(f"Output exists for participant {args.participant}")

    input_dataset_description = None

    if os.path.exists(os.path.join(input_dataset_description, 'dataset_description.json')):
        with open(os.path.join(input_dataset_description, 'dataset_description.json'), 'r') as f:
            input_dataset_description = json.load(f)
    else:
        raise ValueError('input dataset does not contain a dataset_description.json file')

    logger.info("input dataset path: " + input_dataset_description)
    logger.info("Cross-sectional dataset name: " + input_dataset_description['Name'])

    # Create the output dataset and add this container to the GeneratedBy, if needed
    bids_helpers.update_output_dataset(output_dataset, input_dataset_description['Name'] + '_parcellation', [input_dataset])

    with open(os.path.join(output_dataset, 'dataset_description.json'), 'r') as f:
        output_dataset_description = json.load(f)

    logger.info("Output dataset path: " + output_dataset)
    logger.info("Output dataset name: " + output_dataset_description['Name'])

    # preprocessed images to be processed longitudinally
    input_session_preproc_t1w_bids = list()

    # brain masks for the preprocessed images
    cx_brain_mask_bids = list()

    if args.participant_images is not None:
        with open(args.participant_images, 'r') as f:
            t1w_relpaths = [line.strip() for line in f]
            for relpath in t1w_relpaths:
                if not os.path.exists(os.path.join(input_dataset, relpath)):
                    raise ValueError(f"Image {relpath} not found in cross-sectional dataset")
                input_session_preproc_t1w_bids.append(bids_helpers.BIDSImage(input_dataset, relpath))
    else:
        bids_t1w_filter = dict()

        if args.bids_filter_file is not None:
            bids_t1w_filter = bids_helpers.get_modality_filter_query('t1w', args.bids_filter_file)
        else:
            bids_t1w_filter = bids_helpers.get_modality_filter_query('t1w')
            # Need to filter on desc-preproc in addition to the default t1w filter
            bids_t1w_filter['desc'] = 'preproc'

        with tempfile.TemporaryDirectory(suffix=f"antsnetct_bids_{args.participant}.tmpdir") as bids_wd:
            # Have to turn off validation for derivatives
            input_session_preproc_t1w_bids = bids_helpers.find_participant_images(input_dataset, args.participant, bids_wd,
                                                                                  validate=False, **bids_t1w_filter)

    logger.info(f"Using participant images: {[im.get_uri() for im in input_session_preproc_t1w_bids]}")

    for t1w_bids in input_session_preproc_t1w_bids:
        with tempfile.TemporaryDirectory(
                suffix=f"antsnetparc_{system_helpers.get_nifti_file_prefix(t1w_bids.get_path())}.tmpdir") as working_dir:
            try:

                logger.info("Processing T1w image: " + t1w_bids.get_uri(relative=False))

                do_antsnet_parcellation(t1w_bids, working_dir, dkt31=args.dkt31, hoa=args.hoa, cerebellum=args.cerebellum)

            except Exception as e:
                logger.error(f"Caught {type(e)} during processing of {str(t1w_bids)}")
                # Print stack trace
                traceback.print_exc()
                debug_workdir = os.path.join(output_dataset, t1w_bids.get_derivative_rel_path_prefix() + "_workdir")
                if args.keep_workdir.lower() != 'never':
                    logger.info("Saving working directory to " + debug_workdir)
                    shutil.copytree(working_dir, debug_workdir)

    # Optionally parcellate SST images
    if args.parcellate_sst:
        sst_bids = bids_helpers.find_participant_images(input_dataset, args.participant, bids_wd,
                                                        validate=False, datatype='func', suffix='SST')
        if len(sst_bids) == 0:
            raise ValueError(f"No SST images found for participant {args.participant}")
        with tempfile.TemporaryDirectory(
            suffix=f"antsnetparc_{system_helpers.get_nifti_file_prefix(sst_bids.get_path())}.tmpdir") as working_dir:
            try:
                if args.parcellate_sst:
                    logger.info("Processing SST image: " + sst_bids.get_uri(relative=False))
                    parcellate_sst(sst_bids, output_dataset, args.participant, args.longitudinal, args.verbose)
            except Exception as e:
                logger.error(f"Caught {type(e)} during processing of {args.participant}")
                # Print stack trace
                traceback.print_exc()
                debug_workdir = os.path.join(args.output_dataset, f"sub-{args.participant}", f"sub-{args.participant}_workdir")
                if args.keep_workdir.lower() != 'never':
                    logger.info(f"Saving working directory {working_dir} to {debug_workdir}")
                    shutil.copytree(working_dir, debug_workdir)


def do_antsnet_parcellation(t1w_bids, work_dir, dkt31=True, hoa=True, cerebellum=False):
    """Do antsnet parcellation on a T1w image.

    Parameters:
    -----------
    t1w_bids : BIDSImage
        BIDS image object for the T1w image
    work_dir : str
        Working directory for the parcellation
    dkt31 : bool
        Do DKT31 parcellation
    hoa : bool
        Do Harvard-Oxford atlas parcellation
    cerebellum : bool
        Do cerebellum parcellation

    Returns:
    -------
    dict with keys:
        - dkt31
        - hoa
        - cerebellum
    for the selected parcellations.
    """
    logger.info("Doing antsnet parcellation on " + t1w_bids.get_uri(relative=False))

    # Read the T1w image
    t1w_image = ants_image_read(t1w_bids.get_path())

    # Get the brain mask
    brain_mask = preprocessing.get_brain_mask(t1w_image)

    # Do the parcellation
    parcellation_results = dict()

    if dkt31:
        parcellation_results['dkt31'] = cross_sectional_pipeline.do_dkt31_parcellation(t1w_image, work_dir, brain_mask)

    if hoa:
        parcellation_results['hoa'] = cross_sectional_pipeline.do_hoa_parcellation(t1w_image, work_dir, brain_mask)

    if cerebellum:
        parcellation_results['cerebellum'] = cross_sectional_pipeline.do_cerebellum_parcellation(t1w_image, work_dir,
                                                                                               brain_mask)

    return parcellation_results


def parcellate_sst(sst_bids, output_dataset, participant, longitudinal, verbose):
    raise NotImplementedError("SST parcellation not implemented yet")