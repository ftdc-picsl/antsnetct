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
import pandas as pd
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
            "session": "*",
            "suffix": "T1w"
        }

    User-defined filters override this, so they should include the default filter keys. The 'session' can be set to something
    other than '*' to process only specified sessions, but it should not be removed, because it is necessary to filter out
    longitudinal templates.

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
    # Maybe add this later - first get sessions done. Not sure how best to handle SST.
    # Is it best to apply parcellation directly, or merge session results somehow?
    # subject_parser.add_argument("--parcellate-sst", help="Parcellate SST images, if they exist (implies --longitudinal)",
    #                            action='store_true')

    label_parser = parser.add_argument_group('AntsXNet parcellation options')
    label_parser.add_argument("--dkt31", help="Do DKT31 parcellation", action=argparse.BooleanOptionalAction, default=True)
    label_parser.add_argument("--hoa", help="Do Harvard-Oxford atlas parcellation. If you are using atlas labels, this may be "
                              "required", action=argparse.BooleanOptionalAction, default=True)
    label_parser.add_argument("--cerebellum", help="Do cerebellum parcellation", action=argparse.BooleanOptionalAction,
                              default=False)

    template_parser = parser.add_argument_group('Atlas-based labeling arguments')
    template_parser.add_argument("--atlas-label-config", help="JSON file containing the label definitions in group "
                                 "template space", type=str, default=None)

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

    # preprocessed images
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
            bids_t1w_filter['session'] = '*' # process all sessions, but ignore longitudinal templates

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

                # get thickness image
                t1w_thickness = t1w_bids.get_derivative_rel_path_prefix() + "_seg-antsnetct_desc-thickness.nii.gz"

                antsnet_parc = antsnet_parcellation(t1w_bids, working_dir, dkt31=args.dkt31, hoa=args.hoa,
                                                    cerebellum=args.cerebellum)

                # Do thickness stats on dkt labels
                if args.dkt31:
                    logger.info("Collecting thickness stats on " + t1w_bids.get_uri(relative=False))
                    compute_volume_and_thickness_stats(t1w_bids, t1w_thickness, antsnet_parc['dkt31'], working_dir)
                if args.hoa:
                    logger.info("Doing Harvard-Oxford stats on " + t1w_bids.get_uri(relative=False))
                    compute_volume_stats(t1w_bids, antsnet_parc['hoa'], working_dir)


            except Exception as e:
                logger.error(f"Caught {type(e)} during processing of {str(t1w_bids)}")
                # Print stack trace
                traceback.print_exc()
                debug_workdir = os.path.join(output_dataset, t1w_bids.get_derivative_rel_path_prefix() + "_workdir")
                if args.keep_workdir.lower() != 'never':
                    logger.info("Saving working directory to " + debug_workdir)
                    shutil.copytree(working_dir, debug_workdir)

            # Do atlas-based parcellation if requested
            if atlas_label_config is not None:
                try:
                    logger.info("Doing atlas-based parcellation on " + t1w_bids.get_uri(relative=False))
                    parcellation_results = do_atlas_based_parcellation(t1w_bids, atlas_label_config, working_dir, antsnet_parc,
                                                                       args.longitudinal)

                    # Save the parcellation results to the output dataset
                    for key, value in parcellation_results.items():
                        bids_helpers.save_parcellation_results(output_dataset, t1w_bids, key, value)

                except Exception as e:
                    logger.error(f"Caught {type(e)} during atlas-based parcellation of {str(t1w_bids)}")
                    traceback.print_exc()
                    debug_workdir = os.path.join(output_dataset, t1w_bids.get_derivative_rel_path_prefix() + "_workdir")
                    if args.keep_workdir.lower() != 'never':
                        logger.info("Saving working directory to " + debug_workdir)
                        shutil.copytree(working_dir, debug_workdir)

    # # Optionally parcellate SST images
    # if args.parcellate_sst:
    #     sst_bids = bids_helpers.find_participant_images(input_dataset, args.participant, bids_wd,
    #                                                     validate=False, datatype='func', suffix='SST')
    #     if len(sst_bids) == 0:
    #         raise ValueError(f"No SST images found for participant {args.participant}")
    #     with tempfile.TemporaryDirectory(
    #         suffix=f"antsnetparc_{system_helpers.get_nifti_file_prefix(sst_bids.get_path())}.tmpdir") as working_dir:
    #         try:
    #             if args.parcellate_sst:
    #                 logger.info("Processing SST image: " + sst_bids.get_uri(relative=False))
    #                 parcellate_sst(sst_bids, output_dataset, args.participant, args.longitudinal, args.verbose)
    #         except Exception as e:
    #             logger.error(f"Caught {type(e)} during processing of {args.participant}")
    #             # Print stack trace
    #             traceback.print_exc()
    #             debug_workdir = os.path.join(args.output_dataset, f"sub-{args.participant}", f"sub-{args.participant}_workdir")
    #             if args.keep_workdir.lower() != 'never':
    #                 logger.info(f"Saving working directory {working_dir} to {debug_workdir}")
    #                 shutil.copytree(working_dir, debug_workdir)


def antsnet_parcellation(t1w_bids, work_dir, dkt31=True, hoa=True, cerebellum=False):
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
        Do cerebellum parcellation (requires hoa)

    Returns:
    -------
    dict with keys:
        - dkt31
        - hoa
        - cerebellum
    containing BIDSImage objects for the selected parcellations.
    """
    logger.info("Doing antsnet parcellation on " + t1w_bids.get_uri(relative=False))

    # Read the T1w image
    t1w_image = ants_image_read(t1w_bids.get_path())

    # Get the brain mask
    brain_mask = preprocessing.get_brain_mask(t1w_image)

    # Do the parcellation
    parcellation_results = dict()

    if cerebellum:
        hoa = True  # cerebellum parcellation requires hoa

    if dkt31:
        dkt31 = ants_helpers.desikan_killiany_tourville_parcellation(t1w_image, work_dir, brain_mask)
        parcellation_results['dkt31'] = bids_helpers.image_to_bids(
            dkt31,
            t1w_bids.get_ds_path(),
            t1w_bids.get_derivative_rel_path_prefix() + "_seg-dkt31_dseg.nii.gz",
            metadata={'Description': 'ANTsPyNet DKT31', 'Sources': [t1w_bids.get_uri(relative=True)]}
            )

    if hoa:
        hoa = ants_helpers.hoa_parcellation(t1w_image, work_dir, brain_mask)
        parcellation_results['hoa'] = bids_helpers.image_to_bids(
            hoa,
            t1w_bids.get_ds_path(),
            t1w_bids.get_derivative_rel_path_prefix() + "_seg-hoa_dseg.nii.gz",
            metadata={'Description': 'ANTsPyNet Harvard-Oxford Subcortical', 'Sources': [t1w_bids.get_uri(relative=True)]}
            )

    if cerebellum:
        cerebellum_mask_file = system_helpers.get_temp_file(work_dir, prefix='antsnet_parc', suffix='.nii.gz')
        hoa_labels = ants_image_read(parcellation_results['hoa'].get_path())
        # Create a cerebellum mask from the Harvard-Oxford labels
        cerebellum_mask = ants_helpers.threshold_image(parcellation_results['hoa'].get_path(), work_dir, 29, 32)
        cerebellum = ants_helpers.cerebellum_parcellation(t1w_image, work_dir, cerebellum_mask)
        parcellation_results['cerebellum'] = bids_helpers.image_to_bids(
            cerebellum,
            t1w_bids.get_ds_path(),
            t1w_bids.get_derivative_rel_path_prefix() + "_seg-cerebellum_dseg.nii.gz",
            metadata={'Description': 'ANTsPyNet Cerebellum',
                      'Sources': [t1w_bids.get_uri(relative=True), parcellation_results['hoa'].get_uri(relative=True)]}
            )

    return parcellation_results


def parcellate_sst(sst_bids, work_dir):
    raise NotImplementedError("SST parcellation not implemented yet")


def atlas_based_parcellation(t1w_bids, atlas_label_config, work_dir, antsnet_parcellation=None, longitudinal=False):
    """Do atlas-based parcellation on a T1w image.

    Parameters:
    -----------
    t1w_bids : BIDSImage
        BIDS image object for the T1w image
    atlas_label_config : dict
        Atlas label configuration dict
    work_dir : str
        Working directory for the parcellation
    antsnet_parcellation : dict
        Dictionary with antsnet parcellation results. This must contain 'hoa' if the atlas-based parcellation has any labels
        with 'restrict_to_cortex' or 'propagate_to_cortex' set to True.
    longitudinal : bool
        Do longitudinal parcellation

    Returns:
    -------
    dict with keys for each atlas, containing the parcellation results.
    """
    logger.info("Doing atlas-based parcellation on " + t1w_bids.get_uri(relative=False))

    # Read the T1w image
    t1w_image = ants_image_read(t1w_bids.get_path())


    return parcellation_results


def label_stats(label_image, label_definitions, work_dir, scalar_images=None):
    """Compute stats for a label image.

    Parameters:
    -----------
    label_image : BIDSImage
        Label image to compute stats for
    label_definitions : dict
        Label definitions for the label image
    work_dir : str
        Working directory for the stats
    scalar_images : list of BIDSImage, optional
        List of scalar images on which to compute stats.

    Returns:
    -------
    list of pandas DataFrame, one for each scalar image, or a single DataFrame if scalar_images is None.
    """
    label_stats = list()
    if scalar_images is None:
        label_stat = ants_helpers.label_statistics(label_image.get_path(), work_dir, label_definitions)
        # Save the label statistics to the output dataset
        label_file = label_image.get_derivative_rel_path_prefix() + "_desc-labelgeometry.tsv"
        pd.DataFrame.to_csv(label_stat, label_file, sep='\t', index=False)
        label_stats.append(label_file)
    else:
        for scalar in scalar_images:
            if not os.path.exists(scalar.get_path()):
                raise ValueError(f"Scalar image {scalar.get_uri(relative=False)} does not exist")

            logger.info("Computing stats for label image " + label_image.get_uri(relative=False))
            label_stat = ants_helpers.label_statistics(label_image.get_path(), label_definitions, work_dir, scalar.get_path())
            label_file = scalar.get_derivative_rel_path_prefix() + "_seg- desc-labelstats.tsv"
            pd.DataFrame.to_csv(label_stat, label_file, sep='\t', index=False)
            label_stats.append(label_file)
            label_stats.append(label_stat)

