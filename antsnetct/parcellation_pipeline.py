from . import ants_helpers
from . import bids_helpers
from . import cross_sectional_pipeline
from . import system_helpers

from .data import get_label_definitions_path

from .system_helpers import copy_file, PipelineError

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


def run_parcellation_pipeline():

    # Handle args with argparse
    parser = argparse.ArgumentParser(formatter_class=RawDefaultsHelpFormatter, add_help = False,
                                     description='''Parcellation and regional statistics with ANTsPyNet.

    The analysis level for parcellation is the subject. By default, all T1w images for a participant will be
    processed. For per-session control, use the --session option. To process a specific subset of images for a participant,
    use the --participant-images option or a BIDS filter file.

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

    The available parcellations are

        DKT31 - cortical parcellation with 31 labels per hemisphere (https://mindboggle.readthedocs.io/en/latest/labels.html).
        This can optionally be masked by the thickness image to restrict to cortical GM.

        Harvard-Oxford Subcortical - Harvard-Oxford subcortical atlas with 48 labels
        (https://github.com/HOA-2/SubcorticalParcellations).

        Cerebellum - cerebellar segmentation into 3 classes (CSF, GM, WM) and lobular parcellation
        (https://www.cobralab.ca/cerebellum-lobules). Requires HOA parcellation to be done first.


    As with atlas-based parcellations (see below), these parcellations can be post-processed using the thickness or
    segmentation images. The DKT31 parcellation can be masked or propagated to cortical GM. Propagation requires HOA labels to
    be computed first (see "Cortical propagation into hippocampus and amygdala" below). The HOA parcellation can be masked to
    remove CSF voxels from non-CSF labels, similar to the "mask_csf" option for atlas labels. The cerebellum parcellation can
    likewise be masked, which requires masked HOA labels.

    Dependencies for segmentations are added automatically, for example, `--dkt31-propagated` implies both `--dkt31 and --hoa`.


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

    The atlas parcellation configuration file is a JSON file that contains a named dict for each atlas parcellation. Each dict
    must have a unique name, which will be used to name the output parcellation. Each dict must have the following keys:

        "template_label" - the name of the template in which the label image resides, e.g. "MNI152NLin2009cAsym"
        "atlas_label" - the atlas name, e.g. "Schaefer2018"

    Optional keys are:

        "template_resolution" - the resolution of the template, e.g. "01"

        "atlas_description" - the description of the atlas, e.g. "100Parcels7Networks"

        "sample_thickness" - if true, compute statistics on cortical thickness, if thickness exists for the session

        "restrict_to_cortex" - if true, restrict the parcellation to the cortical GM label

        "propagate_to_cortex" - if true, propagate the parcellation to the cortical GM mask of the target image

        "mask_csf" - if true, remove CSF voxels from the parcellation. Use this to remove CSF voxels. Note that using the
        'restrict_to_cortex' or 'propagate_to_cortex' options makes this redundant.


    The label_image must exist at the path "{TEMPLATEFLOW_HOME}/tpl-{template_label}/{label_image}". The {label_image} is
    identified as

      tpl-{template_label}_[res-{template_resolution}]_atlas-{atlas_label}_[desc-{atlas_description}_]dseg.nii.gz

    There must be a corresponding .tsv file with label definitions at the same location.

    Example config:
    {
      "schaefer2018x100x7": {
        "template_label": "MNI152NLin2009cAsym",
        "template_resolution": "01",
        "atlas_label": "Schaefer2018",
        "atlas_description": "100Parcels7Networks",
        "sample_thickness": true,
        "restrict_to_cortex": true,
        "propagate_to_cortex": false
      },
      "brainCOLORSubcortical": {
        "template_label": "ADNINormalAgingANTs",
        "template_resolution": "01",
        "atlas_label": "BrainColor",
        "atlas_description": "subcortical",
        "sample_thickness": false,
        "mask_csf": true,
      }
    }

    All labels are restricted to the mask defined by the antsnetct brain segmentation.

    If thickness is present for the session, it will be used to define a cortical thickness mask. This can be used with the
    "restrict_to_cortex" and "propagate_to_cortex" options. If "restrict_to_cortex" is true, the parcellation will be
    restricted to the cortical mask. If "propagate_to_cortex" is true, the parcellation will be propagated to the cortical mask
    of the target image - this fills in cortical voxels that are missed due to registration error.

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
    required_parser.add_argument("--input-dataset", "--input-dataset", help="BIDS derivatives dataset dir, "
                                 "containing the antsnetct output", type=str, required=True)
    required_parser.add_argument("--participant", "--subject", help="Participant to process", type=str)

    optional_parser = parser.add_argument_group('General optional arguments')
    optional_parser.add_argument("-h", "--help", action="help", help="show this help message and exit")
    optional_parser.add_argument("--keep-workdir", help="Copy working directory to output, for debugging purposes. Either "
                                 "'never', 'on_error', or 'always'.", type=str, default='on_error')
    optional_parser.add_argument("--verbose", help="Verbose output", action='store_true')
    optional_parser.add_argument("--longitudinal", help="Do longitudinal analysis", action='store_true')


    subject_parser = parser.add_argument_group('Input filtering arguments')
    subject_parser.add_argument("--bids-filter-file", help="BIDS filter to apply to the input dataset", type=str, default=None)
    subject_parser.add_argument("--participant-images","--subject-images", help="Text file containing a list of participant "
                                "images to process relative to the input dataset. If not provided, all images for the "
                                "participant will be processed.", type=str, default=None)
    subject_parser.add_argument("--session", help="Process only this session. This overrides any session defined in a "
                                 "BIDS filter file, if any.", type=str, default=None)
    # Maybe add this later - first get sessions done. Not sure how best to handle SST.
    # Is it best to apply parcellation directly, or merge session results somehow?
    # subject_parser.add_argument("--parcellate-sst", help="Parcellate SST images, if they exist (implies --longitudinal)",
    #                            action='store_true')

    label_parser = parser.add_argument_group('AntsXNet parcellation options')
    label_parser.add_argument("--dkt31", help="Do DKT31 parcellation", action=argparse.BooleanOptionalAction, default=False)
    label_parser.add_argument("--dkt31-masked", help="Use cortical thickness to mask the DKT31 parcellation to cortical GM. "
                              "Requires a cortical thickness image for the session.", action=argparse.BooleanOptionalAction,
                              default=False)
    label_parser.add_argument("--dkt31-propagated", help="Propagate the DKT parcellation to cortical GM. Requires a cortical "
                              "thickness image for the session.", action=argparse.BooleanOptionalAction,
                              default=False)
    label_parser.add_argument("--hoa", help="Do Harvard-Oxford atlas parcellation. If you are using atlas labels, this may be "
                              "required", action=argparse.BooleanOptionalAction, default=False)
    label_parser.add_argument("--hoa-masked", help="Mask out CSF from HOA labels. Requires antsnetct tissue segmentation.",
                              action=argparse.BooleanOptionalAction, default=False)
    label_parser.add_argument("--cerebellum", help="Do cerebellum parcellation", action=argparse.BooleanOptionalAction,
                              default=False)
    label_parser.add_argument("--cerebellum-masked", help="Mask out CSF from cerebellum labels. Requires antsnetct tissue "
                              "segmentation.", action=argparse.BooleanOptionalAction, default=False)

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

    atlas_label_config = args.atlas_label_config

    # Only need group template if we are doing atlas-based labeling
    if atlas_label_config is not None:
        if not 'TEMPLATEFLOW_HOME' in os.environ or not os.path.exists(os.environ.get('TEMPLATEFLOW_HOME')):
            raise PipelineError(f"templateflow directory not found at " +
                                f"TEMPLATEFLOW_HOME={os.environ.get('TEMPLATEFLOW_HOME')}")

    input_dataset = args.input_dataset
    output_dataset = input_dataset

    input_dataset_description = None

    if args.dkt31_masked or args.dkt31_propagated:
        args.dkt31 = True

    if os.path.exists(os.path.join(input_dataset, 'dataset_description.json')):
        with open(os.path.join(input_dataset, 'dataset_description.json'), 'r') as f:
            input_dataset_description = json.load(f)
    else:
        raise ValueError('input dataset does not contain a dataset_description.json file')

    logger.info("input dataset path: " + input_dataset)
    logger.info("input dataset name: " + input_dataset_description['Name'])

    # Check we have a derivative dataset
    if input_dataset_description.get('DatasetType', None) != 'derivative':
        raise ValueError('input dataset is not a BIDS derivatives dataset')

    # add this container to the GeneratedBy, if needed. This also checks the consistency of the templateflow location
    bids_helpers.update_output_dataset(args.input_dataset, input_dataset_description['Name'])

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
                    raise ValueError(f"Image {relpath} not found in input dataset")
                input_session_preproc_t1w_bids.append(bids_helpers.BIDSImage(input_dataset, relpath))
    else:
        bids_t1w_filter = dict()

        if args.bids_filter_file is not None:
            bids_t1w_filter = bids_helpers.get_modality_filter_query('t1w', args.bids_filter_file)
        else:
            bids_t1w_filter = bids_helpers.get_modality_filter_query('t1w')
            # Need to filter on desc-preproc in addition to the default t1w filter
            bids_t1w_filter['desc'] = 'preproc'
            if args.longitudinal:
                bids_t1w_filter['session'] = '.+' # process all sessions, but ignore longitudinal templates (no session key)

        if args.session is not None:
            bids_t1w_filter['session'] = args.session

        with tempfile.TemporaryDirectory(suffix=f"antsnetct_bids_{args.participant}.tmpdir") as bids_wd:
            # Have to turn off validation for derivatives
            input_session_preproc_t1w_bids = bids_helpers.find_participant_images(input_dataset, args.participant, bids_wd,
                                                                                  validate=False, regex_search=True,
                                                                                  **bids_t1w_filter)

    logger.info(f"Using participant images: {[im.get_uri() for im in input_session_preproc_t1w_bids]}")

    for t1w_bids in input_session_preproc_t1w_bids:
        with tempfile.TemporaryDirectory(
                suffix=f"antsnetparc_{system_helpers.get_nifti_file_prefix(t1w_bids.get_path())}.tmpdir") as working_dir:
            try:

                logger.info("Processing T1w image: " + t1w_bids.get_uri(relative=False))

                # Get the brain mask for the T1w image
                t1w_brain_mask_bids = bids_helpers.BIDSImage(
                    t1w_bids.get_ds_path(),
                    t1w_bids.get_derivative_rel_path_prefix() + "_desc-brain_mask.nii.gz"
                )

                t1w_thickness_bids = t1w_bids.get_derivative_image("_seg-antsnetct_desc-cortical_thickness.nii.gz")
                t1w_biascorr_bids = t1w_bids.get_derivative_image("_desc-biascorr_T1w.nii.gz")
                seg_bids = t1w_bids.get_derivative_image("_seg-antsnetct_dseg.nii.gz")

                antsnet_parc = antsnet_parcellation(t1w_bids, t1w_brain_mask_bids, working_dir, dkt31=args.dkt31,
                                                    mask_dkt31=args.dkt31_masked, propagate_dkt31=args.dkt31_propagated,
                                                    hoa=args.hoa, mask_hoa=args.hoa_masked, cerebellum=args.cerebellum,
                                                    segmentation_bids=seg_bids, thickness_bids=t1w_thickness_bids,
                                                    t1w_biascorr_bids=t1w_biascorr_bids)

                # Do atlas-based parcellation if requested
                if atlas_label_config is not None:
                    hoa_parc_bids = t1w_bids.get_derivative_image("_seg-hoa_dseg.nii.gz")
                    segmentation_bids = t1w_bids.get_derivative_image("_seg-antsnetct_dseg.nii.gz")
                    atlas_ref_t1w_bids = t1w_biascorr_bids if t1w_biascorr_bids is not None else t1w_bids
                    atlas_based_parcellation(atlas_ref_t1w_bids, t1w_brain_mask_bids, atlas_label_config, working_dir,
                                            longitudinal=args.longitudinal, thickness_bids=t1w_thickness_bids,
                                            hoa_parcellation_bids=hoa_parc_bids, segmentation_bids=segmentation_bids)

                logger.info("Done processing T1w image: " + t1w_bids.get_uri(relative=False))

                if args.keep_workdir.lower() == 'always':
                    complete_workdir = os.path.join(output_dataset, t1w_bids.get_derivative_rel_path_prefix() + "_workdir")
                    logger.info("Saving working directory to " + complete_workdir)
                    shutil.copytree(working_dir, complete_workdir, copy_function=shutil.copy)

            except Exception as e:
                logger.error(f"Caught {type(e)} during processing of {str(t1w_bids)}")
                # Print stack trace
                traceback.print_exc()
                debug_workdir = os.path.join(output_dataset, t1w_bids.get_derivative_rel_path_prefix() + "_workdir")
                if args.keep_workdir.lower() != 'never':
                    logger.info("Saving working directory to " + debug_workdir)
                    shutil.copytree(working_dir, debug_workdir, copy_function=shutil.copy)



def antsnet_parcellation(t1w_bids, brain_mask_bids, work_dir, thickness_bids=None, segmentation_bids=None,
                         t1w_biascorr_bids=None, dkt31=True, mask_dkt31=False, propagate_dkt31=False, hoa=True, mask_hoa=False,
                         cerebellum=False, mask_cerebellum=False):
    """Do antsnet parcellation on a T1w image.

    Parameters:
    -----------
    t1w_bids : BIDSImage
        BIDS image object for the T1w image. This must have skull on and should be minimally preprocessed
        (eg neck trim, voxel orientation).
    brain_mask_bids : BIDSImage
        BIDS image object for the brain mask.
    work_dir : str
        Working directory for the parcellation.
    thickness_bids : BIDSImage
        BIDS image object for the cortical thickness image. This is used for stats.
    segmentation_bids : BIDSImage
        BIDS image object for the antsnetct tissue segmentation. This is used to mask the HOA parcellation to remove CSF.
    t1w_biascorr_bids : BIDSImage
        BIDS image object for the N4 bias-corrected T1w image. If None, the input T1w image is used. This is used to get
        better intensity statistics and QC images.
    dkt31 : bool
        Do DKT31 parcellation.
    mask_dkt31 : bool
        Use cortical thickness to mask the DKT31 parcellation to cortical GM only. Requires a cortical thickness image.
    propagate_dkt31 : bool
        Propagate the DKT31 parcellation to cortical GM. Requires a cortical thickness image. This requires the HOA
        parcellation to be done first, so if propagate_dkt31 is true, hoa will be set to true.
    hoa : bool
        Do Harvard-Oxford atlas parcellation.
    mask_hoa : bool
        Mask out CSF from the HOA parcellation. Requires antsnetct tissue segmentation.
    cerebellum : bool
        Do cerebellum parcellation (requires hoa).
    mask_cerebellum : bool
        Mask out CSF from the cerebellum parcellation. Requires mask_hoa.

    Returns:
    -------
    dict with keys:
        - dkt31
        - hoa
        - cerebellum
    pluse optional masked / propagated versions of the above.
    each key returns a dict containing 'image': BIDSImage objects for the selected parcellation, and 'label_definitions': str
    path to the label definitions tsv file. Thickness stats will be produced for the DKT parcellation if appropriate. Other
    parcellations will get label volumes and mean T1w intensity.
    """
    logger.info("Doing antsnet parcellation on " + t1w_bids.get_uri(relative=False))

    t1w = t1w_bids.get_path()

    if t1w_biascorr_bids is None:
        t1w_biascorr_bids = t1w_bids

    brain_mask = brain_mask_bids.get_path()

    parcellation_results = dict()

    if cerebellum:
        hoa = True  # cerebellum parcellation requires hoa
    if mask_cerebellum:
        cerebellum = True
        hoa = True  # cerebellum masking requires hoa
        mask_hoa = True  # cerebellum masking requires masked hoa

    cortical_mask = None
    cortical_mask_source = None

    if mask_dkt31 or propagate_dkt31:
        dkt31 = True
        if propagate_dkt31:
            hoa = True  # dkt31 propagation requires hoa
        if thickness_bids is not None:
            cortical_mask = ants_helpers.threshold_image(thickness_bids.get_path(), work_dir, lower=0.001)
            cortical_mask_source = thickness_bids.get_uri(relative=True)
        elif segmentation_bids is not None:
            cortical_mask = ants_helpers.threshold_image(segmentation_bids.get_path(), work_dir, 8, 8)
            cortical_mask_source = segmentation_bids.get_uri(relative=True)
        else:
            raise ValueError("antsnet thickness or segmentation image is required for DKT31 masking or propagation")
    if mask_hoa:
        if segmentation_bids is None:
            raise ValueError("antsnetct segmentation is required to mask HOA parcellation")

    if hoa:
        logger.info("Starting Harvard-Oxford subcortical parcellation")
        hoa_bids = t1w_bids.get_derivative_image("_seg-hoa_dseg.nii.gz")
        parcellation_results['hoa'] = dict()

        if hoa_bids is not None:
            logger.info("Harvard-Oxford parcellation already exists at " + hoa_bids.get_uri(relative=False))
            parcellation_results['hoa']['image'] = hoa_bids
            parcellation_results['hoa']['label_definitions'] = hoa_bids.get_path().replace('.nii.gz', '.tsv')
        else:
            hoa_file = ants_helpers.harvard_oxford_subcortical_parcellation(t1w, work_dir, brain_mask)
            hoa_description = 'ANTsPyNet Harvard-Oxford Subcortical'
            hoa_sources = [t1w_bids.get_uri(relative=True)]

            hoa_bids = bids_helpers.image_to_bids(hoa_file, t1w_bids.get_ds_path(),
                                                  t1w_bids.get_derivative_rel_path_prefix() + "_seg-hoa_dseg.nii.gz",
                                                  metadata={'Description': hoa_description, 'Manual': False,
                                                            'Sources': hoa_sources})
            parcellation_results['hoa']['image'] = hoa_bids
            parcellation_results['hoa']['label_definitions'] = hoa_bids.get_path().replace('.nii.gz', '.tsv')
            copy_file(get_label_definitions_path('hoa_subcortical'), parcellation_results['hoa']['label_definitions'])

            if mask_hoa:
                hoa_mask_file = remove_csf_from_hoa_tissue_labels(hoa_file, segmentation_bids.get_path(), work_dir)
                hoa_description = 'ANTsPyNet Harvard-Oxford Subcortical, tissue labels masked to remove CSF voxels',
                hoa_mask_sources = [t1w_bids.get_uri(relative=True), segmentation_bids.get_uri(relative=True)]
                hoa_mask_bids = bids_helpers.image_to_bids(hoa_mask_file, t1w_bids.get_ds_path(),
                                                           t1w_bids.get_derivative_rel_path_prefix() +
                                                           "_seg-hoaMasked_dseg.nii.gz",
                                                           metadata={'Description': hoa_description, 'Manual': False,
                                                                     'Sources': hoa_mask_sources})
                parcellation_results['hoa_masked'] = dict()
                parcellation_results['hoa_masked']['image'] = hoa_mask_bids
                parcellation_results['hoa_masked']['label_definitions'] = hoa_mask_bids.get_path().replace('.nii.gz', '.tsv')
                copy_file(get_label_definitions_path('hoa_subcortical'),
                          parcellation_results['hoa_masked']['label_definitions'])

            hoa_scalar_images = [t1w_biascorr_bids]
            hoa_scalar_descriptions = ['t1wIntensity']

            make_label_stats(hoa_bids, parcellation_results['hoa']['label_definitions'], work_dir,
                             scalar_images=hoa_scalar_images, scalar_descriptions=hoa_scalar_descriptions)
            make_parcellation_qc_plots(t1w_biascorr_bids, brain_mask_bids, hoa_bids, 'hoa', 'HOA', work_dir)

            if mask_hoa:
                make_label_stats(hoa_mask_bids, parcellation_results['hoa_masked']['label_definitions'], work_dir,
                                 scalar_images=hoa_scalar_images, scalar_descriptions=hoa_scalar_descriptions)
                make_parcellation_qc_plots(t1w_biascorr_bids, brain_mask_bids, hoa_mask_bids, 'hoa', 'HOAMasked',
                                           work_dir)

    if dkt31:
        logger.info("Starting DKT31 parcellation")
        dkt31_bids = t1w_bids.get_derivative_image("_seg-dkt31_dseg.nii.gz")
        parcellation_results['dkt31'] = dict()

        if dkt31_bids is not None:
            logger.info("DKT31 parcellation already exists at " + dkt31_bids.get_uri(relative=False))
            parcellation_results['dkt31']['image'] = dkt31_bids
            parcellation_results['dkt31']['label_definitions'] = dkt31_bids.get_path().replace('.nii.gz', '.tsv')
        else:
            dkt31_file = ants_helpers.desikan_killiany_tourville_parcellation(t1w, work_dir, brain_mask)
            dkt31_sources = [t1w_bids.get_uri(relative=True)]

            dkt31_bids = bids_helpers.image_to_bids(
                dkt31_file,
                t1w_bids.get_ds_path(),
                t1w_bids.get_derivative_rel_path_prefix() + "_seg-dkt31_dseg.nii.gz",
                metadata={'Description': 'ANTsPyNet DKT31', 'Manual': False, 'Sources': dkt31_sources}
            )

            parcellation_results['dkt31']['image'] = dkt31_bids
            parcellation_results['dkt31']['label_definitions'] = dkt31_bids.get_path().replace('.nii.gz', '.tsv')
            copy_file(get_label_definitions_path('dkt31'), parcellation_results['dkt31']['label_definitions'])

            if mask_dkt31:
                logger.info("Masking DKT31 parcellation with cortical mask")

                dkt31_mask_sources = [t1w_bids.get_uri(relative=True), cortical_mask_source]
                dkt31_mask_file = ants_helpers.apply_mask(dkt31_file, cortical_mask, work_dir)

                dkt31_mask_bids = bids_helpers.image_to_bids(
                    dkt31_mask_file, t1w_bids.get_ds_path(), t1w_bids.get_derivative_rel_path_prefix() +
                    "_seg-dkt31Masked_dseg.nii.gz", metadata={'Description': 'ANTsPyNet DKT31 masked to cortex',
                                                              'Manual': False, 'Sources': dkt31_mask_sources}
                )
                parcellation_results['dkt31_masked'] = dict()
                parcellation_results['dkt31_masked']['image'] = dkt31_mask_bids,
                parcellation_results['dkt31_masked']['label_definitions'] = dkt31_mask_bids.get_path().replace('.nii.gz',
                                                                                                               '.tsv')
                copy_file(get_label_definitions_path('dkt31'), parcellation_results['dkt31_masked']['label_definitions'])

            if propagate_dkt31:
                logger.info("Propagating DKT31 parcellation to cortical mask")

                dkt31_propagated_sources = [t1w_bids.get_uri(relative=True), cortical_mask_source,
                                            hoa_bids.get_uri(relative=True)]
                # Fill hippocampus and amygdala using HOA labels to prevent propagation of cortical labels into these structures
                [tmp_dkt31, tmp_labels] = ants_helpers.add_labels_to_segmentation(dkt31_file, hoa_bids.get_path(),
                                                                                  [20,21,22,23], work_dir)
                tmp_dkt31 = ants_helpers.propagate_labels_through_mask(cortical_mask, tmp_dkt31, work_dir)
                # Now remove the added labels to make the final output
                dkt31_propagated_file = ants_helpers.remove_labels(tmp_dkt31, tmp_labels, work_dir)

                dkt31_propagated_bids = bids_helpers.image_to_bids(
                    dkt31_propagated_file, t1w_bids.get_ds_path(),
                    t1w_bids.get_derivative_rel_path_prefix() + "_seg-dkt31Propagated_dseg.nii.gz",
                    metadata={'Description': 'ANTsPyNet DKT31 propagated to cortical mask', 'Manual': False,
                              'Sources': dkt31_propagated_sources}
                )
                parcellation_results['dkt31_propagated'] = dict()
                parcellation_results['dkt31_propagated']['image'] = dkt31_propagated_bids
                parcellation_results['dkt31_propagated']['label_definitions'] = \
                    dkt31_propagated_bids.get_path().replace('.nii.gz', '.tsv')
                copy_file(get_label_definitions_path('dkt31'), parcellation_results['dkt31_propagated']['label_definitions'])

            dkt_scalar_images = [t1w_biascorr_bids]
            dkt_scalar_descriptions = ['t1wIntensity']

            if thickness_bids is not None:
                logger.info("Sampling thickness for DKT31 parcellation")
                dkt_scalar_images.append(thickness_bids)
                dkt_scalar_descriptions.append('corticalThickness')

            make_label_stats(dkt31_bids, parcellation_results['dkt31']['label_definitions'], work_dir,
                             scalar_images=dkt_scalar_images, scalar_descriptions=dkt_scalar_descriptions)
            make_parcellation_qc_plots(t1w_biascorr_bids, brain_mask_bids, dkt31_bids, 'dkt31', 'DKT31', work_dir)

            if mask_dkt31:
                make_label_stats(dkt31_mask_bids, parcellation_results['dkt31_masked']['label_definitions'], work_dir,
                             scalar_images=dkt_scalar_images, scalar_descriptions=dkt_scalar_descriptions)
                make_parcellation_qc_plots(t1w_biascorr_bids, brain_mask_bids, dkt31_mask_bids, 'dkt31', 'DKT31Masked',
                                           work_dir)

            if propagate_dkt31:
                make_label_stats(dkt31_propagated_bids, parcellation_results['dkt31_propagated']['label_definitions'],
                                 work_dir, scalar_images=dkt_scalar_images, scalar_descriptions=dkt_scalar_descriptions)
                make_parcellation_qc_plots(t1w_biascorr_bids, brain_mask_bids, dkt31_propagated_bids, 'dkt31',
                                           'DKT31Propagated', work_dir)

    if cerebellum:

        logger.info("Starting cerebellum parcellation")
        cerebellum_bids = t1w_bids.get_derivative_image("_seg-cerebellum_dseg.nii.gz")
        parcellation_results['cerebellum'] = dict()

        if cerebellum_bids is not None:
            logger.info("Cerebellum parcellation already exists at " + cerebellum_bids.get_uri(relative=False))
            parcellation_results['cerebellum']['image'] = cerebellum_bids
            parcellation_results['cerebellum']['label_definitions'] = cerebellum_bids.get_path().replace('.nii.gz', '.tsv')
        else:
            # Create a cerebellum mask from the Harvard-Oxford labels
            cerebellum_mask = ants_helpers.threshold_image(parcellation_results['hoa']['image'].get_path(), work_dir, 29, 32)
            cerebellum_segmentations = ants_helpers.cerebellum_parcellation(t1w, work_dir, cerebellum_mask)
            cerebellum_bids = bids_helpers.image_to_bids(
                cerebellum_segmentations['parcellation'],
                t1w_bids.get_ds_path(),
                t1w_bids.get_derivative_rel_path_prefix() + "_seg-cerebellum_dseg.nii.gz",
                metadata={'Description': 'ANTsPyNet Cerebellum',
                        'Manual': False,
                        'Sources': [t1w_bids.get_uri(relative=True),
                                    parcellation_results['hoa']['image'].get_uri(relative=True)]}
                )

            parcellation_results['cerebellum']['image'] = cerebellum_bids
            # parcellation_results['cerebellum']['tissue_segmentation_image'] = cerebellum_three_tissue_bids
            parcellation_results['cerebellum']['label_definitions'] = \
                parcellation_results['cerebellum']['image'].get_path().replace('.nii.gz', '.tsv')
            copy_file(get_label_definitions_path('antspynet_cerebellum'),
                      parcellation_results['cerebellum']['label_definitions'])

            if mask_cerebellum:
                cerebellum_nocsf_mask = ants_helpers.threshold_image(parcellation_results['hoa_masked']['image'].get_path(),
                                                                     work_dir, 29, 32)
                cerebellum_masked_file = ants_helpers.apply_mask(cerebellum_segmentations['parcellation'],
                                                                 cerebellum_nocsf_mask, work_dir)
                cerebellum_masked_bids = bids_helpers.image_to_bids(
                    cerebellum_masked_file, t1w_bids.get_ds_path(),t1w_bids.get_derivative_rel_path_prefix() +
                    "_seg-cerebellumMasked_dseg.nii.gz",
                    metadata={'Description': 'ANTsPyNet Cerebellum masked to remove CSF voxels', 'Manual': False,
                              'Sources': [t1w_bids.get_uri(relative=True),
                                          parcellation_results['hoa_masked']['image'].get_uri(relative=True)]}
                )
                parcellation_results['cerebellum_masked'] = dict()
                parcellation_results['cerebellum_masked']['image'] = cerebellum_masked_bids
                parcellation_results['cerebellum_masked']['label_definitions'] = \
                    cerebellum_masked_bids.get_path().replace('.nii.gz', '.tsv')
                copy_file(get_label_definitions_path('antspynet_cerebellum'),
                          parcellation_results['cerebellum_masked']['label_definitions'])

            cerebellum_scalar_images = [t1w_biascorr_bids]
            cerebellum_scalar_descriptions = ['t1wIntensity']

            make_label_stats(cerebellum_bids, parcellation_results['cerebellum']['label_definitions'], work_dir,
                             scalar_images=cerebellum_scalar_images, scalar_descriptions=cerebellum_scalar_descriptions)
            make_parcellation_qc_plots(t1w_biascorr_bids, brain_mask_bids, cerebellum_bids, 'cerebellum_parcellation',
                                       'Cerebellum', work_dir)

            if mask_cerebellum:
                make_label_stats(cerebellum_masked_bids, parcellation_results['cerebellum_masked']['label_definitions'],
                                 work_dir, scalar_images=cerebellum_scalar_images,
                                 scalar_descriptions=cerebellum_scalar_descriptions)
                make_parcellation_qc_plots(t1w_biascorr_bids, brain_mask_bids, cerebellum_masked_bids,
                                           'cerebellum_parcellation', 'CerebellumMasked', work_dir)

    return parcellation_results


def parcellate_sst(sst_bids, work_dir):
    raise NotImplementedError("SST parcellation not implemented yet")


def atlas_based_parcellation(t1w_bids, brain_mask_bids, atlas_label_config, work_dir, longitudinal=False, thickness_bids=None,
                             hoa_parcellation_bids=None, segmentation_bids=None):
    """Do atlas-based parcellation on a T1w image.

    Parameters:
    -----------
    t1w_bids : BIDSImage
        BIDS image object for the T1w image, used as a reference space. Any image in the preproc T1w space can be used,
        but the bias-corrected T1w is recommended for consistent intensity statistics.
    brain_mask_bids : BIDSImage
        BIDS image object for the brain mask
    atlas_label_config : str
        Atlas label configuration file, in JSON format
    work_dir : str
        Working directory for the parcellation
    thickness_bids : BIDSImage
        BIDS image object for the cortical thickness image, or None. This is used to define a cortical mask, and is required
        if any atlas has 'restrict_to_cortex' or 'propagate_to_cortex' set to true.
    longitudinal : bool
        Use longitudinal transforms to group template space
    hoa_parcellation_bids : BIDSImage, optional
        Harvard-Oxford parcellation, required if propagate_to_cortex is true for any atlas.
    segmentation_bids : BIDSImage, optional
        antsnetct tissue segmentation, required if mask_csf is true for any atlas.

    Returns:
    -------
    dict with keys for each atlas, containing the parcellation results.
    """
    logger.info("Doing atlas-based parcellation on " + t1w_bids.get_uri(relative=False))

    parcellation_results = dict()

    with open(atlas_label_config, 'r') as f:
        atlas_config = json.load(f)

    # Find the local template transformation for this session
    local_template_transforms = list()

    local_template_name = None

    cortical_mask = None
    not_csf_mask = None

    cortical_mask_source = None

    if segmentation_bids is not None:
        not_csf_mask = ants_helpers.threshold_image(segmentation_bids.get_path(), work_dir, 3, 3, 0, 1)

    if thickness_bids is not None:
        cortical_mask = ants_helpers.threshold_image(thickness_bids.get_path(), work_dir, lower=0.001)
        cortical_mask_source = thickness_bids.get_uri(relative=True)
    elif segmentation_bids is not None:
        cortical_mask = ants_helpers.threshold_image(segmentation_bids.get_path(), work_dir, 8, 8)
        cortical_mask_source = segmentation_bids.get_uri(relative=True)

    if longitudinal:
        ds_path = t1w_bids.get_ds_path()
        subject = t1w_bids.get_entity('sub')

        sst_to_t1w_transform = t1w_bids.get_derivative_path_prefix() + '_from-sst_to-T1w_mode-image_xfm.h5'

        sst_dir = os.path.join(ds_path, f"sub-{subject}", "anat")

        # Use regex to extract template name matching "sub-{subject}_from_T1w_to_([A-Za-z0-9]+)_mode-image_xfm.h5"
        regex = f"sub-{subject}_from-([A-Za-z0-9]+)_to-T1w_mode-image_xfm.h5"
        for file in os.listdir(sst_dir):
            match = re.match(regex, file)
            if match:
                local_template_name = match.group(1)
                local_template_transforms = [sst_to_t1w_transform, os.path.join(sst_dir, file)]
                break

        if local_template_name is None:
            raise ValueError(f"No transforms to group template space found for {t1w_bids.get_uri(relative=False)}")
    else:
        for file in os.listdir(os.path.dirname(t1w_bids.get_path())):
                t1w_derivative_basename = os.path.basename(t1w_bids.get_derivative_path_prefix())
                regex = f"{t1w_derivative_basename}_from-([A-Za-z0-9]+)_to-T1w_mode-image_xfm.h5"
                match = re.match(regex, file)
                if match:
                    local_template_name = match.group(1)
                    local_template_transforms = [os.path.join(os.path.dirname(t1w_bids.get_path()), file)]

        if local_template_name is None:
            raise ValueError(f"No template transforms found for {t1w_bids.get_uri(relative=False)}")

    # Iterate over atlases, which might be defined in another template space
    # We need to find the transform from the atlas template to the local template
    for output_atlas_name, atlas_info in atlas_config.items():
        external_template_name = atlas_info['template_label']

        logger.info(f"Processing atlas labels {output_atlas_name}:\n{json.dumps(atlas_info, indent=4)}")

        # Get the atlas label image
        label_image = bids_helpers.TemplateImage(external_template_name,
                                                 suffix='dseg',
                                                 resolution=atlas_info.get('template_resolution', '01'),
                                                 description=atlas_info.get('atlas_description', None),
                                                 atlas=atlas_info['atlas_label'])

        label_image_dirname = os.path.dirname(label_image.get_path())
        label_image_basename = os.path.basename(label_image.get_path()).replace('.nii.gz', '.tsv')
        label_image_tokens = label_image_basename.split('_')
        label_definitions_tokens = [t for t in label_image_tokens if not t.startswith('res-') and not t.startswith('cohort-')]
        # all this is necessary because there are multiple label files with the same definitions, eg different resolutions
        # or cohort of the same atlas
        label_definitions_file = os.path.join(label_image_dirname, '_'.join(label_definitions_tokens))

        template_to_session_transforms = local_template_transforms.copy()

        if local_template_name != external_template_name:
            template_to_local_template_transform = bids_helpers.find_template_transform(local_template_name,
                                                                                        external_template_name)
            if template_to_local_template_transform is None:
                raise ValueError(f"No transforms found from tpl-{external_template_name} to tpl-{local_template_name}")

            template_to_session_transforms.append(template_to_local_template_transform)

        logger.info(f"Warping atlas labels {label_image.get_uri()} to session space with transforms: " +
                    f"{template_to_session_transforms}")

        # Warp labels to the session space
        parcellation = ants_helpers.apply_transforms(
            t1w_bids.get_path(),
            label_image.get_path(),
            template_to_session_transforms,
            work_dir,
            interpolation='GenericLabel',
            single_precision=True
        )

        parcellation = ants_helpers.apply_mask(parcellation, brain_mask_bids.get_path(), work_dir)

        parcellation_sources = [t1w_bids.get_uri(relative=True), brain_mask_bids.get_uri(relative=True), label_image.get_uri()]

        if atlas_info.get('restrict_to_cortex', False):
            # Restrict the parcellation to the cortical GM
            if cortical_mask is None:
                raise ValueError("antsnet thickness or segmentation image is required for cortical restriction of atlas "
                                 f"{output_atlas_name}")
            parcellation = ants_helpers.apply_mask(parcellation, cortical_mask, work_dir)
            parcellation_sources.append(cortical_mask_source)
        if atlas_info.get('propagate_to_cortex', False):
            if hoa_parcellation_bids is None:
                raise ValueError(f"antsnet parcellation with 'hoa' is required for propagation of atlas {output_atlas_name}")
            [tmp_parcellation, tmp_labels] = ants_helpers.add_labels_to_segmentation(parcellation,
                                                                                     hoa_parcellation_bids.get_path(),
                                                                                     [20,21,22,23],
                                                                                     work_dir)
            tmp_parcellation = ants_helpers.propagate_labels_through_mask(cortical_mask, tmp_parcellation, work_dir)
            # Now remove the added labels
            parcellation = ants_helpers.remove_labels(tmp_parcellation, tmp_labels, work_dir)
            parcellation_sources.extend([cortical_mask_source, hoa_parcellation_bids.get_uri(relative=True)])

        if atlas_info.get('mask_csf', False):
            if segmentation_bids is None:
                raise ValueError(f"antsnetct tissue segmentation is required for CSF masking of atlas {output_atlas_name}")
            # Remove CSF voxels from the parcellation
            parcellation = ants_helpers.apply_mask(parcellation, not_csf_mask, work_dir)
            parcellation_sources.append(segmentation_bids.get_uri(relative=True))

        parcellation_bids = bids_helpers.image_to_bids(
            parcellation,
            t1w_bids.get_ds_path(),
            t1w_bids.get_derivative_rel_path_prefix() + f"_seg-{output_atlas_name}_dseg.nii.gz",
            metadata={'Description': f'antsnetct atlas-based parcellation {output_atlas_name}',
                      'Manual': False,
                      'Sources': parcellation_sources}
            )

        output_label_definitions = parcellation_bids.get_path().replace('.nii.gz', '.tsv')
        copy_file(label_definitions_file, output_label_definitions)
        parcellation_results[output_atlas_name] = dict()
        parcellation_results[output_atlas_name]['segmentation_image'] = parcellation_bids
        parcellation_results[output_atlas_name]['label_definitions'] = output_label_definitions

        # Compute stats
        scalar_images = [t1w_bids]
        scalar_descriptions = ['t1wIntensity']
        if thickness_bids is not None and atlas_info.get('sample_thickness', False):
            scalar_images.append(thickness_bids)
            scalar_descriptions.append('corticalThickness')
        make_label_stats(parcellation_bids, label_definitions_file, work_dir, scalar_images=scalar_images,
                           scalar_descriptions=scalar_descriptions)

        try:
            label_color_map = bids_helpers.load_label_color_map(label_definitions_file)
            make_parcellation_qc_plots(t1w_bids, brain_mask_bids, parcellation_bids, label_color_map, output_atlas_name,
                                       work_dir)
        except Exception as e:
            logger.warning(f"Could not make QC plots for atlas {output_atlas_name}: {type(e)} {str(e)}")

    return parcellation_results


def make_label_stats(label_image_bids, label_def_file, work_dir, compute_label_geometry=True, scalar_images=None,
                     scalar_descriptions=None):
    """Compute stats for a label image, optionally with scalar images. The stats are saved as derivatives of the label image.

    Parameters:
    -----------
    label_image_bids : BIDSImage
        Label image to compute stats on. Should have the entity 'seg' defined for correct naming of the output stats files.
    label_def_file : str
        TSV file containing label definitions
    work_dir : str
        Working directory for the stats
    compute_label_geometry : bool
        If true, compute label geometry stats (volume, surface area, etc.)
    scalar_images : list of BIDSImage, optional
        List of scalar images on which to compute stats.
    scalar_descriptions : list of str, optional
        List of descriptions for the scalar images, used to name the output stats files. Must be the same length as
        scalar_images.

    Returns:
    -------
    list of str: paths to the label stats files
    """
    label_stats_output_files = list()

    label_definitions = bids_helpers.load_label_definitions(label_def_file)

    logger.info(f"Computing stats for label image {label_image_bids.get_uri(relative=False)}")

    if compute_label_geometry:
        logger.info("Computing label geometry stats")
        label_geom_stats = ants_helpers.numpy_label_statistics(label_image_bids.get_path(), label_definitions, work_dir)
        # Save the label statistics to the output dataset
        label_geom_file = label_image_bids.get_derivative_path_prefix() + "_desc-geometry_labelstats.tsv"
        bids_helpers.write_tabular_data(label_geom_stats, label_geom_file)
        label_stats_output_files.append(label_geom_file)

    if scalar_images is not None:
        for scalar, scalar_desc in zip(scalar_images, scalar_descriptions):
            if not os.path.exists(scalar.get_path()):
                raise ValueError(f"Scalar image {scalar.get_uri(relative=False)} does not exist")
            logger.info("Computing scalar stats on image " + scalar.get_uri(relative=False))
            label_stats = ants_helpers.numpy_label_statistics(label_image_bids.get_path(), label_definitions, work_dir,
                                                        scalar.get_path())
            label_stats_file = label_image_bids.get_derivative_path_prefix() + f"_desc-{scalar_desc}_labelstats.tsv"
            bids_helpers.write_tabular_data(label_stats, label_stats_file)
            label_stats_output_files.append(label_stats_file)

    return label_stats_output_files


def make_parcellation_qc_plots(t1w_bids, brain_mask_bids, parcellation_bids, color_map, color_map_title, work_dir):
    """Make QC plots for a parcellation.

    Parameters:
    -----------
    t1w_bids : BIDSImage
        T1w image to use as background
    brain_mask_bids : BIDSImage
        Brain mask for the T1w image
    parcellation_bids : BIDSImage
        Parcellation image to plot
    color_map : str or dict
        Color map to use for the parcellation. This should be a string supported by
        ants_helpers.convert_segmentation_image_to_rgb, or a dict. For example, 'dkt31' or 'hoa'.
    color_map_title : str
        Used to name the output png file.
    work_dir : str
        Working directory for the plots
    """
    logger.info(f"Making QC plots for parcellation {parcellation_bids.get_uri(relative=False)}")

    # Resample everything to 1mm so the PNG plots have a roughly consistent number of slices
    scalar_image = ants_helpers.resample_image_by_spacing(t1w_bids.get_path(), [1, 1, 1], work_dir)

    mask_image = ants_helpers.resample_image_by_spacing(brain_mask_bids.get_path(), [1, 1, 1], work_dir,
                                                       interpolation='NearestNeighbor')

    parcellation_image = ants_helpers.resample_image_by_spacing(parcellation_bids.get_path(), [1, 1, 1], work_dir,
                                                                interpolation='NearestNeighbor')

    # winsorize a bit more aggressively to boost brightness of the brain
    scalar_image = ants_helpers.winsorize_intensity(scalar_image, mask_image, work_dir, lower_percentile=0.0,
                                                    upper_iqr_scale=1.5)

    parcellation_rgb = ants_helpers.convert_segmentation_image_to_rgb(parcellation_image, color_map, work_dir)

    output_desc_ax = f"qcParcellation{color_map_title}Ax"
    output_desc_cor = f"qcParcellation{color_map_title}Cor"

    tiled_ax = ants_helpers.create_tiled_mosaic(scalar_image, mask_image, work_dir, overlay=parcellation_rgb,
                                                      overlay_alpha=0.3, axis=2, pad='mask+5', slice_spec=(3,'mask','mask'))
    tiled_cor = ants_helpers.create_tiled_mosaic(scalar_image, mask_image, work_dir, overlay=parcellation_rgb,
                                                       overlay_alpha=0.3, axis=1, pad='mask+5', slice_spec=(3,'mask','mask'))

    # Could make these derivatives of T1w or the parcellation, use the latter because that's what we do for the TSV files
    system_helpers.copy_file(tiled_ax, parcellation_bids.get_derivative_path_prefix() + f"_desc-{output_desc_ax}.png")
    system_helpers.copy_file(tiled_cor, parcellation_bids.get_derivative_path_prefix() + f"_desc-{output_desc_cor}.png")


def remove_csf_from_hoa_tissue_labels(hoa_file, segmentation_file, work_dir):
    """Masks out CSF from tissue labels in the Harvard-Oxford parcellation using the antsnetct tissue segmentation.

    This does not operate on CSF labels eg (CSF, lateral ventricle) in the HOA parcellation. It only removes CSF voxels from
    tissue labels.

    Parameters:
    -----------
    hoa_file : str
        Path to the Harvard-Oxford parcellation image
    segmentation_file : str
        Path to the antsnetct tissue segmentation image
    work_dir : str
        Working directory for intermediate files

    Returns:
    -------
    str: path to the modified HOA parcellation with CSF labels removed
    """
    # Create a mask of non-CSF voxels from the segmentation
    not_csf_mask = ants_helpers.threshold_image(segmentation_file, work_dir, 3, 3, 0, 1)

    csf_labels = [1,2,3,4,5,6,18,19]
    csf_hoa = ants_helpers.retain_labels(hoa_file, csf_labels, work_dir)
    # Apply the mask to the HOA parcellation
    hoa_nocsf = ants_helpers.apply_mask(hoa_file, not_csf_mask, work_dir)

    # Add back the CSF labels
    [hoa_masked_with_csf, hoa_csf_indices] = ants_helpers.add_labels_to_segmentation(hoa_nocsf, csf_hoa, csf_labels, work_dir,
                                                                                     unique_label_indices=False)

    return hoa_masked_with_csf
