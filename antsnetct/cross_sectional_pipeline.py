import ants_helpers
import bids_helpers

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback

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

    If a brain mask dataset or segmentation dataset are specified at run time, there must be masks or segmentations for the
    input data, or an error will be raised. This is to prevent inconsistent processing.

    --- Brain masking ---

    The script will check for a brain mask from the brain mask dataset, if defined. If not, a brain mask may be found in the
    input dataset. If not, an implicit mask can be derived from segmentation input. If no brain mask is available, one will be
    generated using anyspynet.


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

    ''')

    required_parser = parser.add_argument_group('Required arguments')
    required_parser.add_argument("--input-dataset", help="Input BIDS dataset dir, containing the source images", type=str,
                          required_parser=True)
    required_parser.add_argument("--participant", help="Participant to process", type=str)
    required_parser.add_argument("--session", help="Session to process", type=str)
    required_parser.add_argument("--output-dataset", help="Output BIDS dataset dir", type=str, required_parser=True)

    optional_parser = parser.add_argument_group('Optional arguments')
    optional_parser.add_argument("-h", "--help", action="help", help="show this help message and exit")
    optional_parser.add_argument("--t1w-image-suffix", help="Use a specific T1w head image suffix. Eg, "
                                 "'acq-mprage_T1w.nii.gz' selects "
                                 "sub-participant/ses-session/sub-participant_ses-session_acq-mprage_T1w.nii.gz' ",
                                 type=str, default=None)
    optional_parser.add_argument("--keep-workdir", help="Copy working directory to output, for debugging purposes. Either "
                                 "'never', 'on_error', or 'always'.", type=str, default='on_error')
    optional_parser.add_argument("--verbose", help="Verbose output", action='store_true')

    atlas_parser = parser.add_argument_group('Template arguments')
    atlas_parser.add_argument("--template", help="Template to use for registration, or 'none' to disable this step", type=str,
                              default='MNI152NLin2009cAsym')
    atlas_parser.add_argument("--template-res", help="Resolution of the template, eg 01, 02, etc. Note this is a templateflow "
                              "index and not a physical spacing", type=str, default='01')
    atlas_parser.add_argument("--template-reg-quick", help="Do quick registration to the template", action='store_true')

    brain_mask_parser = parser.add_argument_group('Brain mask arguments')
    brain_mask_parser.add_argument("--brain-mask-dataset", help="Dataset containing brain masks. Masks from here will be used "
                                   "in preference to those in the input dataset.", type=str, default=None)
    brain_mask_parser.add_argument("--brain-mask-method", help="Brain masking method to use with antspynet. Only used if no "
                                   "pre-existing mask is found. Options are 't1', 't1nobrainer', 't1combined'",
                                   type=str, default='t1')
    segmentation_parser = parser.add_argument_group('Segmentation arguments')
    segmentation_parser.add_argument("--segmentation-method", help="Segmentation method to use. Either 'atropos' or "
                                     "'none'. If atropos, probseg images from the segmentation dataset, if defined, or from "
                                     "deep_atropos will be used as priors for segmentation and bias correction. If no "
                                     "segmentation dataset is provided, a segmentation will be generated by antspynet.",
                                     type=str, default='atropos')
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

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    if args.template == 'none':
        args.template = None

    if args.segmentation_method == 'none':
        args.segmentation_method = None

    input_dataset = args.input_dataset
    output_dataset = args.output_dataset

    if (input_dataset == output_dataset):
        raise ValueError('Input and output datasets cannot be the same')

    input_dataset_description = None

    if os.path.exists(os.path.join(input_dataset, 'dataset_description.json')):
        with open(os.path.join(input_dataset, 'dataset_description.json')) as f:
            input_dataset_description = json.load(f)
    else:
        raise ValueError('Input dataset does not contain a dataset_description.json file')

    # There might be multiple T1ws, if so we process them all
    input_t1w_images = None

    if args.participant is None:
        raise ValueError('Participant must be defined')
    if args.session is None:
        raise ValueError('Session must be defined')

    # Find the input T1w image
    if args.t1w_image_suffix is None:
        args.t1w_image_suffix = 'T1w.nii.gz'

    # Returns a list of T1w images and URIs
    bids_input_t1w = bids_helpers.find_images(input_dataset, args.participant, args.session, 'anat', args.t1w_image_suffix)

    # Create the output dataset and add this container to the GeneratedBy, if needed
    bids_helpers.update_output_dataset(output_dataset, input_dataset_description['Name'] + '_antsnetct')

    for bids_t1w in bids_input_t1w:
        print("Processing T1w image: " + bids_t1w['image'])

        match = re.match('(.*)_T1w\.nii\.gz$', bids_t1w['image'])

        t1w_source_entities = match.group(1)

        # Outputs we will define
        output_brain_mask_image = os.path.join(output_dataset, 'sub-' + args.participant, 'ses-' + args.session, 'anat',
                                                t1w_source_entities + '_desc-brain_mask.nii.gz')

        output_anat_prefix = os.path.join(output_dataset, 'sub-' + args.participant, 'ses-' + args.session, 'anat',
                                          t1w_source_entities)

        output_dataset_name = bids_helpers.get_dataset_name(output_dataset)

        output_segmentation_image = output_anat_prefix + '_seg-antsnetct_dseg.nii.gz'
        output_segmentation_label_def = output_anat_prefix + '_seg-antsnetct_dseg.tsv'
        output_segmentation_posteriors = [output_anat_prefix + f"_seg-antsenetct_label-{structure}_probseg.nii.gz"
                                          for structure in ['CSF', 'GM', 'WM', 'SGM', 'BS', 'CBM']]
        output_thickness_image = output_anat_prefix + '_desc-thickness.nii.gz'

        with tempfile.TemporaryDirectory(suffix=f"antsnetct_{t1w_source_entities}.tmpdir") as work_dir_tmpdir:
            working_dir = work_dir_tmpdir.name

            # First find a brain mask using the first available in order of preference:
            # 1. Brain mask dataset
            # 2. Brain mask in input dataset
            # 3. Implicit mask from prior segmentation
            # 4. Generate a brain mask with antspynet
            brain_mask_image = get_brain_mask(input_dataset, bids_t1w, working_dir, args.brain_mask_dataset,
                                                args.segmentation_dataset, args.brain_mask_method)

            # Copy this to output
            shutil.copy(brain_mask_image, output_brain_mask_image)
            shutil.copy(brain_mask_image.replace('.nii.gz', '.json'), output_brain_mask_image.replace('.nii.gz', '.json'))

            brain_mask_uri = bids_helpers.get_uri(output_dataset, output_dataset_name, output_brain_mask_image)

            # With mask defined, we can now do segmentation and bias correction
            # Segmentation is either pre-defined, or generated with antspynet. Either an external segmentation or an
            # antspynet segmentation can be used as priors for iterative segmentation and bias correction with N4 and Atropos.
            # If Atropos is not used, the T1w is bias-corrected separately with N4.
            seg_n4 = segment_and_bias_correct(bids_t1w['image'], brain_mask_image, args.segmentation_dataset,
                                              args.segmentation_method)

            # Copy segmentation to output
            shutil.copy(seg_n4['segmentation_image'], output_segmentation_image)
            # Write label defintion TSV, which is
            #index  name   abbreviation
            #0  Background  BG
            #1  CSF         CSF
            #2  Gray Matter GM
            #3  White Matter WM
            #4  Subcortical Gray Matter SGM
            #5  Brain Stem  BS
            #6  Cerebellum  CBM

            with open(output_segmentation_label_def, 'w') as label_def_out:
                label_def_out.write('\t'.join(['index', 'name', 'abbreviation']) + '\n')
                label_def_out.write('\t'.join(['0', 'Background', 'BG']) + '\n')
                label_def_out.write('\t'.join(['1', 'CSF', 'CSF']) + '\n')
                label_def_out.write('\t'.join(['2', 'Gray Matter', 'GM']) + '\n')
                label_def_out.write('\t'.join(['3', 'White Matter', 'WM']) + '\n')
                label_def_out.write('\t'.join(['4', 'Subcortical Gray Matter', 'SGM']) + '\n')
                label_def_out.write('\t'.join(['5', 'Brain Stem', 'BS']) + '\n')
                label_def_out.write('\t'.join(['6', 'Cerebellum', 'CBM']) + '\n')

            # Copy posteriors to output
            for idx in range(6):
                shutil.copy(seg_n4['posteriors'][idx], output_segmentation_posteriors[idx])



def get_brain_mask(input_dataset, bids_t1w, work_dir, brain_mask_dataset=None, segmentation_dataset=None,
                   brain_mask_method='t1'):
    """Get a brain mask for a T1w image.

    Copies or defines a brain mask using the first available in order of preference:
        1. Brain mask dataset
        2. Brain mask in input dataset
        3. Implicit mask from prior segmentation
        4. Generate a brain mask with antspynet

    Parameters:
    -----------
    input_dataset : str
        Path to the input dataset.
    bids_t1w : dict
        Dictionary containing information about the T1w image in BIDS format.
    work_dir : str
        Path to the working directory.
    brain_mask_dataset : str, optional
        Path to the brain mask dataset. If provided, it is an error to not find a brain mask.
    segmentation_dataset : str, optional
        Path to the segmentation dataset. If provided, it is used to generate a brain mask if no brain mask is found.
        It is an error to not find a segmentation for the T1w image.
    brain_mask_method : str, optional
        Method to generate the brain mask. Default is 't1'.

    Returns:
    --------
    str
        Path to the selected brain mask.

    Raises:
    -------
    ValueError
        If the brain mask dataset is not None and does not contain a brain mask for the specified T1w image.
    """

    selected_brain_mask = os.path.join(work_dir, 'brain_mask.nii.gz')
    selected_brain_mask_sidecar = selected_brain_mask.replace('.nii.gz', '.json')

    # If a mask dataset is defined, it is an error to not find a mask
    if brain_mask_dataset is not None:
        brain_mask = bids_helpers.find_brain_mask(brain_mask_dataset, bids_t1w['uri'])
        if brain_mask['mask_image'] is None:
            raise ValueError('Brain mask dataset does not contain a brain mask for ' + bids_t1w['uri'])
        # Found a brain mask
        print("Using brain mask: " + brain_mask['image'])
        # Copy the mask and sidecar to the output dataset
        shutil.copy(brain_mask['image'], selected_brain_mask)
        shutil.copy(brain_mask['image'].replace('.nii.gz', '.json'), selected_brain_mask_sidecar)

        # Modify sidecar so its source is the dataset from which we got the mask
        bids_helpers.set_sources(selected_brain_mask_sidecar, brain_mask['uri'])

        return selected_brain_mask

    # If no mask dataset is defined, try to find a mask in the input dataset
    brain_mask = bids_helpers.find_brain_mask(input_dataset, bids_t1w['uri'])
    if brain_mask['mask_image'] is not None:
        # Found a brain mask
        print("Using brain mask: " + brain_mask['image'])
        # Copy the mask and sidecar to the output dataset
        shutil.copy(brain_mask['image'], selected_brain_mask)
        shutil.copy(brain_mask['image'].replace('.nii.gz', '.json'), selected_brain_mask_sidecar)
        bids_helpers.set_sources(selected_brain_mask_sidecar, brain_mask['uri'])
        return selected_brain_mask

    if segmentation_dataset is not None:
        segmentation = bids_helpers.find_segmentation_images(segmentation_dataset, bids_t1w['uri'])
        if segmentation['segmentation_image'] is not None:
            # Found a segmentation
            print("Using segmentation: " + segmentation['image'])
            # Generate a brain mask from the segmentation
            selected_brain_mask = ants_helpers.binarize_brain_mask(segmentation['segmentation_image'], work_dir)
            selected_brain_mask_sidecar = selected_brain_mask.replace('.nii.gz', '.json')
            brain_volume_ml = ants_helpers.get_brain_volume(selected_brain_mask)
            brain_mask_sidecar_json = {'Type': 'Brain', 'Sources': [segmentation['segmentation_uri']],
                                        'Volume': brain_volume_ml, 'VolumeUnit': 'ml'}
            with open(selected_brain_mask_sidecar, 'w') as sidecar_out:
                json.dump(brain_mask_sidecar_json, sidecar_out, indent=2, sort_keys=True)
            return selected_brain_mask


    print("No brain mask found, generating one with antspynet")
    selected_brain_mask = ants_helpers.deep_brain_extraction(bids_t1w['image'], work_dir, brain_mask_method)
    # Write sidecar
    selected_brain_mask_sidecar = selected_brain_mask.replace('.nii.gz', '.json')
    brain_volume_ml = ants_helpers.get_brain_volume(selected_brain_mask)
    output_mask_sidecar_json = {'Type': 'Brain', 'Sources': [bids_t1w['uri']],'Volume': brain_volume_ml, 'VolumeUnit': 'ml'}
    with open(selected_brain_mask_sidecar, 'w') as sidecar_out:
        json.dump(output_mask_sidecar_json, sidecar_out, indent=2, sort_keys=True)
    return selected_brain_mask


def segment_and_bias_correct(bids_t1w, brain_mask_image, work_dir, segmentation_dataset=None, segmentation_method='atropos'):
    """Segment and bias correct a T1w image

    If segmentatation_dataset is not None, it is searched for a segmentation for the T1w image. It is an error
    if the dataset does not contain a segmentation for the T1w image.

    If a pre-existing segmentation is found, it can either be copied directly or used as priors for iterative segmentation
    and bias correction with N4 and Atropos.

    If no pre-existing segmentation is found, priors are generated with anyspynet, and either copied directly or used for
    segmentation and bias correction with N4 and Atropos.

    If the segmentation_method is 'none', the prior segmentation (whether from another dataset, or generated with deep_atropos)
    is copied, and the T1w image is bias corrected with N4.

    If the segmentation_method is 'atropos', the priors are used to iteratively refine the bias correction and segmentation
    using `antsAtroposN4.sh`.

    Parameters:
    -----------
    bids_t1w : dict
        Dictionary containing information about the T1w image in BIDS format.
    brain_mask_image : str
        Path to the brain mask image.
    work_dir : str
        Path to the working directory.
    brain_mask_dataset : str, optional
        Path to the brain mask dataset. If provided, it is an error to not find a brain mask.
    segmentation_dataset : str, optional
        Path to the segmentation dataset.
    brain_mask_method : str, optional
        Method to generate the brain mask. Default is 't1'.

    """

    prior_segmentation = None

    # If a segmentation dataset is defined, it is an error to not find a segmentation
    if segmentation_dataset is not None:
        prior_segmentation = bids_helpers.find_segmentation_images(segmentation_dataset, bids_t1w['uri'])
        if prior_segmentation['segmentation_image'] is None:
            raise ValueError('Segmentation dataset does not contain a segmentation for ' + bids_t1w['uri'])
        # Found a segmentation
        print("Using segmentation: " + prior_segmentation['image'])
    else:
        # If no segmentation is found, generate one with antspynet
        print("No segmentation found, generating one with antspynet")
        prior_segmentation = ants_helpers.deep_atropos(bids_t1w['image'], work_dir)

    if segmentation_method == 'atropos':
        # Run antsAtroposN4.sh, using the priors for segmentation and bias correction
        segmentation = ants_helpers.ants_atropos_n4(bids_t1w['image'], brain_mask_image, prior_segmentation['posteriors'],
                                                    work_dir)
    elif segmentation_method == 'none':
        # Copy the prior segmentation
        segmentation = prior_segmentation

        # Add the bias corrected image
        segmentation['bias_corrected'] = ants_helpers.n4_bias_correction(bids_t1w['image'], brain_mask_image, work_dir)
    else:
        raise ValueError('Unknown segmentation method: ' + segmentation_method)
