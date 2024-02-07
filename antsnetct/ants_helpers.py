import ants
import antspynet

import os
import shutil

from system_helpers import run_command

def deep_brain_extraction(anatomical_image, work_dir, modality='t1'):
    """Extract brain from an anatomical image

    Parameters:
    -----------
    anatomical_image: str
        Anatomical image

    work_dir: str
        Path to working directory

    modality: str
        Brain extraction modality, see antspynet.utilities.brain_extraction for a complete list.

    Returns:
    --------
    brain_mask: str
        Path to brain mask
    """

    anat = ants.image_read(anatomical_image)

    be_output = antspynet.brain_extraction(anat, modality=modality, verbose=True)

    brain_mask = ants.iMath_get_largest_component(ants.threshold_image(be_output, 0.5, 1.5))

    return brain_mask


def deep_atropos(anatomical_image, work_dir):
    """Calls antspynet deep_atropos and returns the resulting segmentation and posteriors

    Parameters:
    -----------
    anatomical_image: str
        Anatomical image

    Returns:
    --------
    segmentation: str
        Path to segmentation image
    posteriors: list of str
        List of paths to segmentation posteriors in order 1-6 for CSF, GM, WM, deep GM, brainstem, cerebellum
        The posteriors are written to disk with numeric format %02d, so that they can be read by Atropos.
    posterior_spec: str
        Ants Atropos posterior specification containing the paths to the posteriors with c-style numeric
        format %02d. This can be used as input to ants_atropos_n4.
    """
    anat = ants.image_read(anatomical_image)
    seg = antspynet.deep_atropos(anat)

    # write results to disk
    segmentation_fn = os.path.join(work_dir, '_deep_atropos_segmentation.nii.gz')
    ants.write_image(seg['segmentation_image'], segmentation_fn)

    posteriors_fn = []

    # Write posteriors to disk with numeric format %02d
    for i, p in enumerate(seg['probability_images']):
        posterior_fn = os.path.join(work_dir, '_deep_atropos_posterior%02d.nii.gz' % i)
        ants.write_image(p, posterior_fn)
        posteriors_fn.append(posterior_fn)

    return {'segmentation': segmentation_fn, 'posteriors': posteriors_fn}


def ants_atropos_n4(anatomical_images, brain_mask, priors, work_dir, iterations=3, atropos_iterations=15,
                    atropos_prior_weight=0.25, atropos_mrf_weight=0.1, denoise=True, use_mixture_model_proportions = True,
                    n4_prior_classes=[2,3,4,5,6], n4_spline_spacing=180, n4_convergence='[ 50x50x50x50,1e-7 ]',
                    n4_shrink_factor=3):
    """Segment anatomical image using Atropos and N4

    Parameters:
    -----------
    anatomical_images: list of str
        List of anatomical images
    brain_mask: str
        Path to brain mask
    priors: list of str
        List of priors in order 1-6 for CSF, GM, WM, deep GM, brainstem, cerebellum
    work_dir: str
        Path to working directory
    iterations: int
        Number of iterations for Atropos
    atropos_iterations: int
        Number of iterations for Atropos
    atropos_prior_weight: float
        Prior weight for Atropos
    atropos_mrf_weight: float
        MRF weight for Atropos
    denoise: bool
        Denoise input images
    use_mixture_model_proportions: bool
        Use mixture model proportions
    n4_prior_classes: list of int
        List of prior classes
    n4_spline_spacing: int
        Spline spacing for N4
    n4_convergence: str
        Convergence criteria for N4
    n4_shrink_factor: int
        Shrink factor for N4

    Returns:
    --------
    segmentation_n4_dict: dict
        Dictionary containing the following keys:
        'bias_corrected': list of str
            List of paths to bias corrected images for each input modality
        'segmentation': str
            Path to segmentation image
        'posteriors': list of str
            List of paths to segmentation posteriors in order 1-6 for CSF, GM, WM, deep GM, brainstem, cerebellum

    """

    # Convert to list if only one image is provided
    if isinstance(anatomical_images, str):
        anatomical_images = [anatomical_images]

    # Write list of priors to work_dir in c-style numeric format %02d
    prior_spec = f"{work_dir}/prior_%02d.nii.gz"

    for i, p in enumerate(priors):
        shutil.copy(p, prior_spec % i)

    anatomical_input_string = ' -a '.join(anatomical_images)

    n4_prior_classes_string = ' -y '.join([str(i) for i in n4_prior_classes])

    command = ['antsAtroposN4.sh', '-d', '3', '-a', anatomical_input_string, '-x', brain_mask, '-p', prior_spec,
               '-c', str(len(priors)), '-o', f"{work_dir}/ants_atropos_n4_", '-m', str(iterations),
               '-n', str(atropos_iterations), '-r', f"[ {atropos_mrf_weight}, 1x1x1 ]", '-w', str(atropos_mrf_weight),
               '-g', str(1 if denoise else 0), '-b', 'Socrates [ ' + str(1 if use_mixture_model_proportions else 0) + ' ]',
               '-y', n4_prior_classes_string, '-w', str(atropos_prior_weight), '-e', n4_convergence,
               '-f', str(n4_shrink_factor), '-q', f"[ {n4_spline_spacing} ]"]

    output = run_command(command)

    # Following the bash script, we run antsAtroposN4.sh again
    # using the corrected image as input

    anatomical_input_string = ' -a '.join([f"{work_dir}/ants_atropos_n4_Segmentation{i}N4.nii.gz" \
                                            for i in range(len(anatomical_images))])

    command = ['antsAtroposN4.sh', '-d', '3', '-a', anatomical_input_string, '-x', brain_mask, '-p', prior_spec,
               '-c', str(len(priors)), '-o', f"{work_dir}/ants_atropos_n4_", '-m', str(2),
               '-n', str(atropos_iterations), '-r', f"[ {atropos_mrf_weight}, 1x1x1 ]", '-w', str(atropos_mrf_weight),
               '-g', str(1 if denoise else 0), '-b', 'Socrates [ ' + str(1 if use_mixture_model_proportions else 0) + ' ]',
               '-y', n4_prior_classes_string, '-w', str(atropos_prior_weight), '-e', n4_convergence,
               '-f', str(n4_shrink_factor), '-q', f"[ {n4_spline_spacing} ]"]

    # Output files are
    # {work_dir}/ants_atropos_n4_Segmentation[i]N4..nii.gz for input modality i
    # {work_dir}/ants_atropos_n4_Segmentation.nii.gz
    # {work_dir}/ants_atropos_n4_SegmentationPosteriors_0%2d.nii.gz
    segmentation_n4_dict = {
                            'bias_corrected': [ f"{work_dir}/ants_atropos_n4_Segmentation{i}N4.nii.gz"
                                               for i in range(len(anatomical_images)) ],
                            'segmentation': f"{work_dir}/ants_atropos_n4_Segmentation.nii.gz",
                            'posteriors': [ f"{work_dir}/ants_atropos_n4_SegmentationPosteriors_%02d.nii.gz" % i \
                                for i in range(1,7) ]
                            }

    return segmentation_n4_dict


def n4_bias_correction(anatomical_image, work_dir, segmentation_posteriors, n4_convergence='[ 50x50x50x50,1e-7 ]',
                       n4_shrink_factor=3, n4_spline_spacing=180):
    """Correct bias field in an anatomical image.

    This function corrects bias in a similar way to antsAtroposN4.sh, but does not update the
    segmentation.

    Parameters:
    -----------
    anatomical_image: str
        Anatomical image to correct
    work_dir: str
        Path to working directory
    segmentation_posteriors: (str)
        List of segmentation posteriors in order 1-6 for CSF, GM, WM, deep GM, brainstem, cerebellum. Posteriors
        2-6 are used to create a pure tissue mask for N4 bias correction.
    n4_convergence: str
        Convergence criteria for N4
    n4_shrink_factor: int
        Shrink factor for N4
    n4_spline_spacing: int
        Spline spacing for N4

    Returns:
    --------
    bias_corrected_image: str
        Path to bias corrected image
    """

    # Make a pure tissue mask from the segmentation posteriors
    pure_tissue_mask = f"{work_dir}/pure_tissue_mask.nii.gz"

    # Everything except CSF goes into mask
    command = ['ImageMath', '3', pure_tissue_mask, 'PureTissueN4WeightMask'].extend(segmentation_posteriors[1:])

    bias_corrected_anatomical = f"{work_dir}/n4_bias_corrected.nii.gz"
    shutil.copy(anatomical_image, bias_corrected_anatomical)

    # run iteratively as is done in antsCorticalThickness.sh
    for iteration in range(2):
        # Truncate outliers
        run_command(['ImageMath', '3', bias_corrected_anatomical, 'TruncateImageIntensity', bias_corrected_anatomical, '0.0',
                     '0.995', '256'])
        # bias correct
        run_command(['N4BiasFieldCorrection', '-d', '3', '-i', bias_corrected_anatomical, '-o', bias_corrected_anatomical,
               '-c', n4_convergence, '-s', str(n4_shrink_factor), '-b', str(n4_spline_spacing)])
        # Normalize and rescale
        run_command(['ImageMath', '3', bias_corrected_anatomical, 'Normalize', bias_corrected_anatomical])
        run_command(['ImageMath', '3', bias_corrected_anatomical, 'm', bias_corrected_anatomical, '1000'])

    # Alternative idea: apply bias field manually, then normalize by the mask. This would avoid normalizing the image
    # into the range 0-1000 including the uncorrected background. But we would need to change antsAtroposN4.sh to
    # stay consistent

    return bias_corrected_anatomical


def cortical_thickness(segmentation, work_dir, segmentation_posteriors=None, kk_its=45, grad_update=0.025, grad_smooth=1.5):
    """Compute cortical thickness from an anatomical image.

    Parameters:
    -----------
    segmentation: str
        Path to segmentation image (GM = 2, WM = 3)
    work_dir: str
        Path to working directory
    segmentation_posteriors: (str)
        List of segmentation posteriors in order 1-6 for CSF, GM, WM, deep GM, brainstem, cerebellum
    kk_its: int
        Number of iterations for cortical thickness estimation
    grad_update: float
        gradient descent update parameter
    grad_smooth: float
        gradient field smoothing parameter

    Returns:
    --------
    thickness_image:str
        Path to cortical thickness image
    """

    if segmentation_posteriors is None:
        kk = ants.kelly_kapowski(s=segmentation, its=kk_its, r=grad_update, x=grad_smooth, verbose=True)
    else:
        kk = ants.kelly_kapowski(s=segmentation, g=segmentation_posteriors[2], w=segmentation_posteriors[3], its=kk_its,
                                 r=grad_update, x=grad_smooth, verbose=True)

    thick_file = os.path.join(work_dir, 'cortical_thickness.nii.gz')
    ants.image_write(kk, thick_file)
    return thick_file


def binarize_brain_mask(segmentation, work_dir):
    """Binarize brain mask

    Parameters:
    -----------
    segmentation: str
        Path to labeled segmentation image where all positive labels are part of the brain mask
    work_dir: str
        Path to working directory

    Returns:
    --------
    brain_mask: str
        Path to brain mask
    """
    mask = ants.image_read(segmentation)
    # Binarize mask image
    mask = ants.threshold_image(mask, 1, None, 1, 0)

    mask_file = os.path.join(work_dir, 'binarized_brain_mask.nii.gz')

    ants.image_write(mask, mask_file)

    return mask_file


def brain_volume_ml(mask_image, work_dir):
    """Compute brain volume from a brain mask

    Parameters:
    -----------
    mask_image: str
        Path to brain mask or labeled segmentation, where brain volume is the volume of all voxels >= 1
        Path to working directory

    Returns:
    --------
    brain_volume: float
        Brain volume in mm^3
    """
    mask = ants.image_read(mask_image)
    # Binarize mask image
    mask = ants.threshold_image(mask, 1, None, 1, 0)

    brain_volume_mm3 = sum(mask.numpy().flatten()) * mask.spacing[0] * mask.spacing[1] * mask.spacing[2]

    brain_volume_ml = brain_volume_mm3 / 1000

    return brain_volume_ml

