import ants
import antspynet

import csv
import logging
import numpy as np
import os

from .system_helpers import run_command, get_nifti_file_prefix, copy_file, get_temp_file, get_temp_dir, get_verbose


def apply_mask(image, mask, work_dir):
    """Multiply an image by a mask

    Parameters:
    -----------
    image: str
        Path to image
    mask: str
        Path to mask. Any voxel > 0 is included in the mask
    work_dir: str
        Path to working directory

    Returns:
    --------
    masked_image: str
        Path to masked image
    """
    img = ants.image_read(image)
    msk = ants.image_read(mask)

    msk_thresh = ants.image_clone(msk)

    msk_thresh[msk_thresh > 0] = 1
    msk_thresh[msk_thresh < 0] = 0

    masked_img = img * msk_thresh

    masked_image_file = get_temp_file(work_dir, prefix='apply_mask') + '_masked.nii.gz'

    ants.image_write(masked_img, masked_image_file)

    return masked_image_file

def smooth_image(image, sigma, work_dir):
    """Smooth an image with a Gaussian kernel

    Parameters:
    -----------
    image: str
        Path to image.
    sigma: float
        Standard deviation of the Gaussian kernel in voxel units.
    work_dir: str
        Path to working directory.

    Returns:
    --------
    smoothed_image: str
        Path to smoothed image.
    """
    img = ants.image_read(image)

    smoothed_img = ants.smooth_image(img, sigma)

    smoothed_image_file = get_temp_file(work_dir, prefix='smooth_image') + '_smoothed.nii.gz'

    ants.image_write(smoothed_img, smoothed_image_file)

    return smoothed_image_file

def average_images(images, work_dir):
    """Average a list of 3D images.

    Parameters:
    -----------
    images: list of str
        List of paths to images to average
    work_dir: str
        Path to working directory

    Returns:
    --------
    avg_image: str
        Path to average image
    """
    avg_image_file = get_temp_file(work_dir, prefix='average_images') + '_mean.nii.gz'

    cmd = ['AverageImages', '3', avg_image_file, '0']

    cmd.extend(images)

    run_command(cmd)

    return avg_image_file

def gamma_correction(image, gamma, work_dir):
    """Apply gamma correction to an image

    Parameters:
    -----------
    image: str
        Path to image.
    gamma: float
        Gamma value for gamma correction.
    work_dir: str
        Path to working directory.

    Returns:
    --------
    corrected_image: str
        Path to gamma corrected image.
    """
    img = ants.image_read(image)

    corrected_img = ants.image_clone(img)
    corrected_img[corrected_img > 0] = corrected_img[corrected_img > 0] ** gamma

    corrected_image_file = get_temp_file(work_dir, prefix='gamma_correction') + '_corrected.nii.gz'

    ants.image_write(corrected_img, corrected_image_file)

    return corrected_image_file


def deep_brain_extraction(anatomical_image, work_dir, modality='t1threetissue'):
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

    # Can be a single mask, or a dict containing a mask and probabilities
    be_output = antspynet.brain_extraction(anat, modality=modality, verbose=get_verbose())

    if isinstance(be_output, dict):
        be_output = be_output['segmentation_image']

    brain_mask = ants.iMath_get_largest_component(ants.threshold_image(be_output, 0.5, 1.5))

    mask_image_file = get_temp_file(work_dir, prefix='brain_masking', suffix="_thresholded.nii.gz")

    ants.image_write(brain_mask, mask_image_file)

    return mask_image_file


def deep_atropos(anatomical_image, work_dir, use_legacy_network=False):
    """Calls antspynet deep_atropos and returns the resulting segmentation and posteriors

    Parameters:
    -----------
    anatomical_image: str
        Anatomical image
    work_dir: str
        Path to working directory
    use_legacy_network: bool
        Use the legacy T1w-only network available in ANTsPyNet 0.2.8, instead of the current HCP-trained network.

    Returns:
    --------
    dict with keys:
        segmentation : str
            Path to segmentation image
        posteriors : list of str
            List of paths to segmentation posteriors in order: CSF, GM, WM, deep GM, brainstem, cerebellum.
    """
    anat = ants.image_read(anatomical_image)

    if use_legacy_network:
        seg = antspynet.deep_atropos(anat, do_preprocessing=True, verbose=get_verbose())
    else:
        seg = antspynet.deep_atropos([anat, None, None], do_preprocessing=True, verbose=get_verbose())

    tmp_file_prefix = get_temp_file(work_dir, prefix='deep_atropos')

    # write results to disk
    segmentation_fn = f"{tmp_file_prefix}_segmentation.nii.gz"
    ants.image_write(seg['segmentation_image'], segmentation_fn)

    posteriors_fn = []

    # Don't return the background class
    atropos_classes = seg['probability_images'][1:7]

    # Write posteriors to disk with numeric format %02d
    for i, p in enumerate(atropos_classes):
        posterior_fn = f"{tmp_file_prefix}_" + 'posterior%02d.nii.gz' % (i + 1)
        ants.image_write(p, posterior_fn)
        posteriors_fn.append(posterior_fn)

    return {'segmentation': segmentation_fn, 'posteriors': posteriors_fn}


def segment_and_bias_correct(anatomical_images, brain_mask, priors, work_dir, denoise=True, which_n4=None, **kwargs):
    """Segment anatomical images using Atropos and N4. This calls a Python implementation of the two-stage
    Atropos-N4 loop from antsCorticalThickness.sh.

    Parameters:
    -----------
    anatomical_images : list of str
        List of paths to coregistered anatomical images
    brain_mask : str
        Path to brain mask
    priors: list of str
        List of priors in order 1-6 for CSF, GM, WM, deep GM, brainstem, cerebellum
    work_dir : str
        Path to working directory
    denoise : bool, optional
        Denoise input images. Default is True.
    which_n4 : list of bool, optional
        List of booleans to determine whether bias correction should be run on each anatomical image. Default is None, which
        runs N4 on all images. If not None, should be a list of booleans with the same length as anatomical_images, where True
        indicates that N4 should be run on that image. If this is False for an image, the bias correction will be skipped for
        that image, as will preprocessing such as intensity winsorization and normalization. Denoising will still be applied if
        denoise=True. A typical use case would be for images that have already been bias corrected, or quantitative images like
        FA.

    **kwargs : dict
        Additional keyword arguments to pass to ants_atropos_n4

    Returns:
    --------
    dict with keys:
        'bias_corrected_anatomical_images' : list of str
            List of paths to bias corrected images for each input modality
        'segmentation' : str
            Path to segmentation image, containing labels 1-6 for CSF, GM, WM, deep GM, brainstem, cerebellum.
        'posteriors' : list of str
            List of paths to segmentation posteriors in order: CSF, GM, WM, deep GM, brainstem, cerebellum.

    """
    if isinstance(anatomical_images, str):
        anatomical_images = [anatomical_images]

    if which_n4 is None:
        which_n4 = [True] * len(anatomical_images)

    # Preprocess the images
    anat_preproc = list()
    for i, anat in enumerate(anatomical_images):
        if which_n4[i]:
            anat = winsorize_intensity(anat, brain_mask, work_dir)
            # Normalize intensity within the brain mask
            anat = normalize_intensity(anat, brain_mask, work_dir, label=1)

        if denoise:
            anat_preproc.append(denoise_image(anat, work_dir))
        else:
            anat_preproc.append(anat)

    round_1 = ants_atropos_n4(anat_preproc, brain_mask, priors, work_dir, which_n4=which_n4, **kwargs)

    # round 2 always runs 2 iterations
    kwargs['iterations'] = 2

    round_2 = ants_atropos_n4(round_1['bias_corrected_anatomical_images'], brain_mask, priors, work_dir,
                              which_n4=which_n4, **kwargs)

    normalized_anatomical = list()

    for i, anat in enumerate(round_2['bias_corrected_anatomical_images']):
        if which_n4[i]:
            # Note we use label 3 here because these are still ANTs labels, 3=WM
            normalized_anatomical.append(normalize_intensity(anat, round_2['segmentation'], work_dir, label=3))
        else:
            normalized_anatomical.append(anat)

    normalized_output = {'bias_corrected_anatomical_images': normalized_anatomical, 'segmentation': round_2['segmentation'],
                         'posteriors': round_2['posteriors']}

    return normalized_output


def ants_atropos_n4(anatomical_images, brain_mask, priors, work_dir, iterations=2, atropos_iterations=15,
                    likelihood_model='Gaussian', mrf_weight=0.1, prior_weight=0.25, use_mixture_model_proportions=True,
                    n4_spline_spacing=180, n4_convergence='[50x50x50x50,1e-7]', n4_shrink_factor=3, which_n4=None):
    """Segment anatomical images using Atropos and N4.

    Parameters:
    -----------
    anatomical_images : list of str
        List of paths to coregistered anatomical images
    brain_mask : str
        Path to brain mask
    priors: list of str
        List of priors in order 1-6 for CSF, GM, WM, deep GM, brainstem, cerebellum
    work_dir : str
        Path to working directory
    iterations : int
        Number of iterations of the N4-Atropos loop
    likelihood_model : str
        Likelihood model for Atropos, with optional parameters. Default is 'Gaussian'.
    atropos_iterations : int
        Number of iterations for Atropos inside each N4-Atropos iterations
    prior_weight: float
        Prior weight for Atropos
    mrf_weight : float
        MRF weight for Atropos
    use_mixture_model_proportions : bool
        Use mixture model proportions
    n4_prior_classes : list of int
        List of prior classes
    n4_spline_spacing : int
        Spline spacing for N4
    n4_convergence : str
        Convergence criteria for N4
    n4_shrink_factor : int
        Shrink factor for N4
    which_n4 : list of bool
        List of booleans to determine which images to run N4 on. Default is None, which runs N4 on all images. If not None,
        should be a list of booleans with the same length as anatomical_images, where True indicates that N4 should be run
        on that image.

    Returns:
    --------
    dict with keys:
        'bias_corrected_anatomical_images' : list of str
            List of paths to bias corrected images for each input modality
        'segmentation' : str
            Path to segmentation image
        'posteriors' : list of str
            List of paths to segmentation posteriors in order: CSF, GM, WM, deep GM, brainstem, cerebellum

    """
    # Convert to list if only one image is provided
    if isinstance(anatomical_images, str):
        anatomical_images = [anatomical_images]

    if which_n4 is None:
        which_n4 = [True] * len(anatomical_images)

    tmp_file_prefix = get_temp_file(work_dir, prefix='ants_atropos_n4')

    # Write list of priors to work_dir in c-style numeric format %02d
    prior_spec = f"{tmp_file_prefix}_prior_%02d.nii.gz"

    for i, p in enumerate(priors):
        copy_file(p, prior_spec % (i+1))

    output_prefix = f"{tmp_file_prefix}_atroposn4_"

    # copy priors to posteriors

    posteriors = priors

    for iteration in range(iterations):

        bias_corrected = list()

        for anat, run_n4 in zip(anatomical_images, which_n4):
            if run_n4:
                corrected_anat = n4_bias_correction(anat, brain_mask, work_dir, posteriors, n4_convergence,
                                                    n4_shrink_factor, n4_spline_spacing)
            else:
                corrected_anat = anat
            bias_corrected.append(corrected_anat)

        seg_output = atropos_segmentation(bias_corrected, brain_mask, work_dir, iterations=atropos_iterations,
                                          prior_probabilities=priors, likelihood_model=likelihood_model, mrf_weight=mrf_weight,
                                          prior_weight=prior_weight,
                                          use_mixture_model_proportions=use_mixture_model_proportions)

        posteriors = seg_output['posteriors']

    # return segmentation and bias-corrected anatomical images
    segmentation_n4_dict = {'bias_corrected_anatomical_images': bias_corrected,
                            'segmentation': seg_output['segmentation'],
                            'posteriors': seg_output['posteriors']}

    return segmentation_n4_dict


def _base_ants_atropos_n4(anatomical_images, brain_mask, priors, work_dir, iterations=3, atropos_iterations=15,
                    prior_weight=0.25, mrf_weight=0.1, denoise=True, use_mixture_model_proportions=True,
                    n4_prior_classes=[2,3,4,5,6], n4_spline_spacing=180, n4_convergence='[50x50x50x50,1e-7]',
                    n4_shrink_factor=3):
    """Segment anatomical images using Atropos and N4, via the antsAtroposN4.sh script in ANTs.

    Parameters:
    -----------
    anatomical_images : list of str
        List of paths to coregistered anatomical images
    brain_mask : str
        Path to brain mask
    priors: list of str
        List of priors in order 1-6 for CSF, GM, WM, deep GM, brainstem, cerebellum
    work_dir : str
        Path to working directory
    iterations : int
        Number of iterations of the N4-Atropos loop
    atropos_iterations : int
        Number of iterations for Atropos inside each N4-Atropos iterations
    prior_weight: float
        Prior weight for Atropos
    mrf_weight : float
        MRF weight for Atropos
    denoise : bool
        Denoise input images
    use_mixture_model_proportions : bool
        Use mixture model proportions
    n4_prior_classes : list of int
        List of prior classes
    n4_spline_spacing : int
        Spline spacing for N4
    n4_convergence : str
        Convergence criteria for N4
    n4_shrink_factor : int
        Shrink factor for N4

    Returns:
    --------
    dict with keys:
        'bias_corrected_anatomical_images' : list of str
            List of paths to bias corrected images for each input modality
        'segmentation' : str
            Path to segmentation image
        'posteriors' : list of str
            List of paths to segmentation posteriors in order: CSF, GM, WM, deep GM, brainstem, cerebellum

    """
    # Convert to list if only one image is provided
    if isinstance(anatomical_images, str):
        anatomical_images = [anatomical_images]

    tmp_file_prefix = get_temp_file(work_dir, prefix='ants_atropos_n4')

    # Write list of priors to work_dir in c-style numeric format %02d
    prior_spec = f"{tmp_file_prefix}_prior_%02d.nii.gz"

    for i, p in enumerate(priors):
        copy_file(p, prior_spec % (i+1))

    anatomical_input_args = [arg for anat in anatomical_images for arg in ['-a', anat]]

    n4_prior_classes_args = [arg for tissue_label in n4_prior_classes for arg in ['-y', str(tissue_label)]]

    stage_1_output_prefix = f"{tmp_file_prefix}_stage1_"

    command = ['antsAtroposN4.sh', '-d', '3']
    command.extend(anatomical_input_args)
    command.extend(n4_prior_classes_args)
    command.extend(['-x', brain_mask, '-p', prior_spec, '-c', str(len(priors)), '-o', stage_1_output_prefix, '-m',
                   str(iterations), '-n', str(atropos_iterations), '-r', f"[{mrf_weight}, 1x1x1]", '-g',
                   '1' if denoise else '0', '-b', f"Socrates[{1 if use_mixture_model_proportions else 0}]",
                   '-w', str(prior_weight), '-e', n4_convergence, '-f', str(n4_shrink_factor), '-q',
                   f"[{n4_spline_spacing}]"])

    run_command(command)

    # Following the bash script, we run antsAtroposN4.sh again
    # using the corrected image as input
    stage_2_output_prefix = f"{tmp_file_prefix}_stage_2_"

    anatomical_images = [f"{stage_1_output_prefix}Segmentation{i}N4.nii.gz" for i in range(len(anatomical_images))]
    anatomical_input_args = [arg for anat in anatomical_images for arg in ['-a', anat]]

    command = ['antsAtroposN4.sh', '-d', '3']
    command.extend(anatomical_input_args)
    command.extend(n4_prior_classes_args)
    command.extend(['-x', brain_mask, '-p', prior_spec, '-c', str(len(priors)), '-o', stage_2_output_prefix, '-m', '2',
                    '-n', str(atropos_iterations), '-r', f"[{mrf_weight}, 1x1x1]", '-g', '0', '-b',
                    f"Socrates[{'1' if use_mixture_model_proportions else '0'}]", '-w', str(prior_weight), '-e',
                    n4_convergence, '-f', str(n4_shrink_factor), '-q', f"[{n4_spline_spacing}]"])

    run_command(command)

    segmentation_n4_dict = {
                            'bias_corrected_anatomical_images': [f"{stage_2_output_prefix}Segmentation{i}N4.nii.gz"
                                               for i in range(len(anatomical_images))],
                            'segmentation': f"{stage_2_output_prefix}Segmentation.nii.gz",
                            'posteriors': [f"{stage_2_output_prefix}SegmentationPosteriors%02d.nii.gz" % i \
                                for i in range(1,7)]
                            }

    return segmentation_n4_dict


def denoise_image(anatomical_image, work_dir):
    """Denoise an anatomical image.

    Denoises an anatomical image using the DenoiseImage command from ANTs.

    Parameters:
    -----------
    anatomical_image : str
        Path to the anatomical image to denoise.
    work_dir : str
        Path to working directory.

    Returns:
    --------
    denoised: str
        Path to the denoised image.
    """
    tmp_file_prefix = get_temp_file(work_dir, prefix='denoise')
    denoised = f"{tmp_file_prefix}_denoised.nii.gz"
    command = ['DenoiseImage', '-d', '3', '-i', anatomical_image, '-o', denoised]
    run_command(command)

    return denoised


def n4_bias_correction(anatomical_image, brain_mask, work_dir, segmentation_posteriors=None,
                       n4_convergence='[50x50x50x50,1e-7]', n4_shrink_factor=3, n4_spline_spacing=180):
    """Correct bias field in an anatomical image.

    This function corrects bias in a similar way to antsAtroposN4.sh, but does not update the
    segmentation.

    Parameters:
    -----------
    anatomical_image : str
        Path to the anatomical image to correct
    brain_mask : str
        Path to the brain mask
    work_dir : str
        Path to working directory
    segmentation_posteriors : list of str, optional
        List of segmentation posteriors in order 1-6 for CSF, GM, WM, deep GM, brainstem, cerebellum. Posteriors
        2-6 are used to create a pure tissue mask for N4 bias correction.
    n4_convergence : str, optional
        Convergence criteria for N4
    n4_shrink_factor : int, optional
        Shrink factor for N4
    n4_spline_spacing : int, optional
        Spline spacing for N4

    Returns:
    --------
    bias_corrected_image: str
        Path to bias corrected image
    """
    # Make a pure tissue mask from the segmentation posteriors
    tmp_file_prefix = get_temp_file(work_dir, prefix='n4')

    pure_tissue_mask = None

    if segmentation_posteriors is not None:
        pure_tissue_mask = f"{tmp_file_prefix}_pure_tissue_mask.nii.gz"
        # Everything except CSF goes into mask
        command = ['ImageMath', '3', pure_tissue_mask, 'PureTissueN4WeightMask']
        command.extend(segmentation_posteriors[1:])
        run_command(command)

    bias_corrected_anatomical = f"{tmp_file_prefix}_bias_corrected.nii.gz"
    copy_file(anatomical_image, bias_corrected_anatomical)

    # bias correct
    n4_cmd = ['N4BiasFieldCorrection', '-d', '3', '-i', bias_corrected_anatomical, '-o', bias_corrected_anatomical,
              '-c', n4_convergence, '-s', str(n4_shrink_factor), '-b', f"[{n4_spline_spacing}]", '-x', brain_mask, '-v', '1']

    if pure_tissue_mask is not None:
        n4_cmd.extend(['-w', pure_tissue_mask])

    run_command(n4_cmd)

    return bias_corrected_anatomical


def atropos_segmentation(anatomical_images, brain_mask, work_dir, iterations=15, convergence_threshold=0.00001,
                         kmeans_classes=0, prior_probabilities=None, prior_weight=0.25, likelihood_model='Gaussian',
                         use_mixture_model_proportions=False, mrf_neighborhood='1x1x1', mrf_weight=0.1,
                         adaptive_smoothing_weight=0.0, partial_volume_classes=None, use_random_seed=True):
    """Segment anatomical images using Atropos

    Parameters:
    -----------
    anatomical_image : str or list of str
        List coregistered anatomical image files
    brain_mask : str
        Path to brain mask
    work_dir : str
        Path to working directory
    iterations : int
        Number of iterations for Atropos
    convergence_threshold : float
        Convergence threshold for Atropos
    kmeans_classes : int
        Number of classes for K-means initialization. Default is 0, which implies prior-based initialization.
    prior_probabilities: list of str
        List of prior probability images. For an antsCorticalThickness segmentation, these must be in order 1-6 for CSF, GM,
        WM, deep GM, brainstem, cerebellum. Required if kmeans_classes is 0.
    prior_weight : float
        Prior probability weight, used with prior_probabilities.
    likelihood_model : str
        Likelihood model for Atropos, with optional parameters. Default is 'Gaussian'. Recommended settings are 'Gaussian'
        or 'HistogramParzenWindows'.
    use_mixture_model_proportions : bool
        Use mixture model proportions in posterior calculation.
    mrf_neighborhood : str
        Markov Random Field neighborhood for Atropos. Default is '1x1x1'. Larger neighborhoods are more smooth, but increase
        computation time substantially.
    mrf_weight : float
        MRF weight for Atropos.
    adaptive_smoothing_weight : float
        Adaptive smoothing weight for anatomical images. Default is 0.0.
    partial_volume_classes : list of str
        List of partial volume classes. Default is None. Example: ['1x2'] for partial volume between classes 1 and 2.
    use_random_seed : bool
        Use a variable random seed for Atropos. Default is True. Set to false to use a fixed seed.

    Returns:
    --------
    dict with keys:
        'segmentation' : str
            Path to segmentation image
        'posteriors' : list of str
            List of paths to segmentation posteriors for each class.

    The order of class labels and posteriors is defined by the order of the prior_probabilities list if specified, or
    intensity in the first anatomical image if using kmeans_classes.
    """
    tmp_file_prefix = get_temp_file(work_dir, prefix='atropos')

    if isinstance(anatomical_images, str):
        anatomical_images = [anatomical_images]

    atropos_cmd = ['Atropos', '-d', '3', '--verbose', '-x', brain_mask]

    anatomical_args = list()

    if adaptive_smoothing_weight > 0.0:
        anatomical_args = [arg for anat in anatomical_images for arg in ['-a', f"[{anat}, {adaptive_smoothing_weight}]"]]
    else:
        anatomical_args = [arg for anat in anatomical_images for arg in ['-a', anat]]

    atropos_cmd.extend(anatomical_args)

    num_classes = kmeans_classes

    if prior_probabilities is not None:
        num_classes = len(prior_probabilities)

    if kmeans_classes > 0:
        atropos_cmd.extend(['-i', f"KMeans[{kmeans_classes}]"])
    else:
        if prior_probabilities is None:
            raise ValueError("Prior probabilities must be specified if kmeans_classes is 0.")
        # Write list of priors to work_dir in c-style numeric format %02d
        prior_spec = f"{tmp_file_prefix}_prior_%02d.nii.gz"
        for idx, prior in enumerate(prior_probabilities):
            copy_file(prior, prior_spec % (idx + 1))
        atropos_cmd.extend(['-i', f"PriorProbabilityImages[{num_classes},{prior_spec},{prior_weight}]"])

    seg_out_prefix = tmp_file_prefix + 'output_'

    seg_out_image = f"{seg_out_prefix}Segmentation.nii.gz"

    seg_out_prior_spec = f"{seg_out_prefix}Posteriors%02d.nii.gz"

    atropos_cmd.extend(['-c', f"[{iterations},{convergence_threshold}]", '-k', likelihood_model, '-o',
                       f"[{seg_out_image},{seg_out_prior_spec}]", '-m', f"[{mrf_weight},{mrf_neighborhood}]",
                       '-p', f"Socrates[{1 if use_mixture_model_proportions else 0}]",
                       '-r', str(1) if use_random_seed else str(0), '-e', '0', '-g', '1'])

    if partial_volume_classes is not None:
        if isinstance(partial_volume_classes, str):
            partial_volume_classes = [partial_volume_classes]
        atropos_cmd.extend([arg for pvc in partial_volume_classes for arg in ['-s', str(pvc)]])

    run_command(atropos_cmd)

    posteriors = [f"{seg_out_prefix}Posteriors%02d.nii.gz" % i for i in range(1, num_classes + 1)]

    return {'segmentation': seg_out_image, 'posteriors': posteriors}


def cortical_thickness(segmentation, segmentation_posteriors, work_dir, kk_its=45, grad_update=0.025, grad_smooth=1.5,
                       gm_lab=8, wm_lab=2, sgm_lab=9):
    """Compute cortical thickness from an anatomical image segmentation.

    Following antsCorticalThickness.sh, subcortical GM will be added to the WM label and posterior probabilty, cortical
    thickness is computed from the SGM/WM boundary to the pial surface.

    Parameters:
    -----------
    segmentation : str
        Path to segmentation image.
    segmentation_posteriors : str
        List of segmentation posteriors in order 1-6 for CSF, GM, WM, deep GM, brainstem, cerebellum
    work_dir : str
        Path to working directory
    kk_its : int
        Number of iterations for cortical thickness estimation
    grad_update : float
        gradient descent update step size parameter
    grad_smooth : float
        gradient field smoothing parameter
    gm_lab : int
        Label for cortical gray matter in the segmentation image
    wm_lab : int
        Label for white matter in the segmentation image
    sgm_lab : int
        Label for subcortical gray matter in the segmentation image

    Returns:
    --------
    thickness_image:str
        Path to cortical thickness image
    """
    tmp_file_prefix = get_temp_file(work_dir, prefix='cortical_thickness')

    # Make a temporary copy of the segmentation, we will modify this to merge subcortical GM into WM
    kk_seg = ants.image_read(segmentation)

    # Add subcortical GM to WM
    kk_seg[kk_seg == sgm_lab] = wm_lab

    kk_seg_file = f"{tmp_file_prefix}_kk_seg.nii.gz"

    ants.image_write(kk_seg, kk_seg_file)

    wm_posterior = ants.image_read(segmentation_posteriors[2])
    sgm_posterior = ants.image_read(segmentation_posteriors[3])

    kk_wm_posterior = wm_posterior + sgm_posterior

    kk_wm_posterior_file = f"{tmp_file_prefix}_kk_wm_posterior.nii.gz"

    ants.image_write(kk_wm_posterior, kk_wm_posterior_file)

    thick_file = f"{tmp_file_prefix}_cortical_thickness.nii.gz"
    # We'll do things on the command line so we can access all the options and check the exit code
    # kk = ants.kelly_kapowski(s=kk_seg, g=gm_posterior, w=kk_wm_posterior, its=kk_its, r=grad_update, x=grad_smooth,
    #                         verbose=get_verbose(), gm_label=gm_lab, wm_label=wm_lab)
    # ants.image_write(kk, thick_file)

    # Encode additional defaults from antsCorticalThickness.sh
    # DIRECT_CONVERGENCE="[45,0.0,10]" - iterations modifiable here as in antsCorticalThickness.sh, default same
    #
    # DIRECT_THICKNESS_PRIOR="10" - fixed as in antsCorticalThickness.sh
    #
    # DIRECT_GRAD_STEP_SIZE="0.025" - modifiable but default as in antsCorticalThickness.sh.
    #
    # DIRECT_SMOOTHING_PARAMETER="1.5" - This is modified by turning on b-spline smoothing, which we don't support here.
    #                                    Default to the same value as antsCorticalThickness.sh
    #
    # DIRECT_NUMBER_OF_DIFF_COMPOSITIONS="10" - Fixed as in antsCorticalThickness.sh
    #
    # USE_BSPLINE_SMOOTHING=0 - this is modifiable by the user, but we don't support b-spline smoothing here.
    cmd = ['KellyKapowski', '-d', '3', '-s', f"[{kk_seg_file}, {gm_lab}, {wm_lab}]", '-g', segmentation_posteriors[1],
           '-w', kk_wm_posterior_file, '-o', thick_file, '-r', str(grad_update), '-m', str(grad_smooth),
           '-c', f"[{kk_its},0.0,10]", '-b', '0', '-t', '10', '-n', '10', '-v', '1']

    run_command(cmd)

    return thick_file


def univariate_template_registration(fixed_image, moving_image, work_dir, fixed_mask=None, moving_mask=None,
                                     metric='CC', metric_param_str='4', transform='SyN[0.2,3,0]',
                                     iterations='20x40x60x70x70x10', shrink_factors='8x6x4x3x2x1',
                                     smoothing_sigmas='5x4x3x2x1x0vox', apply_transforms=True):
    """Pairwise registration with defaults selected for population template registration, similar to antsCorticalTHickness.sh.

    Does a linear and non-linear registration of the moving image to the fixed image with antsRegistration. Affine
    parameters are optimized for inter-subject registration.

    The user-specified parameters control the deformable stage of the registration. The default parameters are
    used for the affine stages, similar to antsRegistrationSyN.sh

    Parameters:
    -----------
    fixed_image : str
        Path to fixed image
    moving_image : str
        Path to moving image
    work_dir : str
        Path to working directory
    fixed_mask : str
        Path to fixed metric mask
    moving_mask : str
        Path to moving metric mask
    metric : str
        Image metric to use for registration with parameters. Default is 'CC' for cross-correlation.
    metric_param_str : str
        Parameters for the image metric, appended to the metric argument such that we use
        "{metric_name}[{fixed},{moving},1,{metric_param_str}]". Default is '4' for cross-correlation with a radius of 4 voxels.
    transform : str
        Transformation model, e.g. 'SyN[0.2,3,0]' for symmetric normalization with gradient step length 0.2, 3 voxel smoothing
        of the update field and no smoothing of the deformation field.
    iterations : str
        Number of iterations at each level of the registration. Number of levels must match shrink and smoothing parameters.
    shrink_factors : str
        Shrink factors at each level of the registration. Number of levels must match iterations and smoothing parameters.
    smoothing_sigmas : str
        Smoothing sigmas at each level of the registration. Number of levels must match shrink and iterations parameters.
    apply_transforms : bool
        Apply the resulting transform to the moving and fixed images

    Returns:
    --------
    dict with keys:
        'fwd_transform' : str
            Path to composite forward transform
        'inv_transform' : str
            Path to composite inverse transform
        'moving_image_warped' : str
            Path to warped moving image, if apply_transforms is True
        'fixed_image_warped' : str
            Path to warped fixed image, if apply_transforms is True
    """
    tmp_file_prefix = get_temp_file(work_dir, prefix="reg")

    metric_arg = f"{metric}[{fixed_image},{moving_image},1,{metric_param_str}]"

    mask_arg = f"[{fixed_mask},{moving_mask}]"

    # Run antsRegistration

    output_root = f"{tmp_file_prefix}_movingToFixed_"
    ants_cmd = command = [
        'antsRegistration',
        '--verbose', '1',
        '--dimensionality', '3',
        '--float', '0',
        '--collapse-output-transforms', '1',
        '--output', output_root,
        '--write-composite-transform', '1',
        '--use-histogram-matching', '0',
        '--winsorize-image-intensities', '[0.005,0.995]',
        '--masks', mask_arg,
        '--initial-moving-transform', f"[{fixed_image},{moving_image},1]",
        '--transform', 'Rigid[0.1]',
        '--metric', f"Mattes[{fixed_image},{moving_image},1,32,Regular,0.25]",
        '--convergence', '[100x500x250x0,1e-6,10]',
        '--shrink-factors', '8x4x2x1',
        '--smoothing-sigmas', '3x2x1x0vox',
        '--transform', 'Affine[0.1]',
        '--metric', f"Mattes[{fixed_image},{moving_image},1,32,Regular,0.25]",
        '--convergence', '[1000x500x250x0,1e-6,10]',
        '--shrink-factors', '8x4x2x1',
        '--smoothing-sigmas', '3x2x1x0vox',
        '--transform', transform,
        '--metric', metric_arg,
        '--convergence', f"[{iterations},1e-6,10]",
        '--shrink-factors', shrink_factors,
        '--smoothing-sigmas', smoothing_sigmas
    ]

    run_command(ants_cmd)

    composite_fwd_transform = f"{output_root}Composite.h5"
    composite_inv_transform = f"{output_root}InverseComposite.h5"

    moving_image_warped = None
    fixed_image_warped = None

    if apply_transforms:

        moving_image_warped = f"{output_root}Warped.nii.gz"

        apply_fwd_cmd = [
            'antsApplyTransforms',
            '--dimensionality', '3',
            '--input', moving_image,
            '--reference-image', fixed_image,
            '--output', moving_image_warped,
            '--interpolation', 'BSpline',
            '--transform', composite_fwd_transform,
            '--verbose', '1'
        ]

        run_command(apply_fwd_cmd)

        fixed_image_warped = f"{output_root}InverseWarped.nii.gz"

        apply_inv_cmd = [
            'antsApplyTransforms',
            '--dimensionality', '3',
            '--input', fixed_image,
            '--reference-image', moving_image,
            '--output', fixed_image_warped,
            '--interpolation', 'BSpline',
            '--transform', composite_inv_transform,
            '--verbose', '1'
        ]

        run_command(apply_inv_cmd)

        return {'forward_transform': composite_fwd_transform, 'inverse_transform': composite_inv_transform,
                'moving_image_warped': moving_image_warped, 'fixed_image_warped': fixed_image_warped}

    return {'forward_transform': composite_fwd_transform, 'inverse_transform': composite_inv_transform}


def apply_transforms(fixed_image, moving_image, transforms, work_dir, interpolation='Linear'):
    """Apply transforms, resampling moving image into fixed image space.

    Parameters:
    -----------
    fixed_image : str
        Path to fixed image
    moving_image : str
        Path to moving image
    transforms : str or list of str
        Path to transform file, a list of files, or 'Identity' for an identity transform
    work_dir : str
        Path to working directory
    interpolation : str
        Interpolation method, e.g. 'Linear', 'NearestNeighbor'

    Returns:
    --------
    moving_image_warped : str
        Path to warped moving image
    """
    tmp_file_prefix = get_temp_file(work_dir, prefix="aat")

    moving_image_warped = f"{tmp_file_prefix}_warped.nii.gz"

    apply_cmd = [
        'antsApplyTransforms',
        '--dimensionality', '3',
        '--input', moving_image,
        '--reference-image', fixed_image,
        '--output', moving_image_warped,
        '--interpolation', interpolation,
        '--verbose', '1'
    ]

    if isinstance(transforms, str):
        apply_cmd.extend(['--transform', transforms])
    else:
        apply_cmd.extend([item for t in transforms for item in ('--transform', t)])



    run_command(apply_cmd)

    return moving_image_warped


def reslice_to_reference(reference_image, source_image, work_dir):
    """Reslice an image to conform to a reference image. This is a simple wrapper around apply_transforms, assuming
    the identity transform and NearestNeighbor interpolation.

    Parameters:
    -----------
    reference_image : str
        Path to reference image
    source_image : str
        Path to source image
    work_dir : str
        Path to working directory

    Returns:
    --------
    resliced_image : str
        Path to resliced image
    """
    resliced = apply_transforms(reference_image, source_image, 'Identity', work_dir, interpolation='NearestNeighbor')
    return resliced


def posteriors_to_segmentation(posteriors, work_dir, class_labels=[0, 3, 8, 2, 9, 10, 11]):
    """Convert posteriors to a segmentation image

    Parameters:
    -----------
    posteriors : list of str
        List of paths to segmentation posteriors in order 1-6 for CSF, GM, WM, deep GM, brainstem, cerebellum
    work_dir: str
        Path to working directory
    class_labels : list of int
        List of labels corresponding to the classes in order 0-6 for background, CSF, GM, WM, deep GM, brainstem, cerebellum.
        The default labels use BIDS common imaging derivatives labels. To use antscorticalthickness numeric labels, set this to
        list(range(0,7)).

    Returns:
    --------
    segmentation : str
        Path to segmentation image
    """

    posterior_images = [ants.image_read(p) for p in posteriors]

    posteriors_np = [posterior.numpy() for posterior in posterior_images]

    # Stack these arrays along a new axis (axis=0 by default in np.stack)
    stacked_posteriors = np.stack(posteriors_np, axis=0)

    prob_sum = np.sum(stacked_posteriors, axis=0)

    # Background probability, set to 1-(sum of other probabilities)
    background_np = 1 - prob_sum
    background_np = np.clip(background_np, 0, 1)
    background_np_expanded = np.expand_dims(background_np, axis=0)
    stacked_posteriors_with_background = np.concatenate((background_np_expanded, stacked_posteriors), axis=0)

    # Find the index of the maximum probability for each voxel
    seg_indices = np.argmax(stacked_posteriors_with_background, axis=0)

    # Map these indices to the class_labels
    seg_indices_function = np.vectorize(lambda x: class_labels[x])
    output_seg_indices = seg_indices_function(seg_indices)
    output_seg_indices = output_seg_indices.astype(np.uint32)

    # Convert the numpy array of indices back to an ANTs image if necessary
    # Use one of the original images to get the space information (e.g., spacing, origin, direction)
    reference_image = posterior_images[0]
    seg = ants.from_numpy(output_seg_indices, spacing=reference_image.spacing, origin=reference_image.origin,
                          direction=reference_image.direction)

    tmp_file_prefix = get_temp_file(work_dir, prefix='prob_to_seg')
    seg_file = f"{tmp_file_prefix}_SegFromPosteriors.nii.gz"

    ants.image_write(seg, seg_file)

    return seg_file


def threshold_image(image, work_dir, lower=None, upper=None, inside_value=1, outside_value=0):
    """Threshold an image.

    Parameters:
    -----------
    image : str
        Path to image to threshold
    work_dir : str
        Path to working directory
    lower : float, optional
        Lower threshold value. If none, default to the minimum value in the image minus an epsilon.
    upper : float, optional
        Upper threshold value. If none, default to the maximum value in the image plus an epsilon.
    inside_value : float
        Value to set for voxels within the threshold. Default is 1.
    outside_value : float, optional
        Value to set for voxels outside the threshold. Default is 0.

    Returns:
    --------
    thresholded_image : str
        Path to thresholded image

    """
    tmp_file_prefix = get_temp_file(work_dir, prefix='threshold')

    mask_file = f"{tmp_file_prefix}_thresholded.nii.gz"

    mask = ants.threshold_image(ants.image_read(image), lower, upper, inside_value, outside_value)

    ants.image_write(mask, mask_file)

    return mask_file



def binary_morphology(mask, work_dir, operation='close', radius=2):
    """Apply binary morphological operations to a mask

    Parameters:
    -----------
    mask : str
        Path to binary mask image
    work_dir : str
        Path to working directory
    operation : str
        Morphological operation to apply. Default is 'close'. Options are 'close', 'open', 'dilate', 'erode'.
    radius : int
        Radius of the structuring element in voxels. Default is 2.

    Returns:
    --------
    mask_morph : str
        Path to morphologically processed mask

    """
    mask_image = ants.image_read(mask)

    tmp_file_prefix = get_temp_file(work_dir, prefix='morphology')

    mask_morph_file = f"{tmp_file_prefix}_morph.nii.gz"

    mask_morph = ants.morphology(mask_image, operation, radius, mtype='binary', value=1)

    ants.image_write(mask_morph, mask_morph_file)

    return mask_morph_file


def brain_volume_ml(mask_image):
    """Compute brain volume from a brain mask

    Parameters:
    -----------
    mask_image : str
        Path to brain mask or labeled segmentation, where brain volume is the volume of all voxels >= 1
        Path to working directory

    Returns:
    --------
    brain_volume : float
        Brain volume in mm^3
    """
    mask = ants.image_read(mask_image)
    # Binarize mask image
    mask = ants.threshold_image(mask, 1, None, 1, 0)

    brain_volume_mm3 = sum(mask.numpy().flatten()) * mask.spacing[0] * mask.spacing[1] * mask.spacing[2]

    brain_volume_ml = brain_volume_mm3 / 1000

    return brain_volume_ml


def get_log_jacobian_determinant(reference_image, transform, work_dir, use_geom=False):
    """Compute the log of the determinant of the Jacobian of a transform

    Parameters:
    -----------
    reference_image : str
        Path to reference image.
    transform : str
        Path to transform file in the space of the reference image. This should be a composite h5 forward transform
        from the moving to the fixed space, containing a composite Affine transform and a warp.
    work_dir : str
        Path to working directory.
    use_geom : bool
        If True, use the geometric calculation.

    Returns:
    --------
    log_jacobian : str
        Path to log of the determinant of the Jacobian
    """
    tmp_file_prefix = get_temp_file(work_dir, prefix='jacobian')

    log_jacobian_file = f"{tmp_file_prefix}_logjac.nii.gz"

    # First decompose transform into its components

    decomposed_basename_prefix = f"{tmp_file_prefix}_decomposed"

    cmd = ['CompositeTransformUtil', '--disassemble', transform, os.path.join(work_dir, decomposed_basename_prefix)]

    run_command(cmd)

    warp_file = f"{decomposed_basename_prefix}_01_DisplacementFieldTransform.nii.gz"

    if not os.path.exists(warp_file):
        # The composite transform might not contain a warp, or might be an inverse transform
        # in antsnetct we always use forward transforms, so assume here we were passed an affine only transform,
        # eg for longitudinal registration
        return None

    cmd = ['CreateJacobianDeterminantImage', '3', warp_file, log_jacobian_file, '1', '1' if use_geom else '0']

    run_command(cmd)

    return log_jacobian_file


def winsorize_intensity(image, mask, work_dir, lower_percentile=0.5, upper_iqr_scale=3):
    """Winsorize the intensity of an image, using a mask to calculate bounds.

    The lower bound is the lower_percentile of the image within the mask, after removing any voxels with intensity <= 0.

    The upper bound is set conservatively using the histogram and inter-quartile range of the image within the mask.

    The aim of the winsorization is to remove outliers and improve the robustness of bias correction and denoising, without
    removing contrast within brain tissue.

    These parameters are a compromise designed to work well enough on different contrasts.

    Parameters:
    -----------
    image : str
        Path to image to truncate.
    mask : str
        Path to mask image.
    work_dir : str
        Path to working directory.
    lower_percentile : float, optional
        Lower percentile for winsorization. Default is 0.5. Set to 0 to include all positive values.
    upper_iqr_scale : float, optional
        Scale factor for upper bound, anything over (inter-quartile range) * upper_iqr_scale is winsorized.
    Returns:
    --------
    winsorized_image : str
        Path to winsorized image

    """
    tmp_file_prefix = get_temp_file(work_dir, prefix='winsorbymask')

    img = ants.image_read(image)
    mask = ants.image_read(mask)

    brain = img[mask > 0]

    # Remove anything <= 0
    brain = brain[brain > 0.0]

    low_threshold = np.percentile(brain, lower_percentile)

    quantiles = np.quantile(brain, [0.0, 0.25, 0.5, 0.75, 1.0])

    # Winsorize at median + 3 * IQR
    high_threshold = quantiles[2] + upper_iqr_scale * (quantiles[3] - quantiles[1])

    img_winsor = img.clone()
    img_winsor[img_winsor < low_threshold] = low_threshold
    img_winsor[img_winsor > high_threshold] = high_threshold

    winsor_file = f"{tmp_file_prefix}_winsorized.nii.gz"

    ants.image_write(img_winsor, winsor_file)

    return winsor_file


def winsorize_intensity_with_seg(image, segmentation, low_label, high_label, work_dir):
    """Winsorize the intensity of an image, using a segmentation to calculate bounds.

    The lower bound is taken from the lowest 0.1 percentile of the image, after removing any voxels with intensity <= 0, within
    the segmentation of the low_label tissue class (CSF for T1w). The upper bound is set conservatively using the histogram and
    inter-quartile range of the image within the high_label (WM for T1w). The aim of the winsorization is to remove outliers
    and improve the robustness of bias correction and denoising, without removing contrast within brain tissue.

    Parameters:
    -----------
    image : str
        Path to image to truncate.
    segmentation : str
        Path to segmentation image.
    low_label : int
        Label for lower bound calculation.
    high_label : int
        Label for upper bound calculation.
    work_dir : str
        Path to working directory.
    Returns:
    --------
    truncated_image : str
        Path to truncated image
    """
    tmp_file_prefix = get_temp_file(work_dir, prefix='winsorbylabel')

    img = ants.image_read(image)
    seg = ants.image_read(segmentation)

    low_voxels = img[seg == low_label]
    high_voxels = img[seg == high_label]

    # Remove anything <= 0
    low_voxels = low_voxels[low_voxels > 0.0]
    # Winsorize at 0.1% of the low tissue class
    low_threshold = np.percentile(low_voxels, 0.1)

    high_quantiles = np.quantile(high_voxels, [0.0, 0.25, 0.5, 0.75, 1.0])

    # Winsorize at median + 5 * IQR
    high_threshold = high_quantiles[2] + 5 * (high_quantiles[3] - high_quantiles[1])

    img_winsor = img.clone()
    img_winsor[img_winsor < low_threshold] = low_threshold
    img_winsor[img_winsor > high_threshold] = high_threshold

    winsor_file = f"{tmp_file_prefix}_winsorized.nii.gz"

    ants.image_write(img_winsor, winsor_file)

    return winsor_file


def normalize_intensity(image, segmentation, work_dir, label=2, scaled_label_mean=1000):
    """Normalize intensity of an image so that the mean intensity of a tissue class is a specified value.

    Parameters:
    -----------
    image : str
        Path to image to normalize.
    segmentation : str
        Path to segmentation image.
    work_dir : str
        Path to working directory.
    label : int, optional
        Label of tissue class to normalize. Default is 2 (BIDS common derived label for white matter).
    scaled_label_mean : float
        Mean intensity of the tissue class after normalization.

    """
    tmp_file_prefix = get_temp_file(work_dir, prefix='norm_intensity')

    img = ants.image_read(image)
    seg = ants.image_read(segmentation)

    mask = seg == label

    mean_intensity_in_mask = np.mean(img[mask])

    img_normalized = img * (scaled_label_mean / mean_intensity_in_mask)

    img_normalized_file = f"{tmp_file_prefix}_normalized_to_label_{label}.nii.gz"

    ants.image_write(img_normalized, img_normalized_file)

    return img_normalized_file


def build_sst(images, work_dir, **kwargs):
    """Construct a template from the input images. This is a simplified interface to build_template, with default parameters
    for SST construction.

    Parameters:
    ----------
    images : list
        List of BIDSImage objects, or a list containing one list of images for each modality. For example, a single-modality
        template may have images=['a.nii.gz', 'b.nii.gz', 'c.nii.gz'], while a multi-modality template might have
        images=[['a_t1w.nii.gz', 'b_t1w.nii.gz', 'c_t1w.nii.gz'], ['a_t2w.nii.gz', 'b_t2w.nii.gz', 'c_t2w.nii.gz']]. The
        number of modalities must match the length of the reg_metric_weights, and the images from the same subject must
        be at the same index in each modality list.
    work_dir : str
        Working directory
    **kwargs :
        Additional keyword arguments build_template.

    Returns:
    -------
    dict with keys:
        'template_image' : str
            template image filename
        'template_input_warped' : str
            List of warped input images
        'template_transforms': str
            List of transforms from input images to template

    """
    kwargs.setdefault('initial_templates', None)
    kwargs.setdefault('reg_iterations', '40x40x50x10')
    kwargs.setdefault('reg_metric_weights', None)
    kwargs.setdefault('reg_shrink_factors', '4x3x2x1')
    kwargs.setdefault('reg_smoothing_sigmas', '3x2x1x0vox')
    kwargs.setdefault('reg_transform', 'SyN[0.2, 3, 0.5]')
    kwargs.setdefault('reg_metric', 'CC[3]')
    kwargs.setdefault('template_iterations', '5')
    kwargs.setdefault('template_norm', 'mean')
    kwargs.setdefault('template_sharpen', 'unsharp_mask')

    return build_template(images, work_dir, **kwargs)


def build_template(images, work_dir, initial_templates=None, reg_transform='SyN[0.2, 3, 0]', reg_metric = 'CC[4]',
                   reg_metric_weights=None, reg_iterations='40x40x50x20', reg_shrink_factors='6x4x2x1',
                   reg_smoothing_sigmas='3x2x1x0vox', template_iterations=4, template_norm='mean',
                   template_sharpen='laplacian'):
    """Construct a template from the input images.

    The images should be preprocessed so that they share:
        * a common basic orientation, eg LPI
        * origin coordinates in a similar anatomical location, eg the center of the brain
        * a similar FOV, so that the same anatomy is present in all images
        * a similar intensity range, such that they can be averaged without losing contrast

    Parameters:
    ----------
    images : list
        List of BIDSImage objects, or a list containing one list of images for each modality. For example, a single-modality
        template may have images=['a.nii.gz', 'b.nii.gz', 'c.nii.gz'], while a multi-modality template might have
        images=[['a_t1w.nii.gz', 'b_t1w.nii.gz', 'c_t1w.nii.gz'], ['a_t2w.nii.gz', 'b_t2w.nii.gz', 'c_t2w.nii.gz']]. The
        number of modalities must match the length of the reg_metric_weights, and the images from the same subject must
        be at the same index in each modality list.
    work_dir :str
        Working directory
    initial_templates : str or list of str
        Initial template(s) to use for registration. If None, the first image for each modality is used.
    template_iterations : int
        Number of iterations for template construction.
    reg_transform : str
        Transform for registration.
    reg_metric : str
        Metric for registration. This controls the metric for the final registration. Earlier linear stages use MI. This is
        passed directly to the template script, and hence needs to contain the metric name and parameters, eg 'CC[4]'.
    reg_metric_weights : list
        Weights for the registration metric. Default is None, for equal weights.
    reg_iterations : str
        Number of iterations for registration.
    reg_shrink_factors : str
        Shrink factors for registration
    reg_smoothing_sigmas : str
        Smoothing sigmas for registration
    reg_transform : str
        Transform for registration. Should be Rigid[step], Affine[step], SyN[params] or BSplineSyN[params]. If using a
        deformable transform, affine and rigid stages are prepended automatically.
    template_norm : str
        Template intensity normalization. Options are 'mean', 'normalized_mean', 'median'.
    template_sharpen : str
        Template sharpening. Options are 'none', 'laplacian', 'unsharp_mask'.

    Returns:
    -------
    list of str
        List of paths to template images, one per modality.
    """
    template_workdir = get_temp_dir(work_dir, prefix='build_template')

    output_prefix = os.path.join(template_workdir, 'T_')

    template_norm = template_norm.lower()

    template_norm_options = {'mean': '0', 'normalized_mean': '1', 'median': '2'}

    template_sharpen = template_sharpen.lower()

    template_sharpen_options = {'none': '0', 'laplacian': '1', 'unsharp_mask': '2'}

    num_modalities = 1
    num_images = len(images)

    if isinstance(images[0], list):
        num_modalities = len(images)
        num_images = len(images[0])
        for mod_images in images:
            if len(mod_images) != num_images:
                raise ValueError("All modalities must have the same number of images.")
    else:
        # One modality, passed flat list of images
        images = [images]

    if initial_templates is not None:
        if isinstance(initial_templates, str):
            initial_templates = [initial_templates]

        if len(initial_templates) != num_modalities:
            raise ValueError("The number of modalities must match the length of the initial_templates list.")

    if reg_metric_weights is None:
        reg_metric_weights = [1] * num_modalities
    else:
        if len(reg_metric_weights) != num_modalities:
            raise ValueError("The number of modalities must match the length of the reg_metric_weights list.")

    reg_metric_weights_str = 'x'.join([str(w) for w in reg_metric_weights])

    # Write list of images to a csv file
    image_csv = os.path.join(template_workdir, 'template_image_list.csv')

    if num_modalities > 1:
        # Write a CSV file with one row per subject, one column per modality
        with open(image_csv, 'w', newline='\n') as f:
            writer = csv.writer(f, lineterminator='\n')
            for row in zip(*images):
                writer.writerow(row)
    else:
        # Write a CSV file with one row per subject
        with open(image_csv, 'w', newline='\n') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(images)

    initial_template_params = list()

    for idx in range(num_modalities):
        if initial_templates is not None:
            initial_template_params.extend(['-z', initial_templates[idx]])
        else:
            initial_template_params.extend(['-z', images[idx][0]])

    template_command = ['antsMultivariateTemplateConstruction2.sh', '-d', '3', '-a', template_norm_options[template_norm], '-A',
                        template_sharpen_options[template_sharpen], '-o', output_prefix, '-n', '0', '-k', str(num_modalities),
                        '-w', reg_metric_weights_str, '-i', str(template_iterations), '-f', reg_shrink_factors, '-r', '0',
                        '-s', reg_smoothing_sigmas, '-q', reg_iterations, '-m', reg_metric, '-t', reg_transform]

    template_command.extend(initial_template_params)

    template_command.append(image_csv)

    run_command(template_command)

    template_images = [f"{output_prefix}template{idx}.nii.gz" for idx in range(num_modalities)]

    for idx,template_file in enumerate(template_images):
        if not os.path.exists(template_file):
            raise ValueError(f"Template file {template_file} not found. Template construction failed.")

    return template_images


def multivariate_pairwise_registration(fixed_images, moving_images, work_dir, fixed_mask=None, moving_mask=None,
                                       metric='CC', metric_param_str='4', metric_weights=None, transform='SyN[0.2,3,0]',
                                       iterations='20x30x70x70x10', shrink_factors='8x6x4x2x1', smoothing_sigmas='4x3x2x1x0vox',
                                       apply_transforms=True):
    """Multivariate pairwise registration of images.

    This is a simplified interface to multivariate_registration, with default parameters for pairwise registration. It will
    also work with single-modality images.

    Parameters:
    -----------
    fixed_images : list or str
        List of fixed images, in the same physical space.
    moving_images : list or str
        List of moving images, in the same physical space.
    work_dir : str
        Path to working directory.
    fixed_mask : str
        Path to fixed mask image.
    moving_mask : str
        Path to moving mask image.
    metric : str
        Image metric to use for registration with parameters. Default is 'CC' for cross-correlation.
    metric_param_str : str
        Parameters for the image metric, appended to the metric argument such that we use
        "{metric_name}[{fixed},{moving},{modality_weight}",{metric_param_str}]. Default is '4' for cross-correlation with a
        radius of 4 voxels.
    metric_weights : list
        Weights for the registration metric. Default is None, for equal weights. If not None, must be a list of the same
        length as the number of modalities.
    transform : str
        Transformation model, e.g. 'SyN[0.2,3,0]' for symmetric normalization with gradient step length 0.2, 3 voxel smoothing
        of the update field and no smoothing of the deformation field.
    iterations : str
        Number of iterations at each level of the registration. Number of levels must match shrink and smoothing parameters.
    shrink_factors : str
        Shrink factors at each level of the registration. Number of levels must match iterations and smoothing parameters.
    smoothing_sigmas : str
        Smoothing sigmas at each level of the registration. Number of levels must match shrink and iterations parameters.
    apply_transforms : bool
        If true, apply the resulting transforms to the images.

    Returns:
    --------
    dict with keys:
        'forward_transform' : str
            Forward composite transform
        'inverse_transform' : str
            Inverse composite transform
    if apply_transforms, additional keys are:
        'moving_images_warped' : list of str
            List of warped moving images
        'fixed_images_warped' : list of str
            List of warped fixed images
    """
    if isinstance(fixed_images, str):
        fixed_images = [fixed_images]
    if isinstance(moving_images, str):
        moving_images = [moving_images]

    num_modalities = len(fixed_images)

    if len(moving_images) != num_modalities:
        raise ValueError("The number of modalities must match the length of the moving_images list.")

    if metric_weights is None:
        metric_weights = [1] * num_modalities
    else:
        if len(metric_weights) != num_modalities:
            raise ValueError("The number of modalities must match the length of the metric_weights list.")

    tmp_output_prefix = get_temp_file(work_dir, prefix=f"mv_pairwise_reg")

    transform_prefix = f"{tmp_output_prefix}_moving_to_fixed_"

    linear_metric_params = list()

    # Note, ANTs called from subprocess.run can cope with spaces in brackets around numbers, but not file names
    # So '-c', '[ 100x250x50x0, 1e-6, 10 ]' works but '-r', '[ fixed.nii.gz, moving.nii.gz, 1 ]' does not

    for modality_idx in range(num_modalities):
        linear_metric_params.extend(
            ['--metric', f"MI[{fixed_images[modality_idx]},{moving_images[modality_idx]},{metric_weights[modality_idx]},32]"])

    rigid_stage = ['--transform', 'Rigid[0.1]']
    rigid_stage.extend(linear_metric_params)
    rigid_stage.extend(['--convergence', '[100x250x50x0, 1e-6, 10]', '--shrink-factors', '8x4x2x1', '--smoothing-sigmas',
                    '4x2x1x0vox'])

    affine_stage = ['--transform', 'Affine[0.1]']
    affine_stage.extend(linear_metric_params)
    affine_stage.extend(['--convergence', '[100x250x50x0, 1e-6, 10]', '--shrink-factors', '8x4x2x1', '--smoothing-sigmas',
                    '4x2x1x0vox'])

    mask_arg = f"[{fixed_mask},{moving_mask}]"

    reg_command = ['antsRegistration', '--dimensionality', '3', '--float', '0', '--collapse-output-transforms', '1',
                    '--output', transform_prefix, '--interpolation', 'Linear', '--winsorize-image-intensities',
                    '[0.0,0.995]', '--use-histogram-matching', '0', '--initial-moving-transform',
                    f"[{fixed_images[0]},{moving_images[0]},1]", '--write-composite-transform', '1', '--masks', mask_arg,
                    '--verbose']

    if transform.startswith('Affine'):
        reg_command.extend(rigid_stage)
    elif transform.startswith('Rigid'):
        pass
    else:
        # Assume a deformable transform here
        reg_command.extend(rigid_stage)
        reg_command.extend(affine_stage)

    last_stage_metric_args = list()

    for modality_idx in range(num_modalities):
        last_stage_metric_args.extend(
            ['--metric', f"{metric}[{fixed_images[modality_idx]},{moving_images[modality_idx]}," +
                f"{metric_weights[modality_idx]},{metric_param_str}]"])

    reg_command.extend(['--transform', transform])
    reg_command.extend(last_stage_metric_args)
    reg_command.extend(['--convergence', f"[{iterations}, 1e-6, 10]", '--shrink-factors', shrink_factors,
                        '--smoothing-sigmas', smoothing_sigmas])

    run_command(reg_command)

    forward_transform = f"{transform_prefix}Composite.h5"
    inverse_transform = f"{transform_prefix}InverseComposite.h5"

    if apply_transforms:
        fwd_warped_images = list()
        inv_warped_images = list()
        for modality_idx in range(num_modalities):
            moving_image_warped = os.path.join(
                work_dir, f"{get_nifti_file_prefix(moving_images[modality_idx])}_to_fixed_{modality_idx}_warped.nii.gz")
            apply_fwd_cmd = ['antsApplyTransforms', '--dimensionality', '3', '--input', moving_images[modality_idx],
                             '--reference-image', fixed_images[modality_idx], '--output', moving_image_warped,
                             '--interpolation', 'BSpline', '--transform', forward_transform, '--verbose', '1']
            run_command(apply_fwd_cmd)
            fwd_warped_images.append(moving_image_warped)

            fixed_image_warped = os.path.join(
                work_dir, f"{get_nifti_file_prefix(fixed_images[modality_idx])}_to_moving_{modality_idx}_warped.nii.gz")
            apply_inv_cmd = ['antsApplyTransforms', '--dimensionality', '3', '--input', fixed_images[modality_idx],
                             '--reference-image', moving_images[modality_idx], '--output', fixed_image_warped,
                             '--interpolation', 'BSpline', '--transform', inverse_transform, '--verbose', '1']
            run_command(apply_inv_cmd)
            inv_warped_images.append(fixed_image_warped)

        return {'forward_transform': forward_transform, 'inverse_transform': inverse_transform,
                'moving_images_warped': fwd_warped_images, 'fixed_images_warped': inv_warped_images}
    else:
        return {'forward_transform': forward_transform, 'inverse_transform': inverse_transform}


def multivariate_sst_registration(fixed_images, moving_images, work_dir, **kwargs):
    """Wrapper for multivariate pairwise registration with default parameters for intrasubject registration.

    Parameters:
    -----------
    fixed_images (list):
        List of fixed images, in the same physical space.
    moving_images (list):
        List of moving images, in the same physical space.
    work_dir (str):
        Path to working directory.
    **kwargs: dict
        Additional keyword arguments for multivariate_pairwise_registration.
    """
    # Set default values for any parameters that are not provided
    kwargs.setdefault('fixed_mask', None)
    kwargs.setdefault('moving_mask', None)
    kwargs.setdefault('metric', 'CC')
    kwargs.setdefault('metric_param_str', '3')
    kwargs.setdefault('metric_weights', None)
    kwargs.setdefault('transform', 'SyN[0.2,3,0.5]')
    kwargs.setdefault('iterations', '30x60x70x10')
    kwargs.setdefault('shrink_factors', '4x3x2x1')
    kwargs.setdefault('smoothing_sigmas', '3x2x1x0vox')
    kwargs.setdefault('apply_transforms', True)

    return multivariate_pairwise_registration(fixed_images, moving_images, work_dir, **kwargs)


def combine_masks(masks, work_dir, thresh = 0.0001):
    """Combine a list of binary masks in the same space into a single mask. Masks are added together and thresholded.
    A very small threshold is approximately a union of the masks, while thresh=(number of masks) is approximately the
    intersection.

    Parameters:
    ----------
    masks : list of str
        List of masks to combine. Must be in the same space.
    work_dir : str
        Path to working directory
    thresh : float
        Threshold for inclusion in the combined mask. Voxels whose sum over all masks exceeds this value are included.

    Returns:
    -------
    str: Path to the combined mask
    """
    # Load the first mask
    combined_mask = ants.image_read(masks[0], pixeltype='unsigned char')

    # Add the rest
    for mask in masks[1:]:
        combined_mask = combined_mask + ants.image_read(mask)

    combined_mask = combined_mask > thresh

    tmp_file_prefix = get_temp_file(work_dir, prefix='combine_mask')
    combined_mask_file = f"{tmp_file_prefix}_combined_mask.nii.gz"

    ants.image_write(combined_mask, combined_mask_file)

    return combined_mask_file


def get_image_spacing(image):
    """Get the voxel spacing of an image

    Parameters:
    -----------
    image : str
        Path to image

    Returns:
    --------
    list of float
        Voxel spacing in mm
    """
    img = ants.image_read(image)
    return img.spacing

def get_image_size(image):
    """Get the size of an image

    Parameters:
    -----------
    image : str
        Path to image

    Returns:
    --------
    list of int
        Image size in voxels
    """
    img = ants.image_read(image)
    return img.shape


def resample_image_by_spacing(image, target_spacing, work_dir, interpolation='Linear'):
    """Resample an image to a new voxel spacing.

    Parameters:
    -----------
    image : str
        Path to image
    target_spacing : list of float
        Target voxel spacing in mm
    work_dir : str
        Path to working directory
    interpolation : str
        Interpolation method. Default is 'Linear'.

    Returns:
    --------
    resampled_image : str
        Path to resampled image
    """
    img = ants.image_read(image)

    interp_code = 0

    # one of 0 (linear), 1 (nearest neighbor), 2 (gaussian), 3 (windowed sinc), 4 (bspline)
    if interpolation.lower() == 'linear':
        interp_code = 0
    elif interpolation.lower() == 'nearestneighbor':
        interp_code = 1
    elif interpolation.lower() == 'gaussian':
        interp_code = 2
    elif interpolation.lower() == 'windowedsinc':
        interp_code = 3
    elif interpolation.lower() == 'bspline':
        interp_code = 4
    else:
        raise ValueError(f"Interpolation method {interpolation} not recognized. Must be one of: 'Linear', 'NearestNeighbor', " +
                        "'Gaussian', 'WindowedSinc', 'BSpline'")

    resampled = ants.resample_image(img, target_spacing, use_voxels=False, interp_type=interp_code)

    tmp_file_prefix = get_temp_file(work_dir, prefix='resample_by_spacing')
    resampled_file = f"{tmp_file_prefix}_resampled.nii.gz"
    ants.image_write(resampled, resampled_file)
    return resampled_file


def pad_image(image, pad_spec, work_dir, pad_to_shape=False):
    """Pad an image with zeros

    Parameters:
    -----------
    image : str
        Path to image
    pad_spec : list of int
        Padding in voxels, e.g. [10, 10, 10] pads by 10 voxels on each side in x, y, z.

        If pad_to_shape=True, the image will be padded until it reaches the specified size. If it is already larger, it will
        not be altered.

        If pad_to_shape=False, this can also be a list of list, e.g. [[0, 10], [10, 10], [5, 0]] to pad different amounts in
        each dimension.
    work_dir : str
        Path to working directory

    Returns:
    --------
    padded_image : str
        Path to padded image
    """
    tmp_file_prefix = get_temp_file(work_dir, prefix='pad_image')

    img = ants.image_read(image)

    if pad_to_shape:
        padded = ants.pad_image(img, shape=pad_spec)
    else:
        padded = ants.pad_image(img, pad_width=pad_spec)

    padded_file = f"{tmp_file_prefix}_padded.nii.gz"

    ants.image_write(padded, padded_file)

    return padded_file