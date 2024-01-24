import ants
import antspynet

import os

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


def atropos_segmentation(anatomical_image, brain_mask, priors, work_dir,
                         iterations=5, prior_weight=0.25):
    """Segment anatomical image using Atropos

    Parameters:
    -----------
    anatomical_image: ants.core.ants_image.ANTsImage
        Anatomical image
    brain_mask: ants.core.ants_image.ANTsImage
        Brain mask
    priors: list of ants.core.ants_image.ANTsImage
        List of priors in order 1-6 for CSF, GM, WM, deep GM, brainstem, cerebellum
    work_dir: str
        Path to working directory
    iterations: int
        Number of iterations for Atropos
    prior_weight: float
        Prior weight for Atropos

    Returns:
    --------
    segmentation: ants.core.ants_image.ANTsImage
        Segmentation image
    posteriors: list of ants.core.ants_image.ANTsImage
        List of segmentation posteriors in order 1-6 for CSF, GM, WM, deep GM, brainstem, cerebellum
    """


def ants_atropos_n4(anatomical_images, brain_mask, prior_spec, work_dir, iterations=3, atropos_iterations=5,
                    atropos_prior_weight=0.25, atropos_mrf_weight=0.1, denoise=True, posterior_formulation='Socrates[ 1 ]',
                    n4_prior_classes=[2,3,4,5,6], n4_spline_spacing=180, n4_convergence='[ 50x50x50x50,1e-7 ]', n4_shrink_factor=3):
    """Segment anatomical image using Atropos and N4

    Parameters:
    -----------
    anatomical_images: str or list of str
        Anatomical image(s) to segment

    brain_mask: str
        Path to brain mask

    priors: list of str

    work_dir: str
        Path to working directory

    iterations: int
        Number of iterations for Atropos

    """


def cortical_thickness(segmentation, work_dir, segmentation_posteriors=None,
                       kk_its=45):
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

    Returns:
    --------
    thickness_image: ants.core.ants_image.ANTsImage
        Cortical thickness image
    """