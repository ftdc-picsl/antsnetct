import os
import re

from .system_helpers import run_command, get_nifti_file_prefix, get_temp_file, PipelineError


def trim_neck(input_image, work_dir):
    """Trim the neck from an image

    Parameters:
    ----------
    input_image : str
        Input image filename
    work_dir : str
        Working directory
    Returns:
    -------
    trimmed_image: str
        The trimmed image filename
    """
    output_file_prefix = get_temp_file(work_dir, prefix='trim_neck')
    tmp_image_trim = f"{output_file_prefix}_NeckTrim.nii.gz"

    # trim neck with c3d
    run_command(['trim_neck.sh', '-d', '-c', '20', '-w', work_dir, input_image, tmp_image_trim])

    return tmp_image_trim


def pad_image(input_image, work_dir, pad_mm=10):
    """ Pad an image with zeros

    Parameters:
    ----------
    input_image : str
        Input image filename
    work_dir : str
        Working directory
    pad_mm : float
        Pad size in mm, applied to all sides of the image

    Returns:
    -------
    padded_image (str):
        Padded image filename
    """
    output_file_prefix = get_temp_file(work_dir, prefix='pad_image')
    padded_image = f"{output_file_prefix}_Padded.nii.gz"

    # Pad image with c3d
    padded_image = os.path.join(work_dir, get_nifti_file_prefix(input_image) + '_padded.nii.gz')
    run_command(['c3d', input_image, '-pad', f"{pad_mm}x{pad_mm}x{pad_mm}mm", f"{pad_mm}x{pad_mm}x{pad_mm}mm", '0',
                 '-o', padded_image])
    return padded_image


def conform_image_orientation(input_image, output_orientation, work_dir):
    """Conform an image to a new orientation

    Parameters:
    ----------
    input_image : str
        Input image filename.
    output_orientation : str
        Output orientation defined by a three-letter axis code, e.g. 'LPI'.
    work_dir : str
        Working directory

    Returns:
    -------
    reoriented_image : str
        Output image filename
    """
    output_file_prefix = get_temp_file(work_dir, prefix='conform_orientation')
    reoriented_image = f"{output_file_prefix}_{output_orientation}.nii.gz"

    run_command(['c3d', input_image, '-swapdim', output_orientation, '-o', reoriented_image])

    return reoriented_image


def reset_origin_by_centroid(input_image, centroid_image, work_dir, output_data_type='float'):
    """Reset the origin of an image to the centroid of another image. The two images must be in the same space.

    Parameters:
    ----------
    input_image : str
        Input image filename
    centroid_image : str
        Image to use for the centroid computation. Can be the image itself, or a mask.
    output_data_type : str
        Output data type, e.g. 'uchar' for masks.
    work_dir : str
        Working directory.

    Returns:
    -------
    output_image : str
        Output image filename, with the origin reset.
    """
    output_file_prefix = get_temp_file(work_dir, prefix='reset_origin')
    output_image = f"{output_file_prefix}_origin_reset.nii.gz"

    # Set origin to centroid - this prevents a shift in single-subject template construction
    # because the raw origins are not set consistently across sessions or protocols
    result = run_command(['c3d', centroid_image, '-centroid'])
    centroid_pattern = r'CENTROID_VOX \[([\d\.-]+), ([\d\.-]+), ([\d\.-]+)\]'

    match = re.search(centroid_pattern, result['stdout'])

    if match:
        # Extract the values from the match
        centroid_vox = [float(match.group(1)), float(match.group(2)), float(match.group(3))]
    else:
        raise PipelineError("Could not get centroid from {centroid_image}")

    # Set origin to centroid for both mask and T1w
    centroid_str = str.join('x',[str(c) for c in centroid_vox]) + "vox"
    result = run_command(['c3d', input_image, '-origin-voxel', centroid_str, '-type', output_data_type, '-o', output_image])

    return output_image