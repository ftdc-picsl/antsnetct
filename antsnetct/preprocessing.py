import os

from .system_helpers import run_command, get_nifti_file_prefix

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

    output_file_prefix = get_nifti_file_prefix(input_image)

    # trim neck with c3d
    tmp_image_trim = os.path.join(work_dir, f"{output_file_prefix}_T1wNeckTrim.nii.gz")

    run_command(['trim_neck.sh', '-d', '-c', '20', '-w', work_dir, input_image, tmp_image_trim])

    return tmp_image_trim

def pad_image(input_image, work_dir, pad_mm=10):
    """ Pad an image with zeros

    Parameters:
    ----------
    input_image (str):
        Input image filename
    work_dir (str):
        Working directory
    pad_mm (float):
        Pad size in mm, applied to all sides of the image

    Returns:
    -------
    padded_image (str):
        Padded image filename
    """

    # Pad image with c3d
    padded_image = os.path.join(work_dir, get_nifti_file_prefix(input_image) + '_padded.nii.gz')
    run_command(['c3d', input_image, '-pad', f"{pad_mm}x{pad_mm}x{pad_mm}mm", f"{pad_mm}x{pad_mm}x{pad_mm}mm", '0',
                 '-o', padded_image])
    return padded_image


def conform_image_orientation(input_image, output_orientation, work_dir):
    """Conform an image to a new orientation

    Parameters:
    ----------
    input_image (str):
        Input image filename.
    output_orientation (str):
        Output orientation, e.g. 'LPI'.
    work_dir (str):
        Working directory

    Returns:
    -------
    reoriented_image (str):
        Output image filename
    """
    reoriented_image = os.path.join(work_dir, get_nifti_file_prefix(input_image) + f"_{output_orientation}.nii.gz")

    run_command(['c3d', input_image, '-swapdim', output_orientation, '-o', reoriented_image])

    return reoriented_image