import os

from .system_helpers import run_command, get_nifti_file_prefix

def trim_neck(input_image, work_dir, pad_mm=10):
    """Trim the neck from an image

    Parameters:
    ----------
    input_image : str
        Input image filename
    work_dir : str
        Working directory
    pad_mm : int
        Padding in mm to add to the trimmed image
    Returns:
    -------
    dict: a dictionary with keys
        'trimmed_image' - trimmed image
        'trim_region_input_space' - a mask of the trimmed region in the original space
    """

    output_file_prefix = get_nifti_file_prefix(input_image)

    # trim neck with c3d, reslice mask into trimmed space
    tmp_image_trim = os.path.join(work_dir, f"{output_file_prefix}_T1wNeckTrim.nii.gz")

    # This is in the original space, and contains 1 for voxels in the trimmed output
    # and 0 for voxels outside the trimmed region. Used for QC
    tmp_trim_region_image = os.path.join(work_dir, f"{output_file_prefix}_T1wNeckTrimRegion.nii.gz")

    result = run_command(['trim_neck.sh', '-d', '-c', '20', '-w', work_dir, '-m', tmp_trim_region_image, input_image,
                            tmp_image_trim])

    # Pad image with c3d and reslice mask to same space
    result = run_command(['c3d', tmp_image_trim, '-pad', f"{pad_mm}x{pad_mm}x{pad_mm}mm",
                            f"{pad_mm}x{pad_mm}x{pad_mm}mm", '0', '-o', tmp_image_trim])

    return { 'trimmed_image': tmp_image_trim, 'trim_region_input_space': tmp_trim_region_image }


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