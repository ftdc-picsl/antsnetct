import copy
import json
import os
import re
import shutil


class BIDSImage:
    """
    Represents a BIDS image file, including paths, metadata, and utilities for handling BIDS data.

    Attributes:
        path (str): Absolute path to the image file.
        ds_path (str): Absolute path to the dataset containing the file.
        ds_name (str): Name of the dataset, as specified in dataset_description.json.
        rel_path (str): Relative path from the dataset to the file.
        sidecar_path (str): Absolute path to the associated .json sidecar file.
        metadata (dict): JSON metadata loaded from the sidecar file.
    """

    def __init__(self, dataset, rel_path):
        """
        Initializes the BIDSImage object.

        Parameters:
        ----------
        dataset (str):
            The root directory of the BIDS dataset.
        rel_path (str):
            The relative path from the dataset to the image file.
        """
        self.ds_path = os.path.abspath(dataset)
        self.rel_path = rel_path
        self.path = os.path.join(self.ds_path, self.rel_path)
        self.ds_name = self._load_dataset_name()
        self.sidecar_path = get_image_sidecar(self.path)
        self.metadata = {}

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"{self.path} does not exist")

        self._load_metadata()

    def _load_dataset_name(self):
        """Loads the dataset name from dataset_description.json."""
        description_file = os.path.join(self.ds_path, 'dataset_description.json')
        if not os.path.exists(description_file):
            raise FileNotFoundError("dataset_description.json not found in dataset path")

        with open(description_file, 'r') as f:
            description = json.load(f)

        if 'Name' not in description:
            raise ValueError("Dataset name ('Name') not found in dataset_description.json")

        return description['Name']

    def _load_metadata(self):
        """Loads metadata from the sidecar JSON file, if present."""
        if os.path.exists(self.sidecar_path):
            with open(self.sidecar_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = None

    def copy_image(self, destination_ds):
        """
        Copies the image and its sidecar file to the same relative path in a new dataset.

        Parameters:
        ----------
        destination_ds (str):
            The root directory of the destination dataset.

        Returns:
        --------
        BIDSImage: A new BIDSImage object representing the copied image.
        """
        dest_ds_path = os.path.abspath(destination_ds)
        dest_file_path = os.path.join(dest_ds_path, self.rel_path)
        dest_sidecar_path = get_image_sidecar(dest_file_path)

        os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
        shutil.copy(self.path, dest_file_path)

        if os.path.exists(self.sidecar_path):
            shutil.copy(self.sidecar_path, dest_sidecar_path)

        return BIDSImage(dest_ds_path, self.rel_path)


    def get_metadata(self):
        """
        Returns a copy of the metadata dictionary.

        Returns:
            dict: A copy of the image's metadata.
        """
        return copy.deepcopy(self.metadata)

    def set_metadata(self, metadata):
        """
        Replaces the metadata with a new dictionary. The changes are written immediately to the sidecar file.

        Parameters:
            metadata (dict): A dictionary of metadata.
        """
        self.metadata = copy.deepcopy(metadata)
        with open(self.sidecar_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)

    # Accessor methods with simple docstrings for brevity
    def get_path(self):
        """Returns the absolute path to the image file."""
        return self.path

    def get_ds_path(self):
        """Returns the absolute path to the dataset."""
        return self.ds_path

    def get_ds_name(self):
        """Returns the name of the dataset."""
        return self.ds_name

    def get_rel_path(self):
        """Returns the relative path from the dataset to the image file."""
        return self.rel_path

    def get_uri(self, relative=False):
        """Returns the BIDS URI for the image file.

        Parameters:
        ----------
        relative (bool): If True, returns the relative URI; otherwise, returns the absolute URI.
        """

        if relative:
            return f"bids::{self.rel_path}"
        else:
            return f"bids:{self.ds_name}:{self.path}"

    def get_sidecar_file(self):
        """Returns the path to the sidecar file."""
        return self.sidecar_file

def image_to_bids(src_image, dataset_dir, dest_rel_path, metadata, overwrite=False):
    """Create a new bids image from an existing source image, copying it to the dataset at the specified path.

    Parameters:
    -----------
    src_image: str
        Path to the source image file.
    dataset_dir: str
        Path to the dataset directory.
    dest_rel_path: str
        Relative path from the dataset to the image to be created.
    metadata: dict
        Metadata to be written to the sidecar file.
    overwrite: bool
        If True, overwrite the destination file if it already exists. If False, raise an exception if the file already exists.

    Returns:
    --------
    BIDSImage: BIDSImage object representing the new image
    """

    dest_file_path = os.path.join(dataset_dir, dest_rel_path)
    dest_sidecar_path = get_image_sidecar(dest_file_path)

    if os.path.exists(dest_file_path) and not overwrite:
        raise FileExistsError(f"{dest_file_path} already exists")

    os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
    shutil.copy(src_image, dest_file_path)

    with open(dest_sidecar_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    return BIDSImage(dataset_dir, dest_rel_path)

#
# Create or update a BIDS output dataset
#
def update_output_dataset(output_dataset_dir, output_dataset_name):
    """Create or update a BIDS output dataset

    This is used to make or update an output dataset. If the dataset exists, its GeneratedBy field is updated to include
    this pipeline, if needed.

    Parameters:
    -----------
    output_dataset_dir: str
        Path to the output dataset directory. If the directory does not exist, it will be created.
    output_dataset_name: str
        Name of the output dataset, used if the dataset_description.json file does not exist.
    """

    if not os.path.isdir(output_dataset_dir):
        os.makedirs(output_dataset_dir)

    if not os.path.exists(os.path.join(output_dataset_dir, 'dataset_description.json')):
        # Write dataset_description.json
        output_ds_description = {'Name': output_dataset_name, 'BIDSVersion': '1.8.0',
                                'DatasetType': 'derivative', 'GeneratedBy': get_generated_by()
                                }
        # Write json to output dataset
        with open(os.path.join(output_dataset_dir, 'dataset_description.json'), 'w') as file_out:
            json.dump(output_ds_description, file_out, indent=2, sort_keys=True)
    else:
        # Get output dataset metadata
        try:
            with open(f"{output_dataset_dir}/dataset_description.json", 'r') as file_in:
                output_dataset_json = json.load(file_in)
            # Check dataset name
            if not 'Name' in output_dataset_json:
                raise ValueError(f"Output dataset description is missing Name, check "
                                 f"{output_dataset_dir}/data_description.json")
            # If this container doesn't already exist in the generated_by list, it will be added
            generated_by = get_generated_by(output_dataset_json['GeneratedBy'])
            # If we updated the generated_by, write it back to the output dataset
            old_gen_by = output_dataset_json['GeneratedBy']
            if old_gen_by is None or len(generated_by) > len(old_gen_by):
                output_dataset_json['GeneratedBy'] = generated_by
                with open(f"{output_dataset_dir}/dataset_description.json", 'w') as file_out:
                    json.dump(output_dataset_json, file_out, indent=2, sort_keys=True)
        except (FileNotFoundError, KeyError):
            raise ValueError(f"Output dataset description is missing GeneratedBy, check "
                             f"{output_dataset_dir}/data_description.json")


# Get a dictionary for the GeneratedBy field for the BIDS dataset_description.json
# This is used to record the software used to generate the dataset
# The environment variables DOCKER_IMAGE_TAG and DOCKER_IMAGE_VERSION are used if set
#
# Container type is assumed to be "docker" unless the variable SINGULARITY_CONTAINER
# is defined
def get_generated_by(existing_generated_by=None):

    generated_by = []

    if existing_generated_by is not None:
        generated_by = copy.deepcopy(existing_generated_by)
        for gb in existing_generated_by:
            if gb['Name'] == 'antsnetct' and gb['Container']['Tag'] == os.environ.get('DOCKER_IMAGE_TAG'):
                # Don't overwrite existing generated_by if it's already set to this pipeline
                return generated_by

    container_type = 'docker'

    if 'SINGULARITY_CONTAINER' in os.environ:
        container_type = 'singularity'

    gen_dict = {'Name': 'antsnetct',
                'Version': os.environ.get('DOCKER_IMAGE_VERSION', 'unknown'),
                'CodeURL': os.environ.get('GIT_REMOTE', 'unknown'),
                'Container': {'Type': container_type, 'Tag': os.environ.get('DOCKER_IMAGE_TAG', 'unknown')}
                }

    generated_by.append(gen_dict)
    return generated_by


def set_sources(sidecar_path, sources):
    """Set source URIs in a sidecar JSON file

    This modifies the sidecar file in place.

    Parameters:
    -----------
    sidecar_path: str
        Path to the sidecar JSON file
    sources: str or list of str
        List of source URIs
    """

    if not isinstance(sources, list):
        sources = [sources]

    with open(sidecar_path, 'r') as f:
        sidecar_json = json.load(f)
        sidecar_json['Sources'] = sources

    with open(sidecar_path, 'w') as f:
        json.dump(sidecar_json, f, indent=2, sort_keys=True)

#
# Search a mask dataset for a mask for a given image. Returns the first brain mask sourced from the input image.
#
# Looks for brain mask matching '*desc-brain*_mask.nii.gz'
#
# Return a dict with keys mask_image, mask_uri
#
def find_brain_mask(mask_dataset_directory, input_image_uri):

    # Load mask dataset_description.json
    with open(os.path.join(mask_dataset_directory, 'dataset_description.json')) as f:
        dataset_description = json.load(f)
        mask_dataset_name = dataset_description['Name']

    # Gets the dataset name and relative image path
    input_image_info = parse_bids_uri(input_image_uri)

    # defined if the mask dataset is the same as the input dataset
    input_local_uri = None

    if input_image_info['dataset_name'] == mask_dataset_name:
        mask_in_input_dataset = True
        input_local_uri = 'bids::' + os.path.join(input_image_info['image_directory'], input_image_info['image_filename'])

    modality_dir = os.path.join(mask_dataset_directory, input_image_info['image_directory'])

    # Search all sidecars in the modality directory for a mask with source image matching the input image
    # sidecars are files ending in .json
    for sidecar in os.listdir(modality_dir):
        if sidecar.find('_desc-brain_') > -1 and sidecar.endswith('_mask.json'):
            sidecar_path = os.path.join(modality_dir, sidecar)
            with open(sidecar_path) as f:
                sidecar_json = json.load(f)
                sidecar_image = sidecar_path.replace('.json', '.nii.gz')

                if 'Sources' in sidecar_json:
                    input_image_in_sources = input_image_uri in sidecar_json['Sources']
                    input_image_in_local_sources = mask_in_input_dataset and input_local_uri in sidecar_json['Sources']

                    if input_image_in_sources or input_image_in_local_sources:
                        mask_uri = f"bids:{mask_dataset_name}:" + os.path.join(input_image_info['image_directory'],
                                                                               sidecar_image)
                        return {'mask_image': os.path.join(modality_dir, sidecar_image), 'mask_uri': mask_uri}

    # No mask found
    return {'mask_image': None, 'mask_uri': None}



def find_segmentation_images(seg_dataset_directory, input_image_uri):
    """Search a segmentation dataset for segmentation + posteriors produced from an input image

    Looks for segmentation images containing the input image in the 'Sources' field of the sidecar JSON file. The segmentation
    must be a dseg, with classes 1-6 corresponding to CSF, GM, WM, Deep GM, Brainstem, Cerebellum respectively.

     Returns a dict with keys segmentation_image, segmentation_uri, posteriors, posterior_uris


    Parameters:
    -----------
    seg_dataset_directory: str
        Path to the segmentation dataset directory
    input_image_uri: str
        BIDS URI for the input image. This is used to search for the segmentation and posteriors that used this as a source.

    Returns:
    --------
    dict: A dictionary with keys:
        segmentation_image: BIDSImage
            The segmentation image, if a suitable one is found, or None.
        posterior_images: list of BIDSImage
            Images representing the classes in order: CSF, GM, WM, Deep GM, Brainstem, Cerebellum.



     Posteriors are a list ordered by CSF, GM, WM, Deep GM, Brainstem, Cerebellum
    """

    with open(os.path.join(seg_dataset_directory, 'dataset_description.json')) as f:
        dataset_description = json.load(f)
        seg_dataset_name = dataset_description['Name']

    input_image_info = parse_bids_uri(input_image_uri)

    seg_dict = {'segmentation_image': None, 'segmentation_uri': None, 'posteriors': [None] * 6, 'posterior_uris': [None] * 6}

    # defined if the segmentations dataset is the same as the input dataset
    input_local_uri = None

    if input_image_info['dataset_name'] == seg_dataset_name:
        seg_in_input_dataset = True
        input_local_uri = 'bids::' + os.path.join(input_image_info['image_directory'], input_image_info['image_filename'])

    # Path to the data directory to search, eg /data/segds/sub-01/ses-01/anat
    modality_dir = os.path.join(seg_dataset_directory, input_image_info['image_directory'])

    # Search all sidecars in the modality directory for a mask with source image matching the input image
    # sidecars are files ending in .json
    for sidecar in os.listdir(modality_dir):
        if sidecar.find('_seg-antsct_') > -1:
            sidecar_path = os.path.join(modality_dir, sidecar)
            sidecar_image = sidecar_path.replace('.json', '.nii.gz')
            if sidecar.endswith('_dseg.json'):
                # segmentation image
                with open(sidecar_path) as f:
                    sidecar_json = json.load(f)
                    sources_exist = 'Sources' in sidecar_json
                    if sources_exist:
                        input_image_in_sources = input_image_uri in sidecar_json['Sources']
                        input_image_in_local_sources = seg_in_input_dataset and input_local_uri in sidecar_json['Sources']

                        if input_image_in_sources or input_image_in_local_sources:
                            seg_dict['segmentation_image'] = os.path.join(modality_dir, sidecar_image)
                            seg_dict['segmentation_uri'] = f"bids:{seg_dataset_name}:" + \
                                os.path.join(input_image_info['image_directory'], sidecar_image)
            elif sidecar.endswith('_probseg.json'):
                # posteriors
                posterior_image = None
                posterior_uri = None
                posterior_index = None

                with open(sidecar_path) as f:
                    sidecar_json = json.load(f)
                    sources_exist = 'Sources' in sidecar_json
                    if sources_exist:
                        input_image_in_sources = input_image_uri in sidecar_json['Sources']
                        input_image_in_local_sources = seg_in_input_dataset and input_local_uri in sidecar_json['Sources']

                        if input_image_in_sources or input_image_in_local_sources:
                            posterior_image = os.path.join(modality_dir, sidecar_image)
                            posterior_uri = f"bids:{seg_dataset_name}:" + os.path.join(input_image_info['image_directory'],
                                                                                        sidecar_image)

                # Get the posterior index from the filename
                posterior_label = re.search(r'label-([A-Z]+)', sidecar).match(1)

                if (posterior_label == 'CSF'):
                    posterior_index = 0
                elif (posterior_label == 'GM'):
                    posterior_index = 1
                elif (posterior_label == 'WM'):
                    posterior_index = 2
                elif (posterior_label == 'DeepGM'):
                    posterior_index = 3
                elif (posterior_label == 'Brainstem'):
                    posterior_index = 4
                elif (posterior_label == 'Cerebellum'):
                    posterior_index = 5

                if posterior_index is not None:
                    seg_dict['posteriors'][posterior_index] = posterior_image
                    seg_dict['posterior_uris'][posterior_index] = posterior_uri

    return seg_dict


# Find images in a BIDS dataset directory
# return a list of images and URIs eg [{image: /data/ds/sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz, uri:
# bids:ds:sub-01:ses-01:anat:sub-01_ses-01_T1w.nii.gz}]
def find_images(input_dataset_dir, participant_label, session_label, modality, image_suffix):

    images = list()

    # Path to the data directory to search, eg /data/ds/sub-01/ses-01/anat
    modality_dir = os.path.join(input_dataset_dir, 'sub-' + participant_label, 'ses-' + session_label, modality)

    for image in os.listdir(modality_dir):
        if image.endswith(image_suffix):
            images.append({'image': os.path.join(modality_dir, image), 'uri': f"bids:{input_dataset_dir}:" + \
                           "sub-{participant_label}/ses_{session_label}/{modality}/{image}"})

    return images

# Gets the dataset_name and relative image path from a BIDS URI
# Returns a dictionary of the form:
# {dataset_name: {dataset_name}, image_directory: {image_dir}, image_filename: {image_filename}}
def parse_bids_uri(uri):

    #  capture the dataset name (if present) and the rest of the path
    match = re.match(r'bids:([^:]*):(.+)$', uri)
    if match:
        dataset_name = match.group(1) or None
        full_path = match.group(2)

        # Extract directory and filename
        image_directory, image_filename = os.path.split(full_path)

        return {
            'dataset_name': dataset_name,
            'image_directory': image_directory,
            'image_filename': image_filename
        }
    else:
        raise ValueError("URI does not follow expected format.")


# Resolve a file from a URI
def resolve_uri(dataset_path, file_uri):
    #  capture the dataset name (if present) and the rest of the path
    match = re.match(r'bids:([^:]*):(.+)$', file_uri)
    if match:
        rel_path = match.group(2)
        return os.path.join(dataset_path, rel_path)
    else:
        raise ValueError("URI does not follow expected format.")


def get_uri(dataset_dir, dataset_name, file_path):
    """Get a BIDS URI for a file

    Parameters:
    -----------
    dataset_dir: str
        Path to the dataset directory
    dataset_name: str
        Name of the dataset as specified in the dataset_description.json
    file_path: str
        Path to the file, either absolute or relative to the dataset directory

    Returns:
    --------
    str: BIDS URI for the file
    """

    if file_path.startswith(dataset_dir):
        return f"bids:{dataset_name}:{os.path.relpath(file_path, dataset_dir)}"
    else:
        return f"bids:{dataset_name}:{file_path}"


def get_uri(dataset_dir, file_path):
    """Get a BIDS URI for a file

    Parameters:
    -----------
    dataset_dir: str
        Path to the dataset directory
    file_path: str
        Path to the file, either absolute or relative to the dataset directory

    Returns:
    --------
    str: BIDS URI for the file
    """

    # read dataset name from dataset_description.json
    with open(os.path.join(dataset_dir, 'dataset_description.json')) as f:
        dataset_description = json.load(f)
        dataset_name = dataset_description['Name']

    if file_path.startswith(dataset_dir):
        return f"bids:{dataset_name}:{os.path.relpath(file_path, dataset_dir)}"
    else:
        return f"bids:{dataset_name}:{file_path}"

def get_image_sidecar(image_file):
    """Get the sidecar file for a NIFTI image

    Parameters:
    ----------
    image_file (str):
        The path to the image file.

    Returns:
    -------
    str: The sidecar file for the image.
    """

    if image_file.endswith('.nii.gz'):
        return image_file[:-7] + '.json'
    elif image_file.endswith('.nii'):
        return image_file[:-4] + '.json'
    else:
        raise ValueError(f"Image file {image_file} does not end in .nii or .nii.gz")