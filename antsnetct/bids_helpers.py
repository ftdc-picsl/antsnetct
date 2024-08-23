import copy
import json
import os
import re
import templateflow

from importlib import metadata

from .system_helpers import copy_file


class BIDSImage:
    """
    Represents a BIDS image file, including paths, metadata, and utilities for handling BIDS data.

    Attributes:
        _path (str): Absolute path to the image file.
        _ds_path (str): Absolute path to the dataset containing the file.
        _ds_name (str): Name of the dataset, as specified in dataset_description.json.
        _rel_path (str): Relative path from the dataset to the file.
        _sidecar_path (str): Absolute path to the associated .json sidecar file.
        _metadata (dict): JSON metadata loaded from the sidecar file.
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
        self._ds_path = os.path.abspath(dataset)
        self._rel_path = rel_path
        self._path = os.path.join(self._ds_path, self._rel_path)

        if not os.path.exists(self._path):
            raise FileNotFoundError(f"{self._path} does not exist")

        self._sidecar_path = get_image_sidecar(self._path)

        self._set_ds_name()
        self._set_derivative_path_prefix()
        self._derivative_rel_path_prefix = os.path.relpath(self._derivative_path_prefix, self._ds_path)
        self._set_metadata_from_sidecar()


    def _set_ds_name(self):
        """Loads the dataset name from dataset_description.json."""
        description_file = os.path.join(self._ds_path, 'dataset_description.json')
        if not os.path.exists(description_file):
            raise FileNotFoundError("dataset_description.json not found in dataset path")

        with open(description_file, 'r') as f:
            ds_description = json.load(f)

        if 'Name' not in ds_description:
            raise ValueError("Dataset name ('Name') not found in dataset_description.json")

        self._ds_name = ds_description['Name']


    def _set_metadata_from_sidecar(self):
        """Loads metadata from the sidecar JSON file, if present."""
        if os.path.exists(self._sidecar_path):
            with open(self._sidecar_path, 'r') as f:
                self._metadata = json.load(f)
        else:
            self._metadata = None


    def _set_derivative_path_prefix(self):
        path_no_desc = re.sub(r'_desc-[A-Za-z0-9]+', '', self._path)
        underscore_index = path_no_desc.rindex('_')

        if underscore_index != -1:  # Check if an underscore was found
            self._derivative_path_prefix = path_no_desc[:underscore_index]
        else:
            # shouldn't happen for any valid BIDS file
            raise ValueError(f"File {self._path} does not have a BIDS suffix")


    def copy_image(self, destination_ds):
        """
        Copies the image and its metadata to the same relative path in a new dataset.

        Parameters:
        ----------
        destination_ds : str
            The root directory of the destination dataset.

        Returns:
        --------
        BIDSImage: A new BIDSImage object representing the copied image.
        """
        dest_ds_path = os.path.abspath(destination_ds)
        dest_file_path = os.path.join(dest_ds_path, self._rel_path)
        dest_sidecar_path = get_image_sidecar(dest_file_path)

        os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
        copy_file(self._path, dest_file_path)

        if self._metadata is not None:
            dest_metadata = copy.deepcopy(self._metadata)
            # Replace relative source URIs with absolute URIs
            if 'Sources' in dest_metadata:
                for idx, source in enumerate(dest_metadata['Sources']):
                    if source.startswith('bids::'):
                        # the source is within this dataset, replace bids:: with bids:{self.ds_name}
                        dest_metadata['Sources'][idx] = f"bids:{self._ds_name}:{source[6:]}"

            with open(dest_sidecar_path, 'w') as f:
                json.dump(dest_metadata, f, indent=4, sort_keys=True)

        return BIDSImage(dest_ds_path, self._rel_path)


    def get_metadata(self):
        """
        Returns a copy of the metadata dictionary.

        Returns:
            dict: A copy of the image's metadata.
        """
        return copy.deepcopy(self._metadata)


    def set_metadata(self, metadata):
        """
        Replaces the metadata with a new dictionary. The changes are written immediately to the sidecar file.

        Parameters
        ----------
        metadata : dict
            A dictionary of metadata.
        """
        self._metadata = copy.deepcopy(metadata)
        with open(self._sidecar_path, 'w') as f:
            json.dump(self._metadata, f, indent=4, sort_keys=True)


    def update_metadata(self, extra_metadata):
        """
        Adds or overwrites metadata. The changes are written immediately to the sidecar file.

        Parameters:
        ----------
        extra_metadata : dict
            A dictionary of additional metadata. If a key already exists in the metadata, the value is overwritten. Otherwise,
            it is added to the metadata.
        """
        for key, value in extra_metadata.items():
            self._metadata[key] = value
        with open(self._sidecar_path, 'w') as f:
            json.dump(self._metadata, f, indent=4, sort_keys=True)


    def get_path(self):
        """Returns the absolute path to the image file."""
        return self._path


    def get_ds_path(self):
        """Returns the absolute path to the dataset."""
        return self._ds_path


    def get_ds_name(self):
        """Returns the name of the dataset."""
        return self._ds_name


    def get_rel_path(self):
        """Returns the relative path from the dataset to the image file."""
        return self._rel_path


    def get_uri(self, relative=True):
        """Returns the BIDS URI for the image file.

        Parameters:
        ----------
        relative : bool, optional
            If True, returns the relative URI; otherwise, returns the absolute URI.

        Returns:
        --------
        str: The BIDS URI for the image file.
        """
        if relative:
            return f"bids::{self._rel_path}"
        else:
            return f"bids:{self._ds_name}:{self._rel_path}"


    def get_sidecar_path(self):
        """Returns the path to the sidecar file."""
        return self._sidecar_path


    def get_derivative_path_prefix(self):
        """Get the the prefix for derivatives, which includes the full path to the file without its BIDS suffix and extension.

        For raw data, only the suffix (eg, '_T1w.nii.gz') is removed. For derivatives, the _desc-DESCRIPTION entity is also
        removed.

        """
        return self._derivative_path_prefix


    def get_derivative_rel_path_prefix(self):
        """Get the the prefix for derivatives relative to the dataset

        For raw data, only the suffix is removed. For derivatives, the _desc-DESCRIPTION entity is also removed.
        """
        return self._derivative_rel_path_prefix


    def __str__(self):
        return f"BIDSImage: {self.get_uri()}"


class TemplateImage:
    """Represents a templateflow image.

    Attributes:
        _cohort (str): The cohort of the template or None
        _desc (str): The description of the template or None
        _name (str): The name of the template
        _path (str): The absolute path to the image file
        _resolution (str): The resolution label of the template or None
        _suffix (str): The suffix of the template
        _derivative_space_file_string (str): A string for use in derivatives created in this template space
    """

    def __init__(self, name, suffix, resolution='01', description=None, cohort=None):

        template_metadata = templateflow.api.get_metadata(name)

        # Almost all templates use res-1 or res-01, but if there's no resolution in the metadata, we'll use None
        if not 'res' in template_metadata:
            resolution = None

        res_keys = list(template_metadata['res'].keys())

        template_res_found = False

        if resolution is None or resolution in res_keys:
            template_res_found = True
        else:
            # fmriprep allows res=1 for templates where the resolution label is '01'. We'll allow that, and the converse,
            # by checking the integer representation of the resolution
            # But only do this if the resolution is numeric
            if resolution.isnumeric():
                res_int = int(resolution)
                metatadata_res_keys_int = [int(k) for k in res_keys]
                # if we find res_int in metatadata_res_keys_int, then the resolution to use is res_keys at the same index
                if res_int in metatadata_res_keys_int:
                    resolution = res_keys[metatadata_res_keys_int.index(res_int)]
                    template_res_found = True

        if not template_res_found:
            raise ValueError(f"Resolution {resolution} not found in template metadata")

        template_matches = templateflow.api.get(name, resolution=resolution, desc=description, cohort=cohort, suffix=suffix,
                                                raise_empty=True)

        if type(template_matches) is list:
            raise ValueError(f"Template could not be uniquely identified from the input. Found: {template_matches}")

        self._cohort = cohort
        self._desc = description
        self._name = name
        self._path = str(template_matches)
        self._resolution = resolution
        self._suffix = suffix

        derivative_string = f"space-{self._name}"

        if self._cohort is not None:
            derivative_string += f"_cohort-{self._cohort}"

        if self._resolution is not None:
            derivative_string += f"_res-{self._resolution}"

        self._derivative_space_string = derivative_string

        _uri = f"bids:templateflow:tpl-{self._name}/" + os.path.basename(self._path)


    def get_path(self):
        """Returns the absolute path to the image file."""
        return self._path

    def get_name(self):
        """Returns the name of the template."""
        return self._name

    def get_resolution(self):
        """Returns the resolution of the template."""
        return self._resolution

    def get_desc(self):
        """Returns the description of the template."""
        return self._desc

    def get_derivative_space_string(self):
        """
        Returns a string for use in derivatives created in this space.

        For example, 'space-MNI152NLin2009cAsym_res-01' or 'space-myTemplate_cohort-01_res-1'
        """
        return self._derivative_space_string

    def get_uri(self):
        """Returns the BIDS URI for the image file."""
        return self._uri


def resolve_uri(dataset_path, file_uri):
    """Resolve a BIDS URI to an absolute path.

    Currently, only supports resolution of BIDS URIs in the current dataset.

    Parameters:
    ----------
        dataset_path : str
            The root directory of the BIDS dataset.
        file_uri :str
            The BIDS URI, in the format "bids:DATASET:RELATIVE_PATH" or "bids::RELATIVE_PATH".

    Returns:
    --------
        str: The absolute path to the file.
    """
    #  capture the dataset name (if present) and the rest of the path
    match = re.match(r'bids:([^:]*):(.+)$', file_uri)
    if match:
        rel_path = match.group(2)
        return os.path.join(dataset_path, rel_path)
    else:
        raise ValueError(f"URI {file_uri} does not follow expected format.")


def get_image_sidecar(image_file):
    """Get the sidecar file for a NIFTI image.

    This does not check if the sidecar exists, as it may be used to generate a sidecar file path.

    Parameters:
    ----------
    image_file : str
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


def get_sidecar_image(sidecar_file):
    """Get the image file for a sidecar JSON file

    This checks if either a ".nii.gz" or ".nii" image file exists, and raises a FileNotFoundError if it does not.

    Parameters:
    ----------
    sidecar_file : str
        The path to the sidecar JSON file.

    Returns:
    -------
    str: The image file for the sidecar.
    """
    if sidecar_file.endswith('.json'):
        image_root = sidecar_file[:-5]

        if os.path.exists(image_root + '.nii.gz'):
            return image_root + '.nii.gz'
        elif os.path.exists(image_root + '.nii'):
            return image_root + '.nii'
        else:
            raise FileNotFoundError(f"Image file not found for sidecar {sidecar_file}")
    else:
        raise ValueError(f"Sidecar file {sidecar_file} does not end in .json")


def image_to_bids(src_image, dataset_dir, dest_rel_path, metadata=None, overwrite=False):
    """Create a new bids image from a NIFTI image file, copying it to the dataset at the specified path.

    Parameters:
    -----------
    src_image : str
        Path to the source image file.
    dataset_dir : str
        Path to the dataset directory.
    dest_rel_path : str
        Relative path from the dataset to the image to be created.
    metadata : dict
        Metadata to be written to the sidecar file.
    overwrite : bool
        If True, overwrite the destination file if it already exists. If False, raise an exception if the file already exists.

    Returns:
    --------
    BIDSImage: BIDSImage object representing the new image
    """
    dest_file_path = os.path.join(dataset_dir, dest_rel_path)
    dest_sidecar_path = get_image_sidecar(dest_file_path)

    if os.path.exists(dest_file_path):
        if not overwrite:
            raise FileExistsError(f"{dest_file_path} already exists")
        # Remove existing file and sidecar
        os.remove(dest_file_path)
        if os.path.exists(dest_sidecar_path):
            os.remove(dest_sidecar_path)

    os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
    copy_file(src_image, dest_file_path)

    if metadata is not None:
        with open(dest_sidecar_path, 'w') as f:
            json.dump(metadata, f, indent=4, sort_keys=True)

    return BIDSImage(dataset_dir, dest_rel_path)


def _get_generated_by(existing_generated_by=None):
    """Get a dictionary for the GeneratedBy field for the BIDS dataset_description.json.

    This is used to record the software used to generate the dataset. The environment variables DOCKER_IMAGE_TAG is used if
    set - this is set in the Dockerfile so should be defined if the code is being run inside a container.
    Container type is assumed to be "docker" unless the variable SINGULARITY_CONTAINER is defined.

    Parameters:
    ----------
        existing_generated_by : dict
            The existing generated_by field, if any.

    Returns:
    --------
        dict:
            A dictionary for the GeneratedBy field in the dataset_description.json

    """
    docker_tag = os.environ.get('DOCKER_IMAGE_TAG', 'undefined')

    container_type = 'docker'
    # If in a singularity container built from docker, both DOCKER_IMAGE_TAG and SINGULARITY_CONTAINER will be defined
    if 'SINGULARITY_CONTAINER' in os.environ:
        container_type = 'singularity'

    if docker_tag == 'undefined':
        # Not in a container, or at least not unless someone has hacked the Dockerfile
        container_type = 'not_containerized'

    generated_by = []

    antsnetct_version = metadata.version('antsnetct')

    if existing_generated_by is not None:
        generated_by = copy.deepcopy(existing_generated_by)
        for gb in existing_generated_by:
            if gb['Name'] == 'antsnetct' and gb['Version'] == antsnetct_version and gb['Container']['Tag'] == docker_tag:
                # Don't overwrite existing generated_by if it's already set to this pipeline
                return generated_by

    gen_dict = {
                'Name': 'antsnetct',
                'Version': antsnetct_version,
                'CodeURL': os.environ.get('GIT_REMOTE', 'unknown'),
                'Container': {'Type': container_type, 'Tag': docker_tag}
                }

    generated_by.append(gen_dict)
    return generated_by


def _get_dataset_links(existing_dataset_links, dataset_link_paths):
    """Get a dictionary for the DatasetLinks field for the BIDS dataset_description.json.

    This is used to record links to other datasets. If the dataset link already exists, the URI is checked to ensure it
    matches the existing URI.

    Parameters:
    ----------
        existing_dataset_links : dict
            The existing DatasetLinks field, if any.
        dataset_link_paths : str
            The new dataset links to add.

    Returns:
    --------
        dict :
            A dictionary for the DatasetLinks field in the dataset_description.json

    Raises:
    -------
    ValueError : If a dataset link already exists with a different URI.
    """
    if existing_dataset_links is None:
        dataset_links = {}
    else:
        dataset_links = copy.deepcopy(existing_dataset_links)

    for path in dataset_link_paths:
        # Get the dataset name from the dataset_description.json
        path_link = _get_single_dataset_link(path)

        name = path_link['Name']
        uri = path_link['URI']

        if name in dataset_links:
            if dataset_links[name] != uri:
                raise ValueError(f"Dataset link {name} already exists with URI {existing_dataset_links[name]}, but new URI "
                                 f"{uri} provided")
        else:
            dataset_links[name] = uri

    return dataset_links


def _get_single_dataset_link(dataset_path):
    """Get a dataset link for the BIDS dataset_description.json from a path to a dataset.

    Parameters:
    -----------
    dataset_path : str
        Path to the dataset directory.

    Returns:
    --------
    dict :
        A dictionary for the dataset link, with keys 'Name' and 'URI'. The URI is a file:// URI to the dataset.
    """
    description_file = os.path.join(dataset_path, 'dataset_description.json')
    if not os.path.exists(description_file):
        raise FileNotFoundError(f"dataset_description.json not found in dataset path {dataset_path}")

    with open(description_file, 'r') as f:
        ds_description = json.load(f)

    if 'Name' not in ds_description:
        raise ValueError("Dataset name ('Name') not found in dataset_description.json")

    dataset_link = { 'Name':ds_description['Name'], 'URI': f"file://{os.path.abspath(dataset_path)}" }

    return dataset_link


def update_output_dataset(output_dataset_dir, output_dataset_name, dataset_link_paths=None):
    """Create or update a BIDS output dataset

    This is used to make or update an output dataset. If the dataset exists, its metadata is updated. Specifically, the
    GeneratedBy field is updated to include this pipeline, if needed. If dataset links are provided, they are added to the
    description if needed.

    Parameters:
    -----------
    output_dataset_dir : str
        Path to the output dataset directory. If the directory does not exist, it will be created.
    output_dataset_name : str
        Name of the output dataset, used if the dataset_description.json file does not exist.
    dataset_link_paths : list of str, optional
        List of paths to other datasets, to which the output dataset is linked.

    Raises:
    -------
    ValueError: If dataset_link_paths provides a name that already exists, but with a different URI.
    """
    os.makedirs(output_dataset_dir, exist_ok=True)

    if not os.path.exists(os.path.join(output_dataset_dir, 'dataset_description.json')):
        # Write dataset_description.json
        output_ds_description = {'Name': output_dataset_name, 'BIDSVersion': '1.8.0',
                                'DatasetType': 'derivative', 'GeneratedBy': _get_generated_by()
                                }
        if (dataset_link_paths is not None):
            output_ds_description['DatasetLinks'] = _get_dataset_links(None, dataset_link_paths)
        # Write json to output dataset
        with open(os.path.join(output_dataset_dir, 'dataset_description.json'), 'w') as file_out:
            json.dump(output_ds_description, file_out, indent=4, sort_keys=True)
    else:
        # Get output dataset metadata
        with open(f"{output_dataset_dir}/dataset_description.json", 'r') as file_in:
            output_ds_description = json.load(file_in)
        # Check dataset name
        if not 'Name' in output_ds_description:
            raise ValueError(f"Output dataset description is missing Name, check "
                                f"{output_dataset_dir}/data_description.json")


        old_gen_by = output_ds_description.get('GeneratedBy')

        # If this container doesn't already exist in the generated_by list, it will be added
        output_ds_description['GeneratedBy'] = _get_generated_by(old_gen_by)

        old_ds_links = output_ds_description.get('DatasetLinks')

        output_ds_description['DatasetLinks'] = _get_dataset_links(old_ds_links, dataset_link_paths)

        ds_modified = False

        if old_gen_by is None or output_ds_description['GeneratedBy'] > len(old_gen_by):
            ds_modified = True
        if dataset_link_paths is not None:
            if old_ds_links is None or len(output_ds_description['DatasetLinks']) > len(old_ds_links):
                ds_modified = True

        if ds_modified:
            with open(f"{output_dataset_dir}/dataset_description.json", 'w') as file_out:
                json.dump(output_ds_description, file_out, indent=4, sort_keys=True)


def set_sources(sidecar_path, sources):
    """Set source URIs in a sidecar JSON file

    This modifies the sidecar file in place.

    Parameters:
    -----------
    sidecar_path : str
        Path to the sidecar JSON file
    sources : str or list of str
        List of source URIs
    """

    if not isinstance(sources, list):
        sources = [sources]

    with open(sidecar_path, 'r') as f:
        sidecar_json = json.load(f)
        sidecar_json['Sources'] = sources

    with open(sidecar_path, 'w') as f:
        json.dump(sidecar_json, f, indent=4, sort_keys=True)


def find_brain_mask(mask_dataset_directory, input_image):
    """
    Search a mask dataset for a mask for a given image. Returns the first brain mask derived from the input image.
    Looks for brain mask matching '*desc-brain*_mask.nii.gz' with optional space-orig and res-01 entities.

    Parameters:
    ----------
    mask_dataset_directory : str
        Path to the mask dataset directory
    input_image : BIDSImage
        The input image that the mask is derived from.

    Returns:
    --------
    BIDSImage:
        A BIDSImage object representing the mask, or None if no mask is found.
    """
    # Load mask dataset_description.json
    with open(os.path.join(mask_dataset_directory, 'dataset_description.json')) as f:
        dataset_description = json.load(f)
        mask_dataset_name = dataset_description['Name']

    search_prefix = input_image.get_derivative_rel_path_prefix()
    search_pattern = re.compile(rf"{search_prefix}(?:_space-orig)?(?:_res-01)?_desc-brain_mask.nii.gz")

    search_dir_relpath = os.path.dirname(search_prefix)

    for image_file in os.listdir(os.path.join(mask_dataset_directory, search_dir_relpath)):
        image_rel_path = os.path.join(search_dir_relpath, image_file)
        if search_pattern.match(image_rel_path):
            return BIDSImage(mask_dataset_directory, image_rel_path)

    # No mask found
    return None



def find_segmentation_probability_images(seg_dataset_directory, input_image):
    """Search a segmentation dataset for a segmentation and posteriors produced from a particular input image.

    This function searches the segmentation dataset for segmentation posteriors matching the BIDS
    "Common image-derived labels": CSF, CGM, WM, SCGM, BS, CBM.

    The file names are expected to match the pattern
      '{source_entities}(?:_space-orig)?_(?:seg-[A-Za-z0-9]+)?(?:_res-01)?_label-{label}_probseg.nii.gz'

    where {source_entities} are the file name parts of the derivative prefix of the input image. Note we do not
    check the metadata Sources field, only the file name, because the segmentation may be indirectly derived from the
    input T1w image. For example, if the segmentation input is a bias-corrected derivative of the input image, the original
    input image may not be listed in the Sources field.

    Parameters:
    -----------
    seg_dataset_directory : str
        Path to the segmentation dataset directory
    input_image : BIDSImage
        BIDSImage object representing the input image, which is the source for the segmentations.

    Returns:
    --------
        list of BIDSImage
            Images representing the classes in order: CSF, GM, WM, Deep GM, Brainstem, Cerebellum.
    """
    with open(os.path.join(seg_dataset_directory, 'dataset_description.json')) as f:
        dataset_description = json.load(f)
        seg_dataset_name = dataset_description['Name']

    # the list to be returned, with posteriors in order
    output_posteriors = [None] * 6

    # Search all sidecars in the modality directory for a mask with source image matching the input image
    # sidecars are files ending in .json
    class_labels = {'CSF':0, 'CGM':1, 'WM':2, 'SGM':3, 'BS':4, 'CBM': 5}

    search_prefix = input_image.get_derivative_rel_path_prefix()
    search_dir_relpath = os.path.dirname(search_prefix)
    search_pattern = \
        re.compile(rf"{search_prefix}(?:_space-orig)?(?:_seg-[A-Za-z0-9]+)?(?:_res-01)?_label-([A-Z]+)_probseg.nii.gz")

    for image_file in os.listdir(os.path.join(seg_dataset_directory, search_dir_relpath)):
        image_rel_path = os.path.join(search_dir_relpath, image_file)
        label_match = search_pattern.match(image_rel_path)
        if label_match:
            posterior_label = label_match.group(1)
            posterior_index = None
            if posterior_label in class_labels:
                if (posterior_label == 'CSF'):
                    posterior_index = 0
                elif (posterior_label == 'CGM'):
                    posterior_index = 1
                elif (posterior_label == 'WM'):
                    posterior_index = 2
                elif (posterior_label == 'SGM'):
                    posterior_index = 3
                elif (posterior_label == 'BS'):
                    posterior_index = 4
                elif (posterior_label == 'CBM'):
                    posterior_index = 5

                if posterior_index is not None:
                    output_posteriors[posterior_index] = BIDSImage(seg_dataset_directory, image_rel_path)

    # If we didn't find all the necessary classes, return None
    for i in range(6):
        if output_posteriors[i] is None:
            return None

    return output_posteriors


def find_session_images(input_dataset_dir, participant_label, session_label, modality, bids_suffix):
    """Find images from a session in a BIDS dataset directory.

    Parameters:
    -----------

    input_dataset_dir : str
        Path to the input dataset directory.
    participant_label : str
        Participant label, eg '01'.
    session_label : str
        Session label, eg 'MR1'. Set to None if the dataset does not have sessions.
    modality : str
        Modality, eg 'anat', 'func'.
    bids_suffix : str
        BIDS image suffix, eg "T1w", with additional keys as needed eg "desc-preproc_T1w". Do not include
        the file extension. This function will search for both .nii and .nii.gz files.

    Returns:
    --------
    list of BIDSImage: A list of BIDSImage objects representing the images found.
    """

    images = list()

    # Path to the data directory to search, eg /data/ds/sub-01/ses-01/anat
    if (session_label is None):
        modality_dir = os.path.join(input_dataset_dir, 'sub-' + participant_label, modality)
    else:
        modality_dir = os.path.join(input_dataset_dir, 'sub-' + participant_label, 'ses-' + session_label, modality)

    if not os.path.exists(modality_dir):
        return None

    for image in os.listdir(modality_dir):
        if image.endswith('_' + bids_suffix + '.nii') or image.endswith('_' + bids_suffix + '.nii.gz'):
            images.append(BIDSImage(input_dataset_dir, os.path.relpath(os.path.join(modality_dir, image), input_dataset_dir)))

    return images


def find_participant_images(input_dataset_dir, participant_label, modality, bids_suffix):
    """Find images for a participant in a BIDS dataset directory, across all sessions.

    Parameters:
    -----------

    input_dataset_dir : str
        Path to the input dataset directory.
    participant_label : str
        Participant label, eg '01'.
    modality : str
        Modality, eg 'anat', 'func'.
    bids_suffix : str
        BIDS image suffix, eg "T1w", with additional keys as needed eg "desc-preproc_T1w". Do not include
        the file extension. This function will search for both .nii and .nii.gz files.

    Returns:
    --------
    list of BIDSImage: A list of BIDSImage objects representing the images found.
    """

    sessions = [d for d in os.listdir(os.path.join(input_dataset_dir, 'sub-' + participant_label)) if d.startswith('ses-')]

    if not sessions:
        return find_session_images(input_dataset_dir, participant_label, None, modality, bids_suffix)

    images = list()

    for sess in sessions:
        images.extend(find_session_images(input_dataset_dir, participant_label, sess[4:], modality, bids_suffix))

    return images

