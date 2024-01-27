
# Get a dictionary for the GeneratedBy field for the BIDS dataset_description.json
# This is used to record the software used to generate the dataset
# The environment variables DOCKER_IMAGE_TAG and DOCKER_IMAGE_VERSION are used if set
#
# Container type is assumed to be "docker" unless the variable SINGULARITY_CONTAINER
# is defined
def get_generated_by(existing_generated_by=None):

    import copy

    generated_by = []

    if existing_generated_by is not None:
        generated_by = copy.deepcopy(existing_generated_by)
        for gb in existing_generated_by:
            if gb['Name'] == 'T1wPreprocessing' and gb['Container']['Tag'] == os.environ.get('DOCKER_IMAGE_TAG'):
                # Don't overwrite existing generated_by if it's already set to this pipeline
                return generated_by

    container_type = 'docker'

    if 'SINGULARITY_CONTAINER' in os.environ:
        container_type = 'singularity'

    gen_dict = {'Name': 'T1wPreprocessing',
                'Version': os.environ.get('DOCKER_IMAGE_VERSION', 'unknown'),
                'CodeURL': os.environ.get('GIT_REMOTE', 'unknown'),
                'Container': {'Type': container_type, 'Tag': os.environ.get('DOCKER_IMAGE_TAG', 'unknown')}
                }

    generated_by.append(gen_dict)
    return generated_by