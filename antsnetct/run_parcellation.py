#!/usr/bin/env python
import os
import logging
import tensorflow as tf

from .parcellation_pipeline import run_parcellation_pipeline
from .system_helpers import set_num_threads, set_tf_threads


def main():

    logger = logging.getLogger(__name__)

    # Set up threading parameters before tf or ants get imported
    # Set tensorflow threads - can only be done before initializing tensorflow
    try:
        set_tf_threads()
    except RuntimeError:
        logger.warning("Could not set TensorFlow threads. This is likely because tensorflow was initialized before this "
                    "package was imported.")

    # Set other threading parameters like ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS, using the existing environment if defined
    # Other variables like OMP_NUM_THREADS will be set the same as ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS
    set_num_threads(int(os.getenv('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', '0')))

    logger.info(f"Using {os.getenv('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS')} threads for ITK processes")

    run_parcellation_pipeline()


if __name__ == "__main__":
    main()

