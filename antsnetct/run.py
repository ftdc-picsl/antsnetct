#!/usr/bin/env python
import os
import sys
import tensorflow as tf
import logging

def main():

    logger = logging.getLogger(__name__)

    # Set up threading parameters before tf or ants get imported

    # Set tensorflow threads - can only be done before initializing tensorflow
    try:
        set_tf_threads()
    except RuntimeError:
        logger.warning("Could not set TensorFlow threads. This is likely because tensorflow was initialized before this "
                    "package was imported.")

    logger.info("tensorflow thread settings:\n\tintra_op_parallelism_threads: %d\n\tinter_op_parallelism_threads: %d",
                tf.config.threading.get_intra_op_parallelism_threads(), tf.config.threading.get_inter_op_parallelism_threads())

    # Set other threading parameters like ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS, using the existing environment if defined
    # Other variables like OMP_NUM_THREADS will be set the same as ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS
    set_num_threads(int(os.getenv('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', '0')))

    logger.info(f"Using {os.getenv('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS')} threads for ITK processes")

    from .cross_sectional_pipeline import cross_sectional_analysis
    from .longitudinal_pipeline import longitudinal_analysis

    if '--longitudinal' in sys.argv:
        sys.argv.remove('--longitudinal')
        longitudinal_analysis()
    else:
        cross_sectional_analysis()

if __name__ == "__main__":
    main()


def set_tf_threads():
    """Set the number of threads to use in TensorFlow. This can only be set before TensorFlow is initialized.

    The number of threads for tensorflow defaults to 1 but can be overridden by setting the environment variables
    TF_NUM_INTRAOP_THREADS and TF_NUM_INTEROP_THREADS.

    """
    intra_threads = int(os.getenv('TF_NUM_INTRAOP_THREADS', '1'))
    inter_threads = int(os.getenv('TF_NUM_INTEROP_THREADS', '1'))

    tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
    tf.config.threading.set_inter_op_parallelism_threads(inter_threads)


def set_num_threads(num_threads=0):
    """Set the number of threads to use in system calls. This sets environment variables for ITK, OpenMP, OpenBLAS, and other
    libraries.

    The number of threads to use in ANTs commands is set by the environment variable ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS.

    The number of threads can be set explicitly, or automatically. If num_threads is 0, the number of threads is set
    automatically to min(system_cores, 8).

    This function should be called before importing ANTsPy or any other library that uses ITK.

    Parameters:
    ----------
    num_threads : int, optional
        The number of threads to use. If 0, the number of threads is set automatically.
    """
    if num_threads < 1:
        num_threads = min(os.cpu_count(), 8)

    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)

