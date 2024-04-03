# __init__.py

import tensorflow as tf

from .log_config import configure_logging

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

configure_logging()