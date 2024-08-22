# __init__.py

import tensorflow as tf

from .log_config import configure_logging
from .system_helpers import set_num_threads

configure_logging()

# Set default number of threads - this might be overidden by the user
set_num_threads()