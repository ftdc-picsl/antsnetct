#!/usr/bin/env python
import sys

from .cross_sectional_pipeline import cross_sectional_analysis
from .longitudinal_pipeline import longitudinal_analysis
from .log_config import configure_logging

def main():

    configure_logging()

    if '--longitudinal' in sys.argv:
        sys.argv.remove('--longitudinal')
        longitudinal_analysis()
    else:
        cross_sectional_analysis()

if __name__ == "__main__":
    main()