#!/usr/bin/env python

import os
import sys

import argparse
from cross_sectional_pipeline import cross_sectional_analysis
from longitudinal_pipeline import longitudinal_analysis


def main():

    if '--longitudinal' in sys.argv:
        sys.argv.remove('--longitudinal')
        longitudinal_analysis()
    else:
        cross_sectional_analysis()

if __name__ == "__main__":
    main()