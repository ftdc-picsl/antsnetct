#!/usr/bin/env python

import os
import sys

import argparse
from cross_sectional_pipeline import cross_sectional_analysis
from longitudinal_pipeline import longitudinal_analysis


def main():
    parser = argparse.ArgumentParser(description='ANTsPyNet Cortical thickness')

    parser.add_argument('--longitudinal', action='store_true', help='Perform longitudinal analysis')

    args = parser.parse_args()

    if args.longitudinal:
        longitudinal_analysis(args)
    else:
        cross_sectional_analysis(args)

if __name__ == "__main__":
    main()