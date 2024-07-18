#!/usr/bin/env python

import sys
import re
import subprocess

def run_command(command):
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    return result.stdout.strip(), result.returncode

def check_clean_repository():
    output, _ = run_command('git status --porcelain')
    if output:
        print("Warning: Repository has local modifications.")
        sys.exit(1)

def check_main_branch():
    output, _ = run_command('git branch --show-current')
    if output != 'main':
        print("Warning: main branch is not checked out.")
        sys.exit(1)

def check_existing_tag(tag):
    output, _ = run_command('git tag')
    if tag in output.split('\n'):
        print(f"Warning: Tag {tag} already exists.")
        sys.exit(1)

def update_version_in_pyproject(version):
    with open('pyproject.toml', 'r') as file:
        lines = file.readlines()

    with open('pyproject.toml', 'w') as file:
        for line in lines:
            if line.startswith('version ='):
                file.write(f'version = "{version}"\n')
            else:
                file.write(line)

def main():
    if len(sys.argv) != 2:
        print("Usage: python git_tag.py vX.Y.Z")
        sys.exit(1)

    tag = sys.argv[1]
    if not re.match(r'^v\d+\.\d+\.\d+$', tag):
        print("Error: Tag must be of the form vX.Y.Z where X, Y, and Z are integers.")
        sys.exit(1)

    version = tag[1:]  # Remove the 'v' prefix

    check_clean_repository()
    check_main_branch()
    check_existing_tag(tag)

    confirmation = input(f"Do you want to create the tag {tag}? (y/n): ")
    if confirmation.lower() != 'y':
        print("Exiting without creating tag.")
        sys.exit(0)

    update_version_in_pyproject(version)
    run_command('git add pyproject.toml')
    run_command(f'git commit -m "updating version for tag {tag}"')
    run_command('git push origin main')
    run_command(f'git tag -a {tag} -m "{tag}"')
    run_command('git push --tags')

    print(f"Tag {tag} has been created.")

    # Now update pyproject.toml with the next version development version
    major, minor, patch = version.split('.')

    next_version = f"{major}.{minor}.{int(patch) + 1}dev"

    update_version_in_pyproject(next_version)

    run_command('git add pyproject.toml')
    run_command(f'git commit -m "updating version for development post {tag}"')
    run_command('git push origin main')

if __name__ == '__main__':
    main()
