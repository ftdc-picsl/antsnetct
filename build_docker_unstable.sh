#!/bin/bash

# Get git information and build docker image
forceBuild=0

if [[ $# -gt 0 ]] ; then
    if [[ "$1" == "-h" ]] ; then
        echo "usage: $0 [-h] [-f]"
        echo "Builds a docker image from source, and embeds git info. Run from source directory."
        echo
        echo "The git tag will be used if present, otherwise the docker image tage will be 'unstable'."
        echo
        echo "By default, the script will not build if the git repository is not clean. Override with -f."
        echo
        exit 1
    elif [[ "$1" == "-f" ]] ; then
        forceBuild=1
    fi
fi

# Get git information
status=$( git status -s )

# status should be empty if the repository is clean
if [[ ! -z "$status" ]] ; then
    echo "Repository is not clean - see git status:"
    git status
    echo
    if [[ $forceBuild -eq 0 ]] ; then
        echo "Use -f to force build"
        exit 1
    else
        echo "Building from repository with local modifications"
    fi
fi

gitRemote=$( git remote get-url origin )

# Get the git hash or tag
hash=$( git rev-parse HEAD )

# See if there's a tag
gitTag=$( git describe --tags --abbrev=0 --exact-match $hash 2>/dev/null || echo "" )

dockerVersion="unstable"

if [[ -z "$gitTag" ]]; then
    echo "No tag found for commit $hash"
    gitTag=${hash:0:7}
else
    echo "Tag found: $gitTag"
    # Check tag is acceptable as a docker version tag, ie lowercase alphanumeric and hyphens only
    if [[ ! "$gitTag" =~ ^[a-z0-9-]+$ ]] ; then
        echo "Tag $gitTag is not suitable for use as a docker version tag"
        exit 1
    fi
    dockerVersion="$gitTag"
fi

dockerTag="cookpa/antsnetct:${dockerVersion}"

# Build the docker image
docker build -t "$dockerTag" . \
    --build-arg GIT_REMOTE="$gitRemote" \
    --build-arg GIT_COMMIT="$gitTag" \
    --build-arg DOCKER_IMAGE_TAG="$dockerTag" \
    --build-arg DOCKER_IMAGE_VERSION="$dockerVersion"

if [[ $? -ne 0 ]] ; then
    echo "Docker build failed - see output above"
    exit 1
else
    echo
    echo "Build successful: $dockerTag"
    echo
fi