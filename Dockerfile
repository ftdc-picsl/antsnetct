FROM cookpa/antsnetct:0.6.2 AS base

# Need to redefine these otherwise they get inherited from the base image
ARG DOCKER_IMAGE_TAG="unknown"
ARG DOCKER_IMAGE_VERSION="unknown"
ARG GIT_REMOTE="unknown"
ARG GIT_COMMIT="unknown"

USER root

COPY . /opt/src/antsnetct

RUN pip install --no-cache-dir --force-reinstall --no-deps /opt/src/antsnetct && \
    rm -rf /root/.cache/pip

LABEL maintainer="Philip A Cook (https://github.com/cookpa)"
LABEL description="Containerized BIDS cortical thickness pipelines using antspynet."
LABEL git.remote=$GIT_REMOTE
LABEL git.commit=$GIT_COMMIT

ENV GIT_REMOTE=$GIT_REMOTE
ENV GIT_COMMIT=$GIT_COMMIT
ENV DOCKER_IMAGE_TAG=$DOCKER_IMAGE_TAG
ENV DOCKER_IMAGE_VERSION=$DOCKER_IMAGE_VERSION

ENV LD_LIBRARY_PATH="/opt/ants/lib"
ENV PATH="/opt/bin:/opt/ants/bin:$PATH"

USER antspyuser

ENTRYPOINT ["antsnetct"]
