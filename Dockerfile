FROM cookpa/antspynet:latest

ARG DOCKER_IMAGE_TAG="unknown"
ARG DOCKER_IMAGE_VERSION="unknown"

ARG GIT_REMOTE="unknown"
ARG GIT_COMMIT="unknown"

USER root

# Get c3d
COPY --from=pyushkevich/tk:2023b /tk/c3d/build/c3d /opt/bin/c3d

# Update antspy
RUN pip install -U \
        https://github.com/ANTsX/ANTsPy/releases/download/v0.5.2/antspyx-0.5.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
    && rm -rf /root/.cache/pip

# Get antspynet data that we need
COPY . /opt/src/antsnetct
COPY scripts/trim_neck.sh /opt/bin/trim_neck.sh

RUN /opt/bin/get_antsxnet_data.py /home/antspyuser/.keras 1 \
        /opt/src/antsnetct/selected_antspynet_data.txt /opt/src/antsnetct/selected_antspynet_networks.txt \
    && chmod -R 0755 /home/antspyuser/.keras

# Install the package
RUN pip install /opt/src/antsnetct

LABEL maintainer="Philip A Cook (https://github.com/cookpa)"
LABEL description="Containerized BIDS cortical thickness pipelines using antspynet."
LABEL git.remote=$GIT_REMOTE
LABEL git.commit=$GIT_COMMIT

ENV GIT_REMOTE=$GIT_REMOTE
ENV GIT_COMMIT=$GIT_COMMIT
ENV DOCKER_IMAGE_TAG=$DOCKER_IMAGE_TAG
ENV DOCKER_IMAGE_VERSION=$DOCKER_IMAGE_VERSION

ENV PATH="/opt/bin:$PATH"

USER antspyuser

ENTRYPOINT ["antsnetct"]
CMD ["-h"]