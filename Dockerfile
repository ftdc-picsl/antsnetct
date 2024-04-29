FROM cookpa/antspynet:latest as builder

# Use a builder layer to get data and networks. These change infrequently, so can be downloaded
# once and then cached.
USER root

# Copy only files needed to get data and networks
COPY selected_antspynet_data.txt selected_antspynet_networks.txt /opt

# Get antspynet data that we need
RUN /opt/bin/get_antsxnet_data.py /home/antspyuser/.keras 1 \
        /opt/selected_antspynet_data.txt /opt/selected_antspynet_networks.txt \
    && chmod -R 0755 /home/antspyuser/.keras

FROM cookpa/antspynet:latest as runtime

ARG DOCKER_IMAGE_TAG="unknown"
ARG DOCKER_IMAGE_VERSION="unknown"

ARG GIT_REMOTE="unknown"
ARG GIT_COMMIT="unknown"

USER root

# Get c3d
COPY --from=pyushkevich/tk:2023b /tk/c3d/build/c3d /opt/bin/c3d

# Update antspy
RUN pip install -U \
        templateflow==24.0.0 \
        https://github.com/ANTsX/ANTsPy/releases/download/v0.5.2/antspyx-0.5.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
    && rm -rf /root/.cache/pip

COPY scripts/trim_neck.sh /opt/bin/trim_neck.sh
# Copy data and code from builder
COPY --from=builder /home/antspyuser/.keras /home/antspyuser/.keras
COPY . /opt/src/antsnetct

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