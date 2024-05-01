# antsnetct
Cortical thickness with ANTsPyNet


## Installation

Install a complete runnable container from [Docker
Hub](https://hub.docker.com/repository/docker/cookpa/antsnetct/general).

For non-containerized installation (not recommended), first install the system
requirements [ANTs](https://github.com/ANTsX/ANTs),
[c3d](https://github.com/pyushkevich/c3d), and
[trim_neck.sh](https://github.com/ftdc-picsl/antsnetct/blob/main/scripts/trim_neck.sh).
Then install with pip
```
git clone https://github.com/ftdc-picsl/antsnetct
pip install antsnetct
```

## Configuration

All users must set the environment variable `TEMPLATEFLOW_HOME` to a location containing
the template to be used. You can use any template as long as it has both a `_T1w.nii.gz`
file and an associated brain mask.


## Configuration for docker

```
docker run --rm -it -v /path/to/local/templateflow:/opt/templateflow \
  -e TEMPLATEFLOW_HOME=/opt/templateflow \
  -e ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2 \
  antsnetct:latest --help
```

## Configuration for singularity

Because singularity does not allow the container to set its user, additional options
are required.
```
export SINGULARITYENV_ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2
export SINGULARITYENV_TEMPLATEFLOW_HOME=/opt/templateflow
singularity run --cleanenv --no-home --home /home/antspyuser \
  -B /path/to/local/templateflow:/opt/templateflow \
  antsnetct_latest.sif --help
```


## Cross-sectional thickness

Basic pipeline

* Conform to standard orientation.
* Trim neck (optional).
* Brain extraction (done by antspynet, or supply your own brain mask dataset, or have one
  in the input dataset).
* Segmentation and bias correction. Priors can come from deep_atropos or user-supplied
  priors. Priors can be used as input to antsAtroposN4.sh, or used directly as the final
  segmentation.
* Denoise and bias-correct T1w using the segmentation.
* Thickness computation. Same as antsCorticalThickness.sh.
* Warp to template - similar to antsCorticalThickness.sh
* Generate template space derivatives, Jacobian, GMP, thickness, etc.

See `antsnetct --help` for current usage.


## Longitudinal thickness