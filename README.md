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

First run the cross-sectional (cx) pipeline on all time points.

Defaults to running all session T1w images, but user can input a custom list.

The SST is built from the *processed* T1w images (ie, `desc-biascorr`) but the longitudinal
segmentation is done with the *preprocessed* T1w images (ie, `desc-preproc`), which is oriented
and optionally neck-trimmed, but not denoised or bias-corrected.


Pipeline overview:

* Build SST from the cx-processed T1w 
* Define a common brain mask from the cx sessions (basically the union of the session masks in SST space)
* Segment SST with ANTsPyNet
* Register SST to group template (optional)

For each session image:

* Register to SST
* Warp SST segmentation to each session space
* Run denoising / N4 / Atropos on session T1w, using SST segmentation as priors.
* Compute cortical thickness on session T1w
* Warp derivatives to SST / group template space

See `antsnetct --longitudinal --help` for current usage.
