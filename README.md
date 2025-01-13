# antsnetct
Cortical thickness with ANTsPyNet


## Installation

Install a complete runnable container from [Docker
Hub](https://hub.docker.com/repository/docker/cookpa/antsnetct/general).

Non-containerized installation is not recommended, but is available to advanced users.
First install the system requirements [ANTs](https://github.com/ANTsX/ANTs),
[c3d](https://github.com/pyushkevich/c3d), and
[trim_neck.sh](https://github.com/ftdc-picsl/antsnetct/blob/main/scripts/trim_neck.sh).
These need to be on the system path.

Then install with pip
```
git clone https://github.com/ftdc-picsl/antsnetct
pip install antsnetct
```

If you run into errors with the pip install, you may need to pre-install specific
versions of ANTsPyNet and / or ANTsPy. Note also that cached ANTsPyNet data in `~/.keras`
can get out of date and affect results. The docker containers manage the software and data
dependencies for you, and are the recommended way to run antsnetct.


## Configuration

All users must set the environment variable `TEMPLATEFLOW_HOME` to a location containing
the template to be used. You can use any template as long as it has both a `_T1w.nii.gz`
file and an associated brain mask.

The default is a single thread for all operations. Multi-threading of ITK can be enabled
by setting the environment variable `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`. Set
`ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=0` to use all available cores, or 8 threads,
whichever is smaller. Multi-threading of tensorflow operations in ANTsPyNet processes is
controlled by the `TF_NUM_INTRAOP_THREADS` and `TF_NUM_INTEROP_THREADS` variables.


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

## Cross-sectional output

Output is prefixed with the source entities, which uniquely identifies each T1w input.
Some files have metadate stored in a JSON sidecar.

<table>
  <tr>
    <th>File</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>desc-biascorrbrain_T1w.nii.gz</td>
    <td>Bias-corrected and denoised image, masked with the brain mask.</td>
  </tr>
  <tr>
    <td>desc-biascorr_T1w.nii.gz</td>
    <td>Bias-corrected and denoised image, with the skull on.</td>
  </tr>
  <tr>
    <td>desc-brain_mask.nii.gz</td>
    <td>Binary brain mask for the T1w.</td>
  </tr>
  <tr>
    <td>desc-preproc_T1w.nii.gz</td>
    <td>The input T1w after preprocessing but before any antsnetct processing. Currently, the only preprocessing is neck
    trimming.</td>
  </tr>
  <tr>
    <td>from-{template}_to-T1w_mode-image_xfm.h5</td>
    <td>Composite transform for warping images from the template space to the space of the T1w.</td>
  </tr>
  <tr>
    <td>from-T1w_to-{template}_mode-image_xfm.h5</td>
    <td>Composite transform for warping images from the T1w space to the template.</td>
  </tr>
  <tr>
    <td>seg-antsnetct_desc-thickness.nii.gz</td>
    <td>Cortical thickness in mm.</td>
  </tr>
  <tr>
    <td>seg-antsnetct_dseg.nii.gz</td>
    <td>Six-class brain segmentation, using BIDS common derived segmentation labels.</td>
  </tr>
  <tr>
    <td>seg-antsnetct_label-{label}_probseg.nii.gz</td>
    <td>Probability image for each segmentation label.</td>
  </tr>
  <tr>
    <td>space-{template}[_res-{template_res}]_desc-biascorrbrain_T1w.nii.gz</td>
    <td>Bias-corrected, denoised image in the template space.</td>
  </tr>
  <tr>
    <td>space-{template}[_res-{template_res}]_desc-logjacobian.nii.gz</td>
    <td>Log jacobian of the nonlinear component of the warp to the template, for analysis of nonlinear volume change.</td>
  </tr>
  <tr>
    <td>space-{template}[_res-{template_res}]_desc-thickness.nii.gz</td>
    <td>Cortical thickness in the template space.</td>
  </tr>
  <tr>
    <td>space-ADNINormalAgingANTs_res-01_label-CGM_probseg.nii.gz</td>
    <td>Cortical gray matter probability in the template space, for VBM.</td>
  </tr>
</table>



## Longitudinal thickness

First run the cross-sectional (cx) pipeline on all time points. To make this faster, you can use
the `--longitudinal-preproc` option, which will skip thickness and group template normalization.

The longitudinal pipeline defaults to running all session T1w images, but user can input a custom
list.

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


## Longitudinal output

### Subject-level output

Under `sub-subjeclabel/anat`, subject-level output files are prefixed with `sub-JP01`. The
T1w image in this folder is the single-subject template (SST).

<table>
  <tr>
    <th>File</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>desc-brain_mask.nii.gz</td>
    <td>Brain mask for the SST.</td>
  </tr>
  <tr>
    <th>File</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>desc-brain_T1w.nii.gz</td>
    <td>Brain-extracted SST.</td>
  </tr>
  <tr>
    <td>from-{template}_to-T1w_mode-image_xfm.h5</td>
    <td>Transform from the group template to the SST.</td>
  </tr>
  <tr>
    <td>from-T1w_to-{template}_mode-image_xfm.h5</td>
    <td>Transform from the SST to the group template.</td>
  </tr>
  <tr>
    <td>seg-antsnetct_dseg.nii.gz</td>
    <td>Segmentatation of the SST.</td>
  </tr>
  <tr>
    <td>seg-antsnetct_label-{label}_probseg.nii.gz</td>
    <td>Probability images from the segmentation. These are used as priors for session-level segmentation.</td>
  </tr>
  <tr>
    <td>T1w.nii.gz</td>
    <td>The SST image.</td>
  </tr>
</table>

### Session-level output

Session-level output is similar to the cross-sectional output, except that the transforms
map to the SST space, not the group template space. To warp images to the group template,
combine the warps, for example:
```
antsApplyTransforms \
    -d 3 \
    -i sub-01_ses-01_desc-biascorrbrain_T1w.nii.gz \
    -r tpl-group_T1w.nii.gz \
    -o sub-01_ses-01_space-group_desc-biascorrbrain_T1w.nii.gz \
    -t sub-01_from-T1w_to-group_template_mode-image_xfm.h5  sub-01_ses-01_from-T1w_to-sst_mode-image_xfm.h5
```
