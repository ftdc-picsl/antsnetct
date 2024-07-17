# 
# To run locally, you need ANTs, c3d, and trim_neck.sh on the PATH
#
rm -rf dist antsnetct.egg-info; python -m build .; pip uninstall -y antsnetct; pip install --no-input dist/antsnetct-0.0.3.dev0-py3-none-any.whl
