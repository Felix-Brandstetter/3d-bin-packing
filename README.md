# binpack

## Setup Docker Container
1. Build the docker image using the following command in this folder:
`docker build -t binpack .`
1. After building the image, run the following command from this folder to start the container (tested on Windows):  `docker run -it --shm-size=8gb --rm -v ${PWD}:/home/jovyan/work -p 8888:8888 -p 8265:8265 binpack /bin/bash`
1. In the interactive bash inside the container, run `jupyter notebook` to run the notebook server. Follow the instructions on the command line to connect to jupyter via a browser.


# Optionally
## Prevent Jupyter Notebook outputs from being committed to Git
```
pip install nbstripout nbconvert
cd *this repository*
nbstripout --install
```

## Compile Requirements
To create requirements.txt from requirements.in run: `pip-compile --verbose requirements.in`. For further usage
documentation see: https://github.com/jazzband/pip-tools

## Create License Output
`pip-licenses --order=license --format=markdown`.
Further documentation: https://github.com/raimon49/pip-licenses
