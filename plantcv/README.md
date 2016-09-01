Clowder extractor for plantcv images.

This extractor uses Clowder API to fetch an uploaded plantcv image, invokes appropriate plantcv image analysis tool to analyze it and generates a series of plantcv output images and numerical data.

# Install
## system packages
On a Ubuntu box, you can install dependencies using `apt-get`:

    apt-get install -y -q build-essential unzip cmake libgtk2.0-dev python-dev python-numpy python-gtk2 python-matplotlib libavcodec-dev libavformat-dev libswscale-dev libdc1394-22 libjpeg-dev libpng-dev libjasper-dev libtiff-dev libtbb-dev sqlite3 wget git imagemagick

## opencv
So far, it uses opencv 2.4.8, installed from github source:

    git clone -q https://github.com/Itseez/opencv.git && \
    cd opencv && git checkout -q 2.4.8 && \
    mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=${CLOWDER_HOME}/opencv -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON .. && \
    make && make install

## plantcv

    git clone https://github.com/danforthcenter/plantcv.git

## prepare output dir

    mkdir plantcv-output

## set environment variables
Make sure `PATH` and `LD_LIBRARY_PATH` to include opencv bin and lib dir; make sure that `PYTHONPATH` include python path to opencv and plantcv:

    export PYTHONPATH=$HOME/opencv/lib/python2.7/site-packages:$HOME/plantcv/lib:$PYTHONPATH

## install this extractor
We recommend using python's virtualenv to run this extractor, within which we install extractor dependencies:

    git clone https://opensource.ncsa.illinois.edu/bitbucket/scm/cats/extractors-plantcv.git && \
    cd extractors-plantcv && \
    virtualenv pyenv && \
    . pyenv/bin/activate && \
    pip install pika requests wheel matplotlib && \
    pip install git+https://opensource.ncsa.illinois.edu/stash/scm/cats/pyclowder.git && \
    deactivate

# Run extractor service
First, change `config.py` to set correct value to Clowder rabbimq server and plantcv tool path. Then run the extractor service:

    . extractors-plantcv/pyevn/bin/activate && ./extractors-plantcv/plantcv/ex-plantcv.py >/tmp/extractor.log 2>&1 &

# plantcv Dockerfile
Please refer to plantcv Dockerfile setting if you want to run it in Docker.

# Batch processing mode using a job scheduler
We can also run a lot of extractors simultaneously when we see many image uploads. Code in `hpc/` directory includes :1) an agent watching the number of new uploaded images and launching extractor jobs on a hpc system; 2) HPC-side script to run extractor.
## Deploy extractor on HPC system
- Use git to clone this code repository.
- Install GNU-Parallel on the system.
- Modify `hpc/bashrc.plantcv` to set the software environment and copy it to `$HOME/.bashrc.plantcv`.
- Create a config file for rabbitmq `$HOME/.rabbitmq.plantcv` with a line looking like `user_name:password`. `user_name` is the user name of the rabbitmq server.
- Create a config file for Clowder `$HOME/.clowderaccount` with a line looking like `email:password`, where `email` is your clowder account email.
## Deploy agent
Deploy the agent on a machine with access to both the messaging server and the HPC system (head node).
- Use git to clone this code repository.
- Create a config file for rabbitmq `$HOME/.rabbitmq.plantcv` with a line looking like `user_name:password`. `user_name` is the user name of the rabbitmq server.
- Set up ssh key pairs to be able to do password-less access to HPC system.
- Create the job submission script template for your HPC system by referring to `_t.roger`.
- Modify `plantcv-agent.sh`, the config part.
- Add `plantcv-agent.sh` as a user-level cron job.


Contact: Yan Y. Liu <yanliu@illinois.edu>