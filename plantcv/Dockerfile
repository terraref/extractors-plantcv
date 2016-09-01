FROM ncsa/clowder-python-base
MAINTAINER Yan Y. Liu <yanliu@illinois.edu>

# Setup environment variables. These are passed into the container. You can change
# these to your setup. If RABBITMQ_URI is not set, it will try and use the rabbitmq
# server that is linked into the container. MAIN_SCRIPT is set to the script to be
# executed by entrypoint.sh
ENV CLOWDER_HOME="/home/clowder" \
    EXTRACTOR_HOME="/home/clowder/extractors-lemnatec-indoor/plantcv" \
    RABBITMQ_URI="" \
    RABBITMQ_EXCHANGE="clowder" \
    RABBITMQ_VHOST="%2F" \
    RABBITMQ_QUEUE="terra.plantcv" \
    MAIN_SCRIPT="terra.plantcv.py"

### plantcv part: start
# Install common utils & plantcv dependencies
USER root
RUN apt-get update && \
        apt-get install -y -q build-essential && \
        apt-get install -y -q wget git imagemagick \
            unzip cmake libgtk2.0-dev \
            python-dev python-numpy python-gtk2 python-matplotlib python-pip \
            libavcodec-dev libavformat-dev libswscale-dev libdc1394-22 \
            libjpeg-dev libpng-dev libjasper-dev libtiff-dev libtbb-dev sqlite3 && \
        rm -rf /var/lib/apt/lists/*

USER clowder
RUN cd ${CLOWDER_HOME} && \
    mkdir software && \
    # install opencv from source with python support, plantcv
    cd ${CLOWDER_HOME}/software && \
    git clone -q https://github.com/Itseez/opencv.git && \
    cd opencv && git checkout -q 2.4.8 && \
    mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=${CLOWDER_HOME}/opencv -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON .. && \
    make && make install && \
    cd ${CLOWDER_HOME}/software && rm -fr opencv

# install plantcv & prepare output dir
RUN cd ${CLOWDER_HOME} && \
    mkdir -p plantcv-output && \
    git clone https://github.com/danforthcenter/plantcv.git

# set env
ENV PYTHONPATH="${CLOWDER_HOME}/opencv/lib/python2.7/dist-packages:${CLOWDER_HOME}/plantcv/lib:/usr/lib/python2.7/dist-packages:${PYTHONPATH}"
ENV PATH="${CLOWDER_HOME}/opencv/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CLOWDER_HOME}/opencv/lib:${LD_LIBRARY_PATH}"
### plantcv part: end

# Install the requied software.
USER root
RUN apt-get install -y python-opencv opencv-data && \
    yes | pip install virtualenv && \
RUN chsh -s /bin/bash clowder

# Get git source, fix the default paths.
USER clowder
RUN cd ${CLOWDER_HOME} && \
    git clone https://github.com/terraref/extractors-lemnatec-indoor.git && \
    virtualenv pyenv && \
    . pyenv/bin/activate && \
    pip install pika requests wheel matplotlib && \
    pip install git+https://opensource.ncsa.illinois.edu/stash/scm/cats/pyclowder.git@bugfix/CATS-554-add-pyclowder-support-for-dataset && \
    deactivate

USER root
# Copy the helper script.
COPY entrypoint.sh *.py ${EXTRACTOR_HOME}/
RUN chown clowder ${EXTRACTOR_HOME}/entrypoint.sh && \
    chmod 755 ${EXTRACTOR_HOME}/entrypoint.sh

USER clowder
WORKDIR ${EXTRACTOR_HOME}
CMD ["./entrypoint.sh"]

