FROM terraref/terrautils:plantcv
MAINTAINER Max Burnette <mburnet2@illinois.edu>

# Install any programs needed
# TODO: Can we remove this user ID and mount files as Max?
RUN useradd -u 49044 extractor \
    && mkdir -p /home/extractor/sites/danforth/raw_data \
    && mkdir -p /home/extractor/sites/danforth/Level_1 \
    && chown -R extractor /home/extractor

### plantcv part: start
# Install common utils & plantcv dependencies
#USER root
RUN apt-get -y update \
    && apt-get install -y -q git \
        unzip \
        cmake \
        libgtk2.0-dev \
        python-numpy \
        python-gtk2 \
        python-matplotlib \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libdc1394-22 \
        libjpeg-dev \
        libjasper-dev \
        libtiff-dev \
        libtbb-dev \
        sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# install opencv from source with python support, plantcv
#USER extractor
RUN cd /home/extractor && mkdir software && cd software \
    && git clone -q https://github.com/Itseez/opencv.git \
    && cd opencv && git checkout -q 2.4.8 \
    && mkdir build && cd build \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/home/extractor/opencv -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON .. \
    && make -j8 && make install \
    && cd ../.. && rm -fr opencv

# install plantcv & prepare output dir
RUN cd /home/extractor \
    && mkdir -p plantcv-output \
    && git clone https://github.com/danforthcenter/plantcv.git \
    && cd plantcv \
    && pip install opencv-python numpy --upgrade \
    && pip install -r requirements.txt \
    && python setup.py install

# set env
ENV PYTHONPATH="/home/extractor/opencv/lib/python2.7/dist-packages:/home/extractor/plantcv/lib:/usr/lib/python2.7/dist-packages:${PYTHONPATH}"
ENV PATH="/home/extractor/opencv/bin:${PATH}"
ENV LD_LIBRARY_PATH="/home/extractor/opencv/lib:${LD_LIBRARY_PATH}"
### plantcv part: end

# Install the required software.
#USER root
RUN apt-get -y update \
    && apt-get install -y python-opencv \
        opencv-data \
    && yes | pip install virtualenv
RUN chsh -s /bin/bash extractor

# Copy the helper script.
COPY entrypoint.sh extractor_info.json *.py /home/extractor/

USER extractor
ENTRYPOINT ["/home/extractor/entrypoint.sh"]
CMD ["extractor"]

ENV RABBITMQ_EXCHANGE="terra" \
    RABBITMQ_VHOST="%2F" \
    RABBITMQ_QUEUE="terra.lemnatec.plantcv" \
    MAIN_SCRIPT="terra_plantcv.py"
