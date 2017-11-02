#!/bin/bash
set -e

# rabbitmq
if [ "$RABBITMQ_URI" == "" ]; then
    # configure RABBITMQ_URI if started using docker-compose or --link flag
    if [ -n "$RABBITMQ_PORT_5672_TCP_ADDR" ]; then
        RABBITMQ_URI="amqp://guest:guest@${RABBITMQ_PORT_5672_TCP_ADDR}:${RABBITMQ_PORT_5672_TCP_PORT}/${RABBITMQ_VHOST}"
    else
        RABBITMQ_URI="amqp://guest:guest@localhost:5672/%2F"
    fi

    # configure RABBITMQ_URI if rabbitmq is up for kubernetes
    # TODO needs implementation maybe from NDS people
fi


if [ "$1" = 'extractor' ]; then
    cd /home/extractor

    # Set plantcv env var
    /bin/sed -i -e "s#plantcvOutputDir =.*#plantcvOutputDir = '/home/extractor/plantcv-output'#" config.py

    # fix plancv bugs in analyze_color()
    # analyze_color takes 11 args, but image_analysis scripts put 12
    for d in nir_sv vis_sv  vis_tv
    do
      for f in `ls $/home/extractor/plantcv/scripts/image_analysis/$d/*.py`
      do
        /bin/sed -i -e "s#'all','rgb'#'all'#" $f
      done
    done

    # start the extractor service
    source /home/extractor/pyenv/bin/activate && ./${MAIN_SCRIPT}
fi
