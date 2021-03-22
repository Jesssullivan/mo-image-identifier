#!/bin/bash


## NOT RUN ##


echo -e "...Cleaning compiled @ ~/git/mo-image-identifier/..."


PREFIX="~/git/mo-image-identifier/"


sudo rm -rf ${PREFIX}testing/ ${PREFIX}training/ ${PREFIX}images ${PREFIX}test.tgz ${PREFIX}train.tgz ${PREFIX}images.json ${PREFIX}images.tgz ${PREFIX}categories.json
