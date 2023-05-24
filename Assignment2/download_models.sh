#! /bin/bash

curl -LJO "https://raw.githubusercontent.com/acarcher/hed-opencv-dl/master/hed_model/hed_pretrained_bsds.caffemodel"
curl -LJO "https://raw.githubusercontent.com/acarcher/hed-opencv-dl/master/hed_model/deploy.prototxt" > deploy.prototxt