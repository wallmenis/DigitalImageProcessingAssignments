#! /bin/bash
curl -LJO "https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel"
curl -LJO "https://raw.githubusercontent.com/s9xie/hed/master/examples/hed/deploy.prototxt" > deploy.prototxt