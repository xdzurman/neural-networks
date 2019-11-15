#!/bin/bash
docker run --gpus all -p 8888:8888 -p 6006:6006 -v $(pwd)/:/project -it yimejky/nn-project
