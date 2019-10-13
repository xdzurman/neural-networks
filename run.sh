#!/bin/bash
docker run -u $(id -u):$(id -g) -p 8888:8888 -p 6006:6006 -v $(pwd)/src:/labs -it mpikuliak/nsiete
