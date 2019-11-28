FROM tensorflow/tensorflow:2.0.0-gpu-py3

RUN apt-get update && apt-get install -y apt-utils
RUN apt-get install -y apt-transport-https

COPY ./requirements.txt /project/requirements.txt
WORKDIR /project
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

ENTRYPOINT ["/bin/bash"]
