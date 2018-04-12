FROM nvidia/cuda:9.0-base-ubuntu16.04

RUN apt-get update && apt-get install -y software-properties-common\
    	    	      	      	      	 curl

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.6
RUN ln -s /usr/bin/python3.6 /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp36-cp36m-linux_x86_64.whl

WORKDIR "/root/workdir"
ADD scripts/startup.sh ./

ENTRYPOINT ./startup.sh
