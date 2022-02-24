FROM nvidia/cuda:11.2.1-runtime-ubuntu20.04

# set env vars
ENV DEBIAN_FRONTEND=noninteractive

# update packages
RUN apt-get update -qq

# install reqs
RUN apt-get install -y -qq build-essential git git-lfs wget software-properties-common python3 python3-pip ffmpeg libsm6 libxext6
RUN ln -snf /usr/share/zoneinfo/$(curl https://ipapi.co/timezone) /etc/localtime

# install python packages
COPY requirements.txt .
RUN pip3 install -r requirements.txt
