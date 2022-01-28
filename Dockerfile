FROM nvidia/11.6.0-runtime-ubuntu20.04
COPY requirements.txt .
RUN apt-get update -qq
RUN apt-get install -qq apt-utils curl python3 python3-pip
RUN ln -snf /usr/share/zoneinfo/$(curl https://ipapi.co/timezone) /etc/localtime
RUN pip3 install -r requirements.txt