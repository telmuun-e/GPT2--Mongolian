FROM ubuntu:18.04

RUN apt-get update && apt-get install -y python3 python3-pip sudo
RUN pip3 install --upgrade pip

COPY . /home/app/

RUN cd /home/app && pip3 install -r requirements.txt

WORKDIR /home

EXPOSE 8080

ENTRYPOINT python3 /home/app/app.py
