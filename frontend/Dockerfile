FROM ubuntu:18.04

RUN apt-get update && apt-get install -y sudo
RUN apt update && apt install npm -y 
RUN apt install curl -y && curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash -
RUN apt install nodejs -y && npm install -g npm@latest -y
RUN npm install -g @vue/cli

WORKDIR /home/app/
COPY . /home/app/

RUN npm install

RUN npm install bootstrap bootstrap-vue

ENTRYPOINT ["bash", "/home/app/frontend.sh"]