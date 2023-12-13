# Reference Dockerfile: https://github.com/jupyter/docker-stacks/blob/master/datascience-notebook/Dockerfile
FROM jupyter/minimal-notebook:latest

USER root

RUN apt-get update && apt-get install -y cmake libz-dev graphviz libgraphviz-dev

USER $NB_UID

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
RUN pip uninstall -y atari-py
