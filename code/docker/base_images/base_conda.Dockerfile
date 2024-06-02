# sudo dockerd
FROM continuumio/miniconda
ARG env_yml
# Create the environment:
COPY envs/$env_yml ./environment.yml
RUN conda env create -f environment.yml
WORKDIR .
