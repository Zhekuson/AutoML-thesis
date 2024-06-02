ARG env_name
FROM base_conda:$env_name
#RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get --allow-releaseinfo-change update && apt-get install libgl1 -y
WORKDIR .
COPY ../src src/
ENV PATH /opt/conda/envs/research_env/bin:$PATH
RUN echo "conda activate research_env" >> ~/.bashrc
RUN export PYTHONPATH="$PYTHONPATH:$PWD"
RUN chown root .
RUN chown root /root/
CMD python src/run.py --config_path=${config_path}