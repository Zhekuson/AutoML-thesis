ARG env_name
FROM base_conda:$env_name
WORKDIR .
COPY ../src src/
ENV PATH /opt/conda/envs/research_env/bin:$PATH
RUN echo "conda activate research_env" >> ~/.bashrc
RUN export PYTHONPATH="$PYTHONPATH:$PWD"
RUN chown root .
RUN chown root /root/
CMD python src/benchmark.py --config_path=${config_path}