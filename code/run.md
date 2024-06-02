# HOW TO BUILD AND RUN
```bash
# base conda image
docker build . --tag base_conda:$TAG \
-f docker/base_images/base_conda.Dockerfile \
--build-arg="env_yml=$ENV_YML_PATH"
# build research img
docker build . --tag research_img:$TAG \
-f docker/research.Dockerfile --pull=false \
--build-arg="env_name=$TAG"
# run
docker run \
--mount type=bind,source="$(pwd)"/dataset_sources,target=/dataset_sources,readonly \
--mount type=bind,source="$(pwd)"/configs,target=/configs,readonly \
--memory=8g \
--cpus=4 \
--env-file $DOT_ENV_PATH research_img:$TAG
```


