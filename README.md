### Download the models

```bash
huggingface-cli download black-forest-labs/FLUX.1-schnell --local-dir ./local_models/black-forest-labs/FLUX.1-schnell
huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir ./local_models/black-forest-labs/FLUX.1-dev
```

#### Build the docker image and push to gcr

```bash
gcloud auth configure-docker
docker login gcr.io

docker build -t silk-image-generation .
docker tag silk-image-generation gcr.io/citric-lead-450721-v2/silk-image-generation:1.0.0
docker push gcr.io/citric-lead-450721-v2/silk-image-generation:1.0.0


```
