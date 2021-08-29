# AI Spark Template

## Build Docker
```bash
$ docker build -t ai-spark-template:latest -f Dockerfile .
```

## Run Docker
```bash
docker run -it --name ai-spark-template -v C:/Users/YIsaienkov/Documents/ai-spark-template/:/app/ --rm ai-spark-template /bin/bash
```