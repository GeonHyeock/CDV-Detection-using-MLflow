model_name="yolo6"
model_version="1"
port="5001"
host="0.0.0.0"

docker run --rm -p ${port}:${port} --gpus all -it mlflow_serving \
 mlflow models serve -m models:/${model_name}/${model_version} \
 --no-conda \
 --port ${port} \
 --host ${host}