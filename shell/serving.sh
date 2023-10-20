model_name="cluster"
model_version="2"
port="5001"
host="0.0.0.0"

mlflow models serve -m "models:/$model_name/$model_version" --no-conda --port $port --host $host