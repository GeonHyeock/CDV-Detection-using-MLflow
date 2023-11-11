port="5002"

docker run --rm -p ${port}:${port} -it frontend \
streamlit run streamlit_frontend/serving.py --server.port=${port}