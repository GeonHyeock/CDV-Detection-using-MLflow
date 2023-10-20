import argparse
import requests
import mlflow
import cv2
import streamlit as st
import numpy as np
import pandas as pd
from collections import defaultdict
from infer import draw_bbox_array, Infer


@st.cache_data()
def load_model(logged_model="runs:/096a362268c64363b896555db566d5d5/model"):
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    return loaded_model


def infer(image):
    res = requests.post(
        url=args.uri,
        json={"inputs": image},
        headers={"Content-Type": "application/json"},
    )
    if res.status_code == 200:
        result = res.json()
        return result["predictions"]
    else:
        print("Request failed with status code:", res.status_code)
    pass


def main(args):
    st.title("Clustering")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image = cv2.imdecode(np.fromstring(image_bytes, np.uint8), flags=1)
        col1, col2 = st.columns(2)
        with col1:
            st.write("원본")
            st.image(image)

        # demo
        if "model" not in st.session_state:
            st.session_state.model = load_model()

        draw_img_array = (
            np.expand_dims(np.swapaxes(image, 0, 2), 0).astype(np.float32) / 255
        )
        result = Infer(st.session_state.model, image)[:-1]

        with col2:
            draw_img_array = image

            st.write("결과")
            draw_img_array, det = draw_bbox_array(result, (640, 640), image)
            st.image(draw_img_array)

            csv = defaultdict(list)
            for d in det:
                x, y, x2, y2, c, _ = d
                w, h = x2 - x, y2 - y
                x, y = x + w / 2, y + h / 2

                csv["x_center"] += [int(x)]
                csv["y_center"] += [int(y)]
                csv["width"] += [int(w)]
                csv["height"] += [int(h)]
                csv["confidence"] += [float(c)]
                csv["area"] += [int(w * h)]

            st.download_button(
                label="Download result CSV",
                data=pd.DataFrame(csv).to_csv().encode("utf-8"),
                file_name=f"{'.'.join(uploaded_file.name.split('.')[:-1])}_result.csv",
            )
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uri",
        default="http://127.0.0.1/invocations",
        help="serving된 모델 uri",
    )
    args = parser.parse_args()

    main(args)
