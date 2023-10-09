from glob import glob
from PIL import Image, ImageDraw
import streamlit as st
import numpy as np
import pandas as pd
import os
from model import CNNReLUBlock


def draw_image(
    img_name,
    df,
    draw_bbox,
    net,
    img_path="data/images",
    col=["bbox_x", "bbox_y", "bbox_width", "bbox_height"],
):
    img = Image.open(os.path.join(img_path, img_name)).convert("RGB")
    process_img = net(img)

    for IMG in [img, process_img]:
        if draw_bbox:
            img_df = df[df.image_name == img_name]
            draw = ImageDraw.Draw(IMG)
            for bbox in np.array(img_df.loc[:, col]):
                x, y, w, h = bbox
                draw.rectangle((x, y, x + w, y + h), outline=(255, 0, 0), width=3)
    st.image(img)
    st.image(process_img)


def make_data_fram(label_path="data/labels"):
    df = pd.concat([pd.read_csv(p) for p in glob(f"{label_path}/*.csv")])
    return df


def main():
    df = make_data_fram()
    net = CNNReLUBlock().to("cuda")
    img_list = sorted(df.image_name.unique(), key=lambda x: int(x.split(".")[0]))
    with st.sidebar:
        img_name = st.selectbox("Image Name", img_list)
        draw_bbox = st.checkbox("draw bbox", value=True)

    draw_image(img_name, df, draw_bbox, net)


if __name__ == "__main__":
    main()
