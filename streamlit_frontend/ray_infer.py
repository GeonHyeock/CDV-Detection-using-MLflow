import cv2
import pandas as pd
import os
import zipfile
import io
import ray


def Batch_Infer(
    img_path_list,
    batch_size=2,
    img_shape=(640, 640),
    conf_thres=0.4,
    iou_thres=0.45,
    sic=False,
):
    n = len(img_path_list)
    batch_result = [
        batch_infer.remote(img_path_list[idx : idx + batch_size], conf_thres, iou_thres, img_shape, sic)
        for idx in range(0, n, batch_size)
    ]
    batch_result, buf = ray.get(batch_result), io.BytesIO()
    with zipfile.ZipFile(buf, "x") as csv_zip:
        for csvs, csv_names in batch_result:
            for csv, csv_name in zip(csvs, csv_names):
                csv_zip.writestr(csv_name, pd.DataFrame(csv).to_csv(index=False))
        csv_zip.extractall(f"inputdata2_result/conf{(conf_thres)}_iou{(iou_thres)}")
    return


@ray.remote
def batch_infer(img_path_list, conf_thres, iou_thres, img_shape=(640, 640), sic=False):
    from streamlit_frontend import draw_bbox_array, Infer, make_csv

    if not isinstance(img_path_list, list):
        img_path_list = [img_path_list]
    image = [cv2.imread(f) for f in img_path_list]
    det = Infer(image, conf_thres, iou_thres)
    det = draw_bbox_array(det, img_shape, image, sic, only_det=True)
    csvs = make_csv(det)

    csv_names = [
        ".".join(("/".join(file.split("/")[1:]) + f"_conf{(conf_thres)}_iou{(iou_thres)}").split(".")[:-1] + ["csv"])
        for file in img_path_list
    ]
    return csvs, csv_names


if __name__ == "__main__":
    ray.init()
    img_path_list = [
        root + "/" + file_name
        for root, _, files in os.walk("inputdata2")
        for file_name in files
        if os.path.splitext(file_name)[-1] in [".jpg", ".jpeg", ".png"]
    ]
    result = []
    for conf_thres in [0.1]:
        for iou_thres in [0.1]:
            Batch_Infer(img_path_list)
