import cv2
import pandas as pd
import os
import zipfile
import io
import ray
from stqdm import stqdm
from util import draw_bbox_array, Infer, make_csv


@ray.remote
def batch_infer(
    img_path_list,
    img_shape=(640, 640),
    conf_thres=0.4,
    iou_thres=0.45,
    sic=False,
):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "x") as csv_zip:
        batch_size = 2
        for idx in stqdm(range(0, len(img_path_list), batch_size)):
            image = [cv2.imread(f) for f in img_path_list[idx : idx + batch_size]]
            det = Infer(image, conf_thres, iou_thres)
            det = draw_bbox_array(det, img_shape, image, sic, only_det=True)
            csvs = make_csv(det)
            for i, csv in enumerate(csvs):
                file = img_path_list[idx + i]
                csv_name = ".".join("/".join(file.split("/")[1:]).split(".")[:-1] + ["csv"])
                csv_zip.writestr(csv_name, pd.DataFrame(csv).to_csv(index=False))
        csv_zip.extractall(f"inputdata2_result/conf{(conf_thres)}_iou{(iou_thres)}")
    return


if __name__ == "__main__":
    ray.init()
    img_path_list = [
        root + "/" + file_name
        for root, _, files in os.walk("inputdata2")
        for file_name in files
        if os.path.splitext(file_name)[-1] in [".jpg", ".jpeg", ".png"]
    ][:11]
    result = []
    for conf_thres in [0.1]:
        for iou_thres in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            result.append(batch_infer.remote(img_path_list, conf_thres=conf_thres, iou_thres=iou_thres))
    result
