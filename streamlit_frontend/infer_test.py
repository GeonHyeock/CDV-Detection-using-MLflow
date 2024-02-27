import pandas as pd
import os
import zipfile
import io
import ray
import numpy as np

from infer import online_infer, batch_infer
import time


def Batch_Infer_online(
    img_path_list,
    img_shape=(640, 640),
    conf_thres=0.4,
    iou_thres=0.45,
    sic=False,
):
    buf = io.BytesIO()
    batch_result = [online_infer(f, conf_thres, iou_thres, img_shape, sic, only_det=True) for f in img_path_list[:3]]
    with zipfile.ZipFile(buf, "x") as csv_zip:
        for csv, csv_name in batch_result:
            csv_zip.writestr(csv_name, pd.DataFrame(csv).to_csv(index=False))
        csv_zip.extractall(f"inputdata2_result/conf{(conf_thres)}_iou{(iou_thres)}")


def Batch_Infer_ray(
    img_path_list,
    batch_size=4,
    img_shape=(640, 640),
    conf_thres=0.4,
    iou_thres=0.45,
):

    n = len(img_path_list)
    batch_result = [
        batch_infer.remote(
            img_path_list[idx : idx + batch_size],
            conf_thres,
            iou_thres,
            img_shape,
            os.path.join("inputdata2"),
        )
        for idx in range(0, n, batch_size)
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "x") as csv_zip:
        while len(batch_result):
            done, batch_result = ray.wait(batch_result)
            mini_batch_result = ray.get(done[0])
            for csv, csv_name in zip(*mini_batch_result):
                csv_zip.writestr(csv_name, pd.DataFrame(csv).to_csv(index=False))
        csv_zip.extractall(f"inputdata2_result/conf{(conf_thres)}_iou{(iou_thres)}")


def time_test(img_path_list, test_type):
    if test_type == "ray":
        ray.init()
        Batch_Infer = Batch_Infer_ray
    elif test_type == "online":
        Batch_Infer = Batch_Infer_online

    result = []
    for conf_thres in [0.1]:
        for iou_thres in [0.1]:
            start = time.time()
            Batch_Infer(img_path_list, conf_thres=conf_thres, iou_thres=iou_thres)
            end = time.time()
            result.append(end - start)
    f = open(f"{test_type}_time.txt", "w")
    print(np.mean(result))
    f.write(f"{np.mean(result):.4f}")
    f.close()


if __name__ == "__main__":
    img_path_list = [
        root + "/" + file_name
        for root, _, files in os.walk("inputdata2")
        for file_name in files
        if os.path.splitext(file_name)[-1] in [".jpg", ".jpeg", ".png"]
    ][:2]
    time_test(img_path_list, "online")
    time_test(img_path_list, "ray")
