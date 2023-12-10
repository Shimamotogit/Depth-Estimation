
import cv2
import torch
import midas_run

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords

def detect(source: str = "./input_photo/photo_1.jpg", save_path='./save_data', img_size=640,
           iou_thres: float = 0.45, conf_thres: float = 0.3, *, classes=[0]):

    imgsz = img_size
    detection_target = tuple(classes)
    number = 1
    distance_number = {}
    plot_distance_number = {}

    # YOLOv7モデルの読み込み
    model = attempt_load("./weights/yolov7.pt", map_location="cpu")
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)

    # 画像データセットの構築
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # 各画像に対する処理
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to("cpu")
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # YOLOv7による物体検出
        with torch.no_grad():
            pred = model(img, augment=False)[0]

        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=False)

        # Midasモデルによる奥行き情報の取得
        midas_img = cv2.imread(source)
        if midas_img.ndim == 2:
            midas_img = cv2.cvtColor(midas_img, cv2.COLOR_GRAY2BGR)
        midas_img = cv2.cvtColor(midas_img, cv2.COLOR_BGR2RGB) / 255.0
        distance_data = midas_run.run(input=midas_img)

        # 物体検出結果の処理
        for i, det in enumerate(pred):
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # 物体までの距離の計算と保存
                for *xyxy, conf, cls in reversed(det):
                    if int(cls) in detection_target:
                        center_x = int(xyxy[0] + ((xyxy[2] - xyxy[0]) / 2))
                        center_y = int(xyxy[1] + ((xyxy[3] - xyxy[1]) / 2))
                        distance_number[f"{number}"] = (center_x, center_y)
                        number += 1

                # 描画結果の整理と保存
                for i in range(len(distance_number)):
                    x, y = distance_number[f"{i+1}"]
                    plot_distance_number[f"{distance_data[y][x]}"] = (y, x)

                plot_distance_number = sorted(plot_distance_number.items(), reverse=True)

                for i in range(len(plot_distance_number)):
                    cv2.putText(im0s, text=f'{i+1}', org=(plot_distance_number[i][1][1], plot_distance_number[i][1][0]),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_4)

        # 保存
        cv2.imwrite(f"{save_path}/{source.split('/')[-1]}", im0s)

if __name__ == '__main__':
    detect()
