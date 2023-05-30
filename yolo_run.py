import time
from pathlib import Path

import cv2
import torch
import numpy as np
import midas_run

from models.experimental import attempt_load
from utils.datasets import LoadWebcam, LoadImages
from utils.general import check_img_size, non_max_suppression, \
    scale_coords, xyxy2xywh, set_logging, increment_path
from utils.plots import plot_one_box, plot_one_line, plot_one_fill, max_size
from utils.torch_utils import select_device, time_synchronized, TracedModel


def detect(source:str="./input_photo/photo_1.jpg", weights:str='./weights/yolov7.pt', save_path='./save_data', view_img=False, save_txt=False, img_size=640, no_trace=False,
           exist_ok=True, name:str='exp', project:str='./save_data', augment=False, agnostic_nms=False, nosave=False, save_conf=False,
           device:str='cpu', iou_thres:float=0.45, conf_thres:float=0.3, *, classes=[0]):
    
    imgsz = img_size
    trace = no_trace
    detection_target = tuple(classes)
    number = 1
    distance_number = {}
    
    # save_img = not nosave and not source.endswith('.txt')  # save inference images

    # Directories
    # save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, img_size)
    
    if half:
        model.half()  # to FP16


    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    # dataset = LoadWebcam(pipe=source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        
    old_img_w = old_img_h = imgsz
    old_img_b = 1    

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t3 = time_synchronized()
        
        midas_img = cv2.imread(source)
        if midas_img.ndim == 2:
            midas_img = cv2.cvtColor(midas_img, cv2.COLOR_GRAY2BGR)

        # cv2.imshow("test_midas_", midas_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        midas_img = cv2.cvtColor(midas_img, cv2.COLOR_BGR2RGB) / 255.0
        
        distance_data= midas_run.run(input=midas_img)
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            
            # save_path = str(save_dir / p.name)  # img.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + (f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh                    # xywh is lise
                        
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        # with open(txt_path + '.txt', 'a') as f:
                        #     f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    if int(cls) in detection_target:#save_img or view_img:  # Add bbox to image
                        # label = f'{names[int(cls)]} {conf:.2f}'
                        # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                        center_x = int(int(xyxy[0])+((int(xyxy[2])-int(xyxy[0]))/2))
                        center_y = int(int(xyxy[1])+((int(xyxy[3])-int(xyxy[1]))/2))
                        distance_number[f"{number}"] = (center_x, center_y)
                        number += 1
                        # print(f"{center_x} {center_y}")
                        # cv2.line(im0, (center_x, center_y), (center_x, center_y), (0, 0, 0), thickness=10, lineType=cv2.LINE_AA)    #天の描画
                        label = f'{names[int(cls)]} {conf:.2f}'
                        # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)#line_thickness=2
                        
                        # 斜線表示
                        # plot_one_line(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=10)  #LAPTOP 67 parson0
                        
                        # 塗りつぶし
                        # plot_one_fill(xyxy, im0, color=colors[int(cls)])
                            
                        # 該当の最大サイズのみ表示
                        # if oneplot_max_size < max_size(xyxy):
                        #     plot_max = xyxy

            plot_distance_number = {}
            for i in range(len(distance_number)):
                x, y = distance_number[f"{i+1}"]
                plot_distance_number[f"{distance_data[y][x]}"] = (y, x)

            plot_distance_number = sorted(plot_distance_number.items(), reverse=True)
            
            for i in range(len(plot_distance_number)):
                cv2.putText(im0s, text=f'{i+1}', org=(plot_distance_number[i][1][1], plot_distance_number[i][1][0]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_4)
                
    cv2.imwrite(f"{save_path}/{source.split('/')[-1]}", im0s)
    cv2.imshow("test_midas_", im0s)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    

    # Print time (inference + NMS)
    print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            # if view_img:
                # img1 = img[0 : 50, 0: 50]
                
                # try:#該当の最大サイズの描画　未完成　（座標の指定が正しくない）＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
                #     print(f"-----------------{int(plot_max[0]),int(plot_max[1]),int(plot_max[2]),int(plot_max[3])}")
                #     # cv2.imshow(str(p), cv2.resize(im0s, dsize=None, fx=2, fy=2))
                #     # cv2.imshow("tttt", im0s[int(plot_max[1]):int(plot_max[1])+int(plot_max[2]),int(plot_max[3]):int(plot_max[2])])
                # except:
                #     pass
                # cv2.imshow(str(p), cv2.resize(im0s, dsize=None, fx=2, fy=2))
                # cv2.imshow(str(p), im0s)
                # cv2.waitKey(1)  # 1 millisecond





    print(f'Done. ({time.time() - t0:.3f}s)')

if __name__ == '__main__':
    detect()