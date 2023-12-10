import yolo_run

def command_line_argument():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--img_path", default="./input_photo/photo_1.jpg", type=str, 
                        help="入力画像のパス設定")

    parser.add_argument("-s", "--save_path", default="./save_data", type=str, 
                        help="画像を保存するフォルダのパス設定")

    parser.add_argument("-it", "--iou_thres", default=0.45, type=float, 
                        help="物体検出の際の Intersection over Union (IoU) 閾値設定")

    parser.add_argument("-ct", "--conf_thres", default=0.3, type=float, 
                        help="物体検出の信頼度の閾値設定")

    parser.add_argument("--classes", default=[str(i) for i in range(80)], nargs='*',
                        help="物体クラスの設定")

    args = parser.parse_args()
    return args

args = command_line_argument()

yolo_run.detect(source=args.img_path, save_path=args.save_path,
                iou_thres=args.iou_thres, conf_thres=args.conf_thres, classes=[int(i) for i in args.classes])

# PERSON = 0
# BICYCLE = 1
# CAR = 2
# MOTORBIKE = 3
# AEROPLANE = 4
# BUS = 5
# TRAIN = 6
# TRUCK = 7
# BOAT = 8
# TRAFFIC_LIGHT = 9
# FIRE_HYDRANT = 10
# STOP_SIGN = 11
# PARKING_METER = 12
# BENCH = 13
# BIRD = 14
# CAT = 15
# DOG = 16
# HORSE = 17
# SHEEP = 18
# COW = 19
# ELEPHANT = 20
# BEAR = 21
# ZEBRA = 22
# GIRAFFE = 23
# BACKPACK = 24
# UMBRELLA = 25
# HANDBAG = 26
# TIE = 27
# SUITCASE = 28
# FRISBEE = 29
# SKIS = 30
# SNOWBOARD = 31
# SPORTS_BALL = 32
# KITE = 33
# BASEBALL_BAT = 34
# BASEBALL_GLOVE = 35
# SKATEBOARD = 36
# SURFBOARD = 37
# TENNIS_RACKET = 38
# BOTTLE = 39
# WINE_GLASS = 40
# CUP = 41
# FORK = 42
# KNIFE = 43
# SPOON = 44
# BOWL = 45
# BANANA = 46
# APPLE = 47
# SANDWICH = 48
# ORANGE = 49
# BROCCOLI = 50
# CARROT = 51
# HOT_DOG = 52
# PIZZA = 53
# DONUT = 54
# CAKE = 55
# CHAIR = 56
# SOFA = 57
# POTTEDPLANT = 58
# BED = 59
# DININGTABLE = 60
# TOILET = 61
# TVMONITOR = 62
# LAPTOP = 63
# MOUSE = 64
# REMOTE = 65
# KEYBOARD = 66
# CELL_PHONE = 67
# MICROWAVE = 68
# OVEN = 69
# TOASTER = 70
# SINK = 71
# REFRIGERATOR = 72
# BOOK = 73
# CLOCK = 74
# VASE = 75
# SCISSORS = 76
# TEDDY_BEAR = 77
# HAIR_DRIER = 78
# TOOTHBRUSH = 79
