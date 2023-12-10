
import torch
from midas.model_loader import load_model

def process(device, model, image, target_size):

    # 入力画像をPyTorchのテンソルに変換し、指定されたデバイスに送る
    sample = torch.from_numpy(image).to(device).unsqueeze(0)

    # モデルに画像をforwardさせて、出力を得る
    prediction = model.forward(sample)

    # 出力を補間して指定されたサイズにリサイズ
    prediction = (
        torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=target_size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        .squeeze()
        .cpu()
        .numpy()
    )

    return prediction

def run(input, model_path="./weights/dpt_beit_large_512.pt", model_type="dpt_beit_large_512", optimize=False, height=None, square=True):

    # モデルの読み込みと前処理の設定
    model, transform, net_w, net_h = load_model("cpu", model_path, model_type, optimize, height, square)

    # 入力画像の前処理
    image = transform({"image": input})["image"]

    # モデルによる奥行きマップの計算
    with torch.no_grad():
        prediction = process("cpu", model, image, input.shape[1::-1])

    return prediction
