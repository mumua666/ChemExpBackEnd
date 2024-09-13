import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template
import base64
import os
from datetime import datetime

# 解决 Exception: cannot instantiate 'WindowsPath' on your system.
import platform
import pathlib

plt = platform.system()
if plt != "Windows":
    pathlib.WindowsPath = pathlib.PosixPath

# 解决 NotImplementedError: cannot instantiate 'PosixPath' on your system
import platform
import pathlib

plt = platform.system()
if plt == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath


app = Flask(__name__)

# 全局变量：模型
model = None

# 提前加载模型


def load_model():
    global model
    model = torch.hub.load(
        "yolov5-master", "custom", path="best.pt", source="local", force_reload=True
    )


# 路由处理图片检测请求
@app.route("/predict_image", methods=["POST"])
def predict_image():
    global model

    # 获取图像文件
    file = request.files["image"]
    # 获取文件扩展名
    filename, extension = os.path.splitext(file.filename)
    extension = extension.lower()

    # 读取图像数据并转换为RGB格式
    image_data = file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    # 根据文件格式执行专一化的图像预处理
    if extension == ".png":
        # PNG文件的专一化处理，例如处理透明通道
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif extension == ".jpg" or extension == ".jpeg":
        # JPG文件的专一化处理，例如进行颜色空间转换
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        return jsonify({"error": "Unsupported file format"}), 400

    # 使用YOLO模型进行推理
    results = model(image)

    # 提取检测的标签
    labels = []
    for *box, conf, cls in results.xyxy[0].tolist():
        labels.append(results.names[int(cls)])  # 获取对应的标签名称

    # 获取处理后的图像
    processed_image = results.render()[0]

    # 将模型输出的BGR图像转换为RGB或RGBA
    if extension == ".png":
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGBA)
    else:
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

    # 将图像转换为 base64 编码的字符串
    _, buffer = cv2.imencode(
        extension if extension != ".jpeg" else ".jpg", processed_image
    )
    image_base64 = base64.b64encode(buffer).decode("utf-8")

    # 定义图像扩展名和 MIME 类型的映射
    mime_type_mapping = {".png": "image/png", ".jpg": "image/jpg", ".jpeg": "image/jpg"}

    # 根据文件扩展名获取对应的 MIME 类型前缀
    mime_type_prefix = mime_type_mapping.get(extension, "image/jpg")

    # 拼接带有前缀的 base64 编码图像数据
    image_str = f"data:{mime_type_prefix};base64,{image_base64}"

    # 获取当前时间，并将其格式化为字符串
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")

    # 构建保存路径
    save_dir = "static"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_filename = f"{filename}_{current_time}{extension}"
    save_path = os.path.join(save_dir, save_filename)
    cv2.imwrite(save_path, processed_image)

    # 返回带标签和图像数据的响应
    return jsonify({"image": image_str, "label": labels})


# 函数用于在视频帧上绘制检测结果
def detect_objects(frame, model):
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # 获取检测结果

    # 在帧上绘制检测结果
    for det in detections:
        # 获取边界框信息
        x1, y1, x2, y2, conf, class_id = det[:6]

        # 在帧上绘制边界框
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # 在帧上绘制类别和置信度
        label = f"{model.names[int(class_id)]} {conf:.2f}"
        cv2.putText(
            frame,
            label,
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    return frame


# 路由处理视频检测请求
@app.route("/predict_video", methods=["POST"])
def predict_video():
    global model

    # 从请求中获取视频文件
    video_file = request.files["video"]
    # 保存视频到临时文件
    temp_video_path = "temp_video.mp4"
    video_file.save(temp_video_path)

    # 逐帧读取视频
    video = cv2.VideoCapture(temp_video_path)

    # 获取视频的帧率和尺寸
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 视频写入对象
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    output_video_filename = "output_video_" + date_str + "mp4"
    output_video_path = os.path.join("static", output_video_filename)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 逐帧处理视频并进行目标检测
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # 进行目标检测
        detection_result = detect_objects(frame, model)

        # 将处理后的帧写入输出视频
        out_video.write(detection_result)

    # 释放视频对象
    video.release()
    out_video.release()

    return jsonify({"output_video_path": output_video_filename})


@app.route("/")
def index():
    return render_template("index.html")


# 初始加载模型
load_model()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3730)
