# python -m venv .venv

# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# .venv\Scripts\activate


# .venv\Scripts\activate
# venv\Scripts\activate.bat

# streamlit run app.py

import streamlit as st
import ffmpeg
import subprocess
import tempfile
import os
import sys
import warnings
import cv2
import torch
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
# from yolov7.models.experimental import attempt_load 
# Constants
YOLOV7_PATH = 'D:/WEB BTL/WEB_BTL_AI/yolov7'
WEIGHTS_PATH = 'D:/WEB BTL/WEB_BTL_AI/yolov7/best.pt'
WEIGHTS_PATH_TINY = 'D:/WEB BTL/WEB_BTL_AI/yolov7/best-tiny.pt'
WEIGHTS_PATH_V11 = 'D:/WEB BTL/WEB_BTL_AI/best-yolov11n.pt'
sys.path.append(YOLOV7_PATH)

# Create output directory
output_dir = 'D:/WEB BTL/WEB_BTL_AI/output'
os.makedirs(output_dir, exist_ok=True)

# Function to run the detect.py script
# def run_detect(source, conf_threshold, weightspath="", output_dir=output_dir):
#     command = [
#         "python", "yolov7/detect.py",
#         "--weights", weightspath,
#         "--conf", str(conf_threshold),
#         "--source", source,
#         "--device", "cpu",
#         "--save-txt",
#         "--project", output_dir,
#         "--name", "exp"  # Tên cố định cho thư mục exp
#     ]
#     # Chạy subprocess và chờ kết quả
#     result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     if result.returncode != 0:
#         print("Error running YOLOv7 detect.py:")
#         print(result.stderr.decode())
#     return result

# def run_detect(source, conf_threshold, weightspath="", output_dir=output_dir):
#     command = [
#         "python", "yolov7/detect.py",
#         "--weights", weightspath,
#         "--conf", str(conf_threshold),
#         "--source", source,
#         "--device", "cpu",
#         "--save-txt",
#         "--project", output_dir,
#         "--name", "exp"  # Tên cố định cho thư mục exp
#     ]
#     # Chạy subprocess và chờ kết quả
#     result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     if result.returncode != 0:
#         print("Error running YOLOv7 detect.py:")
#         print(result.stderr.decode())
#     return result

import subprocess

def run_detect(source, conf_threshold, weightspath="", output_dir="output", model_choice="yolov7"):
    if model_choice.lower() in ["yolov7", "yolov7-tiny"]:
        # Lệnh cho YOLOv7 hoặc YOLOv7-TINY
        command = [
            "python", "yolov7/detect.py",
            "--weights", weightspath,
            "--conf", str(conf_threshold),
            "--source", source,
            "--device", "cpu",  # Sử dụng GPU nếu có, đổi thành "0" hoặc thiết bị cụ thể
            "--save-txt",
            "--project", output_dir,
            "--name", "exp"  # Tên cố định cho thư mục exp
        ]
    elif model_choice.lower() == "yolov11":
        # Lệnh cho YOLOv11
        command = [
            "yolo",
            "task=detect",
            "mode=predict",
            f"model={weightspath}",
            f"conf={conf_threshold}",
            f"source={source}",
            "save=True",
            f"project={output_dir}",
            "--device=cpu",  # Sử dụng GPU nếu có, đổi thành "0" hoặc thiết bị cụ thể
            
        ]
    else:
        raise ValueError("Unsupported model type. Choose 'yolov7', 'yolov7-tiny', or 'yolov11'.")

    # Chạy subprocess và chờ kết quả
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Error running {model_choice} detect:")
        print(result.stderr.decode())
        return None
    else:
        print(result.stdout.decode())
        return result



def run_detect_realtime(frame, conf_threshold, camera_no, img_size=640, device='cpu', weightspath="", output_dir='D:/WEB BTL/WEB_BTL_AI/output'):
    # Tạo tệp tạm thời lưu frame đầu vào
    temp_file_path = os.path.join(output_dir, f'temp_frame_camera_{camera_no}.jpg')
    cv2.imwrite(temp_file_path, frame)

    # Đường dẫn ảnh đầu ra
    detected_frame_dir = os.path.join(output_dir, 'exp', 'last')
    detected_frame_path = os.path.join(detected_frame_dir, 'image0.jpg')

    # Xóa file đầu ra cũ (nếu tồn tại)
    if os.path.exists(detected_frame_path):
        os.remove(detected_frame_path)

    # Command để chạy YOLOv7 detect.py
    command = [
        "python", os.path.join(YOLOV7_PATH, "detect.py"),
        "--weights", weightspath,
        "--conf", str(conf_threshold),
        "--img-size", str(img_size),
        "--source", temp_file_path,
        "--save-txt",
        "--project", output_dir,
        "--device", device
    ]

    # Chạy subprocess YOLOv7 detect.py
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Kiểm tra kết quả và trả về đường dẫn ảnh nếu tồn tại
    if os.path.exists(detected_frame_path):
        return detected_frame_path
    return None



# Function to find the latest "exp" folder
def get_latest_exp_folder(output_dir, model_choice="yolov7"):
    exp_prefix = ""
    if (model_choice.lower() == "yolov7") or (model_choice.lower() == "yolov7-tiny"):
        exp_prefix = "exp"
    elif model_choice.lower() == "yolov11":
        exp_prefix = "predict"
    exp_folders = [
        folder for folder in os.listdir(output_dir)
        if folder.startswith(exp_prefix) and os.path.isdir(os.path.join(output_dir, folder))
    ]
    if not exp_folders:
        return None
    exp_folders.sort(key=lambda x: int(x[len(exp_prefix):]) if x[len(exp_prefix):].isdigit() else 0, reverse=True)
    return os.path.join(output_dir, exp_folders[0])

# List available cameras
def list_available_cameras():
    available_cameras = []
    # Try to open camera from index 0 to 3 (or any number of indexes you think might have a camera)
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
        cap.release()
    return available_cameras

def find_exp_max(model_choice="yolov7"):
    exp_max = 0
    exp_prefix = ""
    if (model_choice.lower() == "yolov7") or (model_choice.lower() == "yolov7-tiny"):
        exp_prefix = "exp"
    elif model_choice.lower() == "yolov11":
        exp_prefix = "yolov11"
    exp_folders = [folder for folder in os.listdir(output_dir) if folder.startswith(exp_prefix) and os.path.isdir(os.path.join(output_dir, folder))]
    for folder in exp_folders:
        try:
            exp_number = int(folder[3:])  # Lấy số sau "exp"
            exp_max = max(exp_max, exp_number)  # Cập nhật giá trị lớn nhất
        except ValueError:
            continue  # Nếu không thể chuyển đổi thành số, bỏ qua
    return exp_max

def check_cuda_availability():
    return torch.cuda.is_available()


# Main function
def main():
    st.title("Object Detection using YOLO")
    weightspath = ''
    # Sidebar
    st.sidebar.title("Settings")
    # Combobox để chọn mô hình
    model_choice = st.sidebar.selectbox(
        "Select Model",  # Tiêu đề
        ["YOLOv7", "YOLOv7-TINY", "YOLOv11"],  # Danh sách các lựa chọn
        index=0  # Lựa chọn mặc định
    )

    # Slider điều chỉnh ngưỡng tự tin
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        0.0, 1.0, 0.5, 0.05
        # Min, Max, Default, Step
    )
    
    if model_choice == "YOLOv7":
        weightspath = WEIGHTS_PATH
    elif model_choice == "YOLOv7-TINY":
        weightspath = WEIGHTS_PATH_TINY
    elif model_choice == "YOLOv11":
        weightspath = WEIGHTS_PATH_V11

    # Hiển thị mô hình được chọn trong giao diện (nếu cần kiểm tra)
    st.sidebar.text(f"Selected Model: {model_choice}")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Image Detection", "Video Detection", "Real-time Detection"])

    # Image Detection
    with tab1:
        st.header("Image Detection")
        st.text("Weight path: " + weightspath)

        uploaded_image = st.file_uploader("Upload an image...", type=['jpg', 'jpeg', 'png'])
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            # Save uploaded image
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(uploaded_image.read())
                temp_file_path = temp_file.name

            # Run YOLOv7 detection
            run_detect(temp_file_path, conf_threshold, weightspath=weightspath, model_choice=model_choice)

            # Find latest exp folder
            latest_exp_folder = get_latest_exp_folder(output_dir, model_choice=model_choice)
            st.text(f"Latest exp folder: {latest_exp_folder}")
            if latest_exp_folder:
                detected_images = [
                    f for f in os.listdir(latest_exp_folder) if f.endswith('.jpg')
                ]
                if detected_images:
                    detected_image_path = os.path.join(latest_exp_folder, detected_images[0])
                    st.image(detected_image_path, caption="Processed Image with Detections", use_column_width=True)
                else:
                    st.error("No detected images found in the output folder.")
            else:
                st.error("No output folder.")
            
            # Đường dẫn đến thư mục labels
            exp_max = find_exp_max(model_choice = model_choice)
            exp_folder_path = os.path.join(output_dir, f"exp{exp_max}")
            
            labels_dir = os.path.join(exp_folder_path, "labels")

            # Đếm số lần xuất hiện của mỗi class trong các tệp labels
            class_counts = {}

            # Kiểm tra nếu thư mục labels tồn tại
            if os.path.exists(labels_dir):
                label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
                for label_file in label_files:
                    label_file_path = os.path.join(labels_dir, label_file)
                    with open(label_file_path, 'r') as file:
                        # Đọc từng dòng trong file và lấy class (cột thứ 1 trong mỗi dòng)
                        for line in file:
                            parts = line.strip().split()
                            if parts:  # Kiểm tra nếu dòng không trống
                                class_id = int(parts[0])  # Lớp đối tượng là số ở vị trí đầu tiên
                                class_counts[class_id] = class_counts.get(class_id, 0) + 1  # Đếm số lần xuất hiện của class_id

                # Hiển thị số lần xuất hiện của mỗi class
                if class_counts:
                    st.write("Số lần xuất hiện của mỗi class trong các tệp labels:")
                    for class_id, count in class_counts.items():
                        st.write(f"Class {class_id}: {count} lần")
                else:
                    st.write("Không tìm thấy đối tượng trong các tệp labels.")
            else:
                st.error(f"Thư mục {labels_dir} không tồn tại.")

    # Video Detection
    # Video Detection
    with tab2:
        st.header("Video Detection")
        st.text("Weight path: " + weightspath)
        
        uploaded_video = st.file_uploader("Upload a video...", type=['mp4', 'avi', 'mov'])
        
        if uploaded_video is not None:
            # Save the uploaded video to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(uploaded_video.read())
                temp_file_path = temp_file.name

            # Show the uploaded video
            st.video(temp_file_path)
            st.markdown("**Uploaded Video**")
            print(f"Original video path: {temp_file_path}")  # Debug: Check the path

            # Run YOLOv7 detection
            print('Detecting objects in the uploaded video...')
            run_detect(temp_file_path, conf_threshold, weightspath, model_choice = model_choice)

            # Find the latest exp folder
            latest_exp_folder = get_latest_exp_folder(output_dir, model_choice=model_choice)

            if latest_exp_folder:
                # Get the list of video files in the detected folder
                video_files = [f for f in os.listdir(latest_exp_folder) if f.endswith('.mp4')]

                # Check and display the first video
                if video_files:
                    detected_video_path = os.path.join(latest_exp_folder, video_files[0])
                    print(f"Detected video path: {detected_video_path}")  # Debug: Check the path
                    
                    if os.path.exists(detected_video_path):
                        st.video(detected_video_path)
                        st.markdown("**Processed Video with Detections**")
                    else:
                        st.error(f"Detected video does not exist: {detected_video_path}")
                else:
                    st.error("No detected video files (.mp4) found in the output folder.")
            else:
                st.error("No output folder")

            # # Analyze labels
            # if latest_exp_folder:
            #     labels_dir = os.path.join(latest_exp_folder, "labels")
            #     class_counts = {}

            #     if os.path.exists(labels_dir):
            #         label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
            #         for label_file in label_files:
            #             label_file_path = os.path.join(labels_dir, label_file)
            #             with open(label_file_path, 'r') as file:
            #                 for line in file:
            #                     parts = line.strip().split()
            #                     if parts:  # Ensure the line is not empty
            #                         class_id = int(parts[0])  # Object class is the first number
            #                         class_counts[class_id] = class_counts.get(class_id, 0) + 1

            #         # Display class counts
            #         if class_counts:
            #             st.write("Class occurrence counts in label files:")
            #             for class_id, count in class_counts.items():
            #                 st.write(f"Class {class_id}: {count} times")
            #         else:
            #             st.write("No objects found in the label files.")
            #     else:
            #         st.error(f"Labels directory does not exist: {labels_dir}")
            # else:
            #     st.error("No output folder found to analyze labels.")


    # Real-time Detection
    with tab3:
        st.header("Real-time Detection")

        st.text("Weight path: " + weightspath)

        # Liệt kê các camera có sẵn
        available_cameras = list_available_cameras()

        # Cho người dùng chọn camera
        selected_camera = st.selectbox("Select Camera", available_cameras)

        # Checkbox để bắt đầu camera
        start_camera = st.checkbox("Start Camera")

        # Khởi tạo biến cap
        cap = None

        # Kiểm tra GPU
        device = 'cuda' if check_cuda_availability() else 'cpu'
        if device == 'cuda':
            st.text("GPU Available. Running detection on GPU.")
        else:
            st.text("GPU not available. Running detection on CPU.")

        # Hiển thị video real-time từ camera
        if start_camera:
            cap = cv2.VideoCapture(selected_camera)  # Mở camera đã chọn
            if not cap.isOpened():
                st.error("Không thể mở camera.")
            else:
                frame_placeholder = st.empty()
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Không thể đọc khung hình từ camera.")
                        break

                    # Chuyển đổi frame sang RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Gọi YOLOv7 để phát hiện đối tượng
                    detected_frame_path = run_detect_realtime(
                        frame=rgb_frame,
                        conf_threshold=0.5,
                        camera_no=selected_camera,
                        device='cpu',
                        weightspath=weightspath,
                        output_dir=output_dir,
                    )

                    # Hiển thị kết quả phát hiện nếu tồn tại
                    if detected_frame_path and os.path.exists(detected_frame_path):
                        detected_frame = cv2.imread(detected_frame_path)
                        rgb_detected_frame = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(
                            rgb_detected_frame,
                            caption="Processed Frame with Detections",
                            use_column_width=True,
                        )
                    else:
                        # Hiển thị khung hình gốc nếu không có phát hiện
                        frame_placeholder.image(
                            rgb_frame,
                            caption="Real-time Camera Feed",
                            use_column_width=True,
                        )

                    if not start_camera:
                        st.warning("Camera detection stopped.")
                        break

                # Giải phóng tài nguyên camera
                cap.release()

        else:
            st.warning("Vui lòng chọn 'Start Camera' để bắt đầu sử dụng camera.")

    if __name__ == '__main__':
        st.text("Real-time detection application started.")

if __name__ == '__main__':
    main()
