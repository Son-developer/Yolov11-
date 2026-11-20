import cv2
import threading
import time
import winsound  # Thêm thư viện âm thanh có sẵn của Windows
from ultralytics import YOLO
from flask import Flask, render_template, Response, request, jsonify
from Email import send_email_alert
from Whatsapp import send_whatsapp_alert
from Message import send_sms_alert
from icecream import ic
import os
import shutil

# ====== CẤU HÌNH ======
# Sử dụng model OpenVINO của yolov11n đã export
MODEL_PATH = "yolo11n_openvino_model/" 
IMG_SIZE = 640          
FRAME_SKIP = 1          # Chạy AI mọi frame để đạt độ nhạy tối đa
JPEG_QUALITY = 70       
ALERT_COOLDOWN = 5      # Giây giữa 2 lần gửi cảnh báo
DEVICE = None

model = YOLO(MODEL_PATH, task="detect")

app = Flask(__name__)

fall_detected = False
fall_detected_lock = threading.Lock()
fall_detected_time = None
alert_set = False       
recipient = ""
tonumber = ""
confidence = 0.4        # Ngưỡng tin cậy nhận diện người
cap = None


def process_predictions(results, frame):
    global fall_detected, fall_detected_time, alert_set
    boxes = results[0].boxes
    model_names = model.names

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    fall_in_this_frame = False

    for box in boxes:
        class_id = int(box.cls.item())
        class_name = model_names[class_id]
        conf = box.conf.item()

        # Chỉ xét đối tượng là người (class 'person' trong model chuẩn có id = 0)
        if class_name == "person":
            # Lấy tọa độ bounding box [x1, y1, x2, y2]
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]

            # Tính chiều cao và chiều rộng của khung bao
            box_height = y2 - y1
            box_width = x2 - x1

            # Thuật toán: Khi ngã ngang, chiều rộng lớn hơn chiều cao (hoặc giá trị chênh lệch âm)
            thresh = box_height - box_width

            if thresh < 0:  # Phát hiện ngã
                fall_in_this_frame = True
                current_time = time.time()
                
                if alert_set:
                    if not fall_detected or (current_time - fall_detected_time > ALERT_COOLDOWN):
                        fall_detected = True
                        fall_detected_time = current_time

                        frame_path = os.path.join(output_dir, f"fall_frame_{fall_detected_time}.jpg")
                        if cv2.imwrite(frame_path, frame):
                            
                            # Hàm chạy âm thanh cảnh báo dồn dập (tần số 2500Hz, kêu 3 nhịp)
                            def play_alarm_sound():
                                try:
                                    for _ in range(3):
                                        winsound.Beep(2500, 400)
                                        time.sleep(0.1)
                                except Exception:
                                    pass

                            t_email = threading.Thread(target=send_email_alert, kwargs={
                                "label": "Fall Detected!",
                                "confidence_score": conf,
                                "receiver_email": recipient,
                                "frame_path": frame_path
                            }, daemon=True)
                            t_sms = threading.Thread(target=send_sms_alert, args=(tonumber,), daemon=True)
                            t_whatsapp = threading.Thread(target=send_whatsapp_alert, args=(tonumber,), daemon=True)
                            
                            # Luồng phát còi báo động song song
                            t_sound = threading.Thread(target=play_alarm_sound, daemon=True)

                            t_email.start()
                            t_sms.start()
                            t_whatsapp.start()
                            t_sound.start()

                        ic(f"Fall detected via geometry! Thresh: {thresh:.1f}")

    # Giữ trạng thái fall_detected trong 2 giây để mượt giao diện
    if not fall_in_this_frame:
        if fall_detected and fall_detected_time and (time.time() - fall_detected_time > 2.0):
            fall_detected = False

    return fall_detected


def generate_frames():
    global confidence, cap
    frame_count = 0
    last_annotated_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        annotated_frame = frame  

        if frame_count % FRAME_SKIP == 0:
            predict_kwargs = dict(source=frame, imgsz=IMG_SIZE, conf=confidence, verbose=False)
            if DEVICE:
                predict_kwargs["device"] = DEVICE
            results = model.predict(**predict_kwargs)
            
            process_predictions(results, frame)
            
            # Vẽ khung nhận diện lên ảnh
            last_annotated_frame = results[0].plot()

        if last_annotated_frame is not None:
            annotated_frame = last_annotated_frame

        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(0.01)


@app.route('/')
def index():
    return render_template('index.html', fall_detected=fall_detected, alert_set=alert_set)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/send_details', methods=['POST'])
def send_alert():
    global alert_set, recipient, tonumber, confidence
    data = request.get_json()
    recipient = data.get('email', None)
    tonumber = data.get('phone', None)
    
    raw_conf = data.get('conf', 0.4)
    if raw_conf != "" and raw_conf is not None:
        confidence = float(raw_conf)
        
    ic(data, recipient, tonumber, confidence)
    if recipient and tonumber:
        alert_set = True
        return jsonify({"message": "Email and Phone saved successfully!"})
    return jsonify({"message": "Invalid details"}), 400


@app.route('/fall_status')
def updateFallStatus():
    return jsonify({"status": fall_detected})


if __name__ == "__main__":
    cap = cv2.VideoCapture(1, cv2.CAP_MSMF)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    output_dir = "output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)