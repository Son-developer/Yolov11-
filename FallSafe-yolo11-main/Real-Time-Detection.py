import cv2
import threading
import time
import winsound  # Âm thanh có sẵn của Windows
from ultralytics import YOLO
from flask import Flask, render_template, Response, request, jsonify
from Email import send_email_alert
from Whatsapp import send_whatsapp_alert
from Message import send_sms_alert
from icecream import ic
import os
import shutil

# ====== CẤU HÌNH ======
MODEL_PATH = "yolo11n_openvino_model/"
IMG_SIZE = 640
FRAME_SKIP = 1          # chạy AI mỗi N frame (tăng lên 2 nếu video bị giật)
JPEG_QUALITY = 70
ALERT_COOLDOWN = 5      # giây giữa 2 lần gửi cảnh báo
DEVICE = None

# Nguồn video: đường dẫn file, hoặc số 0 để dùng camera
VIDEO_SOURCE = r"C:\Users\laptop\Downloads\Human-Fall-Detection-Sample.mp4"
LOOP_VIDEO = True       # hết video thì tự phát lại từ đầu
FRAME_WIDTH = 640       # thu nhỏ video về chiều rộng này (giữ nguyên tỉ lệ khung hình)

model = YOLO(MODEL_PATH, task="detect")

app = Flask(__name__)

fall_detected = False
fall_detected_lock = threading.Lock()
fall_detected_time = None
alert_set = False
recipient = ""
tonumber = ""
confidence = 0.4
cap = None
paused = False          # True = video đứng hình và AI ngừng chạy


def resize_keep_ratio(frame, width):
    """Thu nhỏ theo chiều rộng, GIỮ NGUYÊN tỉ lệ.
    Quan trọng: luật phát hiện ngã so sánh chiều rộng và chiều cao khung,
    nếu ép về 640x480 cố định thì video 16:9 bị méo và dễ báo sai."""
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    return cv2.resize(frame, (width, int(h * width / w)))


def play_alarm_sound():
    try:
        for _ in range(3):
            winsound.Beep(2500, 400)
            time.sleep(0.1)
    except Exception:
        pass


def process_predictions(results, frame):
    global fall_detected, fall_detected_time
    boxes = results[0].boxes
    model_names = model.names

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    fall_in_this_frame = False

    for box in boxes:
        class_name = model_names[int(box.cls.item())]
        conf = box.conf.item()

        # Chỉ xét đối tượng là người
        if class_name != "person":
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        thresh = (y2 - y1) - (x2 - x1)   # chiều cao - chiều rộng

        # Nằm ngang (rộng hơn cao) => nghi ngã
        if thresh < 0:
            fall_in_this_frame = True
            current_time = time.time()

            if not fall_detected or (current_time - fall_detected_time > ALERT_COOLDOWN):
                fall_detected = True
                fall_detected_time = current_time
                ic(f"Fall detected via geometry! Thresh: {thresh:.1f}")

                # Chỉ gửi cảnh báo khi người dùng đã nhập thông tin trên web
                if alert_set:
                    frame_path = os.path.join(output_dir, f"fall_frame_{fall_detected_time}.jpg")
                    if cv2.imwrite(frame_path, frame):
                        threading.Thread(target=send_email_alert, kwargs={
                            "label": "Fall Detected!",
                            "confidence_score": conf,
                            "receiver_email": recipient,
                            "frame_path": frame_path
                        }, daemon=True).start()
                        threading.Thread(target=send_sms_alert, args=(tonumber,), daemon=True).start()
                        threading.Thread(target=send_whatsapp_alert, args=(tonumber,), daemon=True).start()
                        threading.Thread(target=play_alarm_sound, daemon=True).start()

    # Giữ trạng thái "ngã" thêm 2 giây cho giao diện đỡ nhấp nháy
    if not fall_in_this_frame:
        if fall_detected and fall_detected_time and (time.time() - fall_detected_time > 2.0):
            fall_detected = False

    return fall_detected


def generate_frames():
    global confidence, cap
    fps = cap.get(cv2.CAP_PROP_FPS) or 30      # tốc độ gốc của video
    frame_interval = 1.0 / fps
    frame_count = 0
    last_annotated_frame = None
    last_jpeg = None        # ảnh cuối cùng đã gửi, dùng để giữ hình khi tạm dừng

    while True:
        loop_start = time.time()

        # Đang tạm dừng: không đọc video, không chạy AI, không gửi cảnh báo,
        # chỉ gửi lại khung hình cuối để video đứng yên
        if paused:
            if last_jpeg is not None:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + last_jpeg + b'\r\n')
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            # Hết video: quay lại đầu để phát lại (nếu bật LOOP_VIDEO)
            if LOOP_VIDEO:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if not ret:
                break

        frame = resize_keep_ratio(frame, FRAME_WIDTH)

        frame_count += 1
        annotated_frame = frame

        if frame_count % FRAME_SKIP == 0:
            predict_kwargs = dict(source=frame, imgsz=IMG_SIZE, conf=confidence,
                                  classes=[0], verbose=False)   # classes=[0]: chỉ tìm "person"
            if DEVICE:
                predict_kwargs["device"] = DEVICE
            results = model.predict(**predict_kwargs)

            process_predictions(results, frame)
            last_annotated_frame = results[0].plot()

        if last_annotated_frame is not None:
            annotated_frame = last_annotated_frame

        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        last_jpeg = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + last_jpeg + b'\r\n')

        # Giữ video chạy đúng tốc độ gốc
        elapsed = time.time() - loop_start
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
        else:
            # Xử lý chậm hơn video: bỏ bớt frame cho khỏi bị tua chậm
            for _ in range(int(elapsed / frame_interval) - 1):
                cap.grab()


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


@app.route('/toggle_pause', methods=['POST'])
def toggle_pause():
    global paused
    paused = not paused
    ic(f"Paused: {paused}")
    return jsonify({"paused": paused})


@app.route('/fall_status')
def updateFallStatus():
    return jsonify({"status": fall_detected})


if __name__ == "__main__":
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print("Cannot open video!")
        exit()

    output_dir = "output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)