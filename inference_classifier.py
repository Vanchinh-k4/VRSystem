import pickle
import os
import time
import serial
import threading
import signal
import sys
from flask import Flask, Response
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# ======================================================
# UART TO ESP32
# ======================================================
try:
    ser = serial.Serial('/dev/serial0', 115200, timeout=1)
    time.sleep(2)
    print("✅ Kết nối UART /dev/serial0 thành công.")
except Exception as e:
    print(f"⚠️ Không thể kết nối UART: {e}. Chạy chế độ không có robot.")
    ser = None

# ======================================================
# LOAD MODEL AI (SỬ DỤNG CHO 2, 3, 4, 0 NGÓN TAY)
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.p')
try:
    model_dict = pickle.load(open(MODEL_PATH, 'rb'))
    model = model_dict['model']
    print("✅ Đã tải file Model AI thành công.")
except Exception as e:
    print(f"❌ LỖI KHÔNG TẢI ĐƯỢC MODEL: {e}")
    sys.exit(1)

# ======================================================
# MEDIAPIPE CONFIGURATION
# ======================================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,         
    model_complexity=0,     
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

labels_dict = {0: 'FORWARD', 1: 'BACKWARD', 2: 'LEFT', 3: 'RIGHT', 4: 'STOP'}
THRESHOLD = 0.75

# ======================================================
# GLOBAL VARIABLES & LOCKS
# ======================================================
frame_lock = threading.Lock()
frame_global = None      # Khung hình thô đọc từ Camera
frame_rendered = None    # Khung hình đã được AI vẽ Landmark/Box lên để Stream lên Web
command = "STOP"
confidence = 0.0
cap = None               

# ======================================================
# HÀM ĐẾM SỐ NGÓN TAY VÀ LOGIC ĐIỀU KHIỂN CHẶT CHẼ
# ======================================================
def count_fingers(hand_landmarks):
    lm = hand_landmarks.landmark
    count = 0

    # Ngón cái (Thumb) - So sánh trục X
    if lm[4].x < lm[3].x:
        count += 1

    # 4 ngón còn lại - So sánh trục Y
    finger_tips = [8, 12, 16, 20]
    for tip in finger_tips:
        if lm[tip].y < lm[tip - 2].y:
            count += 1
    return count

def determine_command(hand_landmarks):
    lm = hand_landmarks.landmark
    fingers = count_fingers(hand_landmarks)

    # 1. Xòe bàn tay (>= 4 ngón) -> STOP khẩn cấp
    if fingers >= 4:
        return "STOP", 1.0

    # 2. Nắm đấm hoàn toàn (0 ngón mở) -> BACKWARD (Đi lùi)
    if fingers == 0:
        return "BACKWARD", 1.0

    # 3. CHỈ DÙNG ĐÚNG 1 NGÓN MỞ DUY NHẤT để điều khiển hướng
    if fingers == 1:
        # Kiểm tra chắc chắn ngón đang mở phải là ngón trỏ
        index_open = lm[8].y < lm[6].y
        if index_open:
            dx = lm[8].x - lm[5].x
            dy = lm[8].y - lm[5].y
            
            if abs(dx) > abs(dy):
                if dx > 0.04: return "LEFT", 1.0
                if dx < -0.04: return "RIGHT", 1.0
            else:
                if dy < -0.04: return "FORWARD", 1.0

    # 4. Nếu giơ 2 ngón, 3 ngón -> Sử dụng Model Machine Learning để dự đoán
    try:
        data_aux = []
        x_ = [landmark.x for landmark in hand_landmarks.landmark]
        y_ = [landmark.y for landmark in hand_landmarks.landmark]

        for landmark in hand_landmarks.landmark:
            data_aux.append(landmark.x - min(x_))
            data_aux.append(landmark.y - min(y_))

        probs = model.predict_proba([np.asarray(data_aux)])
        local_confidence = float(np.max(probs))
        pred_index = np.argmax(probs)

        if local_confidence >= THRESHOLD:
            pred_cmd = labels_dict[pred_index]
            if pred_cmd == "FORWARD" and fingers != 1:
                return "STOP", local_confidence
            return pred_cmd, local_confidence
    except:
        pass

    return "STOP", 0.0

# ======================================================
# THREAD 1: ĐỌC CAMERA LIÊN TỤC (Tốc độ tối đa)
# ======================================================
def video_capture_thread():
    global frame_global, cap
    
    available_ports = [1, 2, 0]
    camera_opened = False

    print("🔍 Hệ thống Pi: Đang tự động quét cổng Camera...")
    for port in available_ports:
        print(f"-> Thử kết nối vào /dev/video{port}...")
        cap = cv2.VideoCapture(port) 
        time.sleep(0.5)
        
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret:
                print(f"🎉 XÁC NHẬN: Camera hoạt động tốt tại cổng /dev/video{port}!")
                camera_opened = True
                break
            else:
                print(f"⚠️ Cổng /dev/video{port} mở được nhưng không xuất được hình.")
                cap.release()
        else:
            cap.release()

    if not camera_opened:
        print("❌ THẤT BẠI: Không tìm thấy camera nào khả dụng. Hãy kiểm tra lại cáp nối!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Xóa bỏ hoàn toàn độ trễ buffer hình cũ

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cảnh báo: Bị mất kết nối frame hình tạm thời.")
            time.sleep(0.1)
            continue
        
        frame = cv2.flip(frame, 1)
        
        with frame_lock:
            frame_global = frame.copy()
            
        time.sleep(0.01)

# ======================================================
# THREAD 2: XỬ LÝ AI & VẼ LANDMARK/BOX
# ======================================================
def ai_processing_thread():
    global command, confidence, frame_rendered
    last_command = "STOP"
    last_send_time = 0
    SEND_DELAY = 0.2  

    while True:
        img_processing = None
        with frame_lock:
            if frame_global is not None:
                img_processing = frame_global.copy()

        if img_processing is None:
            time.sleep(0.01)
            continue

        H, W, _ = img_processing.shape
        frame_rgb = cv2.cvtColor(img_processing, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        local_command = "STOP"
        local_confidence = 0.0
        color = (0, 0, 255) # Mặc định màu Đỏ cho lệnh STOP

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # 1. GIỮ LẠI: Vẽ các điểm kết nối Landmark bàn tay lên hình
            mp_draw.draw_landmarks(
                img_processing, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Lấy tọa độ để tính toán lệnh và vẽ bounding box
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]

            # Xác định lệnh điều khiển dựa trên số ngón tay + AI
            local_command, local_confidence = determine_command(hand_landmarks)
            
            # Đổi màu xanh nếu nhận diện đúng lệnh di chuyển
            if local_command != "STOP":
                color = (0, 255, 0)

            # 2. GIỮ LẠI: Vẽ Khung bao quanh bàn tay (Bounding Box)
            x1, y1 = int(min(x_) * W), int(min(y_) * H)
            x2, y2 = int(max(x_) * W), int(max(y_) * H)
            cv2.rectangle(img_processing, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ chữ text nhãn đè ngay phía trên Box bàn tay
            cv2.putText(
                img_processing, f"{local_command} ({local_confidence:.2f})",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )
        else:
            local_command = "STOP"

        # Cập nhật thông số global
        command = local_command
        confidence = local_confidence

        # Sau khi vẽ xong xuôi toàn bộ Landmark/Box, đẩy ảnh đã xử lý sang cho Flask hiển thị
        with frame_lock:
            frame_rendered = img_processing.copy()

        # Điều khiển UART gửi dữ liệu đi
        current_time = time.time()
        if ser and (command != last_command or current_time - last_send_time > SEND_DELAY):
            try:
                ser.write((command + '\n').encode())
                print("SEND:", command)
            except Exception as e:
                print(f"Lỗi UART: {e}")
            last_command = command
            last_send_time = current_time

        time.sleep(0.02) # Tối ưu tốc độ xử lý khoảng ~40-50 FPS

# ======================================================
# FLASK GENERATE: CHỈ LÀM NHIỆM VỤ ĐẨY HÌNH ĐÃ VẼ LÊN WEB
# ======================================================
def generate():
    while True:
        img_display = None
        with frame_lock:
            # Ưu tiên lấy ảnh đã được luồng AI vẽ Landmark, nếu chưa có thì lấy ảnh thô
            if frame_rendered is not None:
                img_display = frame_rendered.copy()
            elif frame_global is not None:
                img_display = frame_global.copy()

        if img_display is None:
            time.sleep(0.01)
            continue

        # Vẽ thanh trạng thái tổng quát ở góc trên màn hình hình nền
        color = (0, 255, 0) if command != "STOP" else (0, 0, 255)
        cv2.putText(
            img_display, f"SYSTEM CMD: {command}", 
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
        )

        # Nén hình và stream (Chất lượng 75% cân bằng giữa độ nét và độ mượt mạng)
        ret, buffer = cv2.imencode('.jpg', img_display, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.04) # Cố định FPS hiển thị ~25 FPS để chống giật lag mạng

# ======================================================
# UI & ROUTES
# ======================================================
@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>Gesture Robot Live Stream</title>
        <style>
            body { margin: 0; font-family: Arial; background: #1e1e2f; color: white; display: flex; justify-content: center; align-items: flex-start; padding-top: 20px; }
            .container { width: 90vw; max-width: 1000px; text-align: center; }
            h2 { margin-bottom: 15px; color: #00ff99; }
            .video-wrapper { width: 100%; aspect-ratio: 16/9; max-height: 70vh; background: black; border-radius: 15px; overflow: hidden; border: 3px solid #00ff99; }
            .video-wrapper img { width: 100%; height: 100%; object-fit: contain; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🖐️ Gesture Control Robot (High FPS Mode)</h2>
            <div class="video-wrapper"><img src="/video"></div>
        </div>
    </body>
    </html>
    """

@app.route('/video')
def video():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ======================================================
# SAFE SHUTDOWN SYSTEM
# ======================================================
def safe_exit_handler(sig, frame):
    print("\n🛑 Đang tắt hệ thống an toàn...")
    try:
        if cap and cap.isOpened():
            cap.release()
            print("- Đã giải phóng tài nguyên Camera.")
    except:
        pass
    try:
        if ser and ser.is_open:
            ser.close()
            print("- Đã giải phóng cổng UART ESP32.")
    except:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, safe_exit_handler)    

# ======================================================
# RUN & START THREADS
# ======================================================
if __name__ == '__main__':
    t1 = threading.Thread(target=video_capture_thread, daemon=True)
    t1.start()

    t2 = threading.Thread(target=ai_processing_thread, daemon=True)
    t2.start()

    app.run(host='0.0.0.0', port=5000, threaded=True)