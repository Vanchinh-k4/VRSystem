import pickle
from flask import Flask, Response
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# ===== LOAD MODEL =====
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# ===== MEDIAPIPE =====
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

labels_dict = {0: 'A', 1: 'B', 2: 'L'}

# ===== CAMERA =====
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ===== THRESHOLD =====
THRESHOLD = 0.8


# ===== STREAM =====
def generate():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        H, W, _ = frame.shape

        data_aux = []
        x_, y_ = [], []

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

            try:
                # ===== PREDICT + CONFIDENCE =====
                probs = model.predict_proba([np.asarray(data_aux)])
                confidence = np.max(probs)
                pred_index = np.argmax(probs)

                if confidence < THRESHOLD:
                    label = "Unknown"
                    color = (0, 0, 255)  # đỏ
                else:
                    label = labels_dict[pred_index]
                    color = (0, 255, 0)  # xanh

            except:
                label = "Error"
                color = (0, 0, 255)

            # ===== BOX =====
            x1 = int(min(x_) * W)
            y1 = int(min(y_) * H)
            x2 = int(max(x_) * W)
            y2 = int(max(y_) * H)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(frame,
                        f"{label} ({confidence:.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        2)

        # ===== STREAM =====
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ===== UI =====
@app.route('/')
def index():
    return """
    <html>
    <head>
    <style>
        body {
            margin: 0;
            font-family: Arial;
            background: #1e1e2f;
            color: white;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            padding-top: 20px;
        }

        .container {
            width: 90vw;
            max-width: 1000px;
            text-align: center;
        }

        h2 {
            margin-bottom: 15px;
        }

        .video-wrapper {
            width: 100%;
            aspect-ratio: 16/9;
            max-height: 70vh;
            background: black;
            border-radius: 15px;
            overflow: hidden;
            border: 3px solid white;
        }

        .video-wrapper img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
    </style>
    </head>

    <body>
        <div class="container">
            <h2>🖐️ Hand Gesture Detection</h2>

            <div class="video-wrapper">
                <img src="/video">
            </div>
        </div>
    </body>
    </html>
    """


# ===== ROUTE =====
@app.route('/video')
def video():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ===== RUN =====
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)