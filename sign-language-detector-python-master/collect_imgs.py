from flask import Flask, Response, request, redirect
import cv2
import os
import time

app = Flask(__name__)

# ===== CONFIG =====
SAVE_PATH = "data"
MAX_IMAGES = 150  
os.makedirs(SAVE_PATH, exist_ok=True)

# ===== WEBCAM =====
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

count = 0
label = 0
collecting = False
last_save_time = 0


# ===== STREAM =====
def generate():
    global count, collecting, last_save_time

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # ===== TEXT =====
        cv2.putText(frame, f"Class: {label}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.putText(frame, f"Count: {count}/{MAX_IMAGES}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        # ===== SAVE DATA =====
        if collecting and count < MAX_IMAGES:
            if time.time() - last_save_time > 0.1:  # delay tránh trùng
                folder = f"{SAVE_PATH}/{label}"
                os.makedirs(folder, exist_ok=True)

                cv2.imwrite(f"{folder}/{count}.jpg", frame)
                count += 1
                last_save_time = time.time()

        # stop khi đủ
        if count >= MAX_IMAGES:
            collecting = False

        # ===== STREAM =====
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ===== UI =====
@app.route('/')
def index():
    return f"""
    <html>
    <head>
        <title>Collect Data</title>

        <style>
            body {{
                margin: 0;
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #1e3c72, #2a5298);
                display: flex;
                justify-content: center;
                align-items: flex-start;
                padding-top: 20px;
                min-height: 100vh;
                color: white;
            }}

            .container {{
                background: rgba(0,0,0,0.6);
                padding: 20px;
                border-radius: 20px;
                text-align: center;
                box-shadow: 0 10px 30px rgba(0,0,0,0.5);

                width: 90vw;
                max-width: 1000px;
            }}

            h2 {{
                margin-bottom: 10px;
            }}

            .info {{
                margin: 10px 0;
                font-size: 18px;
            }}

            button {{
                margin: 5px;
                padding: 10px 20px;
                border: none;
                border-radius: 10px;
                font-size: 16px;
                cursor: pointer;
                transition: 0.3s;
            }}

            button:hover {{
                transform: scale(1.05);
            }}

            .btn-class {{
                background: #00c6ff;
                color: black;
            }}

            .btn-start {{
                background: #00ff9d;
                color: black;
            }}

            .btn-reset {{
                background: #ff4b5c;
                color: white;
            }}

            /* ===== VIDEO ===== */
            .video-wrapper {{
                margin-top: 15px;
                width: 100%;
                aspect-ratio: 16/9;
                max-height: 65vh;
                background: black;
                border-radius: 15px;
                overflow: hidden;
                border: 3px solid white;
            }}

            .video-wrapper img {{
                width: 100%;
                height: 100%;
                object-fit: contain;
            }}
        </style>
    </head>

    <body>
        <div class="container">
            <h2>📸 Collect Data (Webcam)</h2>

            <div class="info">Class: <b>{label}</b></div>
            <div class="info">Count: <b>{count}/{MAX_IMAGES}</b></div>

            <form action="/set_class">
                <button class="btn-class" name="label" value="0">Class 0</button>
                <button class="btn-class" name="label" value="1">Class 1</button>
                <button class="btn-class" name="label" value="2">Class 2</button>
            </form>

            <form action="/start">
                <button class="btn-start">Start Collect</button>
            </form>

            <form action="/reset">
                <button class="btn-reset">Reset</button>
            </form>

            <div class="video-wrapper">
                <img src="/video">
            </div>
        </div>
    </body>
    </html>
    """


# ===== ROUTES =====
@app.route('/set_class')
def set_class():
    global label, count, collecting
    label = int(request.args.get("label"))
    count = 0
    collecting = False
    return redirect('/')


@app.route('/start')
def start():
    global collecting, count
    count = 0
    collecting = True
    return redirect('/')


@app.route('/reset')
def reset():
    global count, collecting
    count = 0
    collecting = False
    return redirect('/')


@app.route('/video')
def video():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ===== RUN =====
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)