from flask import Flask, Response, request, redirect
import cv2
import os
import time

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(BASE_DIR, "data")

os.makedirs(SAVE_PATH, exist_ok=True)

# ==========================================
# WEBCAM
# ==========================================
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Không mở được webcam")
    exit()

# ==========================================
# LABEL NAMES
# ==========================================
LABELS = {
    0: "FORWARD",
    1: "BACKWARD",
    2: "LEFT",
    3: "RIGHT",
    4: "STOP"
}

# ==========================================
# GLOBAL
# ==========================================
count = 0
label = 0
collecting = False

MAX_IMAGE = 150

# ==========================================
# GENERATE VIDEO
# ==========================================
def generate():

    global count
    global collecting

    while True:

        success, frame = cap.read()

        if not success:
            continue

        # ==========================================
        # FLIP
        # ==========================================
        frame = cv2.flip(frame, 1)

        # ==========================================
        # OVERLAY
        # ==========================================
        cv2.putText(
            frame,
            f"Class: {label} - {LABELS[label]}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"Count: {count}/{MAX_IMAGE}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,255),
            2
        )

        # ==========================================
        # SAVE IMAGE
        # ==========================================
        if collecting and count < MAX_IMAGE:

            folder = f"{SAVE_PATH}/{label}"

            os.makedirs(folder, exist_ok=True)

            filename = f"{folder}/{int(time.time()*1000)}.jpg"

            cv2.imwrite(filename, frame)

            count += 1

            time.sleep(0.05)

        # ==========================================
        # AUTO STOP
        # ==========================================
        if count >= MAX_IMAGE:

            collecting = False

        # ==========================================
        # STREAM
        # ==========================================
        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame +
            b'\r\n'
        )

# ==========================================
# UI
# ==========================================
@app.route('/')
def index():

    return f"""
    <html>

    <head>

        <title>Gesture Dataset Collector</title>

        <style>

            body {{

                margin:0;
                font-family:Arial;

                background:
                linear-gradient(
                    135deg,
                    #1e3c72,
                    #2a5298
                );

                display:flex;

                justify-content:center;
                align-items:center;

                height:100vh;

                color:white;
            }}

            .container {{

                background:rgba(0,0,0,0.6);

                padding:30px;

                border-radius:20px;

                text-align:center;

                box-shadow:
                0 10px 30px rgba(0,0,0,0.5);
            }}

            button {{

                margin:5px;

                padding:12px 20px;

                border:none;

                border-radius:10px;

                font-size:16px;

                cursor:pointer;

                transition:0.3s;
            }}

            button:hover {{

                transform:scale(1.05);
            }}

            .btn-class {{

                background:#00c6ff;
                color:black;
            }}

            .btn-start {{

                background:#00ff9d;
                color:black;
            }}

            .btn-reset {{

                background:#ff4b5c;
                color:white;
            }}

            img {{

                margin-top:15px;

                width:640px;

                border-radius:15px;

                border:3px solid white;
            }}

            h2,h3 {{
                margin:10px;
            }}

        </style>

    </head>

    <body>

        <div class="container">

            <h2>📸 Gesture Dataset Collector</h2>

            <h3>
                Current Class:
                {label}
                -
                {LABELS[label]}
            </h3>

            <h3>
                Count:
                {count}/{MAX_IMAGE}
            </h3>

            <!-- ====================== -->
            <!-- CLASS BUTTON -->
            <!-- ====================== -->

            <form action="/set_class">

                <button
                    class="btn-class"
                    name="label"
                    value="0">

                    FORWARD

                </button>

                <button
                    class="btn-class"
                    name="label"
                    value="1">

                    BACKWARD

                </button>

                <button
                    class="btn-class"
                    name="label"
                    value="2">

                    LEFT

                </button>

                <button
                    class="btn-class"
                    name="label"
                    value="3">

                    RIGHT

                </button>

                <button
                    class="btn-class"
                    name="label"
                    value="4">

                    STOP

                </button>

            </form>

            <!-- ====================== -->
            <!-- START -->
            <!-- ====================== -->

            <form action="/start">

                <button class="btn-start">

                    START COLLECT

                </button>

            </form>

            <!-- ====================== -->
            <!-- RESET -->
            <!-- ====================== -->

            <form action="/reset">

                <button class="btn-reset">

                    RESET

                </button>

            </form>

            <!-- ====================== -->
            <!-- VIDEO -->
            <!-- ====================== -->

            <img src="/video">

        </div>

    </body>

    </html>
    """

# ==========================================
# SET CLASS
# ==========================================
@app.route('/set_class')
def set_class():

    global label
    global count
    global collecting

    label = int(request.args.get("label"))

    count = 0

    collecting = False

    return redirect('/')

# ==========================================
# START
# ==========================================
@app.route('/start')
def start():

    global collecting
    global count

    count = 0

    collecting = True

    return redirect('/')

# ==========================================
# RESET
# ==========================================
@app.route('/reset')
def reset():

    global count
    global collecting

    count = 0

    collecting = False

    return redirect('/')

# ==========================================
# VIDEO STREAM
# ==========================================
@app.route('/video')
def video():

    return Response(
        generate(),
        mimetype=
        'multipart/x-mixed-replace; boundary=frame'
    )

# ==========================================
# RUN
# ==========================================
app.run(
    host='0.0.0.0',
    port=5000
)