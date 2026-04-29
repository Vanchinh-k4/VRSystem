from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import time

app = Flask(__name__)


picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"size": (640, 580)}   
    )
)
picam2.start()

def generate():
    while True:
        frame = picam2.capture_array()

        
        ret, buffer = cv2.imencode(
            '.jpg', frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 35]
        )

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

 
        time.sleep(0.03)

@app.route('/video')
def video():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(host='0.0.0.0', port=5001)