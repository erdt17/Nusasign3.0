from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
import cv2, threading, atexit, json, os
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from gtts import gTTS
import pygame

app = Flask(__name__)

# ─── Model YOLO untuk mode huruf (tidak diubah) ───────────
model_huruf = YOLO("best.pt")

# ─── Arsitektur model kata (harus sama persis saat training)
class BisindoV2(nn.Module):
    def __init__(self, input_size=126, hidden_size=256, num_classes=32):
        super().__init__()
        self.bilstm1 = nn.LSTM(input_size, hidden_size, batch_first=True,
                                bidirectional=True, dropout=0.3, num_layers=2)
        self.bilstm2 = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True,
                                bidirectional=True)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out, _ = self.bilstm1(x)
        out, _ = self.bilstm2(out)
        weights = torch.softmax(self.attention(out), dim=1)
        out = (out * weights).sum(dim=1)
        return self.fc(out)


# ─── Load label & model kata ──────────────────────────────
with open('label_classes.json', encoding='utf-8') as f:
    LABELS = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_kata = BisindoV2(num_classes=len(LABELS)).to(device)
checkpoint = torch.load('bisindo_final.pth',
                        map_location=device, weights_only=False)
model_kata.load_state_dict(checkpoint['model_state'])
model_kata.eval()
print(f"[OK] Model kata loaded — {len(LABELS)} kata, device: {device}")

# ─── MediaPipe Holistic ───────────────────────────────────
from mediapipe.python.solutions import holistic as mp_holistic_module
import mediapipe as mp
holistic = mp_holistic_module.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# ─── Kamera ───────────────────────────────────────────────
cap = cv2.VideoCapture(0)

# ─── State (thread-safe) ──────────────────────────────────
lock = threading.Lock()
state = {
    'mode':             'letter',
    'prediction_huruf': '',
    'prediction_kata':  '',
}

frame_buffer = deque(maxlen=60)
buffer_huruf = deque(maxlen=10)

ENCODE_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 65]

# ─── TTS ──────────────────────────────────────────────────
pygame.mixer.init()
_last_spoken = ""
_tts_lock    = threading.Lock()

def speak(text):
    global _last_spoken
    if not text or text == _last_spoken:
        return
    def _do():
        global _last_spoken
        with _tts_lock:
            _last_spoken = text
            try:
                tts = gTTS(text=text, lang='id', slow=False)
                tts.save('temp_audio.mp3')
                pygame.mixer.music.load('temp_audio.mp3')
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
            except Exception as e:
                print(f"[TTS error] {e}")
    threading.Thread(target=_do, daemon=True).start()


# ─── Ekstrak keypoints tangan dari hasil MediaPipe ────────
def extract_keypoints(results):
    lh = [[l.x, l.y, l.z] for l in results.left_hand_landmarks.landmark] \
         if results.left_hand_landmarks else [[0, 0, 0]] * 21
    rh = [[l.x, l.y, l.z] for l in results.right_hand_landmarks.landmark] \
         if results.right_hand_landmarks else [[0, 0, 0]] * 21
    return np.array(lh + rh, dtype=np.float32).flatten()


# ─── Generator frame kamera ───────────────────────────────
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        with lock:
            current_mode = state['mode']

        if current_mode == 'letter':
            small   = cv2.resize(frame, (416, 416))
            results = model_huruf.predict(source=small, conf=0.5,
                                          imgsz=416, verbose=False)
            annotated_frame = results[0].plot()

            if results[0].boxes:
                cls_id    = int(results[0].boxes[0].cls[0])
                raw_label = model_huruf.names[cls_id]
                buffer_huruf.append(raw_label)
                stable = max(set(buffer_huruf), key=buffer_huruf.count)
                with lock:
                    state['prediction_huruf'] = stable
                cv2.putText(annotated_frame, f"Huruf: {stable}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 255, 0), 2)
            else:
                buffer_huruf.append("")
                with lock:
                    state['prediction_huruf'] = ""

        else:
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            kp      = extract_keypoints(results)
            frame_buffer.append(kp)

            annotated_frame = frame.copy()

            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame, results.right_hand_landmarks,
                    mp_holistic_module.HAND_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame, results.left_hand_landmarks,
                    mp_holistic_module.HAND_CONNECTIONS)

            if len(frame_buffer) == 60:
                seq   = np.array(list(frame_buffer), dtype=np.float32)
                seq_t = torch.FloatTensor(seq).unsqueeze(0).to(device)

                with torch.no_grad():
                    out        = model_kata(seq_t)
                    probs      = torch.softmax(out, dim=1)
                    confidence = probs.max().item()
                    pred_idx   = probs.argmax().item()

                if confidence > 0.75:
                    label = LABELS[pred_idx]
                    with lock:
                        state['prediction_kata'] = label
                    speak(label)
                    cv2.putText(annotated_frame,
                                f"Kata: {label} ({confidence*100:.0f}%)",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 0, 255), 2)
                else:
                    cv2.putText(annotated_frame, "Mendeteksi...",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (128, 128, 128), 2)

        ret, buffer = cv2.imencode('.jpg', annotated_frame, ENCODE_PARAMS)
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buffer.tobytes() + b'\r\n')


# ─── Routes ───────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode', methods=['POST'])
def set_mode():
    data = request.get_json(silent=True) or {}
    mode = data.get('mode', 'letter')
    if mode not in ('letter', 'word'):
        return jsonify({'ok': False, 'error': 'mode tidak valid'}), 400
    with lock:
        state['mode'] = mode
        frame_buffer.clear()
        buffer_huruf.clear()
    return jsonify({'ok': True, 'mode': mode})

@app.route('/get_prediction')
def get_prediction():
    mode = request.args.get('mode', 'letter')
    with lock:
        pred = state['prediction_huruf'] if mode == 'letter' \
               else state['prediction_kata']
    return jsonify({'prediction': pred})

@app.route('/get_letter')
def get_letter():
    with lock:
        return jsonify({'letter': state['prediction_huruf']})

@app.route('/get_phrase')
def get_phrase():
    with lock:
        return jsonify({'phrase': state['prediction_kata']})

@atexit.register
def cleanup():
    cap.release()
    holistic.close()
    cv2.destroyAllWindows()
    if os.path.exists('temp_audio.mp3'):
        os.remove('temp_audio.mp3')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
