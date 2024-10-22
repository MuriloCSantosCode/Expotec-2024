import cv2
import json
import mediapipe as mp
import numpy as np
import keyboard
import time
from flask import Flask, render_template, jsonify, Response, request

app = Flask(__name__)

# Configurações iniciais 
video = cv2.VideoCapture(0)
try:
    with open("data.json", "r", encoding="utf-8") as data:
        coords_salvas = json.load(data)
except FileNotFoundError:
    coords_salvas = {}

# Configurações do MediaPipe para reconhecimento de gestos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
handGesture = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                             min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Dicionário para armazenar gestos reconhecidos e seus tempos
gestos_reconhecidos = {}
tempo_gesto = {}
frase_atual = ""
nome_gesto_form = ""
ultimo_tempo_gesto = 0
intervalo_tempo = 2

def normalizar_landmarks(landmarks):
    if landmarks:
        pulso = landmarks[0]
        landmarks_normalizados = []
        for landmark in landmarks:
            x = (landmark.x - pulso.x)
            y = (landmark.y - pulso.y)
            z = (landmark.z - pulso.z)
            landmarks_normalizados.append((x, y, z))
        
        # Escala com base na distância entre o pulso e outro ponto de referência
        escala = np.sqrt((landmarks[9].x - pulso.x) ** 2 + 
                         (landmarks[9].y - pulso.y) ** 2 +
                         (landmarks[9].z - pulso.z) ** 2)
        
        landmarks_normalizados = [(x / escala, y / escala, z / escala) for (x, y, z) in landmarks_normalizados]
        return landmarks_normalizados
    return []

def verificar_distancia_entre_pulsos(pulso_mão1_atual, pulso_mão2_atual, pulso_mão1_salvo, pulso_mão2_salvo):
    distancia_atual = np.sqrt(np.sum((np.array(pulso_mão1_atual) - np.array(pulso_mão2_atual)) ** 2))
    distancia_salva = np.sqrt(np.sum((np.array(pulso_mão1_salvo) - np.array(pulso_mão2_salvo)) ** 2))
    tolerancia = 0.15
    return abs(distancia_atual - distancia_salva) < tolerancia

def comparar_sinais(landmarks_mão1, landmarks_mão2, coords_salvas):
    for nome_gesto, landmarks_salvas in coords_salvas.items():
        landmarks_salvas_mão1 = landmarks_salvas.get("mão1", None)
        landmarks_salvas_mão2 = landmarks_salvas.get("mão2", None)

        distancias_mão1 = []
        distancias_mão2 = []

        if landmarks_salvas_mão1 and landmarks_mão1:
            for idx, (x_atual, y_atual, z_atual) in enumerate(landmarks_mão1):
                idx_str = str(idx)
                if idx_str in landmarks_salvas_mão1:
                    x_salvo = landmarks_salvas_mão1[idx_str]["x"]
                    y_salvo = landmarks_salvas_mão1[idx_str]["y"]
                    z_salvo = landmarks_salvas_mão1[idx_str]["z"]
                    distancia = np.sqrt((x_atual - x_salvo) ** 2 + (y_atual - y_salvo) ** 2 + (z_atual - z_salvo) ** 2)
                    distancias_mão1.append(distancia)

        if landmarks_salvas_mão2 and landmarks_mão2:
            for idx, (x_atual, y_atual, z_atual) in enumerate(landmarks_mão2):
                idx_str = str(idx)
                if idx_str in landmarks_salvas_mão2:
                    x_salvo = landmarks_salvas_mão2[idx_str]["x"]
                    y_salvo = landmarks_salvas_mão2[idx_str]["y"]
                    z_salvo = landmarks_salvas_mão2[idx_str]["z"]
                    distancia = np.sqrt((x_atual - x_salvo) ** 2 + (y_atual - y_salvo) ** 2 + (z_atual - z_salvo) ** 2)
                    distancias_mão2.append(distancia)

        tolerancia_distancia = 0.15
        media_distancia_mão1 = np.mean(distancias_mão1) if distancias_mão1 else float('inf')
        media_distancia_mão2 = np.mean(distancias_mão2) if distancias_mão2 else float('inf')

        posicao_relativa_adequada = True

        if landmarks_mão1 and landmarks_mão2 and landmarks_salvas_mão1 and landmarks_salvas_mão2:
            pulso_mão1_atual = landmarks_mão1[0]
            pulso_mão2_atual = landmarks_mão2[0]
            pulso_mão1_salvo = (landmarks_salvas_mão1["0"]["x"], landmarks_salvas_mão1["0"]["y"], landmarks_salvas_mão1["0"]["z"])
            pulso_mão2_salvo = (landmarks_salvas_mão2["0"]["x"], landmarks_salvas_mão2["0"]["y"], landmarks_salvas_mão2["0"]["z"])

            posicao_relativa_adequada = verificar_distancia_entre_pulsos(pulso_mão1_atual, pulso_mão2_atual, pulso_mão1_salvo, pulso_mão2_salvo)
        
        if posicao_relativa_adequada and media_distancia_mão1 < tolerancia_distancia and (media_distancia_mão2 < tolerancia_distancia or not landmarks_salvas_mão2):
            return nome_gesto  # Retorna o gesto reconhecido
    return None  # Retorna None se nenhum gesto foi reconhecido

@app.route('/salvar_gesto', methods=['POST'])
def salvar_gesto():
    global landmarks_mão_direita, landmarks_mão_esquerda

    # Pega o nome do gesto enviado pelo formulário
    data = request.get_json()
    nome_gesto = data.get('gestureName')

    # Verifica se há landmarks de uma mão capturadas (direita ou esquerda)
    if landmarks_mão_direita or landmarks_mão_esquerda:
        # Salva o gesto com o nome fornecido
        salvar_sinal(nome_gesto, landmarks_mão_direita, landmarks_mão_esquerda)
        return jsonify({'message': 'Gesto salvo com sucesso!'})
    else:
        return jsonify({'message': 'Nenhuma mão detectada para salvar.'})

def salvar_sinal(nome_gesto, landmarks_mão_direita, landmarks_mão_esquerda):
    with open("data.json", "w", encoding="utf-8") as data:
        coords_salvas[nome_gesto] = {}
        
        if landmarks_mão_direita:
            coords_salvas[nome_gesto]["mão1"] = {}
            for idx, (x, y, z) in enumerate(landmarks_mão_direita):
                coords_salvas[nome_gesto]["mão1"][str(idx)] = {
                    "x": x,
                    "y": y,
                    "z": z
                }
                
        if landmarks_mão_esquerda:
            coords_salvas[nome_gesto]["mão2"] = {}
            for idx, (x, y, z) in enumerate(landmarks_mão_esquerda):
                coords_salvas[nome_gesto]["mão2"][str(idx)] = {
                    "x": x,
                    "y": y,
                    "z": z
                }
                
        json.dump(coords_salvas, data, indent=4)
        print("Coordenadas Salvas")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/gesto_reconhecido')
def gesto_reconhecido():
    global frase_atual
    return jsonify({'frase_atual': frase_atual})



def reconhecer_gesto():
    global frase_atual, landmarks_mão_direita, landmarks_mão_esquerda, ultimo_tempo_gesto
    try:
        _, frame = video.read()
        frame = cv2.flip(frame, 1)  # Corrigindo a inversão da câmera
        rgbConvertedFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = handGesture.process(rgbConvertedFrame)
        hands = output.multi_hand_landmarks
        handedness = output.multi_handedness

        landmarks_mão_direita = None
        landmarks_mão_esquerda = None

        if hands:
            for i, hand_landmarks in enumerate(hands):
                label = handedness[i].classification[0].label
                if label == 'Right':
                    landmarks_mão_direita = normalizar_landmarks(hand_landmarks.landmark)
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                elif label == 'Left':
                    landmarks_mão_esquerda = normalizar_landmarks(hand_landmarks.landmark)
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Verifica se o tempo desde o último gesto é maior que o intervalo
            tempo_atual = time.time()
            if tempo_atual - ultimo_tempo_gesto >= intervalo_tempo:
                # Verifica se um gesto foi reconhecido
                nome_gesto = comparar_sinais(landmarks_mão_direita, landmarks_mão_esquerda, coords_salvas)
                if nome_gesto:
                    frase_atual += nome_gesto  # Acumula o gesto reconhecido na frase
                    print(f"Gestos acumulados: {frase_atual}")  # Para ver o progresso no terminal

                    # Atualiza o tempo do último gesto reconhecido
                    ultimo_tempo_gesto = tempo_atual

        return frame
    
    except Exception as e:
        print(f"Erro ao reconhecer o gesto: {e}")
        return None

def gerar_frames():
    while True:
        frame = reconhecer_gesto()
        if frame is None:
            continue  # Pula o frame se houver algum erro

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gerar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)