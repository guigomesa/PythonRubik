import tensorflow as tf 
import cv2
import numpy as np

model = tf.keras.models.load_model('model.h5')

classes = ['rubiks_cube']

colors = {'rubiks_cube': (255, 0, 0)}

input_size = (416, 416)

cap = cv2.VideoCapture(0)


while True:
    # capturar frame na webcam
    ret, frame = cap.read()

    #redimensionar para o tamanho do modelo
    resized_frame = cv2.resize(frame, input_size)

    # normalizar os pixels
    normalized_frame = resized_frame / 255.

    #adicionar uma dimensao para representar o lote de imagens
    batched_frame = np.expand_dims(normalized_frame, axis=0)

    #fazer a previsão
    detections = model.predict(batched_frame)

    for i in range(detections.shape[1]):
        # Obter a classe e a confiança da detecção
        class_id = np.argmax(detections[0, i, :])
        confidence = detections[0, i, 4+class_id]

        # ignorar o que não for confiavel
        if confidence < 0.5:
            continue

        # obter as coordenadas da caixa delimitadora
        x, y, w, h = detections[0, i, 0:4]

        #converter as coordendas para a escala da imagem original
        x *= frame.shape[1] / input_size[0]
        y *= frame.shape[0] / input_size[1]
        w *= frame.shape[1] / input_size[0]
        h *= frame.shape[0] / input_size[1]

        class_name = classes[class_id]
        cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
        cv2.putText(frame, f'{class_name} {confidence:.2f}', (int(x), int(y)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('Detector Cubo', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()