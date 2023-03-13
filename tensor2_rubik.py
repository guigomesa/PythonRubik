import cv2
import numpy as np
import tensorflow as tf

# Inicializar a câmera
cap = cv2.VideoCapture(0)

# Carregar o modelo de detecção de objetos
model = tf.keras.models.load_model('rubik_cube_detector')

# Loop principal
while True:
    # Capturar um frame da webcam
    ret, frame = cap.read()

    # Detectar as seis faces do cubo de Rubik no frame
    cube_faces = []
    for i in range(6):
        # Selecionar a região correspondente a uma das faces do cubo
        x = i % 3
        y = i // 3
        region = frame[y*100:y*100+100, x*100:x*100+100, :]

        # Detectar a face do cubo usando o modelo de detecção de objetos
        cube_face = region.copy()
        cube_face = cv2.cvtColor(cube_face, cv2.COLOR_BGR2RGB)
        cube_face = cv2.resize(cube_face, (224, 224))
        cube_face = cube_face / 255.0
        cube_face = np.expand_dims(cube_face, axis=0)
        predictions = model.predict(cube_face)
        label = np.argmax(predictions[0])
        color = None
        if label == 0:
            color = 'white'
        elif label == 1:
            color = 'yellow'
        elif label == 2:
            color = 'green'
        elif label == 3:
            color = 'blue'
        elif label == 4:
            color = 'red'
        elif label == 5:
            color = 'orange'

        # Adicionar a cor da face ao array de cores da face
        cube_faces.append(color)

        # Mostrar a imagem com a cor da face detectada
        cv2.rectangle(cube_face, (0, 0), (100, 100), (255, 255, 255), 2)
        cv2.putText(cube_face, color, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(f'Cube Face {i}', cube_face)

    # Salvar as informações das faces do cubo em um arquivo numpy binário
    cube_faces = np.array(cube_faces).reshape((6, 3, 3))
    np.save('cube_faces.npy', cube_faces)

    # Aguardar a tecla 'q' ser pressionada para encerrar o programa
    if cv2.waitKey(1) == ord('q'):
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
