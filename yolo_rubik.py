import cv2
import numpy as np
import darknet


config_file = 'yolo/cfg/yolov4.cfg'
weights_file = 'yolo/weights/yolov4.weights'

# carregando yolo
network, class_names, class_colors = darknet.load_network(
    config_file,
    weights_file,
    batch_size=1,
)

# tamanho do quadro de entrada
width = darknet.network_width(network)
height = darknet.network_height(network)

conf_threshold = 0.5
nms_threshold = 0.4


# webcam

cap = cv2.VideoCapture(0)

#defininindo o preview em tela
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    blob = cv2.dnn.blobFromImage(
        frame,
        1 / 255.0,
        (width, height),
        (0, 0, 0),
        1,
        crop=False,
    )

    #definir a entrada do yolo
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, blob)

    detections = darknet.detect_image(network, darknet_image, conf_threshold,
    nms_threshold)

    for label, confidence, bbox in detections:
        if label == 'cube':
            x,y,w,h = bbox
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Cubo detectado', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

np.savetxt('rubik_dados.txt', corners, fmt='%.2f')
cap.release()
cv2.destroyAllWindows()