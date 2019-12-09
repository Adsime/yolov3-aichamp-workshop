import cv2 as cv
import numpy as np

weigths_path = "data/weights/"
cfg_path = "data/cfg/"
class_path = "data/"

def load(weights, cfg):
    weights = weigths_path + weights
    cfg = cfg_path + cfg
    return cv.dnn.readNet(weights, cfg)

def prepare_classes(classes):
    with(open(class_path + classes, "r")) as f:
        return [line.strip() for line in f.readlines()]

def extract_output_layers(net):
    names = net.getLayerNames()
    return [names[layer[0] - 1] for layer in net.getUnconnectedOutLayers()]

def image_to_blob(image):
    width, height, _ = image.shape
    return cv.dnn.blobFromImage(image, 0.00392, (416, 416), (0,0,0), True, crop=False)

def forward(net, blob, output_layers):
    net.setInput(blob)
    return net.forward(output_layers)

def process_result(image, outputs, classes, threshold=0.8):
    width, height, _ = image.shape

    class_ids = []
    confidences = []
    boxes = []

    for o in outputs:
        for detection in o:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if(confidence > threshold):
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indices = cv.dnn.NMSBoxes(boxes, confidences, threshold, 0.0)

    for i, box in enumerate(boxes):
        if i in indices:
            x,y,w,h = box
            label = str(classes[class_ids[i]])
            cv.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv.putText(image, label, (x, y + 20), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

    return boxes, confidence, class_ids