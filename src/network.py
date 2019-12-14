import cv2 as cv
import numpy as np
from copy import deepcopy

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

def image_to_blob(image, size=(416, 416)):
    return cv.dnn.blobFromImage(image, scalefactor=1/255, size=size, swapRB=True, crop=False)

def forward(net, blob, output_layers):
    net.setInput(blob)
    return net.forward(output_layers)

def postprocess(image, outputs, classes, threshold=0.8, nms_threshold=0.7, box_color=(0, 0, 0), text_color=(255, 255, 255), font_scale=0.5, font=cv.FONT_HERSHEY_SIMPLEX, thickness=2):
    height, width, _ = image.shape

    image = deepcopy(image)

    class_ids = []
    confidences = []
    boxes = []

    for o in outputs:
        for detection in o:
            center_x = detection[0] * width
            center_y = detection[1] * height
            w = detection[2] * width
            h = detection[3] * height
            x = center_x - w/2
            y = center_y - h/2

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            boxes.append([int(x), int(y), int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    indices = cv.dnn.NMSBoxes(boxes, confidences, threshold, nms_threshold)

    for i in indices:
        i = i[0]
        label = str(classes[class_ids[i]])
        draw_box(image, label, boxes[i], box_color, text_color, font_scale, font, thickness)
        
    return image, class_ids, confidences, boxes

def draw_box(image, label, box, box_color, text_color, font_scale, font, thickness):
    x,y,w,h = box
    tw, th = cv.getTextSize(text=label, fontFace=font, fontScale=font_scale, thickness=thickness)[0]
    cv.rectangle(image, (x, y), (x + w, y + h), box_color, thickness=thickness)
    cv.rectangle(image, (x, y), (x + tw + 10, y - th - 10), color=box_color, thickness=cv.FILLED)
    cv.putText(image, label, (x + 5, y - 5), fontFace=font, fontScale=font_scale, color=text_color, thickness=thickness)