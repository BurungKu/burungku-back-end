import numpy as np
import cv2

def preprocess(image, size=640):
    orig_w, orig_h = image.size

    image = image.resize((size, size))
    img = np.array(image)
    img = img[:, :, ::-1]  # RGB → BGR if your model expects it (optional)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)   # (1,3,H,W)
    return img, orig_w, orig_h


# Basic NMS implementation
def nms(boxes, scores, iou_threshold):
    idxs = np.argsort(scores)[::-1]
    keep = []

    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)
        if len(idxs) == 1:
            break
        ious = box_iou(boxes[current], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_threshold]

    return keep


def box_iou(box1, boxes):
    # box = [x1,y1,x2,y2]
    x1 = np.maximum(box1[0], boxes[:, 0])
    y1 = np.maximum(box1[1], boxes[:, 1])
    x2 = np.minimum(box1[2], boxes[:, 2])
    y2 = np.minimum(box1[3], boxes[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = area1 + area2 - inter
    return inter / union


def postprocess(outputs, orig_w, orig_h, conf_thres=0.25, iou_thres=0.45):

    preds = outputs[0][0]

    boxes = []
    scores = []
    classes = []

    for det in preds:
        x1, y1, x2, y2, score, cls = det.tolist()
        if score < conf_thres:
            continue

        x1 *= orig_w / 640
        x2 *= orig_w / 640
        y1 *= orig_h / 640
        y2 *= orig_h / 640

        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        classes.append(int(cls))

    boxes = np.array(boxes)
    scores = np.array(scores)

    if len(boxes) == 0:
        return []

    keep = nms(boxes, scores, iou_thres)

    results = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        results.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "score": float(scores[i]),
            "class": int(classes[i])
        })

    return results
