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
    pred = outputs[0]          # (1, 84, 8400)
    pred = pred[0]             # (84, 8400)
    pred = pred.transpose(1, 0)  # (8400, 84)

    boxes = []
    scores = []
    classes = []

    for det in pred:
        # det = [x, y, w, h, obj_conf, cls1, cls2, ...]
        x, y, w, h = det[:4]
        obj_conf = det[4]
        cls_scores = det[5:]

        cls = np.argmax(cls_scores)
        cls_conf = cls_scores[cls]

        score = obj_conf * cls_conf
        if score < conf_thres:
            continue

        # Convert xywh → xyxy
        x1 = (x - w / 2) * (orig_w / 640)
        y1 = (y - h / 2) * (orig_h / 640)
        x2 = (x + w / 2) * (orig_w / 640)
        y2 = (y + h / 2) * (orig_h / 640)

        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        classes.append(cls)

    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)

    # NMS
    keep = nms(boxes, scores, iou_thres)

    results = []
    for i in keep:
        results.append({
            "bbox": [float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3])],
            "score": float(scores[i]),
            "class": int(classes[i])
        })

    return results

