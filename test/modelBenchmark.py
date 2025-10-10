import numpy as np

def read_detections(txt_path):
    """
    Reads detection txt file in format:
    <frame_id> <label> <x> <y> <width> <height>
    Returns: dict {frame_id: [ [label(str), x, y, w, h], ... ]}
    """
    detections = {}
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            frame_id = int(parts[0])
            label = parts[1]  # label as string
            x, y, w, h = map(float, parts[2:])
            detections.setdefault(frame_id, []).append([label, x, y, w, h])
    return detections

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def benchmark(gt_txt, pred_txt, iou_thresh=0.25):
    gt = read_detections(gt_txt)
    pred = read_detections(pred_txt)
    TP, FP, FN = 0, 0, 0

    for frame_id in gt:
        gt_boxes = gt.get(frame_id, [])
        pred_boxes = pred.get(frame_id, [])
        pred_matched = set()
        for g in gt_boxes:
            best_iou = 0
            best_idx = -1
            for i, p in enumerate(pred_boxes):
                if p[0] == g[0] and i not in pred_matched:  # label match
                    curr_iou = iou(g[1:], p[1:])
                    if curr_iou > best_iou:
                        best_iou = curr_iou
                        best_idx = i
            if best_iou >= iou_thresh and best_idx != -1:
                TP += 1
                pred_matched.add(best_idx)
            else:
                FN += 1
        FP += len(pred_boxes) - len(pred_matched)

    extra_pred_frames = set(pred.keys()) - set(gt.keys())
    for frame_id in extra_pred_frames:
        FP += len(pred[frame_id])
    print(f"TP: {TP}, FP: {FP}, FN: {FN}")
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

if __name__ == "__main__":
    gt_txt = "/home/lavi/Videos/detections.txt"
    pred_txt = "/home/lavi/Pictures/detections.txt"
    benchmark(gt_txt, pred_txt)