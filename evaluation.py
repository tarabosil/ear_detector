import os
import cv2
from bounding_box import BoundingBox
import config
import bcolors
import evaloator


def evaluate():
    predicted_path = config.path_to_predicted_bb
    ground_truth_path = config.path_to_ground_truth_bb

    bounding_box = BoundingBox()
    evaluator = evaloator.Evaluator()

    TP, FP = 0, 0
    all_GT = 0

    true_positives, false_positives = [], []
    precision, recall = [], []

    for img_name in os.listdir(f'{predicted_path}'):
        print(f' {bcolors.BLUE} Starting with image {img_name}... {bcolors.ENDC}')
        predicted_img = cv2.imread(f'{predicted_path}/{img_name}')
        ground_truth_img = cv2.imread(f'{ground_truth_path}/{img_name}')
        p_b = bounding_box.get_bounding_boxes(predicted_img)
        gt_b = bounding_box.get_bounding_boxes(ground_truth_img)

        gt_matched = {k: False for k, v in enumerate(gt_b)}
        all_GT += len(gt_matched)

        for i in range(len(p_b)):
            iou_max = evaluator.get_float_min()
            gt_max = -1
            for j in range(len(gt_b)):
                iou = evaluator.iou(p_b[i], gt_b[j])
                print("IOU: ", iou)
                if iou > iou_max:
                    iou_max = iou
                    gt_max = j

            if iou_max >= config.iou_threshold:
                if not gt_matched[gt_max]:
                    TP += 1
                    gt_matched[gt_max] = True
                else:
                    FP += 1
            else:
                FP += 1

        true_positives.append(TP)
        false_positives.append(FP)

    print("TP: ", TP)
    print("FP: ", FP)
    print("GT: ", all_GT)
    pr = evaluator.precision(TP, FP)
    re = evaluator.recall(TP, all_GT)
    print(f"Precision: {pr}, Recall: {re}, F1-score: {2 * ((pr * re) / (pr + re))}")

    for i in range(len(true_positives)):
        if (true_positives[i] + false_positives[i]) != 0:
            precision.append(true_positives[i] / (true_positives[i] + false_positives[i]))
        else:
            precision.append(0)
        recall.append(true_positives[i] / all_GT)

    """ plot precision_recall curve """
    # evaluator.precision_recall_curve(precision, recall)


if __name__ == "__main__":
    evaluate()