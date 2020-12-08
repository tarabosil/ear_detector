import sys
import matplotlib.pyplot as plt
import numpy as np


class Evaluator:

    def is_intersect(self, first, second):
        """ Check if two boxes intersect """
        if first[0] > second[2] or first[2] < second[0]:
            return False
        if first[1] > second[3] or first[3] < second[1]:
            return False
        return True

    def iou(self, predicted, original):
        """ Calculate IOU - intersection over union. """

        if not self.is_intersect(original, predicted):
            return 0

        x1 = max(original[0], predicted[0])
        y1 = max(original[1], predicted[1])
        x2 = min(original[2], predicted[2])
        y2 = min(original[3], predicted[3])

        intersection = (x2 - x1 + 1) * (y2 - y1 + 1)

        originalArea = (original[2] - original[0] + 1) * (original[3] - original[1] + 1)
        predictedArea = (predicted[2] - predicted[0] + 1) * (predicted[3] - predicted[1] + 1)

        union = float(originalArea + predictedArea - intersection)
        IOU = intersection / union

        return IOU

    def get_float_min(self):
        """ Return float min value. """
        return sys.float_info.min

    def precision(self, TP, FP):
        """ Calculate precision. """
        return TP / (TP + FP)

    def recall(self, TP, all_GT):
        """ Calculate recall. """
        return TP / all_GT

    def precision_recall_curve(self, precision, recall):
        """ Precision Recall Curve """
        plt.title("Precision Recall Curve")
        plt.plot(precision, '-', label='precision', color='orchid')
        plt.plot(recall, '-', label='recall', color='royalblue')
        plt.legend()
        # plt.savefig('images/precision_recall.png')
        plt.show()


    def interpolated_precision(self, precision, recall):
        IP = []
        IP.append(precision[0])
        recall_step = recall[0]
        prec_step = precision[0]
        for i in range(1, len(precision)):
            if recall[i] == recall_step:
                IP.append(prec_step)
            else:
                prec_step = precision[i]
                recall_step = recall[i]
                IP.append(prec_step)

        return IP
