# %%
import os
import json
import numpy as np
import pandas as pd


# %%
class PR_func(object):
    """Create precision-reacll function.
    Return:
        `PR_func` instance, call this instance
        with a recall value and it'll return
        a precision value.
    """
    def __init__(
            self,
            df,
            class_names=["STAS"]):
        class_num = len(class_names)
        self.class_num = class_num
        self.class_names = class_names

        if isinstance(df, pd.DataFrame):
            self.df = df
            recall = df.iloc[0].values[0:]
            precision = df.iloc[1].values[0:]

        self.precisions = [np.append(precision, np.array([0]))]
        self.recalls = [np.append(recall, recall[-1:])]

    def __call__(self, recall, class_idx=0):
        if class_idx >= self.class_num:
            raise IndexError("Class index out of range")
        precisions = self.precisions[class_idx]
        recalls = self.recalls[class_idx]
        pc_idx = (recalls > recall).sum()
        if pc_idx == 0:
            precision = 0
        else:
            precision = precisions[-pc_idx:].max()
        return precision


    def plot_pr_curve(self, class_idx=0,
                      smooth=False,
                      figsize=None,
                      return_fig=False):
        """Plot PR curve
        Args:
            class_idx: An integer, index of class.
            smooth: A boolean,
                if True, use interpolated precision.
            figsize: (float, float), optional, default: None
                width, height in inches.
                If not provided, defaults to [6.4, 4.8].
            return_fig: A boolean, whether to return plt.figure.
        """
        if class_idx >= self.class_num:
            raise IndexError("Class index out of range")
        precisions = self.precisions[class_idx].copy()
        recalls = self.recalls[class_idx]

        if smooth:
            max_pc = 0
            for i in range(len(precisions)-1, -1, -1):
                if precisions[i] > max_pc:
                    max_pc = precisions[i]
                else:
                    precisions[i] = max_pc
        
        fig = plt.figure(figsize=figsize)
        plt.plot(recalls, precisions)
        plt.title("PR curve")
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)

        if return_fig:
            return fig
        else:
            plt.show()


    def get_map(self, mode="area"):
        """Get a mAP table

        Args:
            mode: A string, one of "area", "smootharea".
                "area": calculate the area under precision-recall curve.
                "smootharea": calculate the area 
                    under interpolated precision-recall curve.

        Return:
            A Pandas.Dataframe
        """
        aps = [0 for _ in range(self.class_num)]

        if mode == "area" or mode == "smootharea":
            for class_i in range(self.class_num):
                if mode == "smootharea":
                    precisions = self.precisions[class_i].copy()
                    max_pc = 0
                    for i in range(len(precisions) - 1, -1, -1):
                        if precisions[i] > max_pc:
                            max_pc = precisions[i]
                        else:
                            precisions[i] = max_pc
                else:
                    precisions = self.precisions[class_i]
                recalls = self.recalls[class_i]

                for pr_i in range(0, len(precisions) - 1):
                    delta = recalls[pr_i + 1] - recalls[pr_i]
                    value = ((precisions[pr_i + 1] - precisions[pr_i]) / 2 +
                             precisions[pr_i])
                    aps[class_i] += delta * value
        aps.append(sum(aps) / len(aps))
        
        return aps

def read_json(path):
    input_file = open(path)
    json_array = json.load(input_file)
    input_file.close()
    return json_array

def check_pred_json(gt, pred, upper_limit = 150):
    if  np.array_equal(gt.keys(), pred.keys()):
        for i in pred:
            if len(pred[i]) > upper_limit:
                raise ValueError("One picture predicet too many boxes(>150)")
            for j in pred[i]:
                for x in j[:4]:
                    if isinstance(x, int) == False or x < 0:
                        raise ValueError("Box coordinate type error, must be int and bigger than 0!")
                if isinstance(j[4], float) == False or j[4] < 0 or j[4] > 1 or len(str(j[4])) > 7:
                    raise ValueError("Score type error, must be float and bigger than 0!")

        return True
    else:
        raise ValueError("Predict json name different with Ground truth!")

def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) 
    b: (K, 4)
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua

def get_best_conf_box(pred, best_conf = 0.5):
    best_box = {}
    for i in pred:
        best_box.setdefault(i[0], [])
        for j in i[1]:
            if j[4] >= best_conf:
                best_box[i[0]].append(j)
    return best_box
    


def get_precision_recall(gt, pred, classes = 1, conf_score = 0.05, iou_threshold = 0.5):

    # if check_pred_json(gt, pred):
        
    gt = sorted(gt.items())
    pred = sorted(pred.items())
    s = []
    for label in range(classes):    
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_gts = 0
        box_ids  = np.zeros((0,))

        for i in range(len(gt)):
            detections           = np.array(pred[i][label + 1])
            annotations          = np.array(gt[i][label + 1])
            num_P = len(annotations)

            for d in detections:
                if d[4] >= conf_score:
                    scores = np.append(scores, d[4])
                    if num_P == 0:
                        true_positives  = np.append(true_positives, 0)
                        box_ids  = np.append(box_ids, 0)
                        continue

                    overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap         = overlaps[0, assigned_annotation]
                    box_id_pred = assigned_annotation[0] + num_gts
                    box_ids  = np.append(box_ids, box_id_pred)

                    if max_overlap >= iou_threshold:
                        true_positives  = np.append(true_positives, 1)
                    else:
                        true_positives  = np.append(true_positives, 0)
            num_gts += num_P

        # sort by score
        indices         = np.argsort(-scores)
        true_positives  = true_positives[indices]
        box_ids  = box_ids[indices]
        s.append(scores[indices])
        # compute recall and precision
        precision = []
        recall = []
        f1score = []
        for det_i in range(len(true_positives)):
            obj_mask = true_positives[:det_i + 1].astype("bool")
            num_TP = len(set(box_ids[:det_i + 1][obj_mask]))
            num_dets = det_i + 1
            num_TPP = obj_mask.sum()
            num_FP = num_dets - num_TPP
            Precision = num_TP/(num_TP + num_FP)
            Recall = num_TP/num_gts
            f1 = 2 / ( (1/ Precision) + (1/ Recall) )
            # f1 = Precision + Recall
            precision.append(Precision)
            recall.append(Recall)
            f1score.append(f1)
        precision.append(0)
        recall.append(num_TP/num_gts)
        max_index = np.where(np.array(f1score) == max(f1score))[0][0]
        print('MAX F1: {}, conf_thre:{}, recall: {}, precision: {}'.format(round(max(f1score), 5),  s[0][max_index], round(recall[max_index], 5), round(precision[max_index], 5) ))
        print('recall: {}, precision: {}'.format(round(recall[-2], 5), round(precision[-2], 5)))
        PR_df = pd.DataFrame([recall, precision])
        best_box = get_best_conf_box(pred, best_conf = s[0][max_index])
        print('AP: {}'.format(PR_func(PR_df).get_map()[0]))
        return PR_func(PR_df).get_map()[0], best_box, [precision, recall], max_index

# %%
if __name__ == "__main__":
    gt = read_json('D:/OBJ/VGH_test_GT.json')
    pred = read_json('D:/OBJ/Retinanet_test_005.json')
    # pred = read_json('D:/OBJ/VGH_test_y3_005.json')
    # pred = read_json('D:/OBJ/gflv2_VGH_test_005.json')

    AP, best_box = get_precision_recall(gt, pred, conf_score = 0.05)

    print('AP:', AP)
# %%
