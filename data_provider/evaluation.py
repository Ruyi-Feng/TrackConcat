import pandas as pd
import numpy as np

class Evaluation:

    def __init__(self, data: pd.DataFrame, gt: int, gt_ids: set=None):
        """
        Initializes an Evaluation object with the given data and ground truth.

        Args:
            data: A pandas DataFrame containing the predicted data.
            gt: The total number of ground truth data points.
        """
        self.data = data
        self.gt = gt
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.idsw = 0


        gt_ids = set(data["gt_id"]) if gt_ids is None else gt_ids
        gt_ids.discard(-1)
        for gt_id in gt_ids:
            gt_rows = data[data["gt_id"] == gt_id]
            if not gt_rows.empty:
                self.tp += len(gt_rows)
                self.idsw += len(set(gt_rows["car_id"])) - 1

        self.fp = len(data[data["gt_id"] == -1])
        self.fn = self.gt - self.tp
        self.precision = 1.0 * self.tp / (self.tp + self.fp) if self.tp + self.fp > 0 else 0
        self.recall = 1.0 * self.tp / (self.tp + self.fn) if self.tp + self.fn > 0 else 0
        self.f1_score = 2.0 * self.precision * self.recall / (self.precision + self.recall) if self.precision + self.recall > 0 else 0
        self.mota = 1.0 - (self.fn + self.fp + self.idsw) / self.gt if self.gt > 0 else 0


if __name__ == '__main__':
    for rate in np.arange(0.4, 0.45, 0.05):
        flnm = "data/img/RML7/drf%.2f.csv"%rate
        data = pd.read_csv(flnm)
        rml7 = Evaluation(data, 147844)
        print("----------", rate)
        print("mota", rml7.mota)
        print("idsw", rml7.idsw)
        print("recall", rml7.recall)
        print("f1_score", rml7.f1_score)
