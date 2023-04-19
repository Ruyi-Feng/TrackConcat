import csv
import matplotlib.pyplot as plt

plt.switch_backend('agg')


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    # flnm = "./test_results/withAcc/train.csv"
    # csvFile = open(flnm, "a+", newline='')
    # writer = csv.writer(csvFile)
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
        # writer.writerow(preds)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    # writer.writerow(true)
    # csvFile.close()
