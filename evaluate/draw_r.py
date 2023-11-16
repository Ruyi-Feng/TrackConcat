import pandas as pd
import matplotlib.pyplot as plt


def draw_r(flnm):
    data = pd.read_csv(flnm)
    data = data.sort_values(by=["rate"], axis=0, ascending=True)
    data = data.reset_index(drop=True)

    rate = data["rate"].values
    acc = data["acc"].values
    det = data["det"].values
    comp = data["comprehensive"].values
    fig, ax = plt.subplots(2, 1, figsize=(6, 8))
    # ax[0].set_ylim((-0.2,5.2))
    # ax[1].set_ylim((-0.2,5.2))
    ax[0].plot(rate, acc, color='g', marker='^', label="confirm accuracy")
    ax[1].plot(rate, det, marker='^', label="confirm percentage")
    ax[1].plot(rate, comp, marker='o', label="comprehensive")
    ax[0].legend(loc=3)
    ax[0].grid(True)
    ax[1].legend(loc=3)
    ax[1].grid(True)
    plt.show()


if __name__ == '__main__':
    flnm = "G:\\trajectory_extract_tools\\TrackConcat\\data\\img\\RML7\\Dice-comp\\dicerate.csv"
    draw_r(flnm)