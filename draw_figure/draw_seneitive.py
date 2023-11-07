import matplotlib.pyplot as plt
import pandas as pd

def draw_sensitive(flnm):
    result = pd.read_csv(flnm)
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), dpi=150)
    ax = ax.flatten()
    result = result.sort_values(by="if_ori")
    dic = {0: "original", 1: "completed", 2: "without similarity", 3: "TD Tree"}
    color = {1: "r", 0: "b", 2: "c", 3: "g"}
    for if_ori, data in result.groupby(result["if_ori"]):
        data = data.sort_values(by="detect_rate")
        cols =  data.columns
        for i in range(1, 5):
            ax[i-1].plot(data[cols[0]], data[cols[i]], marker='^', label=dic[if_ori], c=color[if_ori])
            # ax[i-1].title()
            ax[i-1].grid(True)
            ax[i-1].legend()
            if i == 0:
                ax[i-1].set_ylim((-1, 1))
            elif i != 2:
                ax[i-1].set_ylim((0, 1))
            else:
                ax[i-1].set_ylim((0, 1500))
    plt.show()

if __name__ == '__main__':
    flnm = ".\\data\\img\\mota_result.csv"
    draw_sensitive(flnm)