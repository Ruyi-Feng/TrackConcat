import matplotlib.pyplot as plt
import pandas as pd

def draw_sensitive(flnm):
    result = pd.read_csv(flnm)
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), dpi=150)
    ax = ax.flatten()
    result = result.sort_values(by="if_ori")
    dic = {1: "original", 0: "completed"}
    color = {1: "r", 0: "b"}
    for if_ori, data in result.groupby(result["if_ori"]):
        cols =  data.columns
        for i in range(1, len(cols)-1):
            ax[i-1].plot(data[cols[0]], data[cols[i]], marker='^', label=dic[if_ori], c=color[if_ori])
            # ax[i-1].title()
            ax[i-1].grid(True)
            ax[i-1].legend()
            if i != 2:
                ax[i-1].set_ylim((0, 1))
            else:
                ax[i-1].set_ylim((0, 1500))
    plt.show()

if __name__ == '__main__':
    flnm = ".\\data\\img\\mota_result.csv"
    draw_sensitive(flnm)