import json
import pandas as pd
import matplotlib.pyplot as plt

def generate_data(track_nm, match_nm):
    with open(match_nm, 'r') as load_f:
        info = json.load(load_f)
    # {"frame": frame, "ids": k, "candidates": {}, "select": int, "gt_id": int}
    data = pd.read_csv(track_nm)
    print(type(data))
    new_info = dict()
    for k in info:
        print(k)
        new_info[k] = []
        for candi in info[k]["candidates"]:
            candi_gt = data.loc[data["car_id"] == int(candi)]["gt_id"].values[0]
            if int(candi_gt) == int(info[k]["gt_id"]):
                color = 'b'
            else:
                color = 'r'
            tmp = {"dis": info[k]["candidates"][candi]["distance"],
                   "Adis": info[k]["candidates"][candi]["adjust_dis"],
                   "color": color}
            new_info[k].append(tmp)

        # candidate: dis, adis, color
    return new_info


def visual(element):
    print("start visualize")
    count = 1
    fig, ax = plt.subplots(1, 2, figsize=(6, 8))
    for i in element:
        print(i)
        for candi in element[i]:
            ax[0].scatter([count], candi["dis"], c=candi["color"])
            # s1 = 3 if candi["color"] == 'b' else 1
            ax[1].scatter([count], candi["Adis"], c=candi["color"])
        count += 1
    plt.show()


if __name__ == '__main__':
    rate = 0.75
    track_nm = ".\\data\\img\\RML7\\drf%.2f-ori.csv"%rate
    match_nm = ".\\data\\img\\match%.2f.json"%rate
    element = generate_data(track_nm, match_nm)
    visual(element)
