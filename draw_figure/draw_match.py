import json
import pandas as pd
import matplotlib.pyplot as plt

def generate_data(track_nm, match_nm):
    with open(match_nm, 'r') as load_f:
        info = json.load(load_f)
    # {"frame": frame, "ids": k, "candidates": {}, "select": int, "gt_id": int}
    data = pd.read_csv(track_nm)
    # print(type(data))
    new_info = dict()
    for k in info:
        # print(k)
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
    b_num = 1
    r_num = 1
    fig, ax = plt.subplots(1, 2, figsize=(6, 8))
    ax[0].set_ylim((-0.2,5.2))
    ax[1].set_ylim((-0.2,5.2))
    for i in element:
        # print(i)
        for candi in element[i]:
            ax[0].scatter([count], candi["dis"], c=candi["color"])
            ax[1].scatter([count], candi["Adis"], c=candi["color"])
            if candi["Adis"] < 0.5:
                if candi["color"]== 'b':
                    b_num += 1
                else:
                    r_num += 1
        count += 1
    print("b", b_num, "r", r_num)
    plt.show()


if __name__ == '__main__':
    rate = 0.8
    track_nm = ".\\data\\img\\RML7\\drf%.2f-ori.csv"%rate
    match_nm = ".\\data\\img\\match-comp-adjusted\\match%.2fr1.25.json"%rate
    element = generate_data(track_nm, match_nm)
    visual(element)
    # plt.plot([1344, 1383, 1449, 1509, 1563], '-^')
    # plt.plot([1-34/(1327+ 34), 1-36/(1383+36), 1-37/(1451+37), 1-65/(1509+65), 1-218/(1563+218)], '-o')
    # print([1-34/(1327+ 34), 1-36/(1383+36), 1-37/(1451+37), 1-65/(1509+65), 1-218/(1563+218)])
    # plt.show()