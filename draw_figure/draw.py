"""
Created on Wed Mar  9 18:41:22 2022
@author: Ruyi-Feng
"""

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from nstransformer.model.exp_test import Exp_Test
from nstransformer.utils.param import param
import torch
import numpy as np
from scipy.signal import savgol_filter

Frameindex = "frame"
Timeindex = "time"
CarIDindex = "car_id"
Laneindex = "lane_id"
Xindex = "longitude"
Yindex = "latitude"
Spdindex = "speed"
Windex = "length"
Hindex = "width"

def draw_timespace(flnm, section_length, time_range=(0,0)):
    df = pd.read_csv(flnm)
    data = df.sort_values(by=[Laneindex, CarIDindex, Frameindex],
                          axis=0, ascending=[True, True, True])
    data = data.reset_index(drop=True)

    for lane, group in data.groupby(data[Laneindex]):
        if lane != 2:
            continue
        plt.rcParams['figure.figsize'] = (3.0, 8.0)
        plt.figure()
        for car_id, track in group.groupby(group[CarIDindex]):
            plt.scatter(track[Frameindex], track[Xindex], c=list(track[Spdindex]), cmap=cm.jet_r, s=1.5, vmin=0, vmax=80)
        plt.ylim(0, section_length)
        if time_range[1] > time_range[0]:
            plt.xlim(time_range)
        plt.ylim((85, 105))
        plt.title("lane: %d"%lane)
        plt.colorbar()
    print("finish draw!")

def formation(input):
    head_date = input["date"].values[0]
    head_longitude = input["longitude"].values[0]
    input["date"] = input["date"].values - head_date
    input["distance"] = input["longitude"].values - head_longitude
    return input, head_longitude

def typical_predict(flnm, label, car_id, time_head):
    labels = {"LeaderPosSpeedAcc": {"cols": ["date", "leaderpos", "speed", "acceleration", "longitude", "distance"] , "token": 5},
            "HeadwaySpeed": {"cols": ["date", "headway", "speed", "longitude", "distance"] , "token": 4},
            "HeadwaySpeedAcc": {"cols": ["date", "headway", "speed", "acceleration", "longitude", "distance"] , "token": 5},
            "TTCSpeed": {"cols": ["date", "ttc", "speed", "longitude", "distance"] , "token": 4},
            "onlyLong": {"cols": ["date", "longitude", "distance"] , "token": 2},
            "TTCSpeedAcc": {"cols": ["date", "ttc", "speed", "acceleration", "longitude", "distance"] , "token": 5},
            "withAcc": {"cols": ["date", "acceleration", "longitude", "distance"] , "token": 3},
            "withHeadway": {"cols": ["date", "headway", "longitude", "distance"] , "token": 3},
            "withLeaderPos": {"cols": ["date", "leaderpos", "longitude", "distance"] , "token": 3},
            "withSpeed": {"cols": ["date", "speed", "longitude", "distance"] , "token": 3},
            "withTTC": {"cols": ["date", "ttc", "longitude", "distance"] , "token": 3},
            "LeaderPosSpeed": {"cols": ["date", "leaderpos", "speed", "longitude", "distance"] , "token": 4}}
    data = pd.read_csv(flnm)
    args = param(label, "MS", "", labels[label]["token"])
    Exp = Exp_Test
    exp = Exp(args)  # set experiments

    car169 = data.loc[(data["car_id"] == 169) & (data["frame"] >= 2946) & (data["frame"] < time_head+2*args.seq_len)].sort_values(by="frame")
    car169 = car169["longitude"].values
    # prepare input data
    input = data.loc[(data["car_id"] == car_id) & (data["frame"] >= time_head) & (data["frame"] < time_head+args.seq_len)].sort_values(by="frame")
    input = input.reset_index(drop=True)
    input, head_longitude = formation(input)
    input = input[labels[label]["cols"]]
    groundtruth = data.loc[(data["car_id"] == car_id) & (data["frame"] >= time_head+args.seq_len) & (data["frame"] < time_head+2*args.seq_len)].sort_values(by="frame")["longitude"]
    # predict
    output = exp.test(input) + head_longitude
    torch.cuda.empty_cache()

    return output, groundtruth.values, car169


def draw_prediction(save_path, predict, gt=None, time_head=0):
    plt.plot(np.arange(time_head, time_head+len(predict)), savgol_filter(predict, 51, 3), c="b")  # savgol_filter(predict, 31, 5)
    if gt is not None:
        plt.plot(np.arange(time_head, time_head+len(gt)), gt, c="red")
    plt.savefig(save_path+"scr.jpg")
    plt.show()

if __name__ == '__main__':
    time_range = (2700, 3200)
    section_length = 130
    flnm = "data\\EXP\\packed-distance.csv"
    fig_save_path = "G:\\track_experiment\\timespace\\"
    draw_timespace(flnm, section_length, time_range)
    predict, gt, car169 = typical_predict(flnm, "withLeaderPos", 200, 2800)
    cp200 = predict[2946-2800-96:]
    gt200 = gt[2946-2800-96:]
    dis_200 = cp200 - gt200
    dis_169 = car169 - cp200
    diff = dis_169 - dis_200
    print("car169", car169)
    print("cp200", cp200)
    print("dis_169", dis_169)
    print("---")
    print("dis_200", dis_200)
    print("diff", diff)
    draw_prediction(fig_save_path, predict, time_head=2896, gt=None)
    print("gt", gt[-1])
    print("predict", predict[-1])
    # print(predict)
"""
1:36 lane2 time_range: (2700, 3200)
plt.ylim((80, 120))
car_id = 200

"""
