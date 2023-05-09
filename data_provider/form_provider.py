# 把轨迹排成platoon后计算车道gap

import copy
import pandas as pd
import numpy as np
import os




def add_headway(flnm):
    data = pd.read_csv(flnm)
    data["date"] = data["car_id"] * 24 * 60 + data["frame"]
    data = data.sort_values(by=["lane_id", "frame", "longitude"], ascending=[True, True, False])
    data = data.reset_index(drop=True)
    new_data = []
    for lane, track in data.groupby(data["lane_id"]):
        for frame, pos in track.groupby(track["frame"]):
            lastrow = pd.DataFrame()
            for index, row in pos.iterrows():
                if len(lastrow) == 0:
                    lastrow = copy.deepcopy(row)
                    new_data.append(row.values.tolist() + [0, 0])
                    continue
                headway = lastrow["longitude"] - row["longitude"]
                new_data.append(row.values.tolist() + [headway, lastrow["longitude"]])
                lastrow = copy.deepcopy(row)
    new_data = pd.DataFrame(new_data, columns=["frame", "car_id","lane_id", "time","latitude","longitude", "speed", "acceleration", "length", "width", "date", "headway", "leaderpos"])
    new_data["ttc"] = new_data["headway"]/new_data["speed"]
    return new_data

def batch_code(new_data, batch, min_date):
    new_data = new_data.sort_values(by=["car_id", "frame"], ascending=[True, True])
    new_data.reset_index(drop=True)
    new_data2 = []
    max_len = 0
    for car_id, track in new_data.groupby(new_data["car_id"]):
        resu = len(track) % batch
        new_data2 = new_data2 + track.values.tolist()[:-resu]
        max_len = max(len(track.values.tolist()[:-resu]), max_len)
    new_data2 = pd.DataFrame(new_data2, columns=["frame", "car_id","lane_id", "time","latitude","longitude", "speed", "acceleration", "length", "width", "date", "headway", "leaderpos", "ttc"])
    new_data2["date"] = max_len * new_data2["car_id"] + new_data2["frame"] + min_date
    return new_data2

def devide_files(packed, files, outflnm, mark):
    for key in files:
        df =  packed[files[key]]
        if os.path.exists(outflnm + key + "\\" + mark + ".csv"):
            df.to_csv(outflnm + key + "\\" + mark + ".csv", index=None, mode='a', header=None)
        else:
            df.to_csv(outflnm + key + "\\" + mark + ".csv", index=None)

def devide_lane(packed, train_setting):
    train = packed[(packed["lane_id"] == train_setting["train"][0]) | \
        (packed["lane_id"] == train_setting["train"][1]) | (packed["lane_id"] == train_setting["train"][2])] # | \
            # (packed["lane_id"] == train_setting["train"][3]) | (packed["lane_id"] == train_setting["train"][4])]
    train.reset_index(drop=True)
    test = packed[(packed["lane_id"] == train_setting["test"][0])]  #  | (packed["lane_id"] == train_setting["test"][1])]
    test.reset_index(drop=True)
    return train, test


if __name__ == '__main__':
    flnm = ".\\data\\img\\track_raw.csv"
    outflnm = ".\\data\\img\\RML7\\"
    batch = 192
    min_date = 0  # 790560
    ### 1
    # packed = add_headway(flnm)
    # packed.to_csv(outflnm+"packed-rml7.csv", index=None, mode='a')
    print("finish pack!")
    ### 2 筛选车道和分出不同的参数
    packed = pd.read_csv(outflnm + "packed-rml7.csv")
    train_setting = {"train": [2, 3, 4], "test": [1]}  # RML7
    # train_setting = {"train": [2, 3, 7, 8, 9], "test": [6, 7]}  # KZM6

    train, test = devide_lane(packed, train_setting)
    test.to_csv(outflnm+"packed.csv", index=None)
    train = batch_code(train, batch, min_date)
    test = batch_code(test, batch, min_date)

    # files = {"HeadwaySpeed": ["date", "headway", "speed", "longitude"],
    #          "HeadwaySpeedAcc": ["date", "headway", "speed", "acceleration", "longitude"],
    #          "TTCSpeed": ["date", "ttc", "speed", "longitude"],
    #          "TTCSpeedAcc": ["date", "ttc", "speed", "acceleration", "longitude"],
    #          "withAcc": ["date", "acceleration", "longitude"],
    #          "withHeadway": ["date", "headway", "longitude"],
    #          "withLeaderPos": ["date", "leaderpos", "longitude"],
    #          "withSpeed": ["date", "speed", "longitude"],
    #          "withTTC": ["date", "ttc", "longitude"],}
    # devide_files(train, files, outflnm, "train")
    # devide_files(test, files, outflnm, "test")

    files = {"LeaderPosSpeed": ["date", "leaderpos", "speed", "longitude"],
            "LeaderPosSpeedAcc": ["date", "leaderpos", "speed", "acceleration", "longitude"],}
    devide_files(train, files, outflnm, "train")
    devide_files(test, files, outflnm, "test")