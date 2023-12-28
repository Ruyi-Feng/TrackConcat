import copy
import cv2
from nstransformer.model.exp_nstrans import Exp_NSTrans, Exp_Linear
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from siamvgg.model.exp_siam import Exp_Siam
from nstransformer.utils.tools import visual
import torch


class Trackconcat():
    """ func
    demand
    ------
    1. break_list input in func conca
    2. prepare track csv with id renew
    3. prepare img of each obj and named of car_id.jpg

    breaks: dict
    ---
    {
        break_id: {
            "lane": 0,
            "history": {"date": [], "speed": []},
            "predict": [],
            "imagefeature": obj,
            "candidates": {
                candi_1: {"distance": 10, "similarity": 0.5, "adjust_dis": 5},
                candi_2: {"distance": 12, "similarity": 0.2, "adjust_dis": 9},}
            "select" : candi_1
        },
        break_id: {...},
    }

    根据break list id 来初始化breaks dict
    nstransformer去predict，提取obj
    找candidate， generate
    """
    label_dict = {"HeadwaySpeed": ["date", "headway", "speed", "longitude"],
                  "HeadwaySpeedAcc": ["date", "headway", "speed", "acceleration", "longitude"],
                  "TTCSpeed": ["date", "ttc", "speed", "longitude"],
                  "TTCSpeedAcc": ["date", "ttc", "speed", "acceleration", "longitude"],
                  "withAcc": ["date", "acceleration", "longitude"],
                  "withHeadway": ["date", "headway", "longitude"],
                  "withLeaderPos": ["date", "leaderpos", "longitude"],
                  "withSpeed": ["date", "speed", "longitude"],
                  "withTTC": ["date", "ttc", "longitude", ],
                  "LeaderPosSpeed": ["date", "leaderpos", "speed", "longitude"],
                  "LeaderPosSpeedAcc": ["date", "leaderpos", "speed", "acceleration", "longitude"],}


    def __init__(self, args, params, dir, flnm, dis_th=5, r=2.0):
        self.only_predict = False
        self.siam = Exp_Siam(params)
        if args.pred_type == "nsTransformer":
            self.nstrans = Exp_NSTrans(args)
        if args.pred_type == "linear":
            self.linear_pred = Exp_Linear(args)
        self.dir = dir
        self.flnm = dir + "\\" + flnm
        self.seq_len = args.seq_len
        self.label = args.label
        self.pred_type = args.pred_type
        self.dis_th = dis_th
        self.r = r

    def _init_breaks(self, frame, break_list):
        """
        用于读取breaklist中所用目标
        在frame之前96长度的序列
        初始化self.breaks并放在history里
        """
        data = pd.read_csv(self.flnm)
        data = data.sort_values(by=["frame"], ascending=True)
        data = data.reset_index(drop=True)
        pre = data.loc[(data["frame"] <= frame) & (
            data["frame"] >= frame - self.seq_len - 20)].sort_values(by="frame")
        self.post = data.loc[(data["frame"] > frame) & (
            data["frame"] <= frame + self.seq_len)].sort_values(by="frame")
        for id in break_list:
            tmp = pre.loc[pre["car_id"] == id][-self.seq_len:]
            lane = tmp["lane"].values[-1]
            img_id = tmp["gt_id"].values[-1]
            tmp = tmp[self.label_dict[self.label]]
            tmp_dict = copy.deepcopy(tmp.to_dict(orient='list'))
            self.breaks.update(
                {id: {"lane": lane, "history": {}, "predict": [], "candidates": {}, "select": -1, "img_id": img_id}})
            self.breaks[id]["history"].update(tmp_dict)

    def _adjust(self, dis, smlr):
        return dis * (1 - smlr) / self.r

    def _candi(self, frame, output, lane):
        """
        input
        -----
        frame: int
        where the track broken
        output: np.array
        the refer vehicle trace predicted
        lane: int
        which lane the refer vehilce in

        return
        -----
        candidates: dict
        as the candidates structure:
        {
            candi_1: {"distance": 10, "similarity": 0.5, "adjust_dis": 5},
            candi_2: {"distance": 12, "similarity": 0.2, "adjust_dis": 9},
        }
        select: int
        the candidates key with the shortest adjust_dis
        """
        cands = dict()
        select = -1
        min_dis = self.dis_th
        group = self.post.loc[self.post["lane"] == lane]
        group = group.reset_index(drop=True)
        for i in range(len(group)):
            if group["frame"][i] - frame >= self.seq_len:
                continue
            if self.pred_type == "use_last_pos":
                refer_long = output[0]
                dis = group["longitude"][i] - refer_long  # + (group["frame"][i] - frame) * 0.05
                if dis > self.dis_th or (group["longitude"][i] - refer_long) < 0:
                    continue
            else:
                refer_long = output[int(group["frame"][i] - frame)]
                dis = abs(group["longitude"][i] - refer_long)
                if dis > self.dis_th:
                    continue
            cands.setdefault(int(group["car_id"][i]),
                             {"distance": self.dis_th,
                              "similarity": 0,
                              "adjust_dis": self.dis_th})

            candi_img = cv2.imread(self.dir+"img\\%d.jpg" %
                                   group["gt_id"][i])  # 此处存疑2
            # 1. 可以考虑改成多候选一起, 2. gt_id 无真值测试时应是car_id,但是和图片对不上
            if self.only_predict:
                smlr = 0.5
            else:
                smlr = float(self.siam.compare_candidates([candi_img])[0])
            cands[group["car_id"][i]]["similarity"] = smlr
            adjust_dis = self._adjust(dis, smlr)
            if adjust_dis < cands[group["car_id"][i]]["adjust_dis"]:
                cands[group["car_id"][i]]["adjust_dis"] = adjust_dis
                cands[group["car_id"][i]]["distance"] = dis
            if adjust_dis < min_dis:
                select = int(group["car_id"][i])
                min_dis = adjust_dis
        return cands, select

    def _extend(self, input):
        new_input = pd.DataFrame()
        last_date = input["date"].max()
        date_seq = np.arange(last_date - self.seq_len, last_date, 1)
        dates, longitudes = input["date"], input["longitude"]
        slope, intercept = np.polyfit(dates, longitudes, 1)
        longitudes = date_seq * slope + intercept
        other_cols = set(input.columns) - set(["date", "longitude"])
        for col in other_cols:
            pre = np.random.normal(input[col].mean(), input[col].std()/2, self.seq_len - len(input))
            new_input[col] = np.append(pre, input[col])
        new_input["date"] = date_seq
        new_input["longitude"] = longitudes
        return new_input

    def _extend_input(self, input, key):
        if len(input) < 3:  # 0.2 * self.seq_len:
            self.breaks[key]["select"] = key
            return True, None
        input = self._extend(input)
        return False, input

    def add_dis_date(self, input):
        long_head = input["longitude"].values[0]
        input["date"] = np.arange(0, self.seq_len)
        input["distance"] = input["longitude"].values - long_head
        return input

    def _fill_predict(self, input, key) -> list:
        if self.pred_type == "use_last_pos":
            output = [float(input["longitude"].values[-1])]
        if self.pred_type == "nsTransformer":
            start_long = float(input["longitude"].values[0])
            label_long = input["longitude"].values.tolist()
            output = savgol_filter((self.nstrans.test(input) + start_long), 51, 3).tolist()
            # visual(label_long + output.tolist(), name=".\\data\\EXP\\%s\\%d.jpg"%(self.label, key))
        if self.pred_type == "linear":
            output = self.linear_pred.test(input["longitude"].values)
        self.breaks[key]["predict"] = output
        return output


    def concat(self, frame, break_list):
        """
        frame: int
        where the track broken
        break_list: list
        the break track id in this frame
        (frame is the last point of track)
        """
        self.breaks = dict()
        self._init_breaks(frame, break_list)
        for key in self.breaks:
            # --- generate prediction ---
            input = pd.DataFrame(self.breaks[key]["history"])
            if len(input) < self.seq_len:
                frag, input = self._extend_input(input, key)
                if frag:
                    print("# not meet seq lenth key: ", key)
                    continue
            input = self.add_dis_date(input)
            output = self._fill_predict(input, key)
            torch.cuda.empty_cache()

            # --- generate image feature ---
            if not self.only_predict:
                img_refer = cv2.imread(self.dir+"img\\%d.jpg" % self.breaks[key]["img_id"])
                self.siam.define_refer(img_refer)

            # --- generate candidate ---
            candidates, select = self._candi(
                frame, output, self.breaks[key]["lane"])
            self.breaks[key]["candidates"].update(candidates)
            self.breaks[key]["select"] = select if select > 0 else key
            print("---------------")
            print("break id", key)
            print("select id", select)
            # print("select info:", self.breaks[key]["candidates"][select])

        return self.breaks
