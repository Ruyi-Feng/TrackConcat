import copy
import cv2
from nstransformer.model.exp_test import Exp_Test
import pandas as pd
from siamvgg.model.exp_siam import Exp_Siam
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
                  "withTTC": ["date", "ttc", "longitude"], }

    def __init__(self, args, params, dir, flnm, dis_th=5, r=2.0):
        self.siam = Exp_Siam(params)
        self.nstrans = Exp_Test(args)
        self.dir = dir
        self.flnm = dir + "\\" + flnm
        self.seq_len = args.seq_len
        self.label = args.label
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
        pre = data.iloc[(data["frame"] <= frame) & (
            data["frame"] >= frame - self.seq_len)]
        self.post = data.iloc[(data["frame"] > frame) & (
            data["frame"] <= frame + self.seq_len)]
        for id in break_list:
            tmp = pre.iloc[pre["car_id"] == id]
            lane = tmp["lane"][-1]
            tmp = tmp[self.label_dict[self.label]]
            tmp_dict = copy.deepcopy(tmp.to_dict(orient='list'))
            self.breaks.update(
                {id: {"lane": lane, "history": {}, "predict": [], "candidates": {}, "select": -1}})
            self.breaks["history"].update(tmp_dict)

    def _adjust(self, dis, smlr):
        return dis * (1 - smlr / self.r)

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
        min_dis = self.dis_th
        group = self.post.iloc[self.post.iloc["lane"] == lane]
        group = group.reset_index(drop=True)
        for i in range(len(group)):
            refer_long = output[group["frame"][i] - frame]
            dis = abs(group["longitude"][i] - refer_long)
            if dis > self.dis_th:
                continue
            cands.setdefault(group["car_id"][i],
                             {"distance": self.dis_th,
                              "similarity": 0,
                              "adjust_dis": self.dis_th})
            cands[group["car_id"][i]]["distance"] = min(
                dis, cands[group["car_id"][i]])
            candi_img = cv2.imread(dir+"\\%d.jpg" %
                                   group["car_id"][i])  # 此处可以考虑改成多候选一起
            smlr = self.siam.compare_candidates([candi_img])[0]
            cands[group["car_id"][i]]["similarity"] = smlr
            adjust_dis = self._adjust(dis, smlr)
            cands[group["car_id"][i]]["adjust_dis"] = adjust_dis
            if adjust_dis < min_dis:
                select = group["car_id"][i]
                min_dis = adjust_dis
        return cands, select

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
            output = self.nstrans.test(input)
            torch.cuda.empty_cache()
            self.breaks[key]["predict"] = output.tolist()

            # --- generate image feature ---
            img_refer = cv2.imread(self.dir+"//%d.jpg" % key)
            self.siam.define_refer(img_refer)

            # --- generate candidate ---
            candidates, select = self._candi(
                frame, output, self.breaks[key]["lane"])
            self.breaks[key]["candidates"].update(candidates)
            self.breaks[key]["select"] = select

        return self.breaks
