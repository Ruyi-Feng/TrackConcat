"""
1. break_list input in func conca
2. prepare track csv with id renew
3. prepare img of each obj and named of car_id.jpg
"""
from concat.concat import Trackconcat
from concat.complete import Completion
import json
import pandas as pd



class Exp_concat():

    def __init__(self, args, params, dir, flnm, dis_th=5, r=1.5, bound=150):
        self.Tc = Trackconcat(args, params, dir, flnm, dis_th, r)
        self.clp = Completion()
        self.flnm = dir + "\\" + flnm
        self.seq_len = args.seq_len  # 作为循环起点，保证历史数据的完整
        self.bound = bound
        self.ids = dict()
        self.new_id = 0

    def _gener_breaklist(self):
        """
        return
        ------
        break_lists: list
        contain break_list in all frames, no break frame is []
        [i, breaks] where the breaks occur on i last time
        """
        data = pd.read_csv(self.flnm)
        data = data.sort_values(by=["frame"]).reset_index(drop=True)
        last_idset = set()
        last_group = pd.DataFrame()
        break_list = [[]]
        for frame, group in data.groupby(data["frame"]):
            while len(break_list) < frame:
                break_list.append([])
            idset = set(group["car_id"].tolist())
            gone = last_idset - idset
            for k in gone:
                if last_group.loc[last_group["car_id"] == k]["longitude"].values < self.bound:
                    break_list[frame-1].append(k)
            last_idset = idset
            last_group = group
        return break_list

    def _refreash(self):
        # 需要刷新第一个是csv
        df = pd.read_csv(self.flnm)
        base = df["car_id"].max()
        for new_id in self.ids:
            for old_id in self.ids[new_id]:
                df["car_id"].replace(old_id, new_id+base, inplace=True)
        df.to_csv(self.flnm, index=None)

    def _fill_gap(self):
        data = pd.read_csv(self.flnm)
        data = self.clp.run(data)
        data.to_csv(self.flnm, index=None)

    def _reid(self, match):
        """
        collect all match and select result into one dict
        key is the new id
        elements include all matched id
        """
        for brk in match:
            found = False
            for key in self.ids:
                if brk in self.ids[key]:
                    self.ids[key].add(match[brk]["select"])
                    found = True
                    break
            if not found:
                self.new_id += 1
                self.ids.update({self.new_id: set([brk, match[brk]["select"]])})

    def _save_match(self, frame, match, rate):
        """
        frame: int
        match: dict
        {id: {"lane": lane, "history": {}, "predict": [],
            "candidates": {candi_1: {"distance": 10, "similarity": 0.5, "adjust_dis": 5}, },
            "select": -1, "img_id": gt_id}}
        """
        flnm = ".\\data\\img\\match%.2f.json"%rate
        with open(flnm, 'r') as load_f:
            info = json.load(load_f)
        for k in match:
            if len(match[k]["candidates"]) == 0:
                continue
            tmp = {"frame": frame, "ids": k, "candidates": {}, "select": int(match[k]["select"]), "gt_id": int(match[k]["img_id"])}
            tmp["candidates"].update(match[k]["candidates"])
            info.update({"%d_%d"%(frame, k): tmp})
        with open(flnm, "w") as f:
            info = json.dump(info, f)


    def run(self, rate):
        """
        input
        -----
        """
        break_lists = self._gener_breaklist()
        for i in range(len(break_lists)):
            # print("-----")
            if len(break_lists[i]) == 0:
                continue
            print(i, break_lists[i])
            match = self.Tc.concat(i, break_lists[i])
            # print("match:", match)
            self._save_match(i, match, rate)
            self._reid(match)
            print("ids:", self.ids)
        # 对ID已经match但有缺口的补全
        self._refreash()
        self._fill_gap()
