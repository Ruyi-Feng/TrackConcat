"""
1. break_list input in func conca
2. prepare track csv with id renew
3. prepare img of each obj and named of car_id.jpg
"""
from concat.concat import Trackconcat
from complete import Completion
import pandas as pd



class Exp_concat():

    def __init__(self, args, params, dir, flnm, dis_th=5, r=2.0, bound=150):
        self.Tc = Trackconcat(args, params, dir, flnm, dis_th, r)
        self.clp = Completion()
        self.flnm = dir + "\\" + flnm
        self.seq_len = args.seq_len  # 作为循环起点，保证历史数据的完整
        self.bound = bound

    def _gener_breaklist(self):
        """
        return
        ------
        break_lists: list
        contain break_list in all frames, no break frame is []
        """
        data = pd.read_csv(self.flnm)
        data = data.sort_values(by=["frames", ]).reset_index(drop=True)
        last_idset = {}
        break_list = []
        for frame, group in data.groupby(data["frame"]):
            idset = set(group["car_id"].tolist())
            gone = idset - last_idset
            breaks = []
            for k in gone:
                if group.iloc[group["car_id"] == k]["longitude"] < self.bound:
                    breaks.append(k)
            last_idset = idset
            break_list.append(breaks)
        return break_list

    def _refreash(self, match):
        df = pd.read_csv(self.flnm)
        for key in match:
            aim_id = match[key]["select"]
            df["car_id"].replace(aim_id, key, inplace=True)
        df.to_csv(self.flnm, index=None)

    def _fill_gap(self):
        data = pd.read_csv(self.flnm)
        data = self.clp(data)
        data.to_csv(self.flnm, index=None)

    def run(self):
        """
        input
        -----
        """
        break_lists = self._gener_breaklist()
        for i in range(len(break_lists)):
            if len(break_lists[i]) == 0:
                continue
            match = self.Tc.concat(i, break_lists[i])
            self._refreash(match)
        # 对ID已经match但有缺口的补全
        self._fill_gap()
