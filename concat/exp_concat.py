"""
1. break_list input in func conca
2. prepare track csv with id renew
3. prepare img of each obj and named of car_id.jpg
"""
from concat.concat import Trackconcat
from concat.complete import Completion
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
        data = data.sort_values(by=["frame"]).reset_index(drop=True)
        last_idset = set()
        last_group = pd.DataFrame()
        break_list = [[]]
        for frame, group in data.groupby(data["frame"]):
            while len(break_list) < frame:
                break_list.append([])
            idset = set(group["car_id"].tolist())
            print("idset:", idset)
            print("last_idset:", last_idset)
            gone = last_idset - idset
            print("gone:", gone)
            breaks = []
            for k in gone:
                if last_group.loc[last_group["car_id"] == k]["longitude"].values < self.bound:
                    breaks.append(k)
                    break_list[frame-1].append(k)
            print("break_list[frame-1]", break_list[frame-1])
            last_idset = idset
            last_group = group
        return break_list

    def _refreash(self, match):
        df = pd.read_csv(self.flnm)
        for key in match:
            aim_id = match[key]["select"]
            df["car_id"].replace(aim_id, key, inplace=True)
        df.to_csv(self.flnm, index=None)

    def _fill_gap(self):
        data = pd.read_csv(self.flnm)
        data = self.clp.run(data)
        data.to_csv(self.flnm, index=None)

    def run(self):
        """
        input
        -----
        """
        break_lists = self._gener_breaklist()
        for i in range(len(break_lists) - 1):
            if len(break_lists[i]) == 0:
                continue
            print(i, break_lists[i])
            match = self.Tc.concat(i, break_lists[i])
            # print("match:", match)
            self._refreash(match)
        # 对ID已经match但有缺口的补全
        self._fill_gap()
