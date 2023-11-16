
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


class Completion:
    """Completion
    for fill the points in broken track fragements
    """

    def __init__(self, if_ref=True):
        self.ref = if_ref

    def _fit(self, track, head, end):
        dur = pd.DataFrame(columns=track.columns)
        if end < 3:
            # 使用end的全部点
            pre = track.iloc[0: end]
            rest = 6 - end
        else:
            pre = track.iloc[end-2: end]
            rest = 3
        if head - end > 1:
            # 存在中间零碎的参考点，如果不参考的话已经被删了，所以不用判断
            dur = track.iloc[end+1: head]
        for c in range(head, head + rest):
            if c >= len(track) or track["if_fill"][c] == -1:
                rest = c - head
                break
        post = track.iloc[head: head+rest]
        time = np.arange(pre["frame"].iloc[-1]+1, post["frame"].iloc[0])
        t, cx, cy = [], [], []
        t = pre["frame"].tolist() + dur["frame"].tolist() + post["frame"].tolist()
        cx = pre["longitude"].tolist() + dur["longitude"].tolist() + post["longitude"].tolist()
        cy = pre["latitude"].tolist() + dur["latitude"].tolist() + post["latitude"].tolist()

        px, _ = curve_fit(func,
                          t,
                          cx,
                          bounds=([-3, -np.inf, -np.inf, -np.inf], [3., np.inf, np.inf, np.inf]),
                          maxfev=50000)
        py, _ = curve_fit(funcy,
                          t,
                          cy,
                          bounds=([-10, -np.inf], [10., np.inf]),
                          maxfev=50000)
        p1 = np.poly1d(px)
        x = p1(time)
        p2 = np.poly1d(py)
        y = p2(time)

        fragment = pd.DataFrame()
        # data: [frame_id, car_id, left, top, w, h, conf, lane, gt_id, if_fill]
        fragment["frame"] = time
        fragment["longitude"] = x
        fragment["latitude"] = y
        fragment["width"] = np.mean([pre["width"].iloc[-1], post["width"].iloc[0]])
        fragment["length"] = np.mean([pre["length"].iloc[-1], post["length"].iloc[0]])
        fragment["if_fill"] = 10
        col = set(track.columns)
        extra = col - {"frame", "longitude", "latitude", "width", "length", "if_fill"}
        for label in extra:
            fragment[label] = track[label][end]
        return fragment[track.columns]

    def _fill_blank(self, track):
        # filling=polyfit，则使用三次插值计算（至少4个点）
        # 一共4-6个点，断裂点前至少1个，断后+断前至少4个。
        # 控制fit后的第一阶段系数，[-3, 3]
        new_group = np.empty(shape=(0, len(track.columns)))  # new_group 中总是连续更新到最新一时刻的
        end = -1
        for i in range(len(track)):
            if track["if_fill"][i] == -1:
                end = i
            if track["if_fill"][i] == 1:
                head = i
                if end == -1:
                    continue
                frag = self._fit(track, head, end)
                new_group = np.concatenate([new_group, frag.values], axis=0)
            new_group = np.concatenate([new_group, [track.iloc[i].values]], axis=0)
        return pd.DataFrame(new_group, columns=track.columns)

    def _check_fragments(self, track):
        # 给轨迹长段头标1尾标-1，删除短的。
        h_idx, h_frm = 0, 0
        first = True
        track["if_fill"] = 0
        for index, row in track.iterrows():
            if first:
                h_idx, h_frm = index, row[0]
                n_idx, n_frm = index, row[0]
                first = False
                continue
            n_idx = index
            n_frm = row[0]
            idx_len = n_idx - h_idx
            frm_len = n_frm - h_frm
            if idx_len != frm_len:
                # 轨迹断裂了/轨迹段出现， 标记尾部/加入参考点
                if idx_len >= 3:
                    track["if_fill"][n_idx - 1] = -1
                # 标记合理段的头部
                if n_idx < len(track)-2:
                    if track["frame"][n_idx+2] - track["frame"][n_idx] == 2:
                        track["if_fill"][n_idx] = 1
                else:
                    track["if_fill"][n_idx] = 1
                h_frm = n_frm
                h_idx = n_idx
        # new_track = pd.DataFrame(new_track)
        return track.reset_index(drop=True)

    def run(self, data):
        # data: [frame_id, car_id, left, top, w, h, conf, lane, gt_id]
        intered_tracks = pd.DataFrame()
        data = data.sort_values(by=["car_id", "frame"], ascending=[True, True])
        data = data.reset_index(drop=True)
        for car_id, group in data.groupby(data["car_id"]):
            group = group.sort_values(by=["frame"])  # 按照frame排序
            group = group.reset_index(drop=True)
            # 删除短的
            if len(group) < 30:
                continue
            # 直接加入完整的
            if group["frame"][len(group)-1] - group["frame"][0] + 1 <= len(group):
                group["if_fill"] = 0
                intered_tracks = pd.concat([intered_tracks, group])
                continue
            group = self._check_fragments(group)
            # 此时group 为list，cxcy, group[8]代表是否为头尾。
            new_group = self._fill_blank(group)
            intered_tracks = pd.concat([intered_tracks, new_group])
        intered_tracks = intered_tracks.sort_values(
            by=["car_id", "frame"], ascending=[True, True])
        intered_tracks = intered_tracks.reset_index(drop=True)
        intered_tracks["car_id"] = intered_tracks["car_id"].astype('int')
        return intered_tracks



def func(x, A, B, C, D):
    return A*x**3 + B*x**2 + C*x + D


def funcy(y, A, B):
    return A*y + B
