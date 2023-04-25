# 用于从图像坐标中截取每一条轨迹的图存到dir中
import pandas as pd
import cv2

def _screenshot(img, pos, id, save_dir):
    lft, top, w, h = pos
    scr = img[int(top): int(top+h), int(lft): int(lft+w)]
    cv2.imwrite(save_dir+'\\%d.jpg'%id, scr)

def shot_track(flnm, videonm, save_dir):
    """
    flnm: str
    the track of img position csv file
    save_dir: str
    img screen shot save path
    """
    data = pd.read_csv(flnm, header=None)
    data = data.sort_values(by=[0])
    data.reset_index(drop=True)
    cap = cv2.VideoCapture(videonm)
    video_frame = 0
    last_id = {}
    for frame, group in data.groupby(data[0]):
        while video_frame < frame:
            ret, img = cap.read()
            video_frame += 1
        curr_id = set(group[1].tolist())
        add = curr_id - last_id
        for id in add:
            box = group.loc[group[1] == id].values.tolist()[0]
            _screenshot(img, [box[2], box[3], box[4], box[5]], id, save_dir)
        last_id = curr_id

if __name__ == '__main__':
    csvnm = "MOTC-c.csv"
    videos = ""
    imdir = ""
    shot_track(flnm=csvnm, videonm=videos, save_dir=imdir)
