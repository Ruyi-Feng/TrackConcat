from concat.concat import Trackconcat
from nstransformer.utils.param import param


def test_concat():
    label = "TTCSpeedAcc"
    dir = "./data/EXP/"+label+"/"
    flnm = "packed.csv"
    args = param(label, "MS", "", 4)
    params = {"model_path": "./weights/VGG16/Epoch99.pth", "input_shape": (60, 120, 3)}
    tc = Trackconcat(args, params, dir, flnm)
    breaks = tc.concat(96, [])
