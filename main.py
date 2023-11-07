from nstransformer.utils.param import param
from concat.exp_concat import Exp_concat
import numpy as np

if __name__ == '__main__':
    label = "withLeaderPos"
    dir = ".\\data\\img\\RML7"
    args = param(label, "MS", "", 3)
    params = {"model_path": "./weights/VGG16/Dice_Epoch100.pth", "input_shape": (60, 120, 3)}
    for rate in np.arange(0.4, 0.95, 0.05):
        print("rate", rate)
        flnm = "drf%.2f.csv"%rate
        concat = Exp_concat(args, params, dir, flnm, r=1.5)
        concat.run(rate)
    print("done!")

