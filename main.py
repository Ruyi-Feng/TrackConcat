from nstransformer.utils.param import param
from concat.exp_concat import Exp_concat
import numpy as np

if __name__ == '__main__':
    label = "withLeaderPos"
    dir = ".\\data\\img\\RML7\\"
    args = param(label, "MS", "", 3)
    params = {"model_path": "./weights/VGG16/Dice_Epoch100.pth", "input_shape": (60, 120, 3)}
    for r_value in np.arange(1, 2, 0.25):
        rate = 0.8
        print("rate", rate)
        flnm = "Dice-comp\\drf%.2f_r%.2f.csv"%(rate, r_value)
        concat = Exp_concat(args, params, dir, flnm, r=r_value)
        concat.run(rate)
    print("done!")

