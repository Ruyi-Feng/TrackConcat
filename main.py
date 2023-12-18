from nstransformer.utils.param import param
from concat.exp_concat import Exp_concat
import numpy as np

if __name__ == '__main__':
    label = "withLeaderPos"
    dir = "F:\\UTEwork\\KZM9\\track_experiment\\"
    args = param(label, "MS", "", 3)
    params = {"model_path": "./weights/VGG16/Dice_Epoch100.pth", "input_shape": (60, 120, 3)}
    # for r_value in [1.05, 1.10, 1.15, 1.20]:  # np.arange(0.10, 0.35, 0.1):
    #     rate = 0.8
    #     print("rate", rate)
    #     flnm = "Dice-comp\\drf%.2f_r%.2f.csv"%(rate, r_value)
    #     concat = Exp_concat(args, params, dir, flnm, r=r_value)
    #     concat.run(rate)
    # print("done! match")


    # for rate in np.arange(0.45, 0.95, 0.05):
    #     r_value = 1.00
    #     print("rate", rate)
    #     flnm = "Dice-comp\\drf%.2f.csv"%(rate)  # _r%.2f, r_value)
    #     concat = Exp_concat(args, params, dir, flnm, r=r_value)
    #     concat.run(rate)
    # print("done!")

    flnm = "packed-rml9.csv"
    concat = Exp_concat(args, params, dir, flnm, r=1.0)
    concat.run(1.0)