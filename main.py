from nstransformer.utils.param import param
from concat.exp_concat import Exp_concat


if __name__ == '__main__':
    label = "TTCSpeedAcc"
    dir = ".\\data\\img\\RML7"
    args = param(label, "MS", "", 4)
    flnm = "drf0.60.csv"
    params = {"model_path": "./weights/VGG16/Epoch99.pth", "input_shape": (60, 120, 3)}

    concat = Exp_concat(args, params, dir, flnm)
    concat.run()
    print("done!")

