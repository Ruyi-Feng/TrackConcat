from nstransformer.model.exp_test import Exp_Test
from nstransformer.utils.param import param
from nstransformer.utils.tools import visual
import numpy as np
import pandas as pd
import torch

def test_nstransformer():
    label = "TTCSpeedAcc"
    # root_path, data_path, features, target, var_num
    dir = "./data/EXP/"+label+"/"
    args = param(label, "MS", "", 4)
    print('Args in experiment:')
    print(args)

    Exp = Exp_Test
    exp = Exp(args)  # set experiments
    data = pd.read_csv(dir+"test.csv")
    for i in range(len(data) % args.seq_len):
        print(i)
        input = data.iloc[i * args.seq_len: (i+1) * args.seq_len]
        input = input.reset_index(drop=True)
        output = exp.test(input)
        torch.cuda.empty_cache()
        tgt = input.columns[-1]
        track = np.concatenate((input[tgt], output), axis=0)
        visual(track, name=dir+"//fig//%d.jpg"%i)
