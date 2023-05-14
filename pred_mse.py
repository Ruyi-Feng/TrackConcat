import matplotlib.pyplot as plt
import matplotlib
from nstransformer.model.exp_test import Exp_Test
from nstransformer.utils.param import param
from nstransformer.utils.tools import visual
import numpy as np
import os
import pandas as pd
import torch
matplotlib.use('Qt5Agg')

def evaluate_nstransformer(label, variable):
    # root_path, data_path, features, target, var_num
    dir = "./data/EXP/"+label+"/"
    fig_dir = dir + "fig/"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    args = param(label, "MS", "", variable)
    # print('Args in experiment:')
    # print(args)

    Exp = Exp_Test
    exp = Exp(args)  # set experiments
    data = pd.read_csv(dir+"test.csv")
    mse_n = 0
    n = 0
    mse_v2 = 0
    o_offset = 0  # 起始偏移
    h_offset = 0  # 中部偏移
    e_offset = 0  # 末端偏移
    m_offset = 0  # 最大偏移
    for i in range(len(data) // (2*args.seq_len)):
        # print(i)
        input = data.iloc[2 * i * args.seq_len: (2 * i + 1) * args.seq_len]
        true_value = data.iloc[(2 * i + 1) * args.seq_len: (2 * i + 2) * args.seq_len]
        true_value = true_value[true_value.columns[-1]].values
        if input[input.columns[-1]].values[0] > 140:
            continue
        input = input.reset_index(drop=True)
        output = exp.test(input)
        torch.cuda.empty_cache()
        offset = true_value - output
        mse_v2 += np.sum(np.square(offset))
        mse_n += len(offset)
        n += 1
        o_offset += np.abs(offset[0])
        h_offset += np.abs(offset[args.label_len])
        e_offset += np.abs(offset[-1])
        m_offset += np.max(np.abs(offset))
        visual(true_value, output, name=dir+"//fig//%d.jpg"%i)
    mse = mse_v2 / mse_n
    o_offset = o_offset / n
    h_offset = h_offset / n
    e_offset = e_offset / n
    m_offset = m_offset / n
    print("-------")
    print(label)
    print("mse_v2", mse_v2, "n", n)
    print("mse: %.4f, o_offset: %.4f, h_offset: %.4f, e_offset: %.4f, m_offset: %.4f"%(mse, o_offset, h_offset, e_offset, m_offset))


def total_error(label, variable):
    dir = "./data/EXP/"+label+"/"
    fig_dir = dir + "fig/"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    args = param(label, "MS", "", variable)
    n_p = np.zeros(args.seq_len)
    n_m = np.zeros(args.seq_len)
    Exp = Exp_Test
    exp = Exp(args)  # set experiments
    data = pd.read_csv(dir+"test.csv")
    errs_p = np.zeros(args.seq_len)
    errs_m = np.zeros(args.seq_len)

    for i in range(0, len(data) // (2*args.seq_len), 100):
        # print(i)
        input = data.iloc[2 * i * args.seq_len: (2 * i + 1) * args.seq_len]
        true_value = data.iloc[(2 * i + 1) * args.seq_len: (2 * i + 2) * args.seq_len]
        true_value = true_value[true_value.columns[-1]].values
        if input[input.columns[-1]].values[0] > 140:
            continue
        input = input.reset_index(drop=True)
        output = exp.test(input)
        torch.cuda.empty_cache()
        offset = output - true_value
        for j in range(len(offset)):
            if offset[j] > 0:
                n_p[j] += 1
                errs_p[j] += offset[j]
            else:
                n_m[j] += 1
                errs_m[j] += offset[j]
    errs_p = np.nan_to_num(errs_p / n_p)
    errs_m = np.nan_to_num(errs_m / n_m)
    return errs_p, errs_m

def typical_predict(labels, i):
    lines = dict()
    for label in labels:
        dir = "./data/EXP/"+label+"/"
        fig_dir = dir + "fig/"
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        args = param(label, "MS", "", labels[label])
        # print('Args in experiment:')
        # print(args)
        Exp = Exp_Test
        exp = Exp(args)  # set experiments
        data = pd.read_csv(dir+"test.csv")
        # prepare input data
        input = data.iloc[2 * i * args.seq_len: (2 * i + 1) * args.seq_len]
        true_value = data.iloc[(2 * i + 1) * args.seq_len: (2 * i + 2) * args.seq_len]
        true_value = true_value[true_value.columns[-1]].values
        input = input.reset_index(drop=True)
        lines.update({"groundtruth": true_value})
        # predict
        output = exp.test(input)
        torch.cuda.empty_cache()
        offset = true_value - output
        lines.update({label: output})
    return lines

def subplot_a(labels, i):
    lines = typical_predict(labels, i)
    plt.figure()
    for label in lines:
        plt.plot(lines[label], label=label)
    plt.fill_between(np.arange(len(lines["groundtruth"])), lines["groundtruth"] - 3, lines["groundtruth"] + 3, alpha=0.2)
    plt.legend()
    plt.show()

def subplot_b(labels):
    i = 0
    num = len(labels)
    saves = dict()
    plt.figure()
    # order = ["withTTC", "withSpeed", "TTCSpeedAcc", "TTCSpeed", "withLeaderPos", "LeaderPosSpeed","LeaderPosSpeedAcc", "HeadwaySpeedAcc","HeadwaySpeed" ]
    order = ["HeadwaySpeed","HeadwaySpeedAcc", "LeaderPosSpeedAcc", "LeaderPosSpeed"]
    for label in order:
        p, m = total_error(label, labels[label])
        saves.update({label+"P": p, label+"M": m})
        c = plt.cm.jet(i / num)
        i += 1
        plt.fill_between(np.arange(len(p)), p, m, alpha=(0.2+0.2*i/num), label=label, facecolor=c)
    plt.plot([0, len(p)], [0, 0], c="black", linewidth=1)
    plt.plot([0, len(p)], [3, 3], c="black", linestyle="--", linewidth=0.5)
    plt.plot([0, len(p)], [-3, -3], c="black", linestyle="--", linewidth=0.5)
    plt.legend(loc="upper left")
    plt.savefig("./data/img/total_error.png")
    plt.show()
    # saves = pd.DataFrame(saves)
    # saves.to_csv("./data/img/total_error.csv")

def subplot_b_csv(flnm, labels):
    data = pd.read_csv(flnm)
    i = 0
    saves = dict()
    plt.figure()
    order = ["TTCSpeedAcc", "withLeaderPos",  "HeadwaySpeed","HeadwaySpeedAcc", "LeaderPosSpeedAcc","TTCSpeed", "LeaderPosSpeed", ]  #  ,"withTTC","withSpeed", 
    # order = ["HeadwaySpeedAcc", "LeaderPosSpeedAcc", "HeadwaySpeed","LeaderPosSpeed"]
    num = len(order)
    for label in order:
        p, m = data[label+'P'].values, data[label+'M'].values
        c = plt.cm.jet(i / num)
        i += 1
        plt.fill_between(np.arange(len(p)), p, m, alpha=(0.2+0.2*i/num), label=label, facecolor=c)
    plt.plot([0, len(p)], [0, 0], c="black", linewidth=1)
    plt.plot([0, len(p)], [3, 3], c="black", linestyle="--", linewidth=0.5)
    plt.plot([0, len(p)], [-3, -3], c="black", linestyle="--", linewidth=0.5)
    plt.legend(loc="lower left")
    plt.xlim([0, 60])
    plt.savefig("./data/img/total_error.png")
    plt.show()

if __name__ == '__main__':
    labels = {"LeaderPosSpeedAcc": 5,
              "HeadwaySpeed": 4,
              "HeadwaySpeedAcc": 5,
              "TTCSpeed": 4,
              "TTCSpeedAcc": 5,
              "withAcc": 3,
              "withHeadway": 3,
              "withLeaderPos": 3,
              "withSpeed": 3,
              "withTTC": 3,
              "LeaderPosSpeed": 4}

    # generate evaluation
    for label in labels:
        evaluate_nstransformer(label, labels[label])

    # create figure
    # labels = {"HeadwaySpeed": 4,
    #          "HeadwaySpeedAcc": 5,
    #          "TTCSpeed": 4,
    #          "TTCSpeedAcc": 5,
    #          "withLeaderPos": 3,
    #          "withSpeed": 3,
    #          "withTTC": 3}

    # subplot_a(labels, 89)
    # flnm = ".//data//img//total_error.csv"
    # subplot_b_csv(flnm, labels)
    subplot_b(labels)
