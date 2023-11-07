import csv
import matplotlib.pyplot as plt
import matplotlib
from nstransformer.model.exp_test import Exp_Test
from nstransformer.utils.param import param
from nstransformer.utils.tools import visual
import numpy as np
import os
import pandas as pd
from scipy.signal import savgol_filter
import torch
matplotlib.use('Qt5Agg')

def update(offset, label_len, n_p, errs_p, n_m, errs_m, mse_v2, mse_n, n, o_offset, h_offset, e_offset, m_offset):
    for j in range(len(offset)):
        if offset[j] > 0:
            n_p[j] += 1
            errs_p[j] += offset[j]
        else:
            n_m[j] += 1
            errs_m[j] += offset[j]

    mse_v2 += np.sum(np.square(offset))
    mse_n += len(offset)
    n += 1
    o_offset += np.abs(offset[0])
    h_offset += np.abs(offset[label_len])
    e_offset += np.abs(offset[-1])
    m_offset += np.max(np.abs(offset))
    return mse_v2, mse_n, n, o_offset, h_offset, e_offset, m_offset

def predict(exp, data, seq_len, dir, i):
    input = data.loc[data["date"] < seq_len]
    true_value = data.loc[(data["date"] >= seq_len) & (data["date"] < 2 * seq_len)]
    true_value = true_value[true_value.columns[-1]].values
    input = input.reset_index(drop=True)

    output = exp.test(input)
    torch.cuda.empty_cache()
    offset = true_value - savgol_filter(output, 51, 3)
    if i % 200 == 0:
        visual(true_value, output, name=dir+"//fig//%d.jpg"%i)
    return offset

def evaluate_nstransformer(label, variable):
    # root_path, data_path, features, target, var_num
    dir = "./data/EXP/"+label+"/"
    fig_dir = dir + "fig/"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    args = param(label, "MS", "", variable)

    mse_n, n, mse_v2 = 0, 0, 0
    # 起始偏移 中部偏移 末端偏移 最大偏移
    o_offset, h_offset, e_offset, m_offset = 0, 0, 0, 0
    n_p, n_m = np.zeros(args.seq_len), np.zeros(args.seq_len)
    errs_p, errs_m = np.zeros(args.seq_len), np.zeros(args.seq_len)

    Exp = Exp_Test
    exp = Exp(args)  # set experiments

    with open(dir+"test.csv",'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        input = []
        read_col = False
        i = 0
        for line in csv_reader:
            if not read_col:
                cols = line
                read_col = True
                continue
            if float(line[0]) == 0:
                i += 1
                data = pd.DataFrame(input, columns=cols).astype(float)
                input = []
                if len(data) < 2 * args.seq_len or data[cols[-1]].values[0] > 140:
                    input.append(line)
                    continue
                offset = predict(exp, data, args.seq_len, dir, i)
                mse_v2, mse_n, n, o_offset, h_offset, e_offset, m_offset = update(offset, args.label_len, n_p, errs_p, n_m, errs_m, mse_v2, mse_n, n, o_offset, h_offset, e_offset, m_offset)
            input.append(line)

    errs_p = np.nan_to_num(errs_p / n_p)
    errs_m = np.nan_to_num(errs_m / n_m)
    mse = mse_v2 / mse_n
    o_offset = o_offset / n
    h_offset = h_offset / n
    e_offset = e_offset / n
    m_offset = m_offset / n
    print("-------")
    print(label)
    print("mse_v2", mse_v2, "n", n)
    print("mse: %.4f, o_offset: %.4f, h_offset: %.4f, e_offset: %.4f, m_offset: %.4f"%(mse, o_offset, h_offset, e_offset, m_offset))
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
        lines.update({label: output})
    return lines

def subplot_a(labels, i):
    lines = typical_predict(labels, i)
    plt.figure(figsize=(4, 4), dpi=150)
    for label in lines:
        if label == "groundtruth":
            plt.plot(savgol_filter(lines[label], 51, 3), ':', label=label, c='c', linewidth=2)
            continue
        plt.plot(savgol_filter(lines[label], 51, 3), label=label)
    plt.fill_between(np.arange(len(lines["groundtruth"])), lines["groundtruth"] - 5, lines["groundtruth"] + 5, alpha=0.2)
    plt.legend()
    plt.show()


def subplot_b_csv(flnm, labels):
    data = pd.read_csv(flnm)
    i = 0
    saves = dict()
    plt.figure(figsize=(4, 4), dpi=150)
    order = ["LeaderPosSpeedAcc","TTCSpeedAcc", "TTCSpeed","withTTC", "LeaderPosSpeed", "withSpeed", "withHeadway", "HeadwaySpeed", "withLeaderPos"] #, , # ,   ]  #  ,, 
    # order = ["HeadwaySpeed"]
    num = len(labels)
    for label in order:
        if label not in labels:
            continue
        p, m = data[label+'P'].values, data[label+'M'].values
        c = plt.cm.jet(i / num)
        i += 1
        plt.fill_between(np.arange(len(p)), p, m, alpha=(0.2+0.2*i/num), label=label, facecolor=c)
    plt.plot([0, len(p)], [0, 0], c="black", linewidth=1)
    plt.plot([0, len(p)], [5, 5], c="black", linestyle="--", linewidth=0.5)
    plt.plot([0, len(p)], [-5, -5], c="black", linestyle="--", linewidth=0.5)
    plt.legend(loc="lower left")
    # plt.xlim([0, 60])
    plt.savefig("./data/img/total_error.png")
    plt.show()

if __name__ == '__main__':
    labels = {"HeadwaySpeed": 4,
              "withHeadway": 3,
              "withLeaderPos": 3,
              "withSpeed": 3,
              "LeaderPosSpeed": 4}
    # labels = {"LeaderPosSpeedAcc": 5,
    #           "HeadwaySpeed": 4,
    #           "HeadwaySpeedAcc": 5,
    #           "TTCSpeed": 4,
    #           "onlyLong": 2,
    #           "TTCSpeedAcc": 5,
    #           "withAcc": 3,
    #           "withHeadway": 3,
    #           "withLeaderPos": 3,
    #           "withSpeed": 3,
    #           "withTTC": 3,
    #           "LeaderPosSpeed": 4}

    # generate evaluation
    error_flnm = ".//data//img//total_error.csv"
    # saves = dict()
    # for label in labels:
    #     p, m = evaluate_nstransformer(label, labels[label])
    #     saves.update({label+"P": p, label+"M": m})
    # saves = pd.DataFrame(saves)
    # saves.to_csv(error_flnm)

    # create figure
    # labels = {"HeadwaySpeed": 4,
    #          "HeadwaySpeedAcc": 5,
    #          "TTCSpeed": 4,
    #          "TTCSpeedAcc": 5,
    #          "withLeaderPos": 3,
    #          "withSpeed": 3,
    #          "withTTC": 3}

    # subplot_a(labels, 108)
    subplot_b_csv(error_flnm, labels)
