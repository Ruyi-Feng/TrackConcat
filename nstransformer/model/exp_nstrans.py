from nstransformer.utils.test_loader import Dataset_Pred
from nstransformer.model.exp_basic import Exp_Basic
from nstransformer.model import ns_Transformer

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os

import warnings

warnings.filterwarnings('ignore')


class Exp_NSTrans(Exp_Basic):
    def __init__(self, args):
        super(Exp_NSTrans, self).__init__(args)
        self.timeenc = 0 if self.args.embed != 'timeF' else 1
        self.label = args.label
        print('loading model')
        pretrained_dict = torch.load(os.path.join(
            'weights', self.label, 'checkpoint.pth'), map_location='cuda:0')
        pretrained_dict = {key.replace(
            "module.", ""): value for key, value in pretrained_dict.items()}
        self.model.load_state_dict(pretrained_dict)
        self.dataset_prd = Dataset_Pred(flag='pred', size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
                                        features=self.args.features, target=self.args.target, timeenc=self.timeenc, freq=self.args.freq)

    def _build_model(self):
        model_dict = {
            'ns_Transformer': ns_Transformer,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self):
        self.dataset_prd.confer_data(self.df)
        data_loader = DataLoader(
            self.dataset_prd,
            batch_size=1,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=False)
        return data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _standardize(self, df):
        self.sca_min, self.sca_max, self.mean, self.var = np.load("./weights/"+self.label+"/inverse.npy", allow_pickle=True)
        col = df.columns
        df[col[0]] = (df[col[0]] - self.sca_min[0]) / (self.sca_max[0] - self.sca_min[0])
        for i in range(1, len(col)):
            df[col[i]] = (df[col[i]] - self.sca_min[i]) / (self.sca_max[i] - self.sca_min[i])
            df[col[i]] = (df[col[i]] - self.mean[i-1]) / self.var[i-1]
        return df

    def _de_standardize(self, pred):
        pred = pred * self.var[-1] + self.mean[-1]
        pred = pred * (self.sca_max[-1] - self.sca_min[-1]) + self.sca_min[-1]
        return pred


    def test(self, df):
        # 实现的是一条一条的预测。返回预测值。
        self.df = self._standardize(df)  # 考虑如何组合，使得多条轨迹一起被load上来，一起推理
        test_loader = self._get_data()
        preds = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(
                    batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                outputs = outputs.detach().cpu().numpy()
                # outputs.detach().cpu().numpy()  # .squeeze()
                pred = outputs.reshape(-1, self.args.enc_in)[:, f_dim:].reshape(1, 96)
                pred = self._de_standardize(pred)
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1)
        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        return preds

class Exp_Linear:
    def __init__(self, args):
        self.args = args

    def _fit(self, k, b, l):
        y_head = (l + 1) * k + b
        y_end = (l + self.args.pred_len) * k + b
        return np.linspace(y_head, y_end, self.args.pred_len)

    def test(self, seq):
        """test func

        seq: np.array
        需要预测的输入数据
        """
        if len(seq) < 1:
            return seq[0] * np.ones((self.args.pred_len))
        b = seq[0]
        k = (seq[-1] - seq[0]) / len(seq)
        output = self._fit(k, b, len(seq))
        return output.tolist()

