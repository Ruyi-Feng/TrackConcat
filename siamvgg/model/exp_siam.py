
import cv2
import numpy as np
from PIL import Image
from siamvgg.model.siamese import Siamese as siamese
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable


class Exp_Siam(object):
    """ steps
    1. initialize: load pth or weights
    2. upload broken img in class
    3. candidates compaire with broken img
    """
    _defaults = {
        "model_path": 'model_data/Omniglot_vgg.pth',
        "input_shape": (60, 120, 3),
        "cuda": True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, param):
        self._defaults.update(**param)
        self.__dict__.update(self._defaults)
        self._generate()

    def _generate(self):
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = siamese(self.input_shape)
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = model.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

    def _letterbox_image(self, image, size):
        ih, iw, _ = image.shape
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        image = cv2.resize(image, (nw, nh))
        shape = (h, w, 3) if self.input_shape[-1] != 1 else (h, w)
        new_image = np.zeros(shape)
        new_image.fill(128)
        new_image[(h-nh)//2: (h-nh)//2 + nh, (w-nw)//2: (w-nw)//2 + nw] = image
        return new_image

    def _img_process(self, img):
        img = self._letterbox_image(img, [self.input_shape[1], self.input_shape[0]])
        img = np.asarray(img).astype(np.float64)/255
        if self.input_shape[-1] == 1:
            img = np.expand_dims(img, -1)
        with torch.no_grad():
            img = Variable(torch.from_numpy(np.expand_dims(
                np.transpose(img, (2, 0, 1)), 0)).type(torch.FloatTensor))
            if self.cuda:
                img = img.cuda()
        return img

    def define_refer(self, img):
        self.refer_img = self._img_process(img)

    def compare_candidates(self, candi_s: list) -> list:
        similar = []
        for candi in candi_s:
            if candi is None:
                similar.append(torch.nn.Sigmoid()(0))
                continue
            tmp_img = self._img_process(candi)
            simi = self.net([self.refer_img, tmp_img])[0]
            similar.append(torch.nn.Sigmoid()(simi))
        return similar
