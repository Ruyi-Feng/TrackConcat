
import cv2
from siamvgg.model.exp_siam import Exp_Siam


def test_siamvgg():
    parameter = {"model_path": "./weights/VGG16/Epoch99.pth", "input_shape": (60, 120, 3)}
    model = Exp_Siam(parameter)

    image_1 = input('Input image_1 filename:')
    image_1 = cv2.imread(image_1)
    model.define_refer(image_1)

    while True:
        image_2 = input('Input image_2 filename:')
        try:
            image_2 = cv2.imread(image_2)
        except:
            print('Image_2 Open Error! Try again!')
            continue
        probability = model.compare_candidates([image_2])
        print(probability)
