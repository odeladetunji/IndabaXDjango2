import argparse
import numpy as np
import tensorflow as tf
from keras.models import load_model
import params as cfg
from utils import get_img_array
from explain import *
from models import XceptiponModel
from PIL import Image
import secrets
import matplotlib.pyplot as plt

save_img_path = r"C:\Users\User\Desktop\Workspace\interpretable-breast-cancer-diagnosis\predictions"

def showCAMs(img, x, GradCAM, GuidedBP, chosen_class, upsample_size):
    cam3 = GradCAM.compute_heatmap(image=x, classIdx=chosen_class, upsample_size=upsample_size)
    gradcam = overlay_gradCAM(img, cam3)
    gradcam = cv2.cvtColor(gradcam, cv2.COLOR_BGR2RGB)
    # Guided backprop
    gb = GuidedBP.guided_backprop(x, upsample_size)
    gb_im = deprocess_image(gb)
    gb_im = cv2.cvtColor(gb_im, cv2.COLOR_BGR2RGB)
    # Guided GradCAM
    guided_gradcam = deprocess_image(gb*cam3)
    guided_gradcam = cv2.cvtColor(guided_gradcam, cv2.COLOR_BGR2RGB)

    return gradcam, gb_im, guided_gradcam

def numpy2pil(np_array: np.ndarray) -> Image:
    """
    Convert an HxWx3 numpy array into an RGB Image
    """

    assert_msg = 'Input shall be a HxWx3 ndarray'
    assert isinstance(np_array, np.ndarray), assert_msg
    assert len(np_array.shape) == 3, assert_msg
    assert np_array.shape[2] == 3, assert_msg

    img = Image.fromarray(np_array, 'RGB')
    return img


def main(image_path, model_path):
    """

    :param image_path: path to where the raw image is (.tif )
    :param model_path: path to where the model (checkpoint)  is saved (.h5)
    :return:
    """

    img_array = get_img_array(image_path,
                              pre_process=True)
    # Load classification model
    xception = XceptiponModel(input_shape=cfg.input_shape, num_classes=cfg.num_classes)
    # Load model weights
    model = xception.xception_alone()
    model.load_weights(model_path)

    # get prediction probabilities
    predictions = model.predict(img_array)
    predictions = predictions.flatten()     # (1,4) -> (4)

    pred_dict = {'Benign': predictions[0],
                 'InSitu': predictions[1],
                 'Invasive': predictions[2],
                 'Normal': predictions[3]}

    # return the predicted class
    top_pred_idx = tf.argmax(predictions)
    img_class = cfg.classnames[top_pred_idx]

    # get the Heatmap
    gradCAM = GradCAM(model=model, layerName="block14_sepconv2")
    guidedBP = GuidedBackprop(model=model, layerName="block14_sepconv2")
    image = cv2.imread(image_path)
    upsample_size = (cfg.img_height, cfg.img_width)

    # we can make use of the returning variables for further analysis
    # heatmap is a (512, 512, 3) image
    heatmap, _, _ = showCAMs(image, img_array, gradCAM, guidedBP, top_pred_idx , upsample_size)

    return predictions, pred_dict, img_class, heatmap,



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample prediction of Breast Cancer diagnosis')
    parser.add_argument('--img_path', type=str, help="Path to where the image is stored.")
    parser.add_argument('--model_path', type=str, help="Path to where the checkpoint is stored")

    args = parser.parse_args()
    predictions, pred_dict, img_class, heatmap = main(image_path=args.img_path, model_path=args.model_path)
    print(predictions)
    print(pred_dict)
    print(img_class)
    print(heatmap.shape)
