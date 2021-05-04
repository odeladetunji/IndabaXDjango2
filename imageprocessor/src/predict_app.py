import argparse
import numpy as np
import tensorflow as tf
from keras.models import load_model
import params as cfg
from explain import *
from .models import XceptiponModel
import uuid
import matplotlib.pyplot as plt


import cv2
from PIL import Image
import utils as utility

import numpy as np
import itertools
import matplotlib.pyplot as plt

from scipy import ndimage
import keras.preprocessing as kas

from .processing import normalize_staining, hematoxylin_eosin_aug

from tensorflow.keras.models import Model
from tensorflow.python.framework import ops
from tensorflow.keras import backend as K

from django.core.files.storage import FileSystemStorage

# get_img_array

# save_img_path = r"C:\Users\User\Desktop\Workspace\interpretable-breast-cancer-diagnosis\predictions"
save_img_path = "static"
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
    unique_id = uuid.uuid4().hex
    img = Image.fromarray(np_array, 'RGB')
    filename =  save_img_path + "/{}.png".format(unique_id) #string interpolation
    img.save(filename)

    # fs = FileSystemStorage()
    # filename1 = fs.save(filename, filename)
    # uploaded_file_url = fs.url(filename1)
    # shutil.move(filename, "./static/" + filename)

    return filename


def main(image_path, model_path):
    """

    :param image_path: path to where the raw image is (.tif )
    :param model_path: path to where the model (checkpoint)  is saved (.h5)
    :return:
    """

    img_array = get_img_array(image_path,
                              pre_process=True)
                   
    # Load classification model
    xception = XceptiponModel(input_shape=input_shape, num_classes=num_classes)
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
    img_class = classnames[top_pred_idx]

    # get the Heatmap
    gradCAM = GradCAM(model=model, layerName="block14_sepconv2")
    guidedBP = GuidedBackprop(model=model, layerName="block14_sepconv2")
    image = cv2.imread(image_path)
    upsample_size = (img_height, img_width)

    # we can make use of the returning variables for further analysis
    # heatmap is a (512, 512, 3) image
    heatmap, _, _ = showCAMs(image, img_array, gradCAM, guidedBP, top_pred_idx , upsample_size)
    file = numpy2pil(heatmap)
    dictionary_result = { "payload": [
            {
                "predictions": {
                     'Normal': predictions[3],
                     'Benign': predictions[0],
                     'Invasive': predictions[2],
                     'InSitu': predictions[1],
                },

               "nameOfHeatMap": file
            }
    ]}
    print(dictionary_result)
    return dictionary_result

def get_img_array(img_path, size=(512, 512), pre_process=False):
    # `img` is a PIL image of size 299x299
    img = kas.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    img_array = kas.image.img_to_array(img)

    if pre_process:
        img_norm = normalize_staining(img_array)
        # process H&E
        img_array = hematoxylin_eosin_aug(img_norm)

    # We add a dimension to transform our array into a "batch"
    # of size (1, 512, 512, 3)
    array = np.expand_dims(img_array, axis=0)
    return array


class GradVisualizer:
    """Plot gradients of the outputs w.r.t an input image."""

    def __init__(self, positive_channel=None, negative_channel=None):
        if positive_channel is None:
            self.positive_channel = [0, 255, 0]
        else:
            self.positive_channel = positive_channel

        if negative_channel is None:
            self.negative_channel = [255, 0, 0]
        else:
            self.negative_channel = negative_channel

    def apply_polarity(self, attributions, polarity):
        if polarity == "positive":
            return np.clip(attributions, 0, 1)
        else:
            return np.clip(attributions, -1, 0)

    def apply_linear_transformation(
        self,
        attributions,
        clip_above_percentile=99.9,
        clip_below_percentile=70.0,
        lower_end=0.2,
    ):
        # 1. Get the thresholds
        m = self.get_thresholded_attributions(
            attributions, percentage=100 - clip_above_percentile
        )
        e = self.get_thresholded_attributions(
            attributions, percentage=100 - clip_below_percentile
        )

        # 2. Transform the attributions by a linear function f(x) = a*x + b such that
        # f(m) = 1.0 and f(e) = lower_end
        transformed_attributions = (1 - lower_end) * (np.abs(attributions) - e) / (
            m - e
        ) + lower_end

        # 3. Make sure that the sign of transformed attributions is the same as original attributions
        transformed_attributions *= np.sign(attributions)

        # 4. Only keep values that are bigger than the lower_end
        transformed_attributions *= transformed_attributions >= lower_end

        # 5. Clip values and return
        transformed_attributions = np.clip(transformed_attributions, 0.0, 1.0)
        return transformed_attributions

    def get_thresholded_attributions(self, attributions, percentage):
        if percentage == 100.0:
            return np.min(attributions)

        # 1. Flatten the attributions
        flatten_attr = attributions.flatten()

        # 2. Get the sum of the attributions
        total = np.sum(flatten_attr)

        # 3. Sort the attributions from largest to smallest.
        sorted_attributions = np.sort(np.abs(flatten_attr))[::-1]

        # 4. Calculate the percentage of the total sum that each attribution
        # and the values about it contribute.
        cum_sum = 100.0 * np.cumsum(sorted_attributions) / total

        # 5. Threshold the attributions by the percentage
        indices_to_consider = np.where(cum_sum >= percentage)[0][0]

        # 6. Select the desired attributions and return
        attributions = sorted_attributions[indices_to_consider]
        return attributions

    def binarize(self, attributions, threshold=0.001):
        return attributions > threshold

    def morphological_cleanup_fn(self, attributions, structure=np.ones((4, 4))):
        closed = ndimage.grey_closing(attributions, structure=structure)
        opened = ndimage.grey_opening(closed, structure=structure)
        return opened

    def draw_outlines(
        self, attributions, percentage=90, connected_component_structure=np.ones((3, 3))
    ):
        # 1. Binarize the attributions.
        attributions = self.binarize(attributions)

        # 2. Fill the gaps
        attributions = ndimage.binary_fill_holes(attributions)

        # 3. Compute connected components
        connected_components, num_comp = ndimage.measurements.label(
            attributions, structure=connected_component_structure
        )

        # 4. Sum up the attributions for each component
        total = np.sum(attributions[connected_components > 0])
        component_sums = []
        for comp in range(1, num_comp + 1):
            mask = connected_components == comp
            component_sum = np.sum(attributions[mask])
            component_sums.append((component_sum, mask))

        # 5. Compute the percentage of top components to keep
        sorted_sums_and_masks = sorted(component_sums, key=lambda x: x[0], reverse=True)
        sorted_sums = list(zip(*sorted_sums_and_masks))[0]
        cumulative_sorted_sums = np.cumsum(sorted_sums)
        cutoff_threshold = percentage * total / 100
        cutoff_idx = np.where(cumulative_sorted_sums >= cutoff_threshold)[0][0]
        if cutoff_idx > 2:
            cutoff_idx = 2

        # 6. Set the values for the kept components
        border_mask = np.zeros_like(attributions)
        for i in range(cutoff_idx + 1):
            border_mask[sorted_sums_and_masks[i][1]] = 1

        # 7. Make the mask hollow and show only the border
        eroded_mask = ndimage.binary_erosion(border_mask, iterations=1)
        border_mask[eroded_mask] = 0

        # 8. Return the outlined mask
        return border_mask

    def process_grads(
        self,
        image,
        attributions,
        polarity="positive",
        clip_above_percentile=99.9,
        clip_below_percentile=0,
        morphological_cleanup=False,
        structure=np.ones((3, 3)),
        outlines=False,
        outlines_component_percentage=90,
        overlay=True,
    ):
        if polarity not in ["positive", "negative"]:
            raise ValueError(
                f""" Allowed polarity values: 'positive' or 'negative'
                                    but provided {polarity}"""
            )
        if clip_above_percentile < 0 or clip_above_percentile > 100:
            raise ValueError("clip_above_percentile must be in [0, 100]")

        if clip_below_percentile < 0 or clip_below_percentile > 100:
            raise ValueError("clip_below_percentile must be in [0, 100]")

        # 1. Apply polarity
        if polarity == "positive":
            attributions = self.apply_polarity(attributions, polarity=polarity)
            channel = self.positive_channel
        else:
            attributions = self.apply_polarity(attributions, polarity=polarity)
            attributions = np.abs(attributions)
            channel = self.negative_channel

        # 2. Take average over the channels
        attributions = np.average(attributions, axis=2)

        # 3. Apply linear transformation to the attributions
        attributions = self.apply_linear_transformation(
            attributions,
            clip_above_percentile=clip_above_percentile,
            clip_below_percentile=clip_below_percentile,
            lower_end=0.0,
        )

        # 4. Cleanup
        if morphological_cleanup:
            attributions = self.morphological_cleanup_fn(
                attributions, structure=structure
            )
        # 5. Draw the outlines
        if outlines:
            attributions = self.draw_outlines(
                attributions, percentage=outlines_component_percentage
            )

        # 6. Expand the channel axis and convert to RGB
        attributions = np.expand_dims(attributions, 2) * channel

        # 7.Superimpose on the original image
        if overlay:
            attributions = np.clip((attributions * 0.8 + image), 0, 255)
        return attributions

    def visualize(
        self,
        image,
        gradients,
        integrated_gradients,
        polarity="positive",
        clip_above_percentile=99.9,
        clip_below_percentile=0,
        morphological_cleanup=False,
        structure=np.ones((3, 3)),
        outlines=False,
        outlines_component_percentage=90,
        overlay=True,
        figsize=(15, 8),
    ):
        # 1. Make two copies of the original image
        img1 = np.copy(image)
        img2 = np.copy(image)

        # 2. Process the normal gradients
        grads_attr = self.process_grads(
            image=img1,
            attributions=gradients,
            polarity=polarity,
            clip_above_percentile=clip_above_percentile,
            clip_below_percentile=clip_below_percentile,
            morphological_cleanup=morphological_cleanup,
            structure=structure,
            outlines=outlines,
            outlines_component_percentage=outlines_component_percentage,
            overlay=overlay,
        )

        # 3. Process the integrated gradients
        igrads_attr = self.process_grads(
            image=img2,
            attributions=integrated_gradients,
            polarity=polarity,
            clip_above_percentile=clip_above_percentile,
            clip_below_percentile=clip_below_percentile,
            morphological_cleanup=morphological_cleanup,
            structure=structure,
            outlines=outlines,
            outlines_component_percentage=outlines_component_percentage,
            overlay=overlay,
        )

        _, ax = plt.subplots(1, 3, figsize=figsize)
        ax[0].imshow(image)
        ax[1].imshow(grads_attr.astype(np.uint8))
        ax[2].imshow(igrads_attr.astype(np.uint8))

        ax[0].set_title("Input")
        ax[1].set_title("Normal gradients")
        ax[2].set_title("Integrated gradients")
        plt.show()


from keras import backend as K


def recall(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (all_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * ((prec * rec) / (prec + rec + K.epsilon()))


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',
                          cmap=plt.cm.Blues):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion Matrix')

    print(cm)
#     fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))    

class GradCAM:
    # Adapted with some modification from https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
    def __init__(self, model, layerName=None):
        """
        model: pre-softmax layer (logit layer)
        """
        self.model = model
        self.layerName = layerName

        if self.layerName == None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM")

    def compute_heatmap(self, image, classIdx, upsample_size, eps=1e-5):
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output]
        )
        # record operations for automatic differentiation

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOuts, preds) = gradModel(inputs)  # preds after softmax
            loss = preds[:, classIdx]

        # compute gradients with automatic differentiation
        grads = tape.gradient(loss, convOuts)
        # discard batch
        convOuts = convOuts[0]
        grads = grads[0]
        norm_grads = tf.divide(grads, tf.reduce_mean(tf.square(grads)) + tf.constant(eps))

        # compute weights
        weights = tf.reduce_mean(norm_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOuts), axis=-1)

        # Apply reLU
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)
        cam = cv2.resize(cam, upsample_size,cv2.INTER_LINEAR)

        # convert to 3D
        cam3 = np.expand_dims(cam, axis=2)
        cam3 = np.tile(cam3, [1, 1, 3])

        return cam3


def overlay_gradCAM(img, cam3):
    cam3 = np.uint8(255 * cam3)
    cam3 = cv2.applyColorMap(cam3, cv2.COLORMAP_JET)

    new_img = 0.3 * cam3 + 0.5 * img

    return (new_img * 255.0 / new_img.max()).astype("uint8")


@tf.custom_gradient
def guidedRelu(x):
    def grad(dy):
        return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
    return tf.nn.relu(x), grad



@tf.custom_gradient
def guidedRelu(x):
    def grad(dy):
        return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
    return tf.nn.relu(x), grad


# Reference: https://github.com/eclique/keras-gradcam with adaption to tensorflow 2.0  
class GuidedBackprop:
    def __init__(self,model, layerName=None):
        self.model = model
        self.layerName = layerName
        if self.layerName == None:
            self.layerName = self.find_target_layer()
        self.gbModel = self.build_guided_model()
        
    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply Guided Backpropagation")

    def build_guided_model(self):
        gbModel = Model(
            inputs = [self.model.inputs],
            outputs = [self.model.get_layer(self.layerName).output]
        )
        layer_dict = [layer for layer in gbModel.layers[1:] if hasattr(layer,"activation")]
        for layer in layer_dict:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = guidedRelu
        
        return gbModel
    
    def guided_backprop(self, images, upsample_size):
        """Guided Backpropagation method for visualizing input saliency."""
        with tf.GradientTape() as tape:
            inputs = tf.cast(images, tf.float32)
            tape.watch(inputs)
            outputs = self.gbModel(inputs)

        grads = tape.gradient(outputs, inputs)[0]

        saliency = cv2.resize(np.asarray(grads), upsample_size)

        return saliency


def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    # normalize tensor: center on 0., ensure std is 0.25
    x = x.copy()
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x
 


""" Configuration file for an experiment
"""

random_seed = 24
train_dir = 'dataset/train/'
test_dir = "dataset/test/"

model = "xception_alone"

classnames = ['Benign', 'InSitu', 'Invasive', 'Normal']

features_dir = "./extracted_features/"
train_features = "./extracted_features/features_train.npy"
valid_features = "./extracted_features/features_validate.npy"
test_features = "./extracted_features/features_test.npy"

checkpoint_name = "./checkpoints/checkpoint.h5"
model_path = checkpoint_name +'.h5'
result_file = "history.txt"
train_batch_size = 8
val_batch_size = 8
test_batch_size = 4
img_height, img_width = 512, 512
input_shape = (img_height, img_width, 3)
epochs = 30

num_classes = 4
lr = 0.001
beta_1 = 0.6
beta_2 = 0.8

save_feature = True
dropout_rate = 0.5

"""
Example of how to use main to predict
"""
# img_path = r"C:\Users\User\Desktop\Workspace\interpretable-breast-cancer-diagnosis\sample_images\benign3.tif"
# model_path= r"C:\Users\User\Desktop\Workspace\interpretable-breast-cancer-diagnosis\checkpoints\final_model.h5"

# img_path = "\sample_images\benign3.tif"
# model_path= "\checkpoints\final_model.h5"
# prediction_result = main(img_path, model_path)
# print(prediction_result)
