# """ Class for Interpretable methods implementation
#     Methods:
#         
# """

import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.python.framework import ops
from tensorflow.keras import backend as K


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
 

 
# def show_gradCAMs(model, gradCAM, GuidedBP, im_ls, n=3, decode={}):
#     """
#     model: softmax layer
#     """
#     random.shuffle(im_ls)
#     plt.subplots(figsize=(30, 10*n))
#     k=1
#     for i in range(n):
# #         img = cv2.imread(os.path.join(IMAGE_DIR,im_ls[i]))
#         img = cv2.imread(im_ls[i])
#         upsample_size = (img.shape[1],img.shape[0])
#         if (i+1) == len(df):
#             break
#         # Show original image
#         plt.subplot(n,3,k)
#         plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
#         plt.title("Filename: {}".format(im_ls[i]), fontsize=20)
#         plt.axis("off")
#         # Show overlayed grad
#         plt.subplot(n,3,k+1)
# #         im = img_to_array(load_img(os.path.join(IMAGE_DIR,im_ls[i]), target_size=(W,H)))
#         im = img_to_array(load_img(im_ls[i], target_size=(W,H)))
#         x = np.expand_dims(im, axis=0)
#         x = preprocess_input(x)
#         preds = model.predict(x)
#         idx = preds.argmax()
#         if len(decode)==0:
#             res = decode_predictions(preds)[0][0][1:]
#         else:
#             res = [decode[idx],preds.max()]
#         cam3 = gradCAM.compute_heatmap(image=x, classIdx=idx, upsample_size=upsample_size)
#         new_img = overlay_gradCAM(img, cam3)
#         new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
#         plt.imshow(new_img)
#         plt.title("GradCAM - Pred: {}. Prob: {}".format(res[0],res[1]), fontsize=20)
#         plt.axis("off")
#
#         # Show guided GradCAM
#         plt.subplot(n,3,k+2)
#         gb = GuidedBP.guided_backprop(x, upsample_size)
#         guided_gradcam = deprocess_image(gb*cam3)
#         guided_gradcam = cv2.cvtColor(guided_gradcam, cv2.COLOR_BGR2RGB)
#         plt.imshow(guided_gradcam)
#         plt.title("Guided GradCAM", fontsize=20)
#         plt.axis("off")
#
#         k += 3
#     plt.show()