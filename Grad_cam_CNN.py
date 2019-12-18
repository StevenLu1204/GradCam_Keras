from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential, Model, load_model
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2
import os
import argparse


def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path):
    img_path = path
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]

def grad_cam(input_model, image, category_index, layer_name):
    nb_classes = 19
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    x = Lambda(target_layer, output_shape = target_category_loss_output_shape)(input_model.output)
    model = Model(inputs=input_model.input, outputs=x)
    # model.summary()

    loss = K.sum(model.output)
    conv_output =  [l for l in model.layers if l.name == layer_name][0].output
    grads = normalize(_compute_gradients(loss, [conv_output])[0])
    gradient_function = K.function([model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]
    
    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    
    heatmap = cam / np.max(cam)

    # Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    
    # cv2.imshow("img", img)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Grad-CAM of CNN features")
    parser.add_argument('img_dir', type=str, help='Path to target images')
    parser.add_argument('save_dir', type = str, help="Path to save Grad-CAM images")
    parser.add_argument('model_path', type = str, help = "Path to .h5 file")
    parser.add_argument('layer_name', type = str, help = "Target layer that outputs feature")
    args = parser.parse_args()

    image_dir = args.img_dir
    save_dir = args.save_dir
    model_path = args.model_path
    layer_name = args.layer_name

    model = load_model(model_path)
    print(model.summary())

    for files in os.listdir(image_dir):
        preprocessed_input = load_image(os.path.join(image_dir, files))
               
        predictions = model.predict(preprocessed_input)
        predicted_class = np.argmax(predictions)
       
        cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, layer_name)
        cv2.imwrite(os.path.join(save_dir, "{}_{}_gradcam.jpg".format(files, layer_name)), cam)
        print("Grad-cam of {} done".format(files))


