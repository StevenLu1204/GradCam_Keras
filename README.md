# GradCam_Keras
Keras implementation of GradCam for custom CNN model.

Adapted from https://github.com/jacobgil/keras-grad-cam to generate heatmap image from custom CNN model.


<h3> Usage </h3>

```
python Grad_cam_CNN.py <target_image_dir> <output_image_dir> <model_path> <layer_name>
```

- ```<target_image_dir>```: Path to target images.
- ```<output_image_dir>```: Path to save the generated heatmap images.
- ```<model_path>```: Path to model .h5 weight file.
- ```<layer_name>```: Generate heatmap image from specified layer name.

<h3> Examples </h3>

![Input image](/examples/1.jpg)   ![Output image](/examples/1.jpg_conv_pw_13_relu_gradcam.jpg) 
