# Single Image x4 Resolution

I used works by [Dong, et al 2015](https://arxiv.org/pdf/1501.00092.pdf)

The model is simple, with 3 layers. Each is a conv2d layer, with Relu, except final layer is sigmoid.
First is 64x9x9, then 32x5x5, then 3x5x5. No padding.

I used perceptionloss with MobileNet. I chose it since it is small. In other words, I could gain performance with better model like Xception.

Trained using [Animal Faces](https://www.kaggle.com/andrewmvd/animal-faces) on Google Colab (the notebook is also included). 
But I tested using my personal machine with Python 3.7, keras 2.1.3 and tensorflow 2.1. Opencv-python and numpy is also needed, any version would suffice. I did edit a bit of code so that it could run on my personal laptop (for example changing the backend sigmoid to built-in activation for conv2d), but I did make sure that training and prediction code runs perfectly.

Usage is simple, look at [prediction.py](https://github.com/dht7166/x4Resolution/blob/master/prediction.py).

Ofc, this is no SOTA. It is just spare time project. What is cool is that the model is small, the weight, with the model is < 1mb. Ideal for niche scenario.
Output image might have artifacts, the usual convolution blocky stuff. Also 16 pixels from all size will be cropped. I am too lazy to add padding.

Here are some output, there is more in [example_input](https://github.com/dht7166/x4Resolution/tree/master/example_input). From the order of input, prediction, and truth.



<img src=example_input/flickr_cat_000314_input.jpg width="300" height="300">
<img src=example_input/flickr_cat_000314__x4.jpg width="300" height="300">
<img src=example_input/flickr_cat_000314.jpg width="300" height="300">
