# Single Image x4 Resolution

I used works by [Dong, et al 2015](https://arxiv.org/pdf/1501.00092.pdf)

The model is simple, with 3 layers. Each is a conv2d layer, with Relu, except final layer is sigmoid.
First is 64x9x9, then 32x5x5, then 3x5x5. No padding.

I used perceptionloss with MobileNet. I chose it since it is small. In other words, I could gain performance with better model like Xception.

Usage is simple, look at [prediction.py](https://github.com/dht7166/x4Resolution/blob/master/prediction.py).

Ofc, this is no SOTA. It is just spare time project. What is cool is that the model is small, the weight, with the model is < 1mb. Ideal for niche scenario.
Output image might have artifacts, the usual convolution blocky stuff. Also 16 pixels from all size will be cropped. I am too lazy to add padding.

Here are some output, there is more in [example_input](https://github.com/dht7166/x4Resolution/tree/master/example_input). From the order of input, prediction, and truth.



<img src=example_input/flickr_cat_000311_input.jpg width="512" height="612">
<img src=example_input/flickr_cat_000311__x4.jpg width="512" height="612">
<img src=example_input/flickr_cat_000311.jpg width="512" height="612">
