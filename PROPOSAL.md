
# Proposal
**Authors:**
* Nikolas Tomaštík
* Kamil Džurman

## Motivation 
**TODO**
>What task will you solve, why is this task important, what are the open questions.

We will solve colorizing black-and-white images using neural networks. 

With a working model we will be able to restore old photos, which would be very time demanding and often inaccurate to reproduce manually in software such as Photoshop.

Open ended question include those such as what kind of images we will be training on and colorizing (e.g. what domain - old family photos and portraits, buildings, profession bound objects, de-colorized modern world - streets/cities and so on), ...**TODO CONSULT**

## Related Work
**TODO**
>Shortly describe how were similar problems solved by other people.

Colorizing images is a common showcase neural networks task. Other people solve this task using default or custom GAN methods ([NoGAN](https://github.com/jantic/DeOldify#what-is-nogan), [DC-GAN](https://arxiv.org/pdf/1511.06434.pdf), [SAGAN](https://arxiv.org/abs/1805.08318)) or Convolutional Neural Networks.

One approach translates RGB values into [Lab](https://en.wikipedia.org/wiki/CIELAB_color_space) format and predicts two colored layers from the black and white input layer. Other approach extracts features from groups of pixels and tries to predict output color for each pixel., ...**TODO RESEARCH**

## Datasets
>**TODO**
What datasets exist that can be used to train the task, how much data is available, what are
the properties of these data.

There are many sites that provide free photographs - [Unsplash](https://unsplash.com/), [Gratisography](https://gratisography.com/), [Morguefile](https://morguefile.com/), [Pixabay](https://pixabay.com/), [Stockvault](https://www.stockvault.net/) and so on. There are even many public datasets ([dataset/colornet](https://www.floydhub.com/emilwallner/datasets/colornet), [kaggle](https://www.kaggle.com/c/open-images-2019-object-detection/data), [50-ML-image-datasets](https://blog.cambridgespark.com/50-free-machine-learning-datasets-image-datasets-241852b03b49).

It's safe to say that gigabytes of images are available. There are many types of the images, some dedicated to certain domain, other more general. 

## High-Level Solution Proposal 
**TODO**
If you know already, what will be the architecture of the model, what experiments would you like to perform.

