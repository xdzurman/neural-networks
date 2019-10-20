
# Proposal
**Authors:**
* Nikolas Tomaštík
* Kamil Džurman

## Motivation 
We will solve colorizing black-and-white images using neural networks. 

With a working model we will be able to restore old photos, which would be very time demanding and often 
inaccurate to reproduce manually in software such as Photoshop.

Open ended question include those such as what kind of images we will be training on and colorizing 
(e.g. what domain - old family photos and portraits, buildings, profession bound objects, de-colorized 
modern world - streets/cities and so on) and if there can be extracted another information which could 
be useful for colorizing.

## Related Work
Colorizing images is a common showcase neural networks task. Other people solve this task using default 
or custom GAN methods ([NoGAN](https://github.com/jantic/DeOldify#what-is-nogan), 
[DC-GAN](https://arxiv.org/pdf/1511.06434.pdf), [SAGAN](https://arxiv.org/abs/1805.08318)) or 
Convolutional Neural Networks [](https://dl.acm.org/citation.cfm?id=2925974).

In [Colorful Image Colorization](https://arxiv.org/pdf/1603.08511.pdf) they translates RGB values 
into [Lab](https://en.wikipedia.org/wiki/CIELAB_color_space) format and predicts two colored layers
from the black and white input layer. For this mapping they have used CNN with only using convolution 
layers.

Article [Depth-Aware Image Colorization Network](https://dl.acm.org/citation.cfm?id=3267800) tried different 
approach with finding deepness for solving color bleeding problem which is quiet common in this domain.

Another subdomain of this problem is colorizing cartoons or comics/mangas which was researched 
in [Comicolorization: semi-automatic manga colorization](https://dl.acm.org/citation.cfm?id=3149430) 
and [Automatic Cartoon Colorization Based on Convolutional Neural Network](https://dl.acm.org/citation.cfm?id=3095742).

Other approach extracts features from 
groups of pixels and tries to predict output color for each pixel., ...**TODO RESEARCH**

## Datasets

There are many sites that provide free photographs - [Unsplash](https://unsplash.com/), 
[Gratisography](https://gratisography.com/), [Morguefile](https://morguefile.com/), 
[Pixabay](https://pixabay.com/), [Stockvault](https://www.stockvault.net/) and so on. There are even 
many public datasets ([dataset/colornet](https://www.floydhub.com/emilwallner/datasets/colornet), 
[kaggle](https://www.kaggle.com/c/open-images-2019-object-detection/data), 
[50-ML-image-datasets](https://blog.cambridgespark.com/50-free-machine-learning-datasets-image-datasets-241852b03b49).

It's safe to say that gigabytes of images are available. There are many types of the images, some dedicated to certain 
domain, other more general. 

## High-Level Solution Proposal 
> If you know already, what will be the architecture of the model, what experiments would you like to perform.

Our first approach will be to try replicate some baseline from related work and then we will try to make some
improvements.
