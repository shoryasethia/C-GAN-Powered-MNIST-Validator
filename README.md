# Conditional Generative Adversarial Networks based MNIST model validator
## Brief Introduction to Conditional GANs (CGANs)

CGAN is an extension of the standard GAN architecture, where both the generator and discriminator are conditioned on some extra information. In this case, I use mnist digit labels as the conditioning information for generating MNIST-like images based on a given label.

## How Conditional GANs Work ?

### Standard GAN Architecture

A standard GAN consists of two neural networks: `generator` and `discriminator`. These networks are trained simultaneously through a process of adversarial training:

- **Generator**: Takes random noise as input and generates synthetic data.
- **Discriminator**: Takes both real and synthetic data as input and aims to distinguish between the real and synthetic(fake) data.

The generator tries to produce data that is indistinguishable from real data, while the discriminator tries to correctly classify real and synthetic data. This adversarial process continues until the generator produces realistic data that the discriminator can no longer distinguish from real data.

### Conditional GAN Architecture

In a Conditional GAN, one conditions both the generator and the discriminator on additional information, here I conditioned them on class labels. This allows generator to generate data that is not only realistic but also conditioned on specific labels.

Here’s how I incorporate conditioning information into the GAN:

1. **Labels and Images**: Used digit labels from the MNIST dataset as the conditioning information
2. **Label Embeddings**: Convert the labels into embeddings
3. **Concatenation**: Concatenated embeddings with the input noise for the generator and with the real or synthetic images for the discriminator.

![CGAN basic architecture](https://github.com/shoryasethia/GAN/blob/main/conditional-gan-mnist/Conditional-GAN.png)

# Steps to test any MNIST trained model
### Clone repo
```
git clone https://github.com/shoryasethia/C-GAN-Powered-MNIST-Validator.git
```

### [backtester.py](https://github.com/shoryasethia/C-GAN-Powered-MNIST-Validator/blob/main/backtester.py)
Contains necessary libraries to load [trained cgan generator](https://github.com/shoryasethia/GAN/blob/main/conditional-gan-mnist/generator-mnist-cgan.h5) and function to evaluate trained mnist model

### [evaluate.py](https://github.com/shoryasethia/C-GAN-Powered-MNIST-Validator/blob/main/evaluate.py)
* Enter your *keras* based model's path
* Set number of `testing images`; by default it is set to `num_test_images=1000`
* To plot `first 100` predictions for your model, set `plot = 1` else `plot = 0`
> **You can edit `max_plots = 100` in [backtester.py](https://github.com/shoryasethia/C-GAN-Powered-MNIST-Validator/blob/main/backtester.py) as per your needs**
* Run evaluate.py
* Output for [this](https://github.com/shoryasethia/C-GAN-Powered-MNIST-Validator/blob/main/mnist-cnn.h5) trained model
```
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 
Model tested on =1000 images.
Plotting top 100 predicted and ground truth results.
Number of correctly predicted samples: 969
Number of incorrectly predicted samples: 31
Accuracy of the model: 96.90%
```
## Conclusion

My Conditional GAN model effectively generates MNIST-like images based on specified digit labels. This can be particularly useful for generating additional training data or for testing digit recognition models. You can find my other related implementation in the following repositories:
- [Conditional GAN implementation on MNIST](https://github.com/shoryasethia/GAN/tree/main/conditional-gan-mnist)
- Repo for various MNIST's digit recognition models [here](https://github.com/shoryasethia/Digit-Recognition)

> * **Feel free to explore the code and use the files and trained models for your own projects!**
> * **If you liked anything from this repo, give it a star**
> * **Author : [@shoryasethia](https://github.com/shoryasethia/)**
