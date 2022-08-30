# Exploring-Autoencoders
Learning about auto encoders and latent spaces

Requirements:
- Pytorch (torchvision)
- matplotlib



## So far:

### Trivial solution for MNIST reconstruction:

I trained a neural network with no bias and activation with the following architecture on the image reconstruction task:

784 (input) -> 784 (latent) -> (784) output


First, I trained the to achieve a near-perfect photometric loss on the validation set on MNIST, I trained for 100 epochs with LR scheduling. Once I achieved the loss I wanted,
I then computed (W2)(W1) and plotted the heatmap below. 

(W2)(W1):

![w2w1](https://github.com/AditMeh/Exploring-Autoencoders/blob/main/images/trivialsolution.png)

The second set of weights appears to have higher off-diagonal values in the visualization, I believe this is because it has very small negative values of the diagonal, so the
visualization messes up a little bit. Therefore, I clipped the values on the right plot so we can clearly see the identity matrix.



As evidenced, it appears the network has learned an identity transformation, with W1 and W2 being inverses. 

### CAE, L1 constrainted AE:
I've implemented and trained contractive autoencoders and constrained AE. I won't write about them in detail.


### Denoising autoencoder

![w2w1](https://github.com/AditMeh/Exploring-Autoencoders/blob/main/images/denoising.png)


I've implemented a simple demoising autoencoder on MNIST. The visualization above shows how the network has learned to reconstruct the corrupted parts of the image. For the noise, I just dropped out pixels in the image with a probability hyperparameter, basically dropout. 

### VAE

So far, I've trained a VAE with a gaussian posterior on MNIST:

Here is a visualization. For each cell with an image, I am interpolating between two randomly sample images from the dataset:

![w2w1](https://github.com/AditMeh/Exploring-Autoencoders/blob/main/images/test.gif)


### CNN VAE

I've also trained a VAE using a CNN encoder and decoder, where the decoder uses transposed convolutions. Here's a visualization:

![w2w1](https://github.com/AditMeh/Exploring-Autoencoders/blob/main/images/celeba_small.gif)

![w2w1](https://github.com/AditMeh/Exploring-Autoencoders/blob/main/images/animefacedataset_small.gif)
