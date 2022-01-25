# Exploring-Autoencoders
Learning about auto encoders and latent spaces


## So far:

#### Trivial solution for MNIST reconstruction:

I trained a neural network with no bias and activation with the following architecture on the image reconstruction task:

784 (input) -> 784 (latent) -> (784) output


First, I trained the to achieve a near-perfect photometric loss on the validation set on MNIST, I trained for 100 epochs with LR scheduling. Once I achieved the loss I wanted,
I then computed (W1)(W2) and (W2)(W1). Here are their heatmaps below. 

(W1)(W2):

![w1w2](https://github.com/AditMeh/Exploring-Autoencoders/blob/main/w1_w2.png)

(W2)(W1):

![w2w1](https://github.com/AditMeh/Exploring-Autoencoders/blob/main/w2_w1.png)


The second set of weights appears to have higher off-diagonal values in the visualziation, I believe this is because it has very small negative values of the diagonal, so the
visualization messes up a little bit.

Here is a clipped version ([0, inf)).

![w2w1](https://github.com/AditMeh/Exploring-Autoencoders/blob/main/w2_w1_clipped.png)


As evidenced, it appears the network has learned an identity transformation, with W1 and W2 being inverses. 
