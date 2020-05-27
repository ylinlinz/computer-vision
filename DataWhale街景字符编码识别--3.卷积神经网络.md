## DataWhale街景字符编码识别--3.卷积神经网络

Convolutional Neural Networks (CNN) are biologically-inspired variants of MLPs. From Hubel and Wiesel’s early work on the cat’s visual cortex [[Hubel68\]](http://www.deeplearning.net/tutorial/references.html#hubel68), we know the visual cortex contains a complex arrangement of cells. These cells are sensitive to small sub-regions of the visual field, called a *receptive field*. The sub-regions are tiled to cover the entire visual field. These cells act as local filters over the input space and are well-suited to exploit the strong spatially local correlation present in natural images.

Additionally, two basic cell types have been identified: Simple cells respond maximally to specific edge-like patterns within their receptive field. Complex cells have larger receptive fields and are locally invariant to the exact position of the pattern.

The animal visual cortex being the most powerful visual processing system in existence, it seems natural to emulate its behavior. Hence, many neurally-inspired models can be found in the literature. To name a few: the NeoCognitron [[Fukushima\]](http://www.deeplearning.net/tutorial/references.html#fukushima), HMAX [[Serre07\]](http://www.deeplearning.net/tutorial/references.html#serre07) and LeNet-5 [[LeCun98\]](http://www.deeplearning.net/tutorial/references.html#lecun98), which will be the focus of this tutorial.



CNNs exploit spatially-local correlation by enforcing a local connectivity pattern between neurons of adjacent layers. In other words, the inputs of hidden units in layer **m** are from a subset of units in layer **m-1**, units that have spatially contiguous receptive fields. We can illustrate this graphically as follows:

![_images/sparse_1D_nn.png](http://www.deeplearning.net/tutorial/_images/sparse_1D_nn.png)

Imagine that layer **m-1** is the input retina. In the above figure, units in layer **m** have receptive fields of width 3 in the input retina and are thus only connected to 3 adjacent neurons in the retina layer. Units in layer **m+1** have a similar connectivity with the layer below. We say that their receptive field with respect to the layer below is also 3, but their receptive field with respect to the input is larger (5). Each unit is unresponsive to variations outside of its receptive field with respect to the retina. The architecture thus ensures that the learnt “filters” produce the strongest response to a spatially local input pattern.

However, as shown above, stacking many such layers leads to (non-linear) “filters” that become increasingly “global” (i.e. responsive to a larger region of pixel space). For example, the unit in hidden layer **m+1** can encode a non-linear feature of width 5 (in terms of pixel space).



In addition, in CNNs, each filter ![h_i](http://www.deeplearning.net/tutorial/_images/math/d03a4b38f7e618d6826778cf0a8b906fbbaa935a.png) is replicated across the entire visual field. These replicated units share the same parameterization (weight vector and bias) and form a *feature map*.

![_images/conv_1D_nn.png](http://www.deeplearning.net/tutorial/_images/conv_1D_nn.png)

In the above figure, we show 3 hidden units belonging to the same feature map. Weights of the same color are shared—constrained to be identical. Gradient descent can still be used to learn such shared parameters, with only a small change to the original algorithm. The gradient of a shared weight is simply the sum of the gradients of the parameters being shared.

Replicating units in this way allows for features to be detected *regardless of their position in the visual field.* Additionally, weight sharing increases learning efficiency by greatly reducing the number of free parameters being learnt. The constraints on the model enable CNNs to achieve better generalization on vision problems.



A feature map is obtained by repeated application of a function across sub-regions of the entire image, in other words, by *convolution* of the input image with a linear filter, adding a bias term and then applying a non-linear function. If we denote the k-th feature map at a given layer as ![h^k](http://www.deeplearning.net/tutorial/_images/math/052c3da70a50aa71bb1f945ed6a7b4434717c375.png), whose filters are determined by the weights ![W^k](http://www.deeplearning.net/tutorial/_images/math/7fec4dfcd9f8a6cafcee094bcd800c56e4942a84.png) and bias ![b_k](http://www.deeplearning.net/tutorial/_images/math/d459de01b86dc218684c55d28e293cc9e2a203fc.png), then the feature map ![h^k](http://www.deeplearning.net/tutorial/_images/math/052c3da70a50aa71bb1f945ed6a7b4434717c375.png) is obtained as follows (for ![tanh](http://www.deeplearning.net/tutorial/_images/math/43f9bb846583d99234600f15be84220cf11b23b7.png) non-linearities):

![h^k_{ij} = \tanh ( (W^k * x)_{ij} + b_k ).](http://www.deeplearning.net/tutorial/_images/math/1dadc9ccebdcd6b34042c13290c5917736247bc1.png)

Note

Recall the following definition of convolution for a 1D signal. ![o[n] = f[n]*g[n] = \sum_{u=-\infty}^{\infty} f[u] g[n-u] = \sum_{u=-\infty}^{\infty} f[n-u] g[u]](http://www.deeplearning.net/tutorial/_images/math/6a729ee822fbd2e9edb198ebd7677f72e310f90e.png).

This can be extended to 2D as follows: ![o[m,n] = f[m,n]*g[m,n] = \sum_{u=-\infty}^{\infty} \sum_{v=-\infty}^{\infty} f[u,v] g[m-u,n-v]](http://www.deeplearning.net/tutorial/_images/math/37f2a6885f57ec2ac212bd3eb6f7a2d860dc278d.png).

To form a richer representation of the data, each hidden layer is composed of *multiple* feature maps, ![\{h^{(k)}, k=0..K\}](http://www.deeplearning.net/tutorial/_images/math/3ee11605a3f6fe4acc53eba04ac499527a8ea695.png). The weights ![W](http://www.deeplearning.net/tutorial/_images/math/953bde2ab2fca30897f66185e5b37b73747b8b46.png) of a hidden layer can be represented in a 4D tensor containing elements for every combination of destination feature map, source feature map, source vertical position, and source horizontal position. The biases ![b](http://www.deeplearning.net/tutorial/_images/math/57c9d14bb082716df9000146882ce365335d08f1.png) can be represented as a vector containing one element for every destination feature map. We illustrate this graphically as follows:

![_images/cnn_explained.png](http://www.deeplearning.net/tutorial/_images/cnn_explained.png)

**Figure 1**: example of a convolutional layer

The figure shows two layers of a CNN. **Layer m-1** contains four feature maps. **Hidden layer m** contains two feature maps (![h^0](http://www.deeplearning.net/tutorial/_images/math/872b9b9ed45d1d05fcf3aef3d20f49e83e986066.png) and ![h^1](http://www.deeplearning.net/tutorial/_images/math/8827c479816030bc2cfa09e43714fa82968e2900.png)). Pixels (neuron outputs) in ![h^0](http://www.deeplearning.net/tutorial/_images/math/872b9b9ed45d1d05fcf3aef3d20f49e83e986066.png) and ![h^1](http://www.deeplearning.net/tutorial/_images/math/8827c479816030bc2cfa09e43714fa82968e2900.png) (outlined as blue and red squares) are computed from pixels of layer (m-1) which fall within their 2x2 receptive field in the layer below (shown as colored rectangles). Notice how the receptive field spans all four input feature maps. The weights ![W^0](http://www.deeplearning.net/tutorial/_images/math/13ac47178558c52423c845ecc30e28c4bcb62c84.png) and ![W^1](http://www.deeplearning.net/tutorial/_images/math/40a751368804d49a411a8430fe50c417125df1e7.png) of ![h^0](http://www.deeplearning.net/tutorial/_images/math/872b9b9ed45d1d05fcf3aef3d20f49e83e986066.png) and ![h^1](http://www.deeplearning.net/tutorial/_images/math/8827c479816030bc2cfa09e43714fa82968e2900.png) are thus 3D weight tensors. The leading dimension indexes the input feature maps, while the other two refer to the pixel coordinates.

Putting it all together, ![W^{kl}_{ij}](http://www.deeplearning.net/tutorial/_images/math/5eb8ae5950093821264a14787bfeeb092595c84c.png) denotes the weight connecting each pixel of the k-th feature map at layer m, with the pixel at coordinates (i,j) of the l-th feature map of layer (m-1).



Another important concept of CNNs is *max-pooling,* which is a form of non-linear down-sampling. Max-pooling partitions the input image into a set of non-overlapping rectangles and, for each such sub-region, outputs the maximum value.

- Max-pooling is useful in vision for two reasons:

  By eliminating non-maximal values, it reduces computation for upper layers.It provides a form of translation invariance. Imagine cascading a max-pooling layer with a convolutional layer. There are 8 directions in which one can translate the input image by a single pixel. If max-pooling is done over a 2x2 region, 3 out of these 8 possible configurations will produce exactly the same output at the convolutional layer. For max-pooling over a 3x3 window, this jumps to 5/8.Since it provides additional robustness to position, max-pooling is a “smart” way of reducing the dimensionality of intermediate representations.

