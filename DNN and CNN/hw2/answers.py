r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers
import torch.nn

part1_q1 = r"""
1.
a) The shape of the gradient is (batch_size, out_features, batch_size, in_features) so in our case it's
(64, 512, 64, 1024)

b) The above gradient is a tensor of shape (64, 512) where each item in it is a (64,1024) tensor which is the gradient
of $X^TW$ according to the item (i,j) of X i.e $\frac{\partial Y}{\partial x_{i,j}} = \frac{\partial X^TW}{\partial x_{i,j}}$.
But only a few items of the product depend on X(i,j) for each i,j. So in other words, most of the item of the gradient are 0. 

To be more precise in
$ Y_{m,n}$ only the only j-th row of $Y$ depends on $x_{i,j}$ 

Consquently in $(\frac{\partial Y}{\partial x_{i,j}})_{m,n}$ only the the j-th row is non-zero which means that every other entry is zero

c)No we don't need to compute the whole gradient, the trick is to use (his majesty) the chain rule. 
If each function is able to calculate its own derivative we just have to multiply them along the backward path

As we know $\delta\mat{X}=\pderiv{L}{\mat{X}} = \pderiv{L}{\mat{Y}}\pderiv{Y}{\mat{X}}$ 
But remember $Y$ is a Linear Layer which means $Y = X^T W + B$ So the gradient of $Y$ w.r.t to $X$ is easy to compute because it's only $W$

So we have $\delta\mat{X}=\pderiv{L}{\mat{Y}}W$ Where $\pderiv{L}{\mat{Y}}$ and $W$ are known

This method allows use to gain in time and space complexity. 

-Time because with chain rule we don't have to compute a (64, 512, 64, 1024) tensor

-space because we won't have to store a (64, 512, 64, 1024) tensor in memory


2.
a) The dimension of the Jacobian of the loss according to $W$ is 
(64,512,512,1024)

b) The above gradient is a tensor of shape (64, 512) where each item in it is a (512,1024) tensor which is the gradient
of $X^TW$ according to the item (i,j) of W i.e $\frac{\partial Y}{\partial W_{i,j}} = \frac{\partial X^TW}{\partial W_{i,j}}$.
But only a few items of the product depend on X(i,j) for each i,j. So in other words, most of the item of the gradient are 0. 

To be more precise in $Y_{m,n}$ only the only i-th column of $Y$ depends on $W_{i,j}$ 

Consquently in $(\frac{\partial Y}{\partial W_{i,j}})_{m,n}$ only the the i-th column is non-zero which means that every other entry is zero.


c) No we don't need to compute the whole gradient, the trick is to use (his majesty) the chain rule. If each function is able to calculate its own derivative we just have to multiply them along the backward path

As we know  $\delta\mat{W}=\pderiv{L}{\mat{X}} = \pderiv{L}{\mat{Y}}\pderiv{Y}{\mat{X}}$  But remember  $Y$ is a Linear Layer which means  $Y=X^TW+B$  So the gradient of  $Y$  w.r.t to  $W$  is easy to compute because it's only  $X^^$ 

So we have  $\delta W= (\frac{∂L}{∂Y})^T X$  Where  $\frac{∂L}{∂Y}$  and  $X$  are known

This method allows use to gain in time and space complexity.

-Time because with chain rule we don't have to compute a (64, 512, 512, 1024) tensor

-space because we won't have to store a (64, 512, 512, 1024) tensor in memory





"""

part1_q2 = r"""

Decent based algorithms need the gradient in order to make a step along the slope. Back-propagation is really popular in
deep learning because it's an efficient (in terms of time and space) way to compute the gradient of the output according to one parameter, it doesn't
require to compute the whole Jacobian which can have billions of entry and consequently make optimization more efficient. 
However as we have seen above computing the gradient of the loss directly according to a parameter is possible but much 
harder. But to answer to the question yes it's possible to use decent-based algorithms for optimization without using 
back-propagation.
However by using backpropagation, we can make use of automatic differentiation which make the design of NN easier and faster





"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.1
    reg = 0
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.001
    lr_vanilla = 0.01
    lr_momentum = 0.001
    lr_rmsprop = 0.0001
    reg = 0.001
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.001
    lr = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""

1) Yes the graphs match with our expectation.
For the training accuracy, lower the dropout is higher the accuracy is, this make sens since less we
miss information on our training set more our model will fit to the known data and then will be more accurate
on the training. While the testing set contains only unknown data, so missing some of information about the
training set will reduce the error due to overfitting

2)Having an high dropout setting will make more of our weight to 0, which means that we erase most of our knowledge
about the data. So a too high dropout will reduce our performances. But a more reasonable dropout will make our
classifier's decisions less based on the information based during training and consequently reduces the overfitting.
As we can see on the graphs the training accuracy is higher when the dropout is low. However the best testing accuracy 
is obtained when the dropout is set to 0.4 which mean that indeed having some dropout reduces overfitting.
Moreover when the dropout is to 0.8 both training and testing accuracy are low which is the pattern of underfitting. In
fact with such a dropout we cancel 80% of our weights and then the model doesn't have enough knowledge to be accurate. 

"""

part2_q2 = r"""
Even if it seems that accuracy and loss are inversely correlated in the case of Cross Entropy Loss it is possible for 
the test loss to increase when the test accuracy also increases.
Loss measures a difference between raw prediction and class while accuracy measures a difference between 
threshold prediction and class.
Therefore if raw predictions changes, it directly influences the loss whereas in order to affect the accuracy the 
predictions has to go over or under a threshold.
Also the loss function is continuous while accuracy is in a finite set of values. Consequently during some epochs ,where
the model still predict well, for the loss to increase.
So if a few raw predictions were bad but the model still predicts right it causes the loss to increase, while accuracy 
stays the same. At the same time the model is still learning patterns which are useful for generalization therefore 
increases accuracy.
"""

part2_q3 = r"""
1 Back propagation compute the gradient w.r.t to every parameter using chain rule along the backward path while
gradient descent is the algorithm that optimize our parameters by computing the gradient in order to get to the minimum
of the function

2 During gradient descent we compute the exact gradient to reduce the loss at each iteration. In order to do that we
compute the gradient over all data points and then we average them:

$\nabla _wL = \frac{1}{m}∑^m_i \frac{∂L(x^{(i)})}{\partial w}$

Also the loss converges with a linear rate with GD. 

While in Stochastic Gradient Descent we average the gradient w.r.t to the parameter by computing the gradient on one
small set of points $C \subseteq S$  where the samples and the size of $C$ are randoms.

In SGD: 

$\nabla _wL \approx \frac{1}{|C|}∑_{x ⊆C} \frac{∂L(x)}{\partial w}$

Then in both algorithm update the weights like below

$w ← w - \eta \nabla_w L$ Where $\eta \in ℝ$ is the learning rate

3 In practice SGD is more used in Deep Learning for the following two reasons:

1) Computing the exact gradient of the loss according to all the training points is really long knowing that in neural
network architectures we often have millions of parameters and in training set we have also thousands of samples
so for instance if we have $10^6$ parameters and $10^4$ training samples we get $10^{10}$ partial derivatives to compute
at each iterations.

2) GD also suffers from the local minimum problem, indeed if the direction of the gradient leads to a local minimum GD
will get "trapped" in it whereas computing a random gradient can avoid this kind of problems 

4

1) This method isn't exactly as Gradient Descent, in GD the gradient is computed by summing the loss according to all point in the dataset while here the gradient is computed on the sum of the loss. In fact knowing that
the gradient is linear $(\nabla \sum f = \sum \nabla f)$ we get that the gradient of GD multiplied by the number of samples. So we won't have the scaling factor $frac{1}{m}$. But the gradients are equivalent

2)In the back-propagation method all of the layers need to compute the derivative of their output w.r.t. to their input. Conequently all of the layers of the MLP need to store
their input. The out of memory error happened because of the accumulation of all of the inputs in all of the layers. 
Hence, while the forward layer was computed by loading
the data in batches, the layers stockpiled all of the data-points that they have encountered which lead to the out of memory error.

"""


# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers = 4  # number of layers (not including output)
    hidden_dims = 16 # number of output dimensions for each hidden layer
    activation = torch.nn.ReLU()  # activation function to apply after each hidden layer
    out_activation = "softmax"
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    lr = 0.01
    weight_decay = 0.01
    momentum = 0.9
    loss_fn = torch.nn.CrossEntropyLoss()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
1 As we can see our accuracy increases along the epochs and get to more than 92%, so this translates a low optimization error. 
2 As we remember our the i.i.d hypothesis isn't respected since we can observe some distance between the training set and both validation and testing set. This distance is the origin of the majority of the observed error, which mean thata big part of the loss is due to a high generalization error. 
3 Finally as we can see on the decision boundary our model fit well to our samples the we assume that we have a lowapproximation error


"""

part3_q2 = r"""

Be seeing the data generation process we see that in the center of the plot of the training data we have more negative samples while in the validation and testing set we have most positive samples, which is a consequency of the rotation applied on the sets 
Consequently during classification we can expect to have a bigger FNR. 

"""

part3_q3 = r"""
In the first we want a low False Positive rate because if a patient is diagnosed as positive while he's not he will have to make the seconds tests which are really expensive and dangerous
In this case if a positive patient is diagnosed as negative, will develop his symptoms in the future and will be healed. So in that case a high false negative rate does not matter.

In the second case the we want a low False Negative rate, since if the patient is negative while being sick he will die with high probability, so we don't want to miss sick patients. 

"""


part3_q4 = r"""
1) For a fixed width, we can observe a decision boundary with very similar shape. As long as the depth increase we can
observe that the decision boundary is smoother, and less look like some lines that separate the data. This smoothness 
allow us to get better performance on the test set.

2) For fixed depth, as the width increase we can observe that our model is able to fit more precisely to the shape of 
the data. The depth for which is more obvious is 2. For width of 1 it looks like our model use one line to separate
the data. For width of 4 we can see that our model use 6 line to separate and for wider models the number of line used 
in order to separate the classes increase.

3) 1 In this case the model are very similar, because first the shapes look the same (even if we can notice smoother
angles in the model with D4L8 than with D1L32). To confirm our results we can see that the train accuracies are equal
and the test accuracies are almost equals.
2 In this case also the model are really similar, we can se that in both model the decision boundary separate almost
the data in the same place and we can also recognize similar patterns in both model. 
However despite this similarity in the shape of the decision boundaries, we can observe better accuracies in the model
with D=1, L = 128   

4) As we can see for each experiment the test accuracy is higher than the validation accuracy (on which haven't changed 
the threshold). This translates that indeed the threshold selection improved the test accuracy. This is du to the fact
that the validation set had undergone a similar transformation than the test set (rotation of 50° for validation and 
rotation of 40° for the test set)so the improvement that are made on the validation set will results to big improvements
on the testing set. 


"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    lr = 0.1
    weight_decay = 0.001
    momentum = 0.001
    loss_fn = torch.nn.CrossEntropyLoss()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
1) Let's first compute the number of parameters of a regular block which is composed by two convolutional layers with 
$3\times 3 $ kernel and 256 channels as input and output, so the number of parameters of one layer is:
$(3\cdot3\cdot256 + 1)\cdot256$ consequently the number of parameters in the whole block is 
2\cdot((3\cdot3\cdot256 + 1)\cdot256) = 1 180 160

On the other hand the bottleneck block has 3 layers:
-$(1\times 1 256 \rightarrow 64) \rightarrow (3\times 3 64 \rightarrow 64) \rightarrow (1\times 1 64 \rightarrow 256)$

So the total number of params is $(256+1)\cdot64 + (3\cdot3\cdot64 + 1)\cdot64 + (64+1)\cdot256 = 70,016$

2) Assumption: Floating point operation are $+, - , \times$ and ReLu, moreover the size of the input is $(256, H , W)$

Regular block: 

Conv1: $3 \cdot 3 \cdot 256 \cdot H \cdot W \cdot 256$
    
Relu ($\times 2$): $2 \cdot 256 \cdot H \cdot W $
    
Conv2: $ 3 \cdot 3 \cdot 256 \cdot H \cdot W \cdot 256 $
    
Residual connection: $ 256 \cdot H \cdot W $
    
And the sum is: $ 1 180 416 \cdot H \cdot W$

The bottleneck block:
    
Conv1: $1 \cdot 1 \cdot 256 \cdot H \cdot W \cdot 64 $

Relu ($\times 2$): $2 \cdot  64 \cdot H \cdot W $

Conv2: $3 \cdot 3 \cdot 64 \cdot H \cdot W \cdot 64 $ 

Conv3: $1 \cdot 1 \cdot 64 \cdot H \cdot W \cdot 256 $

Residual connection: $256 \cdot H \cdot W $

Relu (256): $256 \cdot H \cdot W $

And this time the sum is $70 272 \cdot H \cdot W$

As we can see, in term of computation cost the bottleneck block is more efficient than the regular block.

3) The regular block mostly effects the input spatially, we can conclude it after seeing that in the input and the 
output the number of feature maps are the same

The bottleneck block effects both spatial and feature map of the input, this is due to the fact that the first
convolution layer is 1X1 and consequently it's  maintaining the spatial aspect of the input and map the input.
After this first layer there are two layers that are similar to the Regular Block and therefore are operating on the
spatial aspect of the input. 
So we can conclude that the bottleneck block effects mostly on the spatial aspect but also on the feature map while the 
Regular block effects only the input spatially.


"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
1. Increasing the network depth, improves accuracy both on train and on test until L=4 (for both K=32 and K=64)
The best depth in this experiment is L=4. With this depth, it seems that we use the potential generalization power of 
this simple CNN network.
For deeper network, we suffer from vanishing gradient and eventually from too small training set.
For shorter network, the modeling power is weaker and we get poorer results.
2. For L=16, the network doesn't manage to learn at all. The reason is that the gradients don't propagate through all the layers (vanishing gradients).
Ways to solve that issue are to add residual blocks with shortcut connections or to add Batch Normalization layers.


"""

part5_q2 = r"""
The results differ for the several L's:
 - L = 2: K=32, 64 perform similarly and better than K=128, 256. With K=64, the network hits early stopping
 earlier than with K=32. K=256 performs even poorer than K=128.
 causes an accuracy loss 
 - L = 4: for all K, the bigger is K, the better accuracy the model achieves.
 - L = 8: until K=128, the bigger is K, the better accuracy the model achieves. For K=256, it performs significantly poorer.

 Insights: it seems that there is a tradeoff between the depth and the number of kernel filers/layers.
 For undeep CNN and for very deep CNN, only relative small number of filters is effective (up to K=64/128)
 However, for mid-deep CNN, increasing the number of filter/ layer has a positive impact on accuracy.  

 Comparing to 1.1: in 1.1 there are distinct accuracy scores for several depth, however in 1.2, the accuracy are closer
 one to each other for several K per L. For example, with L=4, the test accuracies are in a 5% range. Also, there is no
  model which can't learn at all like in 1,1. 

  In both 1.1 and 1.2 models overfit to training set with ~10/15% accuracy gap between train and test.


"""

part5_q3 = r"""
We observe very stable results in terms of accuracy per L value:
The larger is L, the lower are the results. Best accuracy is obtained for L=1 and worse for L=4.
We can easily assume that with K=[64,128,256], the value of L is critical since for each additional L,
we add 3 convolutional layers to the model. As seen before, a too deep CNN is related to vanishing gradient and lack of regularization 
especially with this simple CNN.

Let's observe that for L=4, the model doesn't learn at all because of the reasons cited above.


"""

part5_q4 = r"""
Here again, we get very stable results for several L values for both K=32 and K=[64,128,256]:
- The greater is L, the poorer accuracy the model achieves.
- The models overfit to training set for every configuration with ~20/30% difference between train and test accuracies.
- For K=[64,128,256] and L=8, the model doesn't learn at all. 


Compare with 1.1:
Here again we see a clear correlation between network depth and results. Up to some depth, accuracy increases, but 
starting from some depth, the network is too much deep and doesn't learn from training data.
The difference is that in 1.1 the accuracy gaps between them are smaller than in 1.4.


Compare with 1.3:
Surprisingly, this ResNet performs approximately the same as the CNN in 1.3. We would expect that ResNets perform
better than simple CNNs in 1.3.

"""

part5_q5 = r"""
1. In YourCNN, we added several layers compared to regular CNN. 
    - In feature extractor part : First we added Batch Normalization layers after each 
    convolutional layers. Secondly, we added Dropout layer each activation.
    _ In classifier part: We added Dropout layer after each activation.
    These two augmentations increased the results on both train and test sets.

2. The plots show that the depth L is critical in this network (given K=[32,64,128]). Indeed for L=3, we achieved
    accuracy of ~80% on both test and train sets. However, for bigger L, the network performs very poorly with
    accuracy lower than 60%. Moreover, the bigger is L, the lower is the accuracy.

    Compare to 1: 
    - We see that in experiment 1, we never achieved more than 70% on both train and test, and 
    most of the time, the model overfitted to train set. 
    At the contrary, our model achieved similar accuracies on both train and test set at ~80%. This shows that our model
    addresses the problem of overfitting, which we observed in Experiment 1, by adding regularization (Dropout).
    - Moreover, in Experiment 2, we observe that early stopping is triggered later than in Experiment 1. Thus, it
    seems that our model offers bigger place for training.
    - Experiments 1 and 2 both show that increasing the depth hurts accuracy. It seems that the problem of vanishing 
    gradient was not addressed by the BN layers. This may be due to the fact that skip-
    connections were not added in a smart way and not included at all in CNN and in our custom model.


"""
# ==============

