r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=256,
        seq_len=64,
        h_dim=128,
        n_layers=3,
        dropout=0.4,
        learn_rate=0.001,
        lr_sched_factor=0.5,
        lr_sched_patience=2,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======

    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    temperature = 0.5
    start_seq = ""
    # ========================
    return start_seq, temperature


part1_q1 = r"""
The two reasons of the splittng are:

1.  Making batch that are indepedent will make our model more creative during training in order to reduce overfitting. 
    Indeed training the model by giving to the model only one batch containing the whole corpus will make it remember it 
    instead of creating the ability to generate new and consistent text
2.  For memory reasons, indeed on character in encoded by a 83d vector, the corpus contains millions of caracters, so 
    passing the whole encoded as an huge matrix wouldn't fit in the memory

"""

part1_q2 = r"""
In fact, the generated text has a longer memory than the original sequence length. This due to the fact that the model's
output is composed by the generated text, which has the same lenght as the input, and by the hidden state which acts as
our memory

"""

part1_q3 = r"""
We are not shuffling the order of batches when training, because in text generation the order of charcters has an 
importance, consequently we want the model to learn the patterns of a text and in order to do so, it has to know the 
previous chars. While in image classification, for instance, a cat will always be cat even if the previous 
classification was a dog.

"""

part1_q4 = r"""
1.  The temperature controls the variance of the distribution for the next char in according to the current one and the
    current state of the model.

2.  Higher temperature lead to a bigger variance for the next char distribution, in other words more noise and then 
    increase the chance of choosing a random character for the next char. Consequently with a high temperature we will 
    observe more spelling mistakes. We can use an high temperature in order to increase the creativity of the model

3.  Higher temperature lead to a lower variance for the next char distribution, consequently that will decrease the 
    chance of choosing a random character for the next char. Consequently with a low temperature we will observe less 
    spelling mistakes and our model will generate character based more on what it learned than on randomness



"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 8
    hypers['h_dim'] = 512
    hypers['z_dim'] = 16
    hypers['x_sigma2'] = 0.001
    hypers['learn_rate'] = 1e-4
    hypers['betas'] = (0.9, 0.99)   # ========================
    return hypers


part2_q1 = r"""
The value of sigma allows to control the degree of freedom that our model has in order to generate new samples, by controlling
the importance of the reconstruction loss ($\sigma$ is acting as a scaling factor to the reconstruction loss).
So with a high sigma the model will generate new data that is close to the dataset while with a low value the model 
will be more far from the dataset when generating new samples.

"""

part2_q2 = r"""
The reconstruction loss is measuring the error between the original image and the generated image of the VAE after 
encoding and decoding.
The KL-divergence is the distance between the encoder probability function and latent space P(Z) prior
it focuses on making the latent space continue and be evenly distributed around 0(It actually helps training the encoder)
and it prevents the latent distribution to looks like separated clusters which is bad for later sampling 
and decoding : in this case, even if our reconstruction are promising, we won't be able to generate good images.

"""

part2_q3 = r"""
Maximizing the evidence distribution is necessary in order to make the generated sample the closest to the dataset. 
Indeed maximizing this distribution will increase the probability that the decoded image will be close to the dataset


"""

part2_q4 = r"""
Modeling the log of the latent-space variance instead of using the variance allows us to get numerical stability because
we map small values in the interval [0,1] to (-inf, log(1)] which is much wider so we obtain better numerical stability.
Moreover $\bb{\sigma}^2_{\bb{\alpha}}$ is a relatively small value 1>>$\bb{\sigma}^2_{\bb{\alpha}}$>0.

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0,
        z_dim=0,
        data_label=0,
        label_noise=0.0,
        discriminator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 32
    hypers['z_dim'] = 16
    hypers['label_noise'] = 0.2
    hypers['data_label'] = 1
    hypers['discriminator_optimizer']['type'] = 'Adam'
    hypers['discriminator_optimizer']['lr'] = 0.0002
    hypers['discriminator_optimizer']['betas'] = (0.5, 0.99)
    hypers['generator_optimizer']['type'] = 'Adam'
    hypers['generator_optimizer']['lr'] = 0.0002
    hypers['generator_optimizer']['betas'] = (0.5, 0.99)
    # ========================
    return hypers


part3_q1 = r"""
As we remember GAN models have to main components the discriminator and the generator. 
On the one hand the discriminator simply behaves as a classifier and during training phase we consider the generator as a constant 
(we don't keep the generator gradients) that is here to "help" the discriminator to converge and to allow it to learn 
the generator flaws.
On the other hand during when training the generator we need to keep its gradients in each step, but also we keep the
discriminator gradients because we need to backpropagate all the way from end of the entire model, in other words from
discriminator output.  

"""

part3_q2 = r"""
1.  No we shouldn't stop early the training only based on a low generator loss. It's possible that the generator loss 
    remains very low while the discriminator make an inaccurate job. This kind of observations means that the generator
    is performing well in fooling the discriminator. So with such a "bad" discriminator we won't have good performances 
    on our model 
    
2.  This kind of results means that the discriminator training is ahead from the generators's training. It means that
    discriminator has a good ability to recognize a real image from a generated image, while the generator is still
    trying to fool the discriminator by learning from its results. 

"""

part3_q3 = r"""
Before comparing the results let's remind how the two model work. On the one hand the VAE is learning how to encode 
picture in the best way in order to allow the decoder to generate the most realistic pictures. So in this model the 
encoder and the decoder are "working together". On the other hand, GANs are opposing two components (Generative
**Adversarial** Network). The generator is trained two generate fake image that will be able to fool the discriminator,
while the discriminator tries to understand how the generator is working in order to not be fooled with fake image.
We can observe better results on the VAE, the faces are more precise than the GAN. The reason is that the VAE is 
learning directly from the images, so when encoding the pictures, it will focus the most on the faces that are similar
between images and less focus on the background (which is blur and similar on most of the generated samples). 
Moreover we can assume that the poor results of the GAN is due to the fact that the training is hard, indeed the 
generator and the discriminator are "playing" against each other. So the improvement of one means the deterioration
of the other so this kind of situation may affect our results. While the training of the VAE is similar to the training 
that we are used to see in neural networks 
"""

# ==============

#%%
