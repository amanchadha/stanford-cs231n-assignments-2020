import numpy as np
import tensorflow as tf

NOISE_DIM = 96

def leaky_relu(x, alpha=0.01):
    """Compute the leaky ReLU activation function.
    
    Inputs:
    - x: TensorFlow Tensor with arbitrary shape
    - alpha: leak parameter for leaky ReLU
    
    Returns:
    TensorFlow Tensor with the same shape as x
    """
    # TODO: implement leaky ReLU
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return tf.maximum(alpha*x, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
def sample_noise(batch_size, dim, seed=None):
    """Generate random uniform noise from -1 to 1.
    
    Inputs:
    - batch_size: integer giving the batch size of noise to generate
    - dim: integer giving the dimension of the noise to generate
    
    Returns:
    TensorFlow Tensor containing uniform noise in [-1, 1] with shape [batch_size, dim]
    """
    if seed is not None:
        tf.random.set_seed(seed)
    # TODO: sample and return noise
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return tf.random.uniform([batch_size,dim], -1, 1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
def discriminator(seed=None):
    """Compute discriminator score for a batch of input images.
    
    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]
    
    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score 
    for an image being real for each input image.
    """
    if seed is not None:
        tf.random.set_seed(seed)

    model = None
    ##############################################################################
    # TODO: Implement architecture                                               #
    #                                                                            #
    # HINT: tf.keras.models.Sequential might be helpful.                         #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    model = tf.keras.models.Sequential()

    # (1) Fully connected layer from size 784 to 256 with LeakyReLU with alpha 0.01
    model.add(tf.keras.layers.Dense(units=256, input_shape=(784,), activation=leaky_relu))

    # (2) Fully connected layer from 256 to 256 with LeakyReLU with alpha 0.01
    model.add(tf.keras.layers.Dense(units=256, activation=leaky_relu))

    # (3) Fully connected layer from 256 to 1
    model.add(tf.keras.layers.Dense(units=1))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return model

def generator(noise_dim=NOISE_DIM, seed=None):
    """Generate images from a random noise vector.
    
    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]
    
    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 784].
    """

    if seed is not None:
        tf.random.set_seed(seed)
    model = None
    ##############################################################################
    # TODO: Implement architecture                                               #
    #                                                                            #
    # HINT: tf.keras.models.Sequential might be helpful.                         #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    model = tf.keras.models.Sequential()

    # (1) Fully connected layer from noise.shape to 1024 with ReLU
    model.add(tf.keras.layers.Dense(units=1024, input_shape=(noise_dim,), activation=tf.nn.relu))
    
    # (2) Fully connected layer from 1024 to 1024 with ReLU
    model.add(tf.keras.layers.Dense(units=1024, activation=tf.nn.relu))
    
    # (3) Fully connected layer from 1024 to 784 with TanH (To restrict to [-1,1])
    model.add(tf.keras.layers.Dense(units=784, activation=tf.nn.tanh))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return model

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.
    
    Inputs:
    - logits_real: Tensor of shape (N, 1) giving scores for the real data.
    - logits_fake: Tensor of shape (N, 1) giving scores for the fake data.
    
    Returns:
    - loss: Tensor containing (scalar) the loss for the discriminator.
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    cross_entropy_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real), 
                                                                 logits=logits_real)
    cross_entropy_non_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake), 
                                                                     logits=logits_fake)
    
    loss = tf.reduce_mean(cross_entropy_real) + tf.reduce_mean(cross_entropy_non_fake)
        
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    cross_entropy_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake), 
                                                                 logits=logits_fake)
    
    loss = tf.reduce_mean(cross_entropy_fake)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss


def get_solvers(learning_rate=1e-3, beta1=0.5):
    """Create solvers for GAN training.
    
    Inputs:
    - learning_rate: learning rate to use for both solvers
    - beta1: beta1 parameter for both solvers (first moment decay)
    
    Returns:
    - D_solver: instance of tf.optimizers.Adam with correct learning_rate and beta1
    - G_solver: instance of tf.optimizers.Adam with correct learning_rate and beta1
    """
    # TODO: create an AdamOptimizer for D_solver and G_solver
    D_solver = None
    G_solver = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    D_solver = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta1)
    G_solver = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta1)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return D_solver, G_solver

def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: Tensor of shape (N, 1) giving scores for the real data.
    - scores_fake: Tensor of shape (N, 1) giving scores for the fake data.
    
    Outputs:
    - loss: A Tensor containing the loss.
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    loss = 0.5 * (tf.reduce_mean((scores_real - 1) ** 2) + tf.reduce_mean((scores_fake ** 2)))
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss

def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: Tensor of shape (N, 1) giving scores for the fake data.
    
    Outputs:
    - loss: A Tensor containing the loss.
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    loss = 0.5 * tf.reduce_mean((scores_fake - 1) ** 2)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss

def dc_discriminator():
    """Compute discriminator score for a batch of input images.
    
    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]
    
    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score 
    for an image being real for each input image.
    """
    model = None
    ##############################################################################
    # TODO: Implement architecture                                               #
    #                                                                            #
    # HINT: tf.keras.models.Sequential might be helpful.                         #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    model = tf.keras.models.Sequential()

    # Unflatten input
    # e.g., model.add(tf.keras.layers.Reshape((3, 4), input_shape=(12,)))
    # yields model.output_shape == (None, 3, 4), where "None" is the batch size, per
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Reshape
    model.add(tf.keras.layers.Reshape((28, 28, 1), input_shape=(784,)))
    
    # (1) 32 Filters, 5x5, Stride 1, Leaky ReLU(alpha=0.01) -> (None, 24, 24, 32)
    # feed in leaky_relu() as a parameter to the layers to make it an activation function
    # padding='valid' is the default
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, 
                                     input_shape=(28, 28, 1), activation=leaky_relu))
    
    # (2) Max Pool 2x2, Stride 2 -> (None, 12, 12, 32)
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    
    # (3) 64 Filters, 5x5, Stride 1, Leaky ReLU(alpha=0.01) -> (None, 8, 8, 64)
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, 
                                     activation=leaky_relu))
    
    # (4) Max Pool 2x2, Stride 2 -> (None, 4, 4, 64)
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    
    # (5) Flatten -> (None, 4*4*64)
    model.add(tf.keras.layers.Flatten())
    #or model.add(tf.keras.layers.Reshape((4*4*64,), input_shape=(4,4,64)))
    
    # (6) Fully Connected size 4 x 4 x 64, Leaky ReLU(alpha=0.01)
    model.add(tf.keras.layers.Dense(units=4*4*64, activation=leaky_relu))
    
    # (7) Fully Connected size 1
    model.add(tf.keras.layers.Dense(units=1))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return model


def dc_generator(noise_dim=NOISE_DIM):
    """Generate images from a random noise vector.
    
    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]
    
    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 784].
    """
    model = tf.keras.models.Sequential()
    # TODO: implement architecture
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # (1) Fully connected of size 1024, ReLU
    model.add(tf.keras.layers.Dense(units=1024, input_shape=(noise_dim,), activation=tf.nn.relu))
    
    # (2) BatchNorm
    model.add(tf.keras.layers.BatchNormalization())#training=True))
    
    # (3) Fully connected of size 7 x 7 x 128, ReLU
    model.add(tf.keras.layers.Dense(units=7*7*128, activation=tf.nn.relu))
    
    # (4) BatchNorm
    model.add(tf.keras.layers.BatchNormalization())#training=True))
    
    # (5) Resize into Image Tensor
    model.add(tf.keras.layers.Reshape((7, 7, 128)))#, input_shape=(6272,)))

    # (6) 64 conv2d^T (transpose) filters of 4x4, stride 2, ReLU
    model.add(tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4,
                                              strides=2, padding='same', 
                                              input_shape=(7, 7, 128),
                                              activation=tf.nn.relu))
    
    # (7) BatchNorm
    model.add(tf.keras.layers.BatchNormalization())#training=True)
    
    # (8) 1 conv2d^T (transpose) filter of 4x4, stride 2, TanH
    model.add(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=4, 
                                              strides=2, padding='same', 
                                              activation=tf.nn.tanh))
    
    # (*) Flatten output
    model.add(tf.keras.layers.Reshape((784,)))#, input_shape=(28,28,1)))
    # or model.add(tf.keras.layers.Reshape((-1, 784)))#, input_shape=(28,28,1)))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return model


# a giant helper function
def run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss,\
              show_every=250, print_every=20, batch_size=128, num_epochs=10, noise_size=96):
    """Train a GAN for a certain number of epochs.
    
    Inputs:
    - D: Discriminator model
    - G: Generator model
    - D_solver: an Optimizer for Discriminator
    - G_solver: an Optimizer for Generator
    - generator_loss: Generator loss
    - discriminator_loss: Discriminator loss
    Returns:
        Nothing
    """
    mnist = MNIST(batch_size=batch_size, shuffle=True)
    
    iter_count = 0
    images = []
    for epoch in range(num_epochs):
        for (x, _) in mnist:
            with tf.GradientTape() as tape:
                real_data = x
                logits_real = D(preprocess_img(real_data))

                g_fake_seed = sample_noise(batch_size, noise_size)
                fake_images = G(g_fake_seed)
                logits_fake = D(tf.reshape(fake_images, [batch_size, 784]))

                d_total_error = discriminator_loss(logits_real, logits_fake)
                d_gradients = tape.gradient(d_total_error, D.trainable_variables)      
                D_solver.apply_gradients(zip(d_gradients, D.trainable_variables))
            
            with tf.GradientTape() as tape:
                g_fake_seed = sample_noise(batch_size, noise_size)
                fake_images = G(g_fake_seed)

                gen_logits_fake = D(tf.reshape(fake_images, [batch_size, 784]))
                g_error = generator_loss(gen_logits_fake)
                g_gradients = tape.gradient(g_error, G.trainable_variables)      
                G_solver.apply_gradients(zip(g_gradients, G.trainable_variables))

            if (iter_count % show_every == 0):
                print('Epoch: {}, Iter: {}, D: {:.4}, G:{:.4}'.format(epoch, iter_count,d_total_error,g_error))
                imgs_numpy = fake_images.cpu().numpy()
                images.append(imgs_numpy[0:16])
                
            iter_count += 1
    
    # random noise fed into our generator
    z = sample_noise(batch_size, noise_size)
    # generated images
    G_sample = G(z)
    
    return images, G_sample[:16]

class MNIST(object):
    def __init__(self, batch_size, shuffle=False):
        """
        Construct an iterator object over the MNIST data
        
        Inputs:
        - batch_size: Integer giving number of elements per minibatch
        - shuffle: (optional) Boolean, whether to shuffle the data on each epoch
        """
        train, _ = tf.keras.datasets.mnist.load_data()
        X, y = train
        X = X.astype(np.float32)/255
        X = X.reshape((X.shape[0], -1))
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i+B], self.y[i:i+B]) for i in range(0, N, B)) 

def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def count_params(model):
    """Count the number of parameters in the current TensorFlow graph """
    param_count = np.sum([np.prod(p.shape) for p in model.weights])
    return param_count