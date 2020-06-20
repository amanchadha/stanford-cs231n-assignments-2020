import tensorflow as tf
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images, numpy array of shape (N, H, W, 3)
    - y: Labels for X, numpy of shape (N,)
    - model: A SqueezeNet model that will be used to compute the saliency map.

    Returns:
    - saliency: A numpy array of shape (N, H, W) giving the saliency maps for the
    input images.
    """
    saliency = None
    # Compute the score of the correct class for each example.
    # This gives a Tensor with shape [N], the number of examples.
    #
    # Note: this is equivalent to scores[np.arange(N), y] we used in NumPy
    # for computing vectorized losses.

    ###############################################################################
    # TODO: Produce the saliency maps over a batch of images.                     #
    #                                                                             #
    # 1) Define a gradient tape object and watch input Image variable             #
    # 2) Compute the “loss” for the batch of given input images.                  #
    #    - get scores output by the model for the given batch of input images     #
    #    - use tf.gather_nd or tf.gather to get correct scores                    #
    # 3) Use the gradient() method of the gradient tape object to compute the     #
    #    gradient of the loss with respect to the image                           #
    # 4) Finally, process the returned gradient to compute the saliency map.      #
    ###############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    X_var = tf.Variable(X, dtype="float32")
    y_var = tf.Variable(y, dtype="int32")
    
    # 1) Define a gradient tape object and watch input Image variable             #
    with tf.GradientTape() as tape:
        # keep track of our gradients
        tape.watch(X_var)

        # 2) Compute the “loss” for the batch of given input images.                  #
        #    - get scores output by the model for the given batch of input images     #
        # make a prediction using the model and then calculate the loss
        # Make sure the model is in "test" mode        
        scores = model(X_var, training=False)
        
        # use tf.gather_nd or tf.gather to get correct scores                    #
        correct_scores = tf.gather_nd(scores, tf.stack((tf.range(X_var.shape[0], dtype="int32"), y_var), axis=1))
    
    # 3) Use the gradient() method of the gradient tape object to compute the     #
    #    gradient of the loss with respect to the image                           #
    grads = tape.gradient(correct_scores, X_var)

    # 4) Finally, process the returned gradient to compute the saliency map.      #
    # note that torch uses the channel-first notation so axis=3
    saliency = np.abs(grads).max(axis=3)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return saliency

def make_fooling_image(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image, a numpy array of shape (1, 224, 224, 3)
    - target_y: An integer in the range [0, 1000)
    - model: Pretrained SqueezeNet model

    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
    """

    # Make a copy of the input that we will modify
    X_fooling = X.copy()

    # Step size for the update
    learning_rate = 1

    ##############################################################################
    # TODO: Generate a fooling image X_fooling that the model will classify as   #
    # the class target_y. Use gradient *ascent* on the target class score, using #
    # the model.scores Tensor to get the class scores for the model.image.   #
    # When computing an update step, first normalize the gradient:               #
    #   dX = learning_rate * g / ||g||_2                                         #
    #                                                                            #
    # You should write a training loop, where in each iteration, you make an     #
    # update to the input image X_fooling (don't modify X). The loop should      #
    # stop when the predicted class for the input is the same as target_y.       #
    #                                                                            #
    # HINT: Use tf.GradientTape() to keep track of your gradients and            #
    # use tape.gradient to get the actual gradient with respect to X_fooling.    #
    #                                                                            #
    # HINT 2: For most examples, you should be able to generate a fooling image  #
    # in fewer than 100 iterations of gradient ascent. You can print your        #
    # progress over iterations to check your algorithm.                          #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    X_fooling_var = tf.Variable(X_fooling)

    for i in range(100): # try out 100 iterations max
        with tf.GradientTape() as tape:
            # keep track of our gradients
            tape.watch(X_fooling_var)

            # make a prediction using the model
            # loss is the score we want to maximize            
            scores = model(X_fooling_var, training=False)

            # get the predicted class
            idx = tf.math.argmax(scores, axis=1) # scores is (N, C)

            if (idx != target_y):
                # use the gradient() method of the gradient tape object to compute the
                # gradient of the loss with respect to the image
                grads = tape.gradient(scores[:, target_y], X_fooling_var)
                
                # when computing an update step, first normalize the gradient
                # default for torch.norm is "frobenius norm" for matrices and "2-norm" for vectors
                dX = learning_rate * grads/tf.norm(grads, ord=2)

                # perform gradient ascent
                X_fooling_var = X_fooling_var + dX
            else:
                # stop when the model is fooled
                break 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return X_fooling_var

def class_visualization_update_step(X, model, target_y, l2_reg, learning_rate):
    ########################################################################
    # TODO: Compute the value of the gradient of the score for             #
    # class target_y with respect to the pixels of the image, and make a   #
    # gradient step on the image using the learning rate. You should use   #
    # the tf.GradientTape() and tape.gradient to compute gradients.        #
    #                                                                      #
    # Be very careful about the signs of elements in your code.            #
    ########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    X_var = tf.Variable(X)

    with tf.GradientTape() as tape:
        # keep track of our gradients
        tape.watch(X_var)

        # generate scores
        scores = model(X_var, training=False)

        # compute the gradient of the score for the class target_y with the L2 reg
        loss = scores[:, target_y] - l2_reg * tf.square(tf.norm(X_var, ord=2))

        grads = tape.gradient(loss, X_var)

    # make a gradient step on the image using the learning rate and 
    # gradient of of the score for the class target_y with respect to the pixels of the image
    X_var = X_var + learning_rate * grads 
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return X_var

def blur_image(X, sigma=1):
    X = gaussian_filter1d(X, sigma, axis=1)
    X = gaussian_filter1d(X, sigma, axis=2)
    return X

def jitter(X, ox, oy):
    """
    Helper function to randomly jitter an image.

    Inputs
    - X: Tensor of shape (N, H, W, C)
    - ox, oy: Integers giving number of pixels to jitter along W and H axes

    Returns: A new Tensor of shape (N, H, W, C)
    """
    if ox != 0:
        left = X[:, :, :-ox]
        right = X[:, :, -ox:]
        X = tf.concat([right, left], axis=2)
    if oy != 0:
        top = X[:, :-oy]
        bottom = X[:, -oy:]
        X = tf.concat([bottom, top], axis=1)
    return X
