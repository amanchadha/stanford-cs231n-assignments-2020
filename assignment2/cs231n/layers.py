from builtins import range
import numpy as np



def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dim_size = x[0].shape

    # (N, D) x (D, M) + (1, M)
    out = x.reshape(x.shape[0], np.prod(dim_size)).dot(w) + b.reshape(1, -1) 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dim_shape = np.prod(x[0].shape)
    N = x.shape[0]
    X = x.reshape(N, dim_shape)

    # downstream gradient = upstream gradient * local gradient
    # local gradient is computed by out = wx + b
    dx = dout.dot(w.T) # (N x M) x (M x D) = (N x D)
    dx = dx.reshape(x.shape)

    dw = X.T.dot(dout) # (D x N) x (N x M) = (D x M)
    
    db = dout.sum(axis = 0) # (N x M), so sum over all N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(0, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # in the backward pass, ReLU acts as a switch, letting gradients pass through
    # if they are > 0, else zeroing them out

    dx = dout
    dx[x < 0] = 0

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)
    layernorm = bn_param.get("layernorm", 0)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # compute the sample mean and variance from mini-batch statistics 
        # using minimal-num-of-operations-per-step policy to ease the backward pass

        # (1) mini-batch mean by averaging over each sample (N) in a minibatch 
        # for a particular column / feature dimension (D)
        mean = x.mean(axis = 0) # (D,)
        # can also do mean = 1./N * np.sum(x, axis = 0)

        # (2) subtract mean vector of every training example
        dev_from_mean = x - mean # (N,D)

        # (3) following the lower branch for the denominator
        dev_from_mean_sq = dev_from_mean ** 2 # (N,D)
  
        # (4) mini-batch variance
        var = 1./N * np.sum(dev_from_mean_sq, axis = 0) # (D,)
        # can also do var = x.var(axis = 0)

        # (5) get std dev from variance, add eps for numerical stability
        stddev = np.sqrt(var + eps) # (D,)

        # (6) invert the above expression to make it the denominator
        inverted_stddev = 1./stddev # (D,)

        # (7) apply normalization
        # note that this is an element-wise multiplication using broad-casting
        x_norm = dev_from_mean * inverted_stddev # also called z or x_hat (N,D)

        # (8) apply scaling parameter gamma to x
        scaled_x = gamma * x_norm # (N,D)

        # (9) shift x by beta
        out = scaled_x + beta # (N,D)
        
        # cache values for backward pass
        cache = {'mean': mean, 'stddev': stddev, 'var': var, 'gamma': gamma, 
                 'beta': beta, 'eps': eps, 'x_norm': x_norm, 'dev_from_mean': dev_from_mean,
                 'inverted_stddev': inverted_stddev, 'x': x}

        # since we transpose dout and make it (D,N) during backprop for layernorm
        cache['axis'] = 1 if layernorm else 0

        # also keep an exponentially decaying running weighted mean of the mean and 
        # variance of each feature, to normalize data at test-time
        if not layernorm:
            running_mean = momentum * running_mean + (1 - momentum) * mean
            running_var = momentum * running_var + (1 - momentum) * var

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # normalize the incoming data based on the standard deviation (sqrt(variance))
        z = (x - running_mean)/np.sqrt(running_var + eps)

        # scale and shift
        out = gamma * z + beta        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # forward pass ->
        # sample_mean = x.mean(axis = 0)  # x.shape is (N, D) so summing over 
                                          # each example in a minibatch
        # sample_var = x.var(axis = 0) + eps
        # sample_stddev = np.sqrt(sample_var)
        # z = (x - sample_mean)/sample_stddev
        # out = gamma * z + beta

    # Per, https://piazza.com/class/k76zko2awvo7jc?cid=572
    # batchnorm_backward is supposed to be computed using a staged-computation process
    # similar to https://cs231n.github.io/optimization-2/

    # inspired from:
    # https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html

    # convention used is downstream gradient = local gradient * upstream gradient

    # extract all relevant params
    beta, gamma, x_norm, var, eps, stddev, dev_from_mean, inverted_stddev, x, mean, axis = \
    cache['beta'], cache['gamma'], cache['x_norm'], cache['var'], cache['eps'], \
    cache['stddev'], cache['dev_from_mean'], cache['inverted_stddev'], cache['x'], \
    cache['mean'], cache['axis']
 
    # get the num of training examples and dimensionality of the input (num of features)
    N, D = dout.shape # can also use x.shape

    # (9)
    dbeta = np.sum(dout, axis=axis)
    dscaled_x = dout 

    # (8)
    dgamma = np.sum(x_norm * dscaled_x, axis=axis)
    dx_norm = gamma * dscaled_x

    # (7)
    dinverted_stddev = np.sum(dev_from_mean * dx_norm, axis=0)
    ddev_from_mean = inverted_stddev * dx_norm

    # (6)
    dstddev = -1/(stddev**2) * dinverted_stddev

    # (5)
    dvar = (0.5) * 1/np.sqrt(var + eps) * dstddev

    # (4)
    ddev_from_mean_sq = 1/N * np.ones((N,D)) * dvar # variance of mean is 1/N

    # (3)
    ddev_from_mean += 2 * dev_from_mean * ddev_from_mean_sq

    # (2)
    dx = 1 * ddev_from_mean
    dmean = -1 * np.sum(ddev_from_mean, axis=0)

    # (1)
    dx += 1./N * np.ones((N,D)) * dmean
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # batchnorm_backward_alt is supposed to be computed by first writing our the 
    # gradients using pen and paper, simplifying as much as possible to obtain a 
    # compact expression, and converting that to python code.
    # the sequence of derivatives calculated in the paper refer to this (second) process

    # inspired from: http://cthorey.github.io./backpropagation/

    # convention used is downstream gradient = local gradient * upstream gradient

    # extract all relevant params
    beta, gamma, x_norm, var, eps, stddev, dev_from_mean, inverted_stddev, mean, x, axis = \
    cache['beta'], cache['gamma'], cache['x_norm'], cache['var'], cache['eps'], \
    cache['stddev'], cache['dev_from_mean'], cache['inverted_stddev'], cache['mean'], \
    cache['x'], cache['axis']

    # get the num of training examples and dimensionality of the input (num of features)
    N = dout.shape[0] # can also use x.shape

    # (9)
    dbeta = np.sum(dout, axis=axis)
    dscaled_x = dout 

    # (8)
    dgamma = np.sum((x - mean) * (var + eps)**(-1. / 2.) * dout, axis=axis)

    dmean = 1/N * np.sum(dout, axis=0)
    dvar = 2/N * np.sum(dev_from_mean * dout, axis=0)
    dstddev = dvar/(2 * stddev)
    dx = gamma*((dout - dmean)*stddev - dstddev*(dev_from_mean))/stddev**2    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # under layernorm, all the hidden units in a layer share the same normalization 
    # terms (mean and variance), but different training cases have different 
    # normalization terms. also, unlike batchnorm, layernorm does not impose any 
    # constraint on the size of a mini-batch and it can be used in the pure online 
    # regime with batch size

    ln_param['mode'] = 'train' # same as batchnorm in train mode
    
    # forward pass remains the same as batchnorm since the below matrices are 
    # transposed, this gets passed on to the backward pass and changes it a bit
    ln_param['layernorm'] = 1 
    
    # transpose x, gamma and beta
    out, cache = batchnorm_forward(x.T, gamma.reshape(-1,1), \
                                   beta.reshape(-1,1), ln_param)

    # transpose output to get original dims
    out = out.T # (N,D) 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # transpose dout because we transposed the input, x, during the forward pass
    dx, dgamma, dbeta = batchnorm_backward(dout.T, cache)

    # transpose gradients w.r.t. input, x, to their original dims
    dx = dx.T # (N,D)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # dropout mask. note "/p" for inverted dropout!
        # use a uniform probability distribution so as to be able to compare with 'p' easily
        mask = (np.random.rand(*x.shape) < p) / p 

        out = x * mask
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # convention used is downstream gradient = local gradient * upstream gradient
        dx = mask * dout

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = 1 * dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # extract params 
    pad = conv_param['pad']
    stride = conv_param['stride']
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    # if "(N + 2 * pad - F)/s" does not yield an int, that means our pad/stride 
    # setting is wrong
    assert (H + 2 * pad - HH) % stride == 0, '[Sanity Check] [FAIL]: Conv Layer Failed in Height'
    assert (W + 2 * pad - WW) % stride == 0, '[Sanity Check] [FAIL]: Conv Layer Failed in Width'

    # output volume size
    # note that the // division yields an int (while / yields a float)
    Hout = (H + 2 * pad - HH) // stride + 1 
    Wout = (W + 2 * pad - WW) // stride + 1

    # create output volume tensor after convolution
    out = np.zeros((N, F, Hout, Wout))

    # pad H and W axes of the input data, 0 is the default constant for np.pad
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    # naive Loops
    for n in range(N): # for each neuron
        for f in range(F): # for each filter/kernel
            for i in range(0, Hout): # for each y activation
                for j in range(0, Wout): # for each x activation
                    # each neuron in a particular depth slide in the output volume
                    # shares weights over the same HH x WW x C region they're 
                    # looking at in the image; also one bias/filter
                    out[n, f, i, j] = (x_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] * w[f, :, :, :]).sum() + b[f]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # extract params 
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param.get('stride', 1)
    pad = conv_param.get('pad', 0)

    # pad H and W axes of the input data, 0 is the default constant for np.pad
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    # output volume size
    # note that the // division yields an int (while / yields a float)
    Hout = (H + 2 * pad - HH) // stride + 1 
    Wout = (W + 2 * pad - WW) // stride + 1

    # construct output
    dx_pad = np.zeros_like(x_pad)
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    
    # naive Loops
    for n in range(N): # for each neuron
        for f in range(F): # for each filter/kernel
            db[f] += dout[n, f].sum() # one bias/filter
            for i in range(0, Hout): # for each y activation
                for j in range(0, Wout): # for each x activation
                    dw[f] += x_pad[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW] * dout[n, f, i, j]
                    dx_pad[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW] += w[f] * dout[n, f, i, j]
    
    # extract dx from dx_pad since dx.shape needs to match x.shape
    dx = dx_pad[:, :, pad:pad+H, pad:pad+W]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # extract params 
    N, C, H, W = x.shape
    HH = pool_param.get('pool_height', 2)
    WW = pool_param.get('pool_width', 2)
    stride = pool_param.get('stride', 2)

    # if "(N - F)/s" does not yield an int, that means our pad/stride 
    # setting is wrong
    assert (H - HH) % stride == 0, '[Sanity Check] [FAIL]: Conv Layer Failed in Height'
    assert (W - WW) % stride == 0, '[Sanity Check] [FAIL]: Conv Layer Failed in Width'

    # output volume size
    # note that the // division yields an int (while / yields a float)
    Hout = (H - HH) // stride + 1
    Wout = (W - WW) // stride + 1

    # create output volume tensor after maxpool
    out = np.zeros((N, C, Hout, Wout)) # output has same dims NCHW format as input

    # naive Loops
    for n in range(N): # for each neuron
        for i in range(Hout): # for each y activation
            for j in range(Wout): # for each x activation
                out[n, :, i, j] = np.amax(x[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW], axis=(-1, -2))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Extract constants and shapes
    x, pool_param = cache
    N, C, H, W = x.shape
    HH = pool_param.get('pool_height', 2)
    WW = pool_param.get('pool_width', 2)
    stride = pool_param.get('stride', 2)

    # output volume size
    # note that the // division yields an int (while / yields a float)
    Hout = (H - HH) // stride + 1 
    Wout = (W - WW) // stride + 1

    # output volume size
    dx = np.zeros_like(x)
    
    # naive loops
    for n in range(N): # for each neuron
        for c in range(C): # for each channel
            for i in range(Hout): # for each y activation
                for j in range(Wout): # for each x activation
                    # pass gradient only through indices of max pool
                    ind = np.argmax(x[n, c, i*stride:i*stride+HH, j*stride:j*stride+WW])
                    ind1, ind2 = np.unravel_index(ind, (HH, WW))
                    dx[n, c, i*stride:i*stride+HH, j*stride:j*stride+WW][ind1, ind2] = dout[n, c, i, j]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape

    # transpose to a channel-last notation (N, H, W, C) and then reshape it to 
    # norm over N*H*W for each C
    x = x.transpose(0, 2, 3, 1).reshape(N*H*W, C)

    out, cache = batchnorm_forward(x, gamma, beta, bn_param)

    # transpose the output back to N, C, H, W
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape

    # transpose to a channel-last notation (N, H, W, C) and then reshape it to 
    # norm over N*H*W for each C
    dout = dout.transpose(0, 2, 3, 1).reshape(N*H*W, C)

    dx, dgamma, dbeta = batchnorm_backward_alt(dout, cache)

    # transpose the output back to N, C, H, W
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)    

    # transpose the output back to N, C, H, W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # key idea of Groupnorm: compute mean and variance statistics by dividing 
    # each datapoint into G groups 
    # gamma/beta (shift/scale) are per channel

    # using minimal-num-of-operations-per-step policy to ease the backward pass  

    N, C, H, W = x.shape
    size = (N*G, C//G * H * W) # in groupnorm, D = C//G * H * W

    # (0) rehsape X to accommodate G
    # divide each sample into G groups (G new samples)
    x = x.reshape((N*G, -1)) # reshape to same as size # reshape NxCxHxW ==> N*GxC/GxHxW =N1*C1 (N1>N*Groups)

    # (1) mini-batch mean by averaging over a particular column / feature dimension (D)
    # over each sample (N) in a minibatch 
    mean = x.mean(axis = 1, keepdims= True) # (N,1) # sum through D
    # can also do mean = 1./N * np.sum(x, axis = 1)

    # (2) subtract mean vector of every training example
    dev_from_mean = x - mean # (N,D)

    # (3) following the lower branch for the denominator
    dev_from_mean_sq = dev_from_mean ** 2 # (N,D)

    # (4) mini-batch variance
    var = 1./size[1] * np.sum(dev_from_mean_sq, axis = 1, keepdims= True) # (N,1)
    # can also do var = x.var(axis = 0)

    # (5) get std dev from variance, add eps for numerical stability
    stddev = np.sqrt(var + eps) # (N,1)

    # (6) invert the above expression to make it the denominator
    inverted_stddev = 1./stddev # (N,1)

    # (7) apply normalization
    # note that this is an element-wise multiplication using broad-casting
    x_norm = dev_from_mean * inverted_stddev # also called z or x_hat (N,D) 
    x_norm = x_norm.reshape(N, C, H, W)

    # (8) apply scaling parameter gamma to x
    scaled_x = gamma * x_norm # (N,D)

    # (9) shift x by beta
    out = scaled_x + beta # (N,D)

    # backprop sum axis
    axis = (0, 2, 3)

    # cache values for backward pass
    cache = {'mean': mean, 'stddev': stddev, 'var': var, 'gamma': gamma, \
             'beta': beta, 'eps': eps, 'x_norm': x_norm, 'dev_from_mean': dev_from_mean, \
             'inverted_stddev': inverted_stddev, 'x': x, 'axis': axis, 'size': size, 'G': G, 'scaled_x': scaled_x}
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # convention used is downstream gradient = local gradient * upstream gradient
    # extract all relevant params
    beta, gamma, x_norm, var, eps, stddev, dev_from_mean, inverted_stddev, x, mean, axis, size, G, scaled_x = \
    cache['beta'], cache['gamma'], cache['x_norm'], cache['var'], cache['eps'], \
    cache['stddev'], cache['dev_from_mean'], cache['inverted_stddev'], cache['x'], \
    cache['mean'], cache['axis'], cache['size'], cache['G'], cache['scaled_x']

    N, C, H, W = dout.shape
    
    # (9)
    dbeta = np.sum(dout, axis = (0,2,3), keepdims = True) #1xCx1x1
    dscaled_x = dout # N1xC1xH1xW1

    # (8)
    dgamma = np.sum(dscaled_x * x_norm,axis = (0,2,3), keepdims = True) # N = sum_through_D,W,H([N1xC1xH1xW1]xN1xC1xH1xW1)
    dx_norm = dscaled_x * gamma # N1xC1xH1xW1 = [N1xC1xH1xW1] x[1xC1x1x1]
    dx_norm = dx_norm.reshape(size) #(N1*G,C1//G*H1*W1)

    # (7)
    dinverted_stddev = np.sum(dx_norm * dev_from_mean, axis = 1, keepdims = True) # N = sum_through_D([NxD].*[NxD]) =4×60
    ddev_from_mean = dx_norm * inverted_stddev #[NxD] = [NxD] x [Nx1]

    # (6)
    dstddev = (-1/(stddev**2)) * dinverted_stddev # N = N x [N]

    # (5)
    dvar = 0.5 * (1/np.sqrt(var + eps)) * dstddev # N = [N+const]xN

    # (4)
    ddev_from_mean_sq = (1/size[1]) * np.ones(size) * dvar # NxD = NxD*N

    # (3)    
    ddev_from_mean += 2 * dev_from_mean * ddev_from_mean_sq # [NxD] = [NxD]*[NxD]

    # (2)
    dx = (1) * ddev_from_mean # [NxD] = [NxD]
    dmean = -1 * np.sum(ddev_from_mean, axis = 1, keepdims = True) # N = sum_through_D[NxD]

    # (1) cache
    dx += (1/size[1]) * np.ones(size) * dmean # NxD (N= N1*Groups) += [NxD]XN

    # (0):
    dx = dx.reshape(N, C, H, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
