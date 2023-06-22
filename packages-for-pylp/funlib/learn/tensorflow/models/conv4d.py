# -*- coding: utf-8 -*-
import tensorflow as tf


def conv4d(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1, 1, 1),
        padding='valid',
        data_format='channels_last',
        dilation_rate=(1, 1, 1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        trainable=True,
        name=None,
        reuse=None):
    '''Performs a 4D convolution of the ``(t, z, y, x)`` dimensions of a tensor
    with shape ``(b, c, l, d, h, w)`` with ``k`` filters. The output tensor
    will be of shape ``(b, k, l', d', h', w')``. ``(l', d', h', w')`` will be
    smaller than ``(l, d, h, w)`` if a ``valid`` padding was chosen.

    This operator realizes a 4D convolution by performing several 3D
    convolutions. The following example demonstrates how this works for a 2D
    convolution as a sequence of 1D convolutions::

        I.shape == (h, w)
        k.shape == (U, V) and U%2 = V%2 = 1

        # we assume kernel is indexed as follows:
        u in [-U/2,...,U/2]
        v in [-V/2,...,V/2]

        (k*I)[i,j] = Σ_u Σ_v k[u,v] I[i+u,j+v]
                   = Σ_u (k[u]*I[i+u])[j]
        (k*I)[i]   = Σ_u k[u]*I[i+u]
        (k*I)      = Σ_u k[u]*I_u, with I_u[i] = I[i+u] shifted I by u

        Example:

            I = [
                [0,0,0],
                [1,1,1],
                [1,1,0],
                [1,0,0],
                [0,0,1]
            ]

            k = [
                [1,1,1],
                [1,2,1],
                [1,1,3]
            ]

            # convolve every row in I with every row in k, comments show output
            # row the convolution contributes to
            (I*k[0]) = [
                [0,0,0], # I[0] with k[0] ⇒ (k*I)[ 1] ✔
                [2,3,2], # I[1] with k[0] ⇒ (k*I)[ 2] ✔
                [2,2,1], # I[2] with k[0] ⇒ (k*I)[ 3] ✔
                [1,1,0], # I[3] with k[0] ⇒ (k*I)[ 4] ✔
                [0,1,1]  # I[4] with k[0] ⇒ (k*I)[ 5]
            ]
            (I*k[1]) = [
                [0,0,0], # I[0] with k[1] ⇒ (k*I)[ 0] ✔
                [3,4,3], # I[1] with k[1] ⇒ (k*I)[ 1] ✔
                [3,3,1], # I[2] with k[1] ⇒ (k*I)[ 2] ✔
                [2,1,0], # I[3] with k[1] ⇒ (k*I)[ 3] ✔
                [0,1,2]  # I[4] with k[1] ⇒ (k*I)[ 4] ✔
            ]
            (I*k[2]) = [
                [0,0,0], # I[0] with k[2] ⇒ (k*I)[-1]
                [4,5,2], # I[1] with k[2] ⇒ (k*I)[ 0] ✔
                [4,2,1], # I[2] with k[2] ⇒ (k*I)[ 1] ✔
                [1,1,0], # I[3] with k[2] ⇒ (k*I)[ 2] ✔
                [0,3,1]  # I[4] with k[2] ⇒ (k*I)[ 3] ✔
            ]

            # the sum of all valid output rows gives k*I (here shown for row 2)
            (k*I)[2] = (
                [2,3,2] +
                [3,3,1] +
                [1,1,0] +
            ) = [6,7,3]
    '''

    # check arguments
    assert len(inputs.get_shape().as_list()) == 6, (
        "Tensor of shape (b, c, l, d, h, w) expected")
    assert isinstance(kernel_size, int) or len(kernel_size) == 4, (
        "kernel size should be an integer or a 4D tuple, not %s" % kernel_size)
    if isinstance(strides, int):
        strides = (strides,)*4
    assert strides == (1, 1, 1, 1), (
        "Strides other than 1 not yet implemented")
    assert data_format == 'channels_first', (
        "Data format other than 'channels_first' not yet implemented")
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate,)*4
    assert dilation_rate == (1, 1, 1, 1), (
        "Dilation rate other than 1 not yet implemented")

    if not name:
        name = 'conv4d'

    # input, kernel, and output sizes
    (b, c_i, l_i, d_i, h_i, w_i) = tuple(inputs.get_shape().as_list())
    if isinstance(kernel_size, int):
        (l_k, d_k, h_k, w_k) = (kernel_size,)*4
    else:
        (l_k, d_k, h_k, w_k) = kernel_size

    # output size for 'valid' convolution
    if padding.lower() == 'valid':
        l_o = l_i - l_k + 1
    else:
        l_o = l_i

    if b is None:
        b = -1

    # output tensors for each 3D frame
    frame_results = [None]*l_o

    # convolve each kernel frame i with each input frame j
    for i in range(l_k):

        # reuse variables of previous 3D convolutions for the same kernel
        # frame (or if the user indicated to have all variables reused)
        reuse_kernel = reuse

        for j in range(l_i):

            # add results to this output frame
            out_frame = j - (i - l_k//2) - (l_i - l_o)//2
            if out_frame < 0 or out_frame >= l_o:
                continue

            # convolve input frame j with kernel frame i
            frame_conv3d = tf.layers.conv3d(
                tf.reshape(inputs[:, :, j, :], (b, c_i, d_i, h_i, w_i)),
                filters,
                kernel_size=(d_k, h_k, w_k),
                padding=padding,
                data_format='channels_first',
                activation=None,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                trainable=trainable,
                name=name + '_3dchan%d' % i,
                reuse=reuse_kernel)

            # subsequent frame convolutions should use the same kernel
            reuse_kernel = True

            if frame_results[out_frame] is None:
                frame_results[out_frame] = frame_conv3d
            else:
                frame_results[out_frame] += frame_conv3d

    output = tf.stack(frame_results, axis=2)

    if activation:
        if isinstance(activation, str):
            output = tf.keras.activations.get(activation)(output)
        else:
            output = activation(output)

    return output
