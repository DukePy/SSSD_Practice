import tensorflow.keras as keras
import tensorflow as tf
import opt_einsum as oe
from einops import rearrange, repeat
import tensorflow_addons as tfa
import math
import numpy as np
import scipy.special as ss

contract = oe.contract
contract_expression = oe.contract_expression

print(tf.__version__)


class TFGLU(keras.layers.Layer):
    def __init__(self, dim=-1):
        super(TFGLU, self).__init__()
        self.dim = dim

    def call(self, x):
        x1, x2 = tf.split(x, 2, self.dim)
        x2 = keras.activations.sigmoid(x2)
        y = tf.multiply(x1, x2)
        return y


def Activation(activation=None, dim=-1):
    if activation in [None, 'id', 'identity', 'linear']:
        return tf.identity
    elif activation == 'tanh':
        return keras.layers.Activation("tanh")
    elif activation == 'relu':
        return keras.layers.Activation("relu")
    elif activation == 'gelu':
        return keras.layers.Activation("gelu")
    elif activation in ['swish', 'silu']:
        return keras.layers.Activation("swish")
    elif activation == 'glu':
        return TFGLU(dim=dim)
    elif activation == 'sigmoid':
        return keras.layers.Activation("sigmoid")
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))


def get_initializer(name, activation=None):
    if activation in [None, 'id', 'identity', 'linear', 'modrelu']:
        nonlinearity = 'linear'
    elif activation in ['relu', 'tanh', 'sigmoid']:
        nonlinearity = activation
    elif activation in ['gelu', 'swish', 'silu']:
        nonlinearity = 'relu'  # Close to ReLU so approximate with ReLU's gain
    else:
        raise NotImplementedError(f"get_initializer: activation {activation} not supported")

    if name == 'uniform':
        initializer = keras.initializers.HeUniform()
    elif name == 'normal':
        initializer = keras.initializers.HeNormal()
    elif name == 'xavier':
        initializer = keras.initializers.GlorotNormal()
    elif name == 'zero':
        initializer = keras.initializers.Zeros()
    elif name == 'one':
        initializer = keras.initializers.Ones()
    else:
        raise NotImplementedError(f"get_initializer: initializer type {name} not supported")

    return initializer


class TransposedLinear(keras.layers.Layer):
    def __init__(self, d_input, d_output, biasIF=True, weightNorm=False, **kwargs):
        super(TransposedLinear, self).__init__(**kwargs)
        self.biasIF = biasIF
        self.d_input = d_input
        self.d_output = d_output
        self.weightNorm = weightNorm
        self.weight_initializer = keras.initializers.HeUniform()
        if self.biasIF:
            bound = 1 / math.sqrt(self.d_input)
            self.bias_initializer = keras.initializers.RandomUniform(minval=-bound, maxval=bound)
        else:
            self.bias_initializer = keras.initializers.Zeros()

    def build(self, input_shape):
        if self.weightNorm:
            self.g = tf.Variable(initial_value=keras.initializers.Ones()(shape=(self.d_output, 1), dtype="float32"),
                                 trainable=True)
            self.v = tf.Variable(initial_value=self.weight_initializer(shape=(1, self.d_input), dtype="float32"),
                                 trainable=True)
        else:
            self.kernel = tf.Variable(
                initial_value=self.weight_initializer(shape=(self.d_output, self.d_input), dtype="float32"),
                trainable=True)
        self.bias = tf.Variable(initial_value=self.bias_initializer(shape=(self.d_output, 1), dtype="float32"),
                                trainable=True)
        self.built = True

    def setWeightInitializer(self, initializer):
        self.weight_initializer = initializer

    def setBiasInitializer(self, initializer):
        self.bias_initializer = initializer

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_input": self.d_input,
            "d_output": self.d_output,
            "bias": self.bias,
        })
        return config

    def call(self, x):
        if self.weightNorm:
            normW = tf.matmul(self.g, self.v) / tf.norm(self.v)
            return contract('... u l, v u -> ... v l', x, normW) + self.bias
        else:
            return contract('... u l, v u -> ... v l', x, self.kernel) + self.bias


class SimpleLinear(keras.layers.Layer):
    def __init__(self, d_input, d_output, biasIF=True, weightNorm=False, **kwargs):
        super(SimpleLinear, self).__init__(**kwargs)
        self.biasIF = biasIF
        self.d_input = d_input
        self.d_output = d_output
        self.weightNorm = weightNorm
        self.weight_initializer = keras.initializers.HeUniform()
        if self.biasIF:
            bound = 1 / math.sqrt(self.d_input)
            self.bias_initializer = keras.initializers.RandomUniform(minval=-bound, maxval=bound)
        else:
            self.bias_initializer = keras.initializers.Zeros()

    def build(self, input_shape):
        if self.weightNorm:
            self.g = tf.Variable(initial_value=keras.initializers.Ones()(shape=(self.d_input, 1), dtype="float32"),
                                 trainable=True)
            self.v = tf.Variable(initial_value=self.weight_initializer(shape=(1, self.d_output), dtype="float32"),
                                 trainable=True)
        else:
            self.kernel = tf.Variable(
                initial_value=self.weight_initializer(shape=(self.d_input, self.d_output), dtype="float32"),
                trainable=True)
        self.bias = tf.Variable(initial_value=self.bias_initializer(shape=(1, self.d_output), dtype="float32"),
                                trainable=True)
        self.built = True

    def setWeightInitializer(self, initializer):
        self.weight_initializer = initializer

    def setBiasInitializer(self, initializer):
        self.bias_initializer = initializer

    def get_config(self):
        config = super(SimpleLinear, self).get_config()
        config.update({
            "d_input": self.d_input,
            "d_output": self.d_output,
            "biasIF": self.biasIF,
        })
        return config

    def call(self, x):
        if self.weightNorm:
            normW = tf.matmul(self.g, self.v) / tf.norm(self.v)
            return contract('... l u, u v -> ... l v', x, normW) + self.bias
        else:
            return contract('... l u, u v -> ... l v', x, self.kernel) + self.bias


def LinearActivation(
        d_input, d_output, bias=True,
        zero_bias_init=False,
        transposed=False,
        initializer=None,
        activation=None,
        activate=False,  # Apply activation as part of this module
        weight_norm=False,
        **kwargs,
):
    """ Returns a linear nn.Module with control over axes order, initialization, and activation """

    # Construct core module
    if activation == 'glu':
        d_output *= 2
        acti = None
    else:
        acti = activation

    if transposed:
        dim = -2
        linear = TransposedLinear(d_input, d_output, biasIF=bias, weightNorm=weight_norm, **kwargs)
    else:
        dim = -1
        linear = SimpleLinear(d_input, d_output, biasIF=bias, weightNorm=weight_norm, **kwargs)

    # Initialize weight
    if initializer is not None:
        linear.setWeightInitializer(get_initializer(initializer, acti))

    # Initialize bias
    if bias and zero_bias_init:
        linear.setBiasInitializer(keras.initializers.Zeros())

    # Weight norm

    if activate and activation is not None:
        activation = Activation(activation, dim=dim)
        linear = keras.Sequential([linear, activation])

    return linear


""" Misc functional utilities """


def krylov(L, A, b, c=None, return_power=False):
    """
    Compute the Krylov matrix (b, Ab, A^2b, ...) using the squaring trick.

    If return_power=True, return A^{L-1} as well
    """
    # TODO There is an edge case if L=1 where output doesn't get broadcasted, which might be an issue if caller is
    #  expecting broadcasting semantics... can deal with it if it arises

    x = tf.expand_dims(b, axis=-1)  # (..., N, 1)
    A_ = A

    AL = None
    if return_power:
        AL = tf.eye(A.shape[-1], dtype=A.dtype)
        _L = L - 1

    done = L == 1
    # loop invariant: _L represents how many indices left to compute
    while not done:
        if return_power:
            if _L % 2 == 1:
                AL = tf.matmul(A_, AL)
            _L //= 2

        # Save memory on last iteration
        l = x.shape[-1]
        if L - l <= l:
            done = True
            _x = x[..., :L - l]
        else:
            _x = x

        _x = tf.matmul(A_, _x)
        x = tf.concat([x, _x], axis=-1)  # there might be a more efficient way of ordering axes
        if not done:
            A_ = tf.matmul(A_, A_)

    assert x.shape[-1] == L

    if c is not None:
        x = oe.contract("...nl, ...n -> ...l", x, c)
    # x = x.contiguous()  # WOW!!
    if return_power:
        return x, AL
    else:
        return x


def power(L, A, v=None):
    """ Compute A^L and the scan sum_i A^i v_i

    A: (..., N, N)
    v: (..., N, L)
    """

    I = tf.eye(A.shape[-1])

    powers = [A]
    l = 1
    while True:
        if L % 2 == 1:
            I = powers[-1] @ I
            I = tf.matmul(powers[-1], I)
        L //= 2
        if L == 0:
            break
        l *= 2
        powers.append(tf.matmul(powers[-1], powers[-1]))

    if v is None:
        return I

    # Invariants:
    # powers[-1] := A^l
    # l := largest po2 at most L

    # Note that an alternative divide and conquer to compute the reduction is possible and can be embedded into the above loop without caching intermediate powers of A
    # We do this reverse divide-and-conquer for efficiency reasons:
    # 1) it involves fewer padding steps for non-po2 L
    # 2) it involves more contiguous arrays

    # Take care of edge case for non-po2 arrays
    # Note that this initial step is a no-op for the case of power of 2 (l == L)
    k = v.shape[-1] - l
    v_ = tf.matmul(powers.pop(), v[..., l:])
    v = v[..., :l]
    v[..., :k] = v[..., :k] + v_

    # Handle reduction for power of 2
    while v.size(-1) > 1:
        v = rearrange(v, '... (z l) -> ... z l', z=2)
        v = v[..., 0, :] + tf.matmul(powers.pop(), v[..., 1, :])
    return I, tf.expand_dims(v, axis=-1)


def embed_c2r(A):
    A = rearrange(A, '... m n -> ... m () n ()')
    A = np.pad(A, ((0, 0), (0, 1), (0, 0), (0, 1))) + \
        np.pad(A, ((0, 0), (1, 0), (0, 0), (1, 0)))
    return rearrange(A, 'm x n y -> (m x) (n y)')


def transition(measure, N, **measure_args):
    """ A, B transition matrices for different measures

    measure: the type of measure
      legt - Legendre (translated)
      legs - Legendre (scaled)
      glagt - generalized Laguerre (translated)
      lagt, tlagt - previous versions of (tilted) Laguerre with slightly different normalization
    """
    # Laguerre (translated)
    if measure == 'lagt':
        b = measure_args.get('beta', 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    # Generalized Laguerre
    # alpha 0, beta small is most stable (limits to the 'lagt' measure)
    # alpha 0, beta 1 has transition matrix A = [lower triangular 1]
    elif measure == 'glagt':
        alpha = measure_args.get('alpha', 0.0)
        beta = measure_args.get('beta', 0.01)
        A = -np.eye(N) * (1 + beta) / 2 - np.tril(np.ones((N, N)), -1)
        B = ss.binom(alpha + np.arange(N), np.arange(N))[:, None]

        L = np.exp(.5 * (ss.gammaln(np.arange(N) + alpha + 1) - ss.gammaln(np.arange(N) + 1)))
        A = (1. / L[:, None]) * A * L[None, :]
        B = (1. / L[:, None]) * B * np.exp(-.5 * ss.gammaln(1 - alpha)) * beta ** ((1 - alpha) / 2)
    # Legendre (translated)
    elif measure == 'legt':
        Q = np.arange(N, dtype=np.float64)
        R = (2 * Q + 1) ** .5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.) ** (i - j), 1) * R[None, :]
        B = R[:, None]
        A = -A
    # Legendre (scaled)
    elif measure == 'legs':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = np.matmul(np.matmul(T, M), np.linalg.inv(T))
        B = np.diag(T)[:, None]
        B = B.copy()  # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
    elif measure == 'fourier':
        freqs = np.arange(N // 2)
        d = np.stack([freqs, np.zeros(N // 2)], axis=-1).reshape(-1)[:-1]
        A = 2 * np.pi * (np.diag(d, 1) - np.diag(d, -1))
        A = A - embed_c2r(np.ones((N // 2, N // 2)))
        B = embed_c2r(np.ones((N // 2, 1)))[..., :1]
    elif measure == 'random':
        A = np.random.randn(N, N) / N
        B = np.random.randn(N, 1)
    elif measure == 'diagonal':
        A = -np.diag(np.exp(np.random.randn(N)))
        B = np.random.randn(N, 1)
    else:
        raise NotImplementedError

    return A, B


def rank_correction(measure, N, rank=1, dtype=tf.float32):
    """ Return low-rank matrix L such that A + L is normal """

    if measure == 'legs':
        assert rank >= 1
        P = tf.expand_dims(tf.math.sqrt(.5 + tf.range(N, dtype=dtype)), axis=0)  # (1 N)
    elif measure == 'legt':
        assert rank >= 2
        P = tf.math.sqrt(1 + 2 * tf.range(N, dtype=dtype))  # (N)
        P0 = tf.tile(P, [1, 1])
        P0[0::2] = 0.
        P1 = tf.tile(P, [1, 1])
        P1[1::2] = 0.
        P = tf.stack([P0, P1], axis=0)  # (2 N)
    elif measure == 'lagt':
        assert rank >= 1
        P = .5 ** .5 * tf.ones([1, N], dtype=dtype)
    elif measure == 'fourier':
        P = tf.ones(N, dtype=dtype)  # (N)
        P0 = tf.tile(P, [1, 1])
        P0[0::2] = 0.
        P1 = tf.tile(P, [1, 1])
        P1[1::2] = 0.
        P = tf.stack([P0, P1], axis=0)  # (2 N)
    else:
        raise NotImplementedError

    d = P.shape[0]
    if rank > d:
        P = tf.concat([P, tf.zeros([rank - d, N], dtype=dtype)], axis=0)  # (rank N)
    return P


def nplr(measure, N, rank=1, dtype=tf.float32):
    """ Return w, p, q, V, B such that
    (w - p q^*, B) is unitarily equivalent to the original HiPPO A, B by the matrix V
    i.e. A = V[w - p q^*]V^*, B = V B
    """
    assert dtype == tf.float32 or tf.complex64
    if measure == 'random':
        dtype = tf.complex64 if dtype == tf.float32 else tf.complex128

        w = -tf.exp(tf.random.normal(N // 2)) + 1j * tf.random.normal(N // 2)
        P = tf.random.normal((rank, N // 2), dtype=dtype)
        B = tf.random.normal(N // 2, dtype=dtype)
        V = tf.eye(N, dtype=dtype)[..., :N // 2]  # Only used in testing
        return w, P, B, V

    A, B = transition(measure, N)
    A = tf.convert_to_tensor(A, dtype=dtype)  # (N, N)
    B = tf.convert_to_tensor(B, dtype=dtype)[:, 0]  # (N,)

    P = rank_correction(measure, N, rank=rank, dtype=dtype)
    AP = A + tf.reduce_sum(P[..., None, :, :] * P[..., None, :], axis=-3)
    w, V = tf.linalg.eig(AP)  # (..., N) (..., N, N)

    # Only keep one of the conjugate pairs
    w = w[..., 0::2].numpy()
    V = V[..., 0::2].numpy()

    V_inv = tf.linalg.inv(V)

    B = contract('ij, j -> i', V_inv, B)  # V^* B
    P = contract('ij, ...j -> ...i', V_inv, P)  # V^* P

    return w, P, B, V


class S4(keras.Model):

    def __init__(
            self,
            d_model,
            d_state=64,
            l_max=1,
            # Maximum length of sequence. Fine if not provided: the kernel will keep doubling in length until longer than sequence. However, this can be marginally slower if the true length is not a power of 2
            channels=1,  # maps 1-dim to C-dim
            bidirectional=False,
            # Arguments for FF
            activation='gelu',  # activation in between SS and FF
            postact=None,  # activation after FF
            initializer=None,  # initializer on FF
            weight_norm=False,  # weight normalization on FF
            hyper_act=None,  # Use a "hypernetwork" multiplication
            dropout=0.0,
            transposed=True,  # axis ordering (B, L, D) or (B, D, L)
            verbose=False,
            # SSM Kernel arguments
            **kernel_args,
    ):

        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum sequence length, also denoted by L
          if this is not known at model creation, set l_max=1
        channels: can be interpreted as a number of "heads"
        bidirectional: bidirectional
        dropout: standard dropout argument
        transposed: choose backbone axis ordering of (B, L, H) or (B, H, L) [B=batch size, L=sequence length, H=hidden dimension]

        Other options are all experimental and should not need to be configured
        """

        super().__init__()
        if verbose:
            import src.utils.train
            log = src.utils.train.get_logger(__name__)
            log.info(f"Constructing S4 (H, N, L) = ({d_model}, {d_state}, {l_max})")

        self.h = d_model
        self.n = d_state
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed

        # optional multiplicative modulation GLU-style
        # https://arxiv.org/abs/2002.05202
        self.hyper = hyper_act is not None
        if self.hyper:
            channels *= 2
            self.hyper_activation = Activation(hyper_act)

        self.D = tf.Variable(tf.random.normal((channels, self.h)))

        if self.bidirectional:
            channels *= 2

        # SSM Kernel
        self.kernel = HippoSSKernel(self.h, N=self.n, L=l_max, channels=channels, verbose=verbose, **kernel_args)

        # Pointwise
        self.activation = Activation(activation)
        dropout_fn = keras.layers.SpatialDropout2D if self.transposed else keras.layers.SpatialDropout1D
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else tf.identity()

        # position-wise output transform to mix features
        self.output_linear = LinearActivation(
            self.h * self.channels,
            self.h,
            transposed=self.transposed,
            initializer=initializer,
            activation=postact,
            activate=True,
            weight_norm=weight_norm,
        )

        # self.time_transformer = get_torch_trans(heads=8, layers=1, channels=self.h)

    def call(self, u, **kwargs):  # absorbs return_output and transformer src mask
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """
        if not self.transposed: u = keras.backend.permute_dimensions(u, (0, 2, 1))
        L = u.shape[-1]

        # Compute SS Kernel
        k = self.kernel(L=L)  # (C H L) (B C H L)

        # Convolution
        if self.bidirectional:
            k0, k1 = rearrange(k, "(s c) h l -> s c h l", s=2)
            k = tf.pad(k0, [[0, 0], [0, 0], [0, L]]) + tf.pad(tf.reverse(k1, axis=[-1]), [[0, 0], [0, 0], [L, 0]])

        k_f = tf.signal.rfft(k, fft_length=[2 * L])
        u_f = tf.signal.rfft(u, fft_length=[2 * L])
        y_f = contract('bhl,chl->bchl', u_f, k_f)  # k_f.unsqueeze(-4) * u_f.unsqueeze(-3) # (B C H L)
        y = tf.signal.irfft(y_f, fft_length=[2 * L])[..., :L]  # (B C H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + contract('bhl,ch->bchl', u, self.D)  # u.unsqueeze(-3) * self.D.unsqueeze(-1)

        # Optional hyper-network multiplication
        if self.hyper:
            y, yh = rearrange(y, 'b (s c) h l -> s b c h l', s=2)
            y = self.hyper_activation(yh) * y

        # Reshape to flatten channels
        y = rearrange(y, '... c h l -> ... (c h) l')

        y = self.dropout(self.activation(y))

        if not self.transposed: y = keras.backend.permute_dimensions(y, (0, 2, 1))

        y = self.output_linear(y)

        # ysize = b, k, l, requieres l, b, k
        # y = self.time_transformer(y.permute(2,0,1)).permute(1,2,0)

        return y, None

    def step(self, u, state):
        """ Step one time step as a recurrent model. Intended to be used during validation.

        u: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """
        assert not self.training

        y, next_state = self.kernel.step(u, state)  # (B C H)
        y = y + tf.expand_dims(u, -2) * self.D
        y = rearrange(y, '... c h -> ... (c h)')
        y = self.activation(y)
        if self.transposed:
            y = tf.squeeze(self.output_linear(tf.expand_dims(y, -1)), -1)
        else:
            y = self.output_linear(y)
        return y, next_state

    def default_state(self, *batch_shape, device=None):
        return self.kernel.default_state(*batch_shape)

    @property
    def d_state(self):
        return self.h * self.n

    @property
    def d_output(self):
        return self.h

    @property
    def state_to_tensor(self):
        return lambda state: rearrange('... h n -> ... (h n)', state)


class S4Layer(keras.Model):
    def __init__(self, features, lmax, N=64, dropout=0.0, bidirectional=True, layer_norm=True):
        super().__init__()
        self.s4_layer = S4(d_model=features,
                           d_state=N,
                           l_max=lmax,
                           bidirectional=bidirectional)

        self.norm_layer = keras.layers.LayerNormalization() if layer_norm else tf.identity()
        self.dropout = keras.layers.Dropout(dropout) if dropout > 0 else tf.identity()

    def call(self, x):
        # x has shape seq, batch, feature
        x = keras.backend.permute_dimensions(x, (
            1, 2, 0))  # batch, feature, seq (as expected from S4 with transposed=True)
        xout, _ = self.s4_layer(x)  # batch, feature, seq
        xout = self.dropout(xout)
        xout = xout + x  # skip connection   # batch, feature, seq
        xout = keras.backend.permute_dimensions(xout, (2, 0, 1))  # seq, batch, feature
        return self.norm_layer(xout)
