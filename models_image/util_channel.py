import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as layers
import tensorflow_probability as tfp

class Channel(layers.Layer):                                                       #自定义的通道层，根据指定的通道类型，在输入信号上调用相应的通道函数
    """"implement power constraint"""

    def __init__(self, channel_type, name="channel", **kwargs):
        super(Channel, self).__init__(name=name, **kwargs)
        self.channel_type = channel_type
        self.max_usage_count = 100
        #self.current_usage_count = tf.Variable(0, trainable=False)

    def call(self, features, snr_db=None, h_real=None, h_imag=None, b_prob=None, b_stddev=None):
        inter_shape = tf.shape(features)
        f = layers.Flatten()(features)
        # convert to complex channel signal
        dim_z = tf.shape(f)[1] // 2
        z_in = tf.complex(f[:, :dim_z], f[:, dim_z:])
        # power constraint, the average complex symbol power is 1
        norm_factor = tf.reduce_sum(
            tf.math.real(z_in * tf.math.conj(z_in)), axis=1, keepdims=True
        )
        z_in_norm = z_in * tf.complex(
            tf.sqrt(tf.cast(dim_z, dtype=tf.float32) / norm_factor), 0.0
        )



        # Add channel noise
        if self.channel_type == 'awgn':
            if snr_db is None:
                raise Exception("This input snr should exist!")
            z_out = awgn(z_in_norm, snr_db)
        elif self.channel_type == 'slow_fading':
            if snr_db is None or h_real is None or h_imag is None:
                raise Exception("This input snr,h_real,h_imag should exist!")
            z_out = slow_fading(z_in_norm, snr_db, h_real, h_imag)
        elif self.channel_type == 'slow_fading_eq':
            if snr_db is None or h_real is None or h_imag is None:
                raise Exception("This input snr,h_real,h_imag should exist!")
            z_out = slow_fading_eq(z_in_norm, snr_db, h_real, h_imag)
        elif self.channel_type == 'burst':
            if snr_db is None or b_prob is None or b_stddev is None:
                raise Exception("This input snr,b_prob,b_stddev should exist!")
            z_out = burst(z_in_norm, snr_db, b_prob, b_stddev)
        else:
            raise Exception("This option shouldn't be an option!")
        # convert signal back to intermediate shape
        z_out = tf.concat([tf.math.real(z_out), tf.math.imag(z_out)], 1)
        z_out = tf.reshape(z_out, inter_shape)
        return z_out


def awgn(x, snr_db):                                                            #添加高斯白噪声（AWGN）到输入信号x中，其中snr_db是信噪比，单位为分贝
    noise_stddev = tf.sqrt(10 ** (-snr_db / 10))
    noise_stddev = tf.complex(noise_stddev, 0.)
    awgn = tf.complex(
        tf.random.normal(tf.shape(x), 0, 1 / np.sqrt(2)),
        tf.random.normal(tf.shape(x), 0, 1 / np.sqrt(2)),
    )
    return x + noise_stddev * awgn


def slow_fading(x, snr_db, h_real, h_imag):                                     #在输入信号x上添加慢衰落信道，其中snr_db是信噪比，h_real和h_imag是衰落信道的实部和虚部
    noise_stddev = tf.sqrt(10 ** (-snr_db / 10))
    noise_stddev = tf.complex(noise_stddev, 0.)
    h = tf.complex(h_real, h_imag)
    h = tf.reshape(h, (tf.shape(h)[0], 1))
    awgn = tf.complex(
        tf.random.normal(tf.shape(x), 0, 1 / np.sqrt(2)),
        tf.random.normal(tf.shape(x), 0, 1 / np.sqrt(2)),
    )
    return h * x + noise_stddev * awgn


def slow_fading_eq(x, snr_db, h_real, h_imag):                                 #在输入信号x上添加慢衰落信道，并进行等化处理
    noise_stddev = tf.sqrt(10 ** (-snr_db / 10))
    noise_stddev = tf.complex(noise_stddev, 0.)
    h = tf.complex(h_real, h_imag)
    h = tf.reshape(h, (tf.shape(h)[0], 1))
    awgn = tf.complex(
        tf.random.normal(tf.shape(x), 0, 1 / np.sqrt(2)),
        tf.random.normal(tf.shape(x), 0, 1 / np.sqrt(2)),
    )
    return x + noise_stddev * awgn / h


def burst(x, snr_db, b_prob, b_stddev):                                       #在输入信号x上添加突发性干扰，其中snr_db是信噪比，b_prob是突发干扰出现的概率，b_stddev是突发干扰的标准差
    noise_stddev = tf.sqrt(10 ** (-snr_db / 10))
    noise_stddev = tf.complex(noise_stddev, 0.)
    awgn = tf.complex(
        tf.random.normal(tf.shape(x), 0, 1 / np.sqrt(2)),
        tf.random.normal(tf.shape(x), 0, 1 / np.sqrt(2)),
    )
    bb = tfp.distributions.Bernoulli(probs=b_prob,dtype=tf.float32)
    b_sample = bb.sample()
    b_sample = tf.complex(b_sample, 0.)
    b_stddev = tf.complex(b_stddev, 0.)
    burst_noise = b_sample * b_stddev * tf.complex(
        tf.random.normal(tf.shape(x), 0, 1 / np.sqrt(2)),
        tf.random.normal(tf.shape(x), 0, 1 / np.sqrt(2)),
    )
    return x + noise_stddev * awgn + burst_noise
