import tensorflow_compression as tfc
import tensorflow as tf
from tensorflow.keras.layers import PReLU, Activation, GlobalAveragePooling2D, Dense, Concatenate, Conv2D, Multiply
#这些函数可以根据需要进行组合和调用，用于构建不同的编码-解码模型，并在图像处理任务中进行特征提取和图像重建

def GFR_Encoder_Module(inputs, name_prefix, num_filter, kernel_size, stride, activation=None):        #实现编码器的一个模块，包括一次卷积操作和激活函数
    conv = tfc.SignalConv2D(num_filter, kernel_size, corr=True, strides_down=stride, padding="same_zeros",
                            use_bias=True, activation=tfc.GDN(), name=name_prefix + '_conv')(inputs)
    if activation == 'prelu':
        conv = PReLU(shared_axes=[1,2], name=name_prefix + '_prelu')(conv)
    return conv



def GFR_Decoder_Module(inputs, name_prefix, num_filter, kernel_size, stride, activation=None):        #实现解码器的一个模块，包括一次反卷积操作和激活函数
    conv = tfc.SignalConv2D(num_filter, kernel_size, corr=False, strides_up=stride, padding="same_zeros", use_bias=True,
                            activation=tfc.GDN(inverse=True), name=name_prefix + '_conv')(inputs)
    if activation == 'prelu':
        conv = PReLU(shared_axes=[1,2], name=name_prefix + '_prelu')(conv)
    elif activation == 'sigmoid':
        conv = Activation('sigmoid', name=name_prefix + '_sigmoid')(conv)
    return conv



def Attention_Encoder(inputs, snr, tcn):                                    
    en1 = GFR_Encoder_Module(inputs, 'en1', 256, (9, 9), 2, 'prelu')
    en1 = AF_Module(en1, snr, 'en1')
    en2 = GFR_Encoder_Module(en1, 'en2', 256, (5, 5), 2, 'prelu')
    en2 = AF_Module(en2, snr, 'en2')
    en3 = GFR_Encoder_Module(en2, 'en3', 256, (5, 5), 1, 'prelu')
    en3 = AF_Module(en3, snr, 'en3')
    en4 = GFR_Encoder_Module(en3, 'en4', 256, (5, 5), 1, 'prelu')
    en4 = AF_Module(en4, snr, 'en4')
    en5 = GFR_Encoder_Module(en4, 'en5', tcn, (5, 5), 1)
    return en5


def Attention_Decoder(inputs, snr):                                                                  #带自注意力模块的解码器，通过调用解码器模块和自注意力模块构建
    de1 = GFR_Decoder_Module(inputs, 'de1', 256, (5, 5), 1, 'prelu')
    de1 = AF_Module(de1, snr, 'de1')
    de2 = GFR_Decoder_Module(de1, 'de2', 256, (5, 5), 1, 'prelu')
    de2 = AF_Module(de2, snr, 'de2')
    de3 = GFR_Decoder_Module(de2, 'de3', 256, (5, 5), 1, 'prelu')
    de3 = AF_Module(de3, snr, 'de3')
    de4 = GFR_Decoder_Module(de3, 'de4', 256, (5, 5), 2, 'prelu')
    de4 = AF_Module(de4, snr, 'de4')
    de5 = GFR_Decoder_Module(de4, 'de5', 3, (9, 9), 2, 'sigmoid')
    return de5



def AF_Module(inputs, snr, name_prefix):                                                             #实现自适应滤波模块，根据信噪比和输入特征图，对特征图进行加权
    (_, width, height, ch_num) = inputs.shape
    m = GlobalAveragePooling2D(name=name_prefix + '_globalpooling')(inputs)
    m = Concatenate(name=name_prefix + 'concat')([m, snr])
    m = Dense(ch_num//16, activation='relu', name=name_prefix + '_dense1')(m)
    m = Dense(ch_num, activation='sigmoid', name=name_prefix + '_dense2')(m)
    out = Multiply(name=name_prefix + 'mul')([inputs, m])
    return out


def Basic_Encoder(inputs, tcn):                                                                       #实现基本的编码器，通过调用多个编码器模块来构建完整的编码器
    en1 = GFR_Encoder_Module(inputs, 'en1', 256, (9, 9), 2, 'prelu')
    en2 = GFR_Encoder_Module(en1, 'en2', 256, (5, 5), 2, 'prelu')
    en3 = GFR_Encoder_Module(en2, 'en3', 256, (5, 5), 1, 'prelu')
    en4 = GFR_Encoder_Module(en3, 'en4', 256, (5, 5), 1, 'prelu')
    en5 = GFR_Encoder_Module(en4, 'en5', tcn, (5, 5), 1)
    return en5


def Basic_Decoder(inputs):                                                                            #实现基本的解码器，通过调用多个解码器模块来构建完整的解码器
    de1 = GFR_Decoder_Module(inputs, 'de1', 256, (5, 5), 1, 'prelu')
    de2 = GFR_Decoder_Module(de1, 'de2', 256, (5, 5), 1, 'prelu')
    de3 = GFR_Decoder_Module(de2, 'de3', 256, (5, 5), 1, 'prelu')
    de4 = GFR_Decoder_Module(de3, 'de4', 256, (5, 5), 2, 'prelu')
    de5 = GFR_Decoder_Module(de4, 'de5', 3, (9, 9), 2, 'sigmoid')
    return de5


def AF_Module_H(inputs, snr, h_real, h_imag, name_prefix):                                          #带通道相关信息的自适应滤波模块，根据信噪比、输入特征图以及通道相关信息进行加权
    (_, width, height, ch_num) = inputs.shape
    m = GlobalAveragePooling2D(name=name_prefix + '_globalpooling')(inputs)
    m = Concatenate(name=name_prefix + 'concat')([m, snr, h_real, h_imag])
    m = Dense(ch_num//16, activation='relu', name=name_prefix + '_dense1')(m)
    m = Dense(ch_num, activation='sigmoid', name=name_prefix + '_dense2')(m)
    out = Multiply(name=name_prefix + 'mul')([inputs, m])
    return out


def Attention_Encoder_H(inputs, snr, h_real, h_imag, tcn):                                          #带通道相关信息的自注意力编码器，通过调用编码器模块和自注意力模块构建
    en1 = GFR_Encoder_Module(inputs, 'en1', 256, (9, 9), 2, 'prelu')
    en1 = AF_Module_H(en1, snr, h_real, h_imag, 'en1')
    en2 = GFR_Encoder_Module(en1, 'en2', 256, (5, 5), 2, 'prelu')
    en2 = AF_Module_H(en2, snr, h_real, h_imag, 'en2')
    en3 = GFR_Encoder_Module(en2, 'en3', 256, (5, 5), 1, 'prelu')
    en3 = AF_Module_H(en3, snr, h_real, h_imag, 'en3')
    en4 = GFR_Encoder_Module(en3, 'en4', 256, (5, 5), 1, 'prelu')
    en4 = AF_Module_H(en4, snr, h_real, h_imag, 'en4')
    en5 = GFR_Encoder_Module(en4, 'en5', tcn, (5, 5), 1)
    return en5


def Attention_Decoder_H(inputs, h_real, h_imag, snr):                                              #带通道相关信息的自注意力解码器，通过调用解码器模块和自注意力模块构建
    de1 = GFR_Decoder_Module(inputs, 'de1', 256, (5, 5), 1, 'prelu')
    de1 = AF_Module_H(de1, snr, h_real, h_imag, 'de1')
    de2 = GFR_Decoder_Module(de1, 'de2', 256, (5, 5), 1, 'prelu')
    de2 = AF_Module_H(de2, snr, h_real, h_imag, 'de2')
    de3 = GFR_Decoder_Module(de2, 'de3', 256, (5, 5), 1, 'prelu')
    de3 = AF_Module_H(de3, snr, h_real, h_imag, 'de3')
    de4 = GFR_Decoder_Module(de3, 'de4', 256, (5, 5), 2, 'prelu')
    de4 = AF_Module_H(de4, snr, h_real, h_imag, 'de4')
    de5 = GFR_Decoder_Module(de4, 'de5', 3, (9, 9), 2, 'sigmoid')
    return de5