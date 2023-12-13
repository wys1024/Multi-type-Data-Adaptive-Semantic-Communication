# !usr/bin/env python
# -*- coding:utf-8 _*-
import os
import json
import torch
import argparse
import numpy as np
from data.dataset_text import EurDataset, collate_data
from models_text.transceiver import DeepSC
from torch.utils.data import DataLoader
from models_text.utils import BleuScore, SNR_to_noise, greedy_decode, SeqtoText
from tqdm import tqdm
from keras import Input
from keras.layers import Lambda
from keras import Model
from keras.layers import Layer
from models_image.util_channel import Channel
from models_image.util_module import Attention_Encoder, Attention_Decoder
from models_speech.models import sem_enc_model, chan_enc_model, Chan_Model, chan_dec_model, sem_dec_model
from data import dataset_cifar10
from skimage.metrics import structural_similarity as ssim

import librosa

import tensorflow as tf
from tensorflow.keras.models import load_model
from w3lib.html import remove_tags

from DQN.env import MyEnvironment
from DQN.DQN import DQN

parser = argparse.ArgumentParser()

parser.add_argument("command", default='eval_DQN/eval')########################在此修改测试模式


parser.add_argument('--data-dir', default='europarl/train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str)

parser.add_argument('--model-path-DQN', default='models_DQN', type=str)
parser.add_argument('--model-path', default='models/1', type=str)##################################在此选择所取固定带宽模型############################################

parser.add_argument('--channel', default='AWGN', type=str)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type = int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=50, type=int)
parser.add_argument('--epochs', default=2, type = int)
parser.add_argument('--bert-config-path', default='bert/cased_L-12_H-768_A-12/bert_config.json', type = str)
parser.add_argument('--bert-checkpoint-path', default='bert/cased_L-12_H-768_A-12/bert_model.ckpt', type = str)
parser.add_argument('--bert-dict-path', default='bert/cased_L-12_H-768_A-12/vocab.txt', type = str)
parser.add_argument('--tcn', default=16, type=int)
parser.add_argument("--validset_tfrecords_path", type=str,default="/home/src/data/data_speech/tfrecord/validset.tfrecords",  help="tfrecords path of validset.")                   


parser.add_argument("--num_frame", type=int, default=128, help="number of frames in each batch")
# parameter of semantic coding and channel coding
parser.add_argument("--sem_enc_outdims", type=list, default=[32, 128, 128, 128, 128, 128, 128],
                    help="output dimension of SE-ResNet in semantic encoder.") 
parser.add_argument("--chan_enc_filters", type=list, default=[128],
                    help="filters of CNN in channel encoder.")
parser.add_argument("--chan_dec_filters", type=list, default=[128],
                    help="filters of CNN in channel decoder.")
parser.add_argument("--sem_dec_outdims", type=list, default=[128, 128, 128, 128, 128, 128, 32],
                    help="output dimension of SE-ResNet in semantic decoder.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_cpus = os.cpu_count()
AUTOTUNE = tf.data.experimental.AUTOTUNE


def map_function(example):
    feature_map = {"wav_raw": tf.io.FixedLenFeature([], tf.string)}
    parsed_example = tf.io.parse_single_example(example, features=feature_map)
    
    wav_slice = tf.io.decode_raw(parsed_example["wav_raw"], out_type=tf.int16)
    wav_slice = tf.cast(wav_slice, tf.float32) / 2**15

    return wav_slice



def eval_text(args, SNR, net, action):
    action = int(action*16)             
    bleu_score_1gram = BleuScore(1, 0, 0, 0)
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0, pin_memory=True, collate_fn=collate_data)
    StoT = SeqtoText(token_to_idx, end_idx)
    score = []
    score2 = []
    net.eval()
    with torch.no_grad():
        for epoch in range(args.epochs):
            Tx_word = []
            Rx_word = []

            for snr in tqdm(SNR):
                word = []
                target_word = []
                noise_std = SNR_to_noise(snr)

                for sents in test_iterator:

                    sents = sents.to(device)
                    target = sents
                    out = greedy_decode(net, sents, noise_std, args.MAX_LENGTH, pad_idx, start_idx, args.channel, action)                                                        
                    sentences = out.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, sentences))
                    word = word + result_string
                    target_sent = target.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, target_sent))
                    target_word = target_word + result_string
                Tx_word.append(word)
                Rx_word.append(target_word)
            bleu_score = []
            sim_score = []
            for sent1, sent2 in zip(Tx_word, Rx_word):
                bleu_score.append(bleu_score_1gram.compute_blue_score(sent1, sent2)) 
            bleu_score = np.array(bleu_score)
            bleu_score = np.mean(bleu_score, axis=1)
            score.append(bleu_score)
    score1 = np.mean(np.array(score), axis=0)
    return score1



def eval_image(args, image_encoder_model, image_decoder_model, action, SNR):
    action = int(action * 48)
    snr_list = []
    ssim_list = []

    for snr in SNR:
        issim = []

        for i in range(0, 20):
            (train_ds, train_nums), (test_ds, test_nums) = dataset_cifar10.get_dataset_snr(snr)

            test_ds = test_ds.shuffle(buffer_size=test_nums)
            test_ds = test_ds.batch(args.batch_size)
            test2 = test_ds.prefetch(buffer_size=AUTOTUNE)
            iterator = iter(test2)
            first_element = next(iterator)
            test2 = first_element[0]
            image = test2[0]
            tensor = tf.convert_to_tensor(snr)
            snr_tensor = tf.constant(tensor, shape=(args.batch_size, 1))

            encoded_data = image_encoder_model([image, snr_tensor])


            encoded_data = tf.Variable(encoded_data)
            # 对最后一维进行切片
            sliced_tensor = encoded_data[..., :-action]
            # 创建一个与切片后张量形状相同的零张量
            zero_values = tf.zeros_like(sliced_tensor)
            # 将切片后张量的值赋为零张量
            encoded_data[..., :-action].assign(zero_values)
            encoded_data = encoded_data.value()


            rv_imgs = image_decoder_model([encoded_data, snr_tensor])
            # 将 TensorFlow 张量转换为 NumPy 数组
            image = np.array(image)
            rv_imgs = np.array(rv_imgs)
            # 计算 SSIM
            ssim_value = ssim(image, rv_imgs, win_size=3, data_range=rv_imgs.max() - rv_imgs.min())
            issim.append(ssim_value)

        avg_ssim = np.mean(issim)
        snr_list.append(snr)
        ssim_list.append(avg_ssim)
        print(f'SNR (dB): {snr}, SSIM: {avg_ssim}')

    return snr_list, ssim_list



def eval_speech(_input, std, action):
    action = int(action*128)
    std = tf.cast(std, dtype=tf.float32)
    _output, batch_mean, batch_var = sem_enc(_input)
    _output1 = chan_enc(_output)

    _output1 = _output[:, :, :, :-action]        ####################????????????////
    zero_slices = tf.zeros_like(_output[:, :, :, :action])
    _output = tf.concat([_output1, zero_slices], axis=-1)
    
    _output = chan_layer(_output, std)
    _output = chan_dec(_output)
    _output = sem_dec([_output, batch_mean, batch_var])
   

    #loss_value = mse_loss(_input, _output)
    #ssim = librosa.effects.structural_similarity(_input, _output)

    # 将输入张量转换为NumPy数组
    input_array = _input.numpy()
    output_array = _output.numpy()

    # 计算data_range
    data_range = np.max(output_array) - np.min(output_array)

    # 计算SSIM
    ssim_value = ssim(input_array, output_array, win_size=3, data_range=data_range)
  
    #loss_value = mse_loss(_input, _output)
    return ssim_value






if __name__ == '__main__':
    args = parser.parse_args()
    SNR = [0,6,12,18]

    args.vocab_file = '/home/src/data/' + args.vocab_file
    vocab = json.load(open(args.vocab_file, 'rb'))

    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    #构建文本模型
    deepsc = DeepSC(args.tcn, args.num_layers, num_vocab, num_vocab,
                        num_vocab, num_vocab, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)
    

    #构建图像模型
    input_imgs = Input(shape=(32, 32, 3))
    input_snrdb = Input(shape=(1,))
    normal_imgs = Lambda(lambda x: x / 255, name='normal')(input_imgs)
    encoder = Attention_Encoder(normal_imgs, input_snrdb, 48)
    rv = Channel(channel_type='awgn')(encoder, input_snrdb)
    image_encoder_model = Model(inputs=[input_imgs, input_snrdb], outputs=rv)

    decoder = Attention_Decoder(rv, input_snrdb)
    rv_imgs = Lambda(lambda x: x * 255, name='denormal')(decoder)
    image_decoder_model = Model(inputs=[rv, input_snrdb], outputs=rv_imgs)

    #image_encoder_model.compile(Adam(args.learning_rate), 'mse')
    #image_decoder_model.compile(Adam(args.learning_rate), 'mse')



    #构建音频模型
    frame_length = int(48000*0.016)
    stride_length = int(48000*0.016)

    sem_enc = sem_enc_model(frame_length, stride_length, args)
    # define channel encoder
    chan_enc = chan_enc_model(frame_length, args)
    # define channel model
    chan_layer = Chan_Model(name="Channel_Model")
    # define channel decoder
    chan_dec = chan_dec_model(frame_length, args)
    # define semantic decoder
    sem_dec = sem_dec_model(frame_length, stride_length, args)
    mse_loss = tf.keras.losses.MeanSquaredError(name="mse_loss")

    # read .tfrecords file
    validset = tf.data.TFRecordDataset(args.validset_tfrecords_path)
    validset = validset.map(map_func=map_function, num_parallel_calls=num_cpus)
    validset = validset.batch(batch_size=args.batch_size)
    validset = validset.prefetch(buffer_size=args.batch_size)
    valid_loss_epoch = []
    valid_loss = 0.0





if args.command == 'eval_DQN':                           
    
    action = 0.25                             ###########################################在此修改测试带宽################################################
    ''' 
    #加载文字模型
    checkpoint = torch.load(args.model_path_DQN + "/text")
    deepsc.load_state_dict(checkpoint)
    print('text_DQN model load!')
    bleu_score = eval_text(args, SNR, deepsc, action) #snr = [0,6,12,18]
    print('text_DQN: ', bleu_score)

    #加载图像模型
    image_encoder_model.load_weights(args.model_path_DQN + "/image_encoder.h5")  #snr = [0-20]
    image_decoder_model.load_weights(args.model_path_DQN + "/image_decoder.h5")
    print('image_DQN model load!')
    eval_image(args, image_encoder_model, image_decoder_model, action, SNR)
    '''
    #加载音频模型
    sem_enc = load_model(args.model_path_DQN + "/sem_enc.h5")
    chan_enc = load_model(args.model_path_DQN + "/chan_enc.h5")
    chan_dec = load_model(args.model_path_DQN + "/chan_dec.h5")
    sem_dec = load_model(args.model_path_DQN + "/sem_dec.h5")
    print('speech_DQN model load!')


    for snr_dB in SNR:
        snr = pow(10, (snr_dB / 10))
        std = np.sqrt(1 / (2 * snr))

        for step, _input in enumerate(validset):
            loss_value = eval_speech(_input, std, action)
            loss_float = float(loss_value)
            valid_loss_epoch.append(loss_float)
            valid_loss += loss_float
        valid_loss /= (step + 1)
        print("SNR (dB):", snr_dB, "SSIM:", valid_loss)





elif args.command == 'eval':  
    action = 1              ###########################################在此修改测试带宽################################################
    '''
    #加载文字模型
    checkpoint = torch.load(args.model_path + "/text")
    deepsc.load_state_dict(checkpoint)
    print('text model load!')
    bleu_score = eval_text(args, SNR, deepsc, action )  
    print('text: ', bleu_score)

    #加载图像模型
    image_encoder_model.load_weights(args.model_path + "/image_encoder.h5")  
    image_decoder_model.load_weights(args.model_path + "/image_decoder.h5")
    print('image model load!')
    eval_image(args, image_encoder_model, image_decoder_model, action, SNR)
    '''
    #加载音频模型
    sem_enc = load_model(args.model_path + "/sem_enc.h5")
    chan_enc = load_model(args.model_path + "/chan_enc.h5")
    chan_dec = load_model(args.model_path + "/chan_dec.h5")
    sem_dec = load_model(args.model_path + "/sem_dec.h5")
    print('speech model load!')

    for snr_dB in SNR:
        snr = pow(10, (snr_dB / 10))
        std = np.sqrt(1 / (2 * snr))

        for step, _input in enumerate(validset):
            loss_value = eval_speech(_input, std, action)
            loss_float = float(loss_value)
            valid_loss_epoch.append(loss_float)
            valid_loss += loss_float
        valid_loss /= (step + 1)
        print("SNR (dB):", snr_dB, "SSIM:", valid_loss)