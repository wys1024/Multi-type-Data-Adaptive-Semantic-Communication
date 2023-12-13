# -*- coding: utf-8 -*-
import time
import torch
import random
import torch.nn as nn

from keras import Input
from keras.layers import Lambda
from keras import Model
from keras.layers import Layer
import keras
from keras import backend as K
from keras.optimizers import Adam


from DQN.env import MyEnvironment
from DQN.DQN import DQN
from DQN.utils import create_directory
from data import dataset_cifar10
from models_image.util_channel import Channel
from models_image.util_module import Attention_Encoder, Attention_Decoder

from models_text.utils import SNR_to_noise, initNetParams, train_step, val_step
from data.dataset_text import EurDataset, collate_data
from models_text.transceiver import DeepSC
from torch.utils.data import DataLoader
from tqdm import tqdm

import tensorflow as tf
import numpy as np
import argparse
import os
import json
from models_speech.models import sem_enc_model, chan_enc_model, Chan_Model, chan_dec_model, sem_dec_model
from tensorflow.keras.layers import Concatenate
#from skimage.metrics import peak_signal_noise_ratio as compute_pnsr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
AUTOTUNE = tf.data.experimental.AUTOTUNE
num_cpus = os.cpu_count()



def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help='train/train_DQN', default=train_DQN)

    # parameter of frame
    parser.add_argument("--sr", type=int, default=48000, help="sample rate for wav file")
    parser.add_argument("--num_frame", type=int, default=128, help="number of frames in each batch")
    parser.add_argument("--frame_size", type=float, default=0.016, help="time duration of each frame")
    parser.add_argument("--stride_size", type=float, default=0.016, help="time duration of frame stride")

    # parameter of semantic coding and channel coding
    parser.add_argument("--sem_enc_outdims", type=list, default=[32, 128, 128, 128, 128, 128, 128],
                        help="output dimension of SE-ResNet in semantic encoder.") 
    parser.add_argument("--chan_enc_filters", type=list, default=[128],
                        help="filters of CNN in channel encoder.")
    parser.add_argument("--chan_dec_filters", type=list, default=[128],
                        help="filters of CNN in channel decoder.")
    parser.add_argument("--sem_dec_outdims", type=list, default=[128, 128, 128, 128, 128, 128, 32],
                        help="output dimension of SE-ResNet in semantic decoder.")

    # path of tfrecords files
    parser.add_argument("--trainset_tfrecords_path", type=str, default="/home/src/data/data_speech/tfrecord/trainset.tfrecords",  help="tfrecords path of trainset.")
    # epoch and learning rate

    parser.add_argument("--batch_size", type=int, default=40, help="batch size.")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate.")
    parser.add_argument("-lr", "--learning_rate", help="learning_rate for training", default=0.0001, type=float)
    parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str)
    parser.add_argument('--checkpoint-path-DQN', default='models_DQN', type=str)
    parser.add_argument('--checkpoint-path', default='models/0.25', type=str)            #############################################在此现在训练模型###########################
    parser.add_argument('--channel', default='AWGN', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
    parser.add_argument('--MAX-LENGTH', default=30, type=int)
    parser.add_argument('--MIN-LENGTH', default=4, type=int)
    parser.add_argument('--d-model', default=128, type=int)
    parser.add_argument('--dff', default=512, type=int)
    parser.add_argument('--num-layers', default=4, type=int)
    parser.add_argument('--num-heads', default=8, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--tcn', default=16, type=int)                  
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/DQN/')
    parser.add_argument('--vailsnr', default = 1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def validate(epoch, args, net):
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0, pin_memory=True, collate_fn=collate_data)
    net.eval()
    pbar = tqdm(test_iterator)
    total = 0
    with torch.no_grad():
        for sents in pbar:
            sents = sents.to(device)
            loss = val_step(net, sents, sents, args.vailsnr, pad_idx,
                             criterion, args.channel)

            total += loss
            pbar.set_description(
                'Epoch: {}; Type: VAL; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )
    return total/len(test_iterator)


def map_function(example):
    feature_map = {"wav_raw": tf.io.FixedLenFeature([], tf.string)}
    parsed_example = tf.io.parse_single_example(example, features=feature_map)
    
    wav_slice = tf.io.decode_raw(parsed_example["wav_raw"], out_type=tf.int16)
    wav_slice = tf.cast(wav_slice, tf.float32) / 2**15

    return wav_slice


'''def PSNR(test_input, test_rec):
    psnr_i1 = np.zeros((test_input.shape[0]))
    for j in range(0, test_input.shape[0]):
        psnr_i1[j] = compute_pnsr(test_input[j, :], test_rec[j, :])
    psnr_ave = np.mean(psnr_i1)
    return psnr_ave'''


def train_speech(_input, snr, sem_enc, chan_enc, chan_layer, chan_dec, sem_dec, action3, optimizer3):
    snr = tf.cast(snr, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(sem_enc.trainable_variables + chan_enc.trainable_variables + chan_dec.trainable_variables + sem_dec.trainable_variables)
        _output, batch_mean, batch_var = sem_enc(_input, training=True)
        _output = chan_enc(_output, training=True)


        _output = tf.Variable(_output, trainable=True)
        # 对最后一维进行切片
        sliced_tensor = _output[..., :-action3]
        # 创建一个与切片后张量形状相同的零张量
        zero_values = tf.zeros_like(sliced_tensor)
        # 将切片后张量的值赋为零张量
        _output[..., :-action3].assign(zero_values)
        _output = _output.value()


        _output = chan_layer(_output, snr)
        _output = chan_dec(_output, training=True)
        _output = sem_dec([_output, batch_mean, batch_var], training=True)
        mse_loss = tf.keras.losses.MeanSquaredError()
        loss_value = mse_loss(_input, _output)
    gradients = tape.gradient(loss_value, sem_enc.trainable_variables + chan_enc.trainable_variables + chan_dec.trainable_variables + sem_dec.trainable_variables)

    optimizer3.apply_gradients(zip(gradients, sem_enc.trainable_variables + chan_enc.trainable_variables + chan_dec.trainable_variables + sem_dec.trainable_variables))
    return loss_value*10000



def train_image(train_data, snr, image_encoder_model, image_decoder_model, action2, optimizer2):
    iterator = iter(train_data)
    first_element = next(iterator)
    test2 = first_element[0]
    image = test2[0]
    tensor = tf.convert_to_tensor(snr)
    snr = tf.constant(tensor, shape=(args.batch_size, 1))

    with tf.GradientTape() as tape:
        encoded_data = image_encoder_model([image, snr], training=True)

        encoded_data = tf.Variable(encoded_data, trainable=True)
        # 对最后一维进行切片
        sliced_tensor = encoded_data[..., :-action2]
        # 创建一个与切片后张量形状相同的零张量
        zero_values = tf.zeros_like(sliced_tensor)
        # 将切片后张量的值赋为零张量
        encoded_data[..., :-action2].assign(zero_values)
        encoded_data = encoded_data.value()

        rv_imgs = image_decoder_model([encoded_data, snr], training=True)
        mse = tf.keras.losses.mean_squared_error(image, rv_imgs)
        mse = tf.reduce_mean(mse)
    gradients = tape.gradient(mse, image_encoder_model.trainable_variables + image_decoder_model.trainable_variables)
    optimizer2.apply_gradients(zip(gradients, image_encoder_model.trainable_variables + image_decoder_model.trainable_variables))
    mse = mse.numpy() / 983.04
    return mse




def train(epoch, args, net, image_encoder_model, image_decoder_model, action):  #固定带宽模型训练
    snr = 8 #默认初始值
    # 准备text数据
    train_eur = EurDataset('train')
    test1 = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0, pin_memory=True, collate_fn=collate_data)
    
    # 准备image数据
    (train_ds, train_nums), (test_ds, test_nums) = dataset_cifar10.get_dataset_snr(snr)
    train_ds = train_ds.shuffle(buffer_size=train_nums)
    train_ds = train_ds.batch(args.batch_size)
    test_ds = test_ds.shuffle(buffer_size=test_nums)
    test_ds = test_ds.batch(args.batch_size)
    test2 = train_ds.prefetch(buffer_size=AUTOTUNE)

    # 准备speech数据
    trainset = tf.data.TFRecordDataset(args.trainset_tfrecords_path)
    trainset = trainset.map(map_func=map_function, num_parallel_calls=num_cpus)
    trainset = trainset.shuffle(buffer_size=args.batch_size * 657, reshuffle_each_iteration=True)
    trainset = trainset.batch(batch_size=args.batch_size)
    test3 = trainset.prefetch(buffer_size=args.batch_size)


    noise_std = np.random.uniform(SNR_to_noise(0), SNR_to_noise(20), size=(1))
    snr1 = noise_std[0]                       
    print('snr1:', snr1)
    snr2 = random.randint(0, 20)
    print('snr2:', snr2)

    action1 = int(16*action) 
    action2 = int(48*action)
    action3 = int(128*action)

    print('###########开始传输数据##############')
    '''
    mse1_sum = 0.0
    num_samples = 0
    for sents in test1:
        sents = sents.to(device)
        mse1 = train_step(net, sents, sents, snr1, pad_idx, optimizer, criterion, args.channel, action1)
        mse1_sum += mse1
        num_samples += 1
    mse1 = mse1_sum / num_samples
    print('#####文字数据传输完毕######',mse1) 
   
    mse2 = train_image(test2, snr2, image_encoder_model, image_decoder_model, action2, optimizer2)
    print('#####图像数据传输完毕######',mse2)
    '''
    
    subset_size = 100  # 子集大小
    trainset_subset = trainset.take(subset_size)
    train_loss = 0
    for step,test in enumerate(trainset_subset):
        loss_value = train_speech(test, snr2, sem_enc, chan_enc, chan_layer, chan_dec, sem_dec, action3, optimizer3)
        loss_float = float(loss_value)
        train_loss += loss_float
        train_loss /= (step + 1)
    mse3=train_loss
    print('#####音频数据传输完毕######',mse3)

    #mse = mse1 + mse2 + mse3
    print('###########结束传输数据##############')

       








def train_DQN(epoch, args, net, image_encoder_model, image_decoder_model, reward_list):
    snr = 8 #默认初始值
    # 准备text数据
    train_eur = EurDataset('train')
    test1 = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0, pin_memory=True, collate_fn=collate_data)
    
    # 准备image数据
    (train_ds, train_nums), (test_ds, test_nums) = dataset_cifar10.get_dataset_snr(snr)
    train_ds = train_ds.shuffle(buffer_size=train_nums)
    train_ds = train_ds.batch(args.batch_size)

    test_ds = test_ds.shuffle(buffer_size=test_nums)
    test_ds = test_ds.batch(args.batch_size)

    test2 = train_ds.prefetch(buffer_size=AUTOTUNE)

    # 准备speech数据
    trainset = tf.data.TFRecordDataset(args.trainset_tfrecords_path)
    trainset = trainset.map(map_func=map_function, num_parallel_calls=num_cpus)
    trainset = trainset.shuffle(buffer_size=args.batch_size * 657, reshuffle_each_iteration=True)
    trainset = trainset.batch(batch_size=args.batch_size)
    test3 = trainset.prefetch(buffer_size=args.batch_size)

    print('###########开始DQN初始化##############')
    env = MyEnvironment()
    create_directory(args.ckpt_dir, sub_dirs=['Q_eval', 'Q_target'])
    mse1 = mse2 = mse3 = 0
    
    agent = DQN(alpha=0.001, state_dim=env.state_dim, action_dim1=env.action_dim1, action_dim2=env.action_dim2, action_dim3=env.action_dim3, fc1_dim=256, fc2_dim=256, ckpt_dir=args.ckpt_dir)
    total_rewards, avg_rewards, eps_history = [], [], []  # 列表存储数据
    total_reward = 0
    done = False
    observation = env.reset(test1, test2, test3, snr)
    observation = agent.observation_to_state(observation)
    print('###########完成DQN初始化##############')
    


    while not done:
        noise_std = np.random.uniform(SNR_to_noise(0), SNR_to_noise(20), size=(1))
        snr = noise_std[0]
        snr2 = random.randint(0, 20)

        action1 = agent.choose_action1(observation, isTrain=True)  # 根据观测值observation选择要执行的动作 
        action2 = agent.choose_action2(observation, isTrain=True)
        action3 = agent.choose_action3(observation, isTrain=True)
        print('选择带宽： ', action1, action2, action3)


        print('################开始传输数据################')
        '''
        mse1_sum = 0.0
        num_samples = 0
        for sents in test1:
           sents = sents.to(device)
           mse1 = train_step(net, sents, sents, snr, pad_idx, optimizer, criterion, args.channel, action1)
           mse1_sum += mse1
           num_samples += 1
        mse1 = mse1_sum / num_samples
        print('#####文字数据传输完毕#####',mse1)  #4.1
        '''
        mse2 = train_image(test2, snr2, image_encoder_model, image_decoder_model, action2, optimizer2)   #0.04
        print('#####图像数据传输完毕#####',mse2)
        '''
        train_loss = 0
        for step,test in enumerate(trainset):
           loss_value = train_speech(test, snr2, sem_enc, chan_enc, chan_layer, chan_dec, sem_dec, action3, optimizer3)
           loss_float = float(loss_value)
           train_loss += loss_float
           train_loss /= (step + 1)
        mse3=train_loss                                          #1.15
        print('#####音频数据传输完毕#####',mse3)
        
        #mse = mse1 + mse2 + mse3
        '''
        print('################结束传输数据################')

        observation_ = env.reset(test1, test2, test3, snr2)
        observation_ = agent.observation_to_state(observation_)
        reward = 100-mse2
        print('reward: ', reward)
        reward_list.append(reward)
        print(reward_list)
        print()
        if reward > 98:
            done = True
        agent.remember(observation, action1, action2, action3, reward, observation_, done)  # 将结果存入记忆体   obs空
        agent.learn()  # 学习更新网络
        total_reward += reward
        observation = observation_

    total_rewards.append(total_reward)
    avg_reward = np.mean(total_rewards[-100:])
    avg_rewards.append(avg_reward)
    eps_history.append(agent.epsilon)

    print('EP:{} reward:{} avg_reward:{} epsilon:{}'.              #DQN
            format(epoch + 1, total_reward, avg_reward, agent.epsilon))
    if (epoch + 1) % 50 == 0:
        agent.save_models(epoch + 1)




if __name__ == '__main__':
    args = parse_args()

#text  ——torch
    args.vocab_file = '/home/src/data/' + args.vocab_file
    """ preparing the dataset """
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    """ define optimizer and loss function """
    deepsc = DeepSC(args.tcn, args.num_layers, num_vocab, num_vocab,num_vocab, num_vocab, args.d_model, args.num_heads,args.dff, 0.1).to(device)

    criterion = nn.CrossEntropyLoss(reduction = 'none')
    optimizer = torch.optim.Adam(deepsc.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay = 5e-4)
    initNetParams(deepsc)
    """ define optimizer and loss function """


#image  ——tensorflow keras
    input_imgs = Input(shape=(32, 32, 3))
    input_snrdb = Input(shape=(1,))
    normal_imgs = Lambda(lambda x: x / 255, name='normal')(input_imgs)
    encoder = Attention_Encoder(normal_imgs, input_snrdb, 48)
    rv = Channel(channel_type='awgn')(encoder, input_snrdb)
    image_encoder_model = Model(inputs=[input_imgs, input_snrdb], outputs=rv)

    decoder = Attention_Decoder(rv, input_snrdb)
    rv_imgs = Lambda(lambda x: x * 255, name='denormal')(decoder)
    image_decoder_model = Model(inputs=[rv, input_snrdb], outputs=rv_imgs)

    optimizer2 = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    loss_function = tf.keras.losses.MeanSquaredError()
    image_encoder_model.compile(optimizer2, loss=loss_function)
    image_decoder_model.compile(optimizer2, loss=loss_function)


#speech  ——tensorflow keras
    frame_length = int(48000*0.016)
    stride_length = int(48000*0.016)

    sem_enc = sem_enc_model(frame_length, stride_length, args)
    chan_enc = chan_enc_model(frame_length, args)
    chan_layer = Chan_Model(name="Channel_Model")
    chan_dec = chan_dec_model(frame_length, args)
    sem_dec = sem_dec_model(frame_length, stride_length, args)

    mse_loss = tf.keras.losses.MeanSquaredError
    optimizer3 = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    sem_enc.compile(optimizer3, loss='mse')
    chan_enc.compile(optimizer3, loss='mse')
    
    chan_dec.compile(optimizer3, loss='mse')
    sem_dec.compile(optimizer3, loss='mse')
    





if args.command == 'train_DQN':
    reward_list = []
    for epoch in range(args.epochs):
        print('开始第:{}次train'.format(epoch))
        train_DQN(epoch, args, deepsc, image_encoder_model, image_decoder_model, reward_list)
 
        if epoch == args.epochs-1:  
            '''
            #保存文字模型
            with open(args.checkpoint_path_DQN + "/text", 'wb') as f:
                torch.save(deepsc.state_dict(), f)
            print('文字模型保存完毕') 
            
            #保存图像模型
            with open(args.checkpoint_path_DQN + "/image_encoder.h5", 'wb') as f:
                image_encoder_model.save_weights(f.name)
            with open(args.checkpoint_path_DQN + "/image_decoder.h5", 'wb') as f:
                image_decoder_model.save_weights(f.name)
            print('图像模型保存完毕')
            
            #保存音频模型
            saved_model_dir = args.checkpoint_path_DQN
            # semantic_encoder
            sem_enc_h5 = saved_model_dir + "/sem_enc.h5"
            sem_enc.save(sem_enc_h5)
            # channel_encoder
            chan_enc_h5 = saved_model_dir + "/chan_enc.h5"
            chan_enc.save(chan_enc_h5)
            # channel_decoder
            chan_dec_h5 = saved_model_dir + "/chan_dec.h5"
            chan_dec.save(chan_dec_h5)
            # semantic_decoder
            sem_dec_h5 = saved_model_dir + "/sem_dec.h5"
            sem_dec.save(sem_dec_h5)
            print('音频模型保存完毕')
        '''
    print(reward_list)

elif args.command == 'train':
    for epoch in range(args.epochs):
        print('开始第:{}次train'.format(epoch))
        action = 0.25                                                         ######################在此修改固定带宽###################
        train(epoch, args, deepsc, image_encoder_model, image_decoder_model, action)

        if epoch == args.epochs-1:  

            #保存文字模型
            '''
            with open(args.checkpoint_path + "/text", 'wb') as f:
                torch.save(deepsc.state_dict(), f)   
            print('文字模型保存完毕')   
           
            #保存图像模型
            with open(args.checkpoint_path + "/image_encoder.h5", 'wb') as f:
                image_encoder_model.save_weights(f.name)
            with open(args.checkpoint_path + "/image_decoder.h5", 'wb') as f:
                image_decoder_model.save_weights(f.name)
            print('图像模型保存完毕')    
            '''    
            #保存音频模型
            saved_model_dir = args.checkpoint_path
            # semantic_encoder
            sem_enc_h5 = saved_model_dir + "/sem_enc.h5"
            sem_enc.save(sem_enc_h5)
            # channel_encoder
            chan_enc_h5 = saved_model_dir + "/chan_enc.h5"
            chan_enc.save(chan_enc_h5)
            # channel_decoder
            chan_dec_h5 = saved_model_dir + "/chan_dec.h5"
            chan_dec.save(chan_dec_h5)
            # semantic_decoder
            sem_dec_h5 = saved_model_dir + "/sem_dec.h5"
            sem_dec.save(sem_dec_h5)
            print('音频模型保存完毕')
            
