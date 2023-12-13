# -*- coding: utf-8 -*-
import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np

from DQN.env import MyEnvironment
from DQN.DQN import DQN
from DQN.utils import create_directory

from utils import SNR_to_noise, initNetParams, train_step, val_step, train_mi
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

parser = argparse.ArgumentParser()
#parser.add_argument('--data-dir', default='data/train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/Rayleigh_8', type=str)  #修改模型存放位置##############################################################################################
parser.add_argument('--channel', default='Rayleigh', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--tcn', default=16, type=int)                  
parser.add_argument('--ckpt_dir', type=str, default='/home/src/checkpoints/DQN/')

device = torch.device("cuda:0")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def validate(epoch, args, net):
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    net.eval()
    pbar = tqdm(test_iterator)
    total = 0
    with torch.no_grad():
        for sents in pbar:
            sents = sents.to(device)
            loss = val_step(net, sents, sents, 0.1, pad_idx,
                             criterion, args.channel)

            total += loss
            pbar.set_description(
                'Epoch: {}; Type: VAL; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )
    return total/len(test_iterator)


def train_DQN(epoch, args, net):
    train_eur = EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_iterator)
    noise_std = np.random.uniform(SNR_to_noise(0), SNR_to_noise(20), size=(1))
    create_directory(args.ckpt_dir, sub_dirs=['Q_eval', 'Q_target'])
    print('###########DQN初始化##############')
    env = MyEnvironment()
    mse = 0
    snrdb_test = noise_std[0]
    data_type = 1
    batch_size = 128
    sequence_length = 3
    test_ds = torch.empty(batch_size, sequence_length)

    agent = DQN(alpha=0.0001, state_dim=env.state_dim, action_dim=env.action_dim, fc1_dim=256, fc2_dim=256, ckpt_dir=args.ckpt_dir)
    total_rewards, avg_rewards, eps_history = [], [], []  # 列表存储数据
    total_reward = 0
    done = False
    observation = env.reset(snrdb_test, test_ds, data_type)
    observation = agent.observation_to_state(observation)
    #########################DQN初始化
    while not done:
        action = agent.choose_action(observation, isTrain=True)  # 根据观测值observation选择要执行的动作
        print('选择带宽： ', action)
        print('###########开始传输数据##############')
        for sents in pbar:
           sents = sents.to(device)
           mse = train_step(net, sents, sents, noise_std[0], pad_idx, optimizer, criterion, args.channel, action)

        print('###########结束传输数据##############')
        
        observation_ = env.reset(noise_std[0], sents, data_type)
        observation_ = agent.observation_to_state(observation_)
        reward = 100-mse
        print('reward', reward)
        if reward > 99:
            done = True
        agent.remember(observation, action, reward, observation_, done)  # 将结果存入记忆体   obs空
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







def train(epoch, args, net):
    train_eur = EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_iterator)
    noise_std = np.random.uniform(SNR_to_noise(0), SNR_to_noise(20), size=(1))
    
    print('###########开始传输数据##############')
    for sents in pbar:
       sents = sents.to(device)
       mse = train_step(net, sents, sents, noise_std[0], pad_idx, optimizer, criterion, args.channel, 8)            ##################################在这修改带宽################################################
    print('###########结束传输数据##############')





if __name__ == '__main__':
    # setup_seed(10)
    args = parser.parse_args()
    args.vocab_file = '/home/src/data/' + args.vocab_file
    """ preparing the dataset """
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]


    deepsc = DeepSC(args.tcn, args.num_layers, num_vocab, num_vocab, num_vocab, num_vocab, args.d_model, args.num_heads,args.dff, 0.1).to(device)
    criterion = nn.CrossEntropyLoss(reduction = 'none')
    optimizer = torch.optim.Adam(deepsc.parameters(),
            lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay = 5e-4)
    initNetParams(deepsc)

    for epoch in range(args.epochs):
        start = time.time()
        record_acc = 10
        
        print('开始第:{}次train'.format(epoch))
        train(epoch, args, deepsc)
        #train_DQN(epoch, args, deepsc)                        ##############################################################################################
        print('开始第:{}次val'.format(epoch))
        avg_acc = validate(epoch, args, deepsc)
        if avg_acc < record_acc:
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            with open(args.checkpoint_path + '/checkpoint_{}.pth'.format(str(epoch + 1).zfill(2)), 'wb') as f:
                torch.save(deepsc.state_dict(), f)
            record_acc = avg_acc
    record_loss = []


    

        


