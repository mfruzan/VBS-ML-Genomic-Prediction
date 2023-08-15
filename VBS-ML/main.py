import pandas as pd
from os import path
import os

import numpy as np
from util import Gen_Data_id
from dataset import TrainDataSet, TestDataSet
from model import testModel, ELBO,Residual_Block
import torch.optim as optim
from init_weights import init_weights
import torch.nn as nn
import torch
from util import load_checkpoint
# import logging
# import sys

from torch.utils.data import DataLoader



#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = {
    ## test set
    "GenData": False,
    'GenDataPerce': 0.1,
    ##
    'datapath': '/home/python_code/Challenge2/2016_Data',
    'codepath': '/home/python_code/Challenge2',
    'epoch': 10000,
    'lr': 0.0001,
    'batch_size': 4096,
    'half': False,
    'phase': 'train',

    # 'checkpoint_path': 'checkpoint/epoch_1.pt',
    'checkpoint_path': '/home/python_code/Challenge2/checkpoint/epoch_8000.pt',
}




class Train(object):
    def __init__(self, config):


        self.model = (testModel(threshold=3.0)).cuda()
        #self.model = (testModel(Residual_Block)).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])


        # init_weights(self.model, "kaiming")

        # self.model.load_state_dict(torch.load(config['checkpoint_path']), strict=False)

        # dataset

        dataset = TrainDataSet(config)
        # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        self.train_data_loader = DataLoader(dataset,
                                            batch_size=config['batch_size'],
                                            # num_workers=1,
                                            )

        #loss
        self.elbo = ELBO(self.model, len(self.train_data_loader.dataset)).cuda()
        self.kl_weight = 1.0
        # other
        self.epoch = config['epoch']
        self.iter_num = 0
        self.print_frequency = 400

        # save
        self.model_save_dir = 'checkpoint'
        self.model_save_frequency = 1

        if not os.path.exists(self.model_save_dir):
            os.mkdir(self.model_save_dir)

        # mkdir(self.model_save_dir)
        # sys.stdout = Logger(join(self.model_save_dir, '{}_infor.txt'.format(config['type'])))
        # print('train data:{}'.format(len(dataset)))
        # print(config)



        # define test
        dataset_test = TestDataSet(config)
        # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        self.test_data_loader = DataLoader(dataset_test,
                                            batch_size=config['batch_size'])

        self.loss = nn.L1Loss()
        self.bestloss = 1000.0;
    def test(self,epoch):
        self.model.eval()

        ErrRela = list()
        for img, label in self.test_data_loader:
            img, label = img.cuda(), label.cuda()

            output = self.model(img)
            loss = self.loss(output, label)
            ErrRela.append((abs(output - label) / label).cpu().data.numpy())
            ErrRela_avge = np.mean(np.array(ErrRela))
        print('test loss: {:0.4f}, best loss: {:0.4f}'.format(ErrRela_avge, self.bestloss))
        if ErrRela_avge < self.bestloss:
            self.save_best(epoch)
            self.bestloss = ErrRela_avge



    def train(self, epoch):
        self.model.train()
        filename = path.join(self.model_save_dir, 'log.txt')

        for img, label in self.train_data_loader:

            img, label = img.cuda(), label.cuda()
            # torch.cuda.synchronize()
            #img = img.reshape(img.shape[0], 1, img.shape[1])
            #label = label.reshape(label.shape[0], 1, label.shape[1])
            # iter_start = time.time()
            output = self.model(img)

            loss, kl, mse, sparsity = self.elbo(output, label, self.kl_weight)

            self.optimizer.zero_grad()
            if config['half']:
                # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                #     scaled_loss.backward()
                loss.backward()
            else:
                loss.backward()
            self.optimizer.step()

            # torch.cuda.synchronize()
            # iter_end = time.time()
            self.iter_num += 1
            print('loss: {:0.4f}, KL: {:0.4f}, MSE: {:0.4f}, sparsity level: {:0.4f}'.format(loss,kl,mse,sparsity[0]))


            with open(filename, 'a') as f:
                f.write(('loss: {:0.4f}, KL: {:0.4f}, MSE: {:0.4f}, sparsity level: {:0.4f}\n'.format(loss,kl,mse,sparsity[0])))

            # if self.iter_num % self.print_frequency == 0:
            #     print(
            #         'epoch:{:0>3d} | iter:{:<4}| type:{}|loss:{:.4f}|recall:{:.4f}|precision:{:.4f}|dice:{:.4f}|{:.2f}'
            #             .format(epoch, self.iter_num, 'single', loss.data.cpu().item(),
            #                     s_recall, s_precision, s_dice, (iter_end - iter_start)))

            # start_time = time.time()
    def start(self):
        for epoch in range(1, self.epoch + 1):

            print('epoch:{:0>5d}, kl_weight:{:0>5f}\n'.format(epoch, self.kl_weight))
            # train
            self.kl_weight = min(self.kl_weight + 0.02, 1)
            self.train(epoch)

            if epoch % 50 == 0:
                config['phase']='test'
                self.test(epoch)
                config['phase'] = 'train'

            if self.iter_num % self.print_frequency == 0:
            # save model
                self.save(epoch)



    def save(self, epoch):
        if epoch % self.model_save_frequency == 0:
            save_path = path.join(self.model_save_dir, 'epoch_{}.pt'.format(epoch))
            torch.save(self.model.state_dict(), save_path)
    def save_best(self, epoch):
        if epoch % self.model_save_frequency == 0:
            save_path = path.join(self.model_save_dir, 'best_epoch.pt')
            torch.save(self.model.state_dict(), save_path)



class Test(object):
    def __init__(self, config):


        self.model = get_net(config['checkpoint_path'])

        # log_alpha = self.model.sbp1.log_sigma2 - 2.0 * torch.log(abs(self.model.sbp1.mu) + 1e-8)
        # log_alpha = torch.clamp(log_alpha, -10.0, 10.0)
        # mask = (log_alpha < 0.0).float()
        # result1 = np.array(mask.cpu().int())
        # np.savetxt('npresultbest.txt', result1)

        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.loss = nn.L1Loss()

        # dataset
        dataset = TestDataSet(config)
        # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        self.test_data_loader = DataLoader(dataset,
                                            batch_size=1,
                                            # num_workers=1,
                                            )

        # save
        self.result_save_dir = 'result'
        if not os.path.exists(self.result_save_dir):
            os.mkdir(self.result_save_dir)

    def test(self):
        self.model.eval()
        filename = path.join(self.result_save_dir, 'L1.txt')

        for img, label in self.test_data_loader:

            img, label = img.cuda(), label.cuda()

            output = self.model(img)
            loss = self.loss(output, label)
            loss1 = loss/(label)
            loss2 = loss1.squeeze()
            print('loss: {:0.4f}'.format(loss2))

            with open(filename, 'a') as f:
                f.write('{:0.4f}\n'.format(loss2))


def get_net(checkpoint_path):

    model = (testModel(threshold=3)).cuda()
    # model = torch.nn.DataParallel(model)
    # load_checkpoint(model, checkpoint_path=checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    return model



if __name__ == '__main__':


    # Get the test set
    if config['GenData']:
        yieldPath = path.join(config['datapath'], 'Grain.yield_adjusted.csv')
        yieldData = pd.read_csv(yieldPath, usecols=['Genotype', 'Adj.Grain.Yield']).values
        TotalSamples = yieldData.shape[0]
        TestSet = config['GenDataPerce']
        Gen_Data_id(config['codepath'], TotalSamples, TestSet)

    if config['phase'] == 'train':
        trainer = Train(config)
        trainer.start()
    else:
        test = Test(config)
        test.test()