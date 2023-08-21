import pandas as pd
import os
import numpy as np
import time


from util import Gen_Data_id
from dataset import TrainDataSet, TestDataSet, ValidateDataSet
from model import testModel
import torch.optim as optim
from init_weights import init_weights
import torch.nn as nn
import torch
from util import load_checkpoint

from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = {
    ## Generating train and test sets
    "GenData": False,
    'GenDataPerce': 0.2, # the percentage of test set

    ##
    'datapath': '../2014_Data', # the data path of challenge2
    'codepath': '.', # The path for generating the train and test IDs
    'epoch': 10000,
    'lr': 0.00001,
    'batch_size': 2048,
    'half': False,
    'Reload': False,
    'phase': 'train',
    'checkpoint_path': 'checkpoint/epoch_10000.pt' # The model path, it can be modified to any model in the 'checkpoint' folder
}




class Train(object):
    def __init__(self, config):

        if config['Reload']:
            self.model = get_net(config['checkpoint_path'])
        else:
            self.model = (testModel()).cuda()
            init_weights(self.model, "kaiming")


        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'], weight_decay=0)
        self.loss = nn.L1Loss()


        # dataset
        dataset = TrainDataSet(config)
        # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        self.train_data_loader = DataLoader(dataset,
                                            batch_size=config['batch_size'],
                                            # num_workers=1,
                                            )

        Vali_dataset = ValidateDataSet(config)   # Note that we use validation to choose the best results on test data
        # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        self.validate_data_loader = DataLoader(Vali_dataset,
                                            batch_size=1,
                                            # num_workers=1,
                                            )

        # other
        self.epoch = config['epoch']
        self.iter_num = 0
        self.print_frequency = 500

        # save
        self.model_save_dir = 'checkpoint'
        self.model_save_frequency = 1

        if not os.path.exists(self.model_save_dir):
            os.mkdir(self.model_save_dir)

        # mkdir(self.model_save_dir)
        # sys.stdout = Logger(join(self.model_save_dir, '{}_infor.txt'.format(config['type'])))
        # print('train data:{}'.format(len(dataset)))
        # print(config)

    def train(self, epoch):
        self.model.train()
        filename = os.path.join(self.model_save_dir, 'log.txt')

        for img, label in self.train_data_loader:

            img, label = img.cuda(), label.cuda()
            # torch.cuda.synchronize()
            # iter_start = time.time()
            output = self.model(img)
            loss = self.loss(output, label)

            self.optimizer.zero_grad()
            if config['half']:
                L1_reg = 0
                for param in self.model.parameters():   # L1 regularization for the parameters
                    L1_reg += torch.sum(torch.abs(param))
                loss += 0.0001*L1_reg
                loss.backward()
            else:
                L1_reg = 0
                for param in self.model.parameters():
                    L1_reg += torch.sum(torch.abs(param))
                loss += 0.0001*L1_reg
                loss.backward()
            self.optimizer.step()

            self.iter_num += 1
            print('loss: {:0.4f}'.format(loss))

            with open(filename, 'a') as f:
                f.write('loss: {:0.4f}\n'.format(loss))

        self.model.eval()
        filename = os.path.join(self.model_save_dir, 'ValiRes.txt')
        print('---------Validate------------')

        ErrL1 = list()
        ErrRela = list()
        ErrMSE = list()
        for img, label in self.validate_data_loader:

            img, label = img.cuda(), label.cuda()

            output = self.model(img)
            loss = self.loss(output, label)
          #  print('Validate_loss: {:0.4f}'.format(loss))
            ErrL1.append(loss.cpu().data.numpy())
            ErrRela.append((abs(output-label)/label).cpu().data.numpy())
            ErrMSE.append(np.square((output-label).cpu().data.numpy()))



        ErrL1_avge = np.mean(np.array(ErrL1))
        ErrL1_var = np.std(np.array(ErrL1), ddof=1)
        ErrRela_avge = np.mean(np.array(ErrRela))
        ErrRela_var = np.std(np.array(ErrRela), ddof=1)
        print('Validate_loss: {:0.4f}'.format(ErrRela_avge))
        # ErrMSE_avge = np.mean(np.array(ErrMSE))
        ## ErrMSE_var = np.std(np.array(ErrMSE), ddof=1)
        with open(filename, 'a') as f:
        #    f.write("L1 loss are {:.4f} and {:.4f} ".format(ErrL1_avge, ErrL1_var))
            f.write("Relative Error are {:.4f} and {:.4f}\n".format(ErrRela_avge, ErrRela_var))

        return ErrRela_avge



    def start(self):
        beforeRes = 1

        for epoch in range(1, self.epoch + 1):
            print('best loss {:0.4f}'.format(beforeRes))

            print('epoch:{:0>5d}\n'.format(epoch))
            # train
            ErrRela_avge = self.train(epoch)

            # save model
            if self.iter_num % self.print_frequency == 0:
                self.save(epoch)
            # save the best model
            if beforeRes > ErrRela_avge and epoch > 10:
                beforeRes = ErrRela_avge
                save_path = os.path.join(self.model_save_dir, 'epoch_best.pt')
                torch.save(self.model.state_dict(), save_path)



    def save(self, epoch):
        if epoch % self.model_save_frequency == 0:
            save_path = os.path.join(self.model_save_dir, 'epoch_{}.pt'.format(epoch))
            torch.save(self.model.state_dict(), save_path)




class Test(object):
    def __init__(self, config):


        self.model = get_net(config['checkpoint_path'])
        self.loss = nn.L1Loss()

        # dataset
        dataset = TestDataSet(config)
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
        filename = os.path.join(self.result_save_dir, 'Res.txt')
        pred_file = os.path.join(self.result_save_dir, 'predictions.txt')

        ErrL1 = list()
        ErrRela = list()
        ErrMSE = list()
        pred, obs = [], []
        for img, label in self.test_data_loader:

            img, label = img.cuda(), label.cuda()

            output = self.model(img)
            pred.append(float(output))
            obs.append(float(label))
            loss = self.loss(output, label) # predicting the yield
            print('loss: {:0.4f}'.format(loss))
            ErrL1.append(loss.cpu().data.numpy())
            ErrRela.append((abs(output-label)/label).cpu().data.numpy())
            ErrMSE.append(np.square((output-label).cpu().data.numpy()))

            with open(filename, 'a') as f:
                f.write('loss: {:0.4f}\n'.format(loss))

        ErrL1_avge = np.mean(np.array(ErrL1))
        ErrL1_var = np.std(np.array(ErrL1), ddof=1)
        ErrRela_avge = np.mean(np.array(ErrRela))
        ErrRela_var = np.std(np.array(ErrRela), ddof=1)
        ErrMSE_avge = np.mean(np.array(ErrMSE))
        ErrMSE_var = np.std(np.array(ErrMSE), ddof=1)

        print("The average and variance of L1 loss are {:.4f} and {:.4f}\n".format(ErrL1_avge, ErrL1_var))
        print("The average and variance of MSE are {:.4f} and {:.4f}\n".format(ErrMSE_avge, ErrMSE_var))
        print("The average and variance of Relative Error are {:.4f} and {:.4f}\n".format(ErrRela_avge, ErrRela_var))
        with open(pred_file, 'w') as file:
            file.write('Observation\tPrediction\n')
            for idx, prd in enumerate(pred):
                file.write('{:0.3f}\t{:0.3f}\n'.format(obs[idx],prd))




def get_net(checkpoint_path):

    model = (testModel()).cuda()
    # model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    return model

start_time = time.time()

if __name__ == '__main__':


    # Generating the train and test IDs
    if config['GenData']:
        yieldPath = os.path.join(config['datapath'], 'Grain.yield_adjusted.csv')
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

end_time = time.time()
print('Time taken = {} sec'.format(end_time - start_time))