# Compute the mean relative error for a trained model.
#
# Code author: Russell Edson, Biometry Hub 26/10/2020

import os
import glob
import numpy as np
import torch
import time

import torch.optim as optim
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from main import get_net
from dataset import read_data, get_id
from model import testModel, ELBO


# Based on the TrainDataSet/TestDataSet classes, but removes the
# duplication and parameterises on the training/eval data subset
# list so that we can effortlessly train/eval different splits.
class GenDataset(Dataset):

    def __init__(self, data_path, list_path):
        self.data_path = data_path
        self.data_list = get_id(list_path)
        self.data, self.label = read_data(self.data_path)

    def __getitem__(self, index):
        case_id = int(self.data_list[index])
        data = self.data[case_id]
        label = self.label[case_id, :]
        return torch.FloatTensor(data), torch.FloatTensor(label)

    def __len__(self):
        return len(self.data_list)


def mre(pred, obs, n=1000):
    """Return the mean relative error for the given predictions and
    observations (up to a maximum of N pred/obs pairs).

    Arguments:
    pred -- a list or array of prediction values
    obs -- a list or array of observation values
    N -- the maximum number of pred/obs pairs to compare (default=1000)

    Returns:
    the mean relative error of the predictions.

    Usage:
    mre([1, 5, 2.222], [1.1, 5, 2.12])
    """
    pred = np.array(pred)
    obs = np.array(obs)
    n = min(n, len(pred))
    return sum(abs(obs[0:n] - pred[0:n]) / abs(obs[0:n])) / n

start_time = time.time()
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CODE_PATH = os.path.abspath('.')
DATA_PATH = os.path.abspath('../2014_Data')


# Neural network training parameters (constant, matches with existing
# model runs)
EPOCHS = 10000
LEARN_RATE = 0.0001
BATCH_SIZE = 4096
INIT_KL_WEIGHT = 1.0
THRESHOLD = 3.0
LOSS = nn.L1Loss()
INIT_BEST_LOSS = 1000.0
PRINT_FREQUENCY = 1  # iterations (~3 per epoch)
TEST_FREQUENCY = 50  # epochs
SAVE_FREQUENCY = 100  # epochs

# Compute MREs for all of the splits, training different NNs
splits = [os.path.abspath(file) for file in glob.glob('../2014_splits/*')]
for split in splits:
    # Construct the training and test dataset loaders
    train_list = os.path.join(split, 'trainID.txt')
    train_dataset = GenDataset(DATA_PATH, train_list)
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    test_list = os.path.join(split, 'testID.txt')
    test_dataset = GenDataset(DATA_PATH, test_list)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    output_path = os.path.join(split, 'run_output')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    log_file = os.path.join(output_path, 'log.txt')
    mre_file = os.path.join(output_path, 'mre.txt')
    pred_file = os.path.join(output_path, 'predictions.txt')

    # Train the neural network
    model = testModel(threshold=THRESHOLD).cuda()
    optimiser = optim.Adam(model.parameters(), lr=LEARN_RATE)
    elbo = ELBO(model, len(train_data_loader.dataset)).cuda()
    kl_weight = INIT_KL_WEIGHT
    best_loss = INIT_BEST_LOSS
    running_mre = best_loss + 1.0
    iter = 0

    for epoch in range(1, EPOCHS + 1):
        print('Epoch:{:0>5d}, kl_weight:{:0>5f}\n'.format(epoch, kl_weight))
        kl_weight = min(kl_weight + 0.02, 1)

        model.train()
        for img, label in train_data_loader:
            img, label = img.cuda(), label.cuda()
            output = model(img)
            loss, kl, mse, sparsity = elbo(output, label, kl_weight)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            iter += 1

            # Print progress at specified iterations
            if iter % PRINT_FREQUENCY == 0:
                res = 'Loss:{:0.4f}, KL:{:0.4f}, '.format(loss, kl)
                res += 'MSE:{:0.4f}, Sparsity:{:0.4f}'.format(mse, sparsity[0])

                print(res)
                with open(log_file, 'a') as file:
                    file.write(res + '\n')

        # Toggle between training/evaluating at specified epochs
        if epoch % TEST_FREQUENCY == 0:
            model.eval()

            rel_errors = []
            for img, label in test_data_loader:
                img, label = img.cuda(), label.cuda()
                output = model(img)
                loss = LOSS(output, label)
                rel_error = abs(output - label) / abs(label)
                rel_errors.append(rel_error.cpu().data.numpy())

                running_mre = np.mean(np.array(rel_errors))
                res = 'Test loss:{:0.4f}, '.format(running_mre)
                res += 'Best loss:{:0.4f}'.format(best_loss)
                print(res)
                with open(log_file, 'a') as file:
                    file.write(res + '\n')

            if running_mre < best_loss:
                best_loss = running_mre
                print('New best loss:{:0.4f}'.format(best_loss))
                save_path = os.path.join(output_path, 'best_epoch.pt')
                torch.save(model.state_dict(), save_path)

        # Save the current NN model state at specified epochs
        if epoch % SAVE_FREQUENCY == 0:
            print('Saving model for epoch {:0>5d}'.format(epoch))
            save_path = os.path.join(output_path, 'epoch_{}.pt'.format(epoch))
            torch.save(model.state_dict(), save_path)

    # Evaluate MREs in the test datasets (first 1000) using the best
    # epoch from the training
    model = get_net(os.path.join(output_path, 'best_epoch.pt'))
    model.eval()
    test_data_loader = DataLoader(test_dataset, batch_size=1)

    pred, obs = [], []
    with torch.no_grad():
        for img, label in test_data_loader:
            img, label = img.cuda(), label.cuda()
            output = model(img)
            pred.append(float(output))
            obs.append(float(label))

    with open(mre_file, 'a') as file:
        file.write('MRE (first 1000):{:0.6f}\n'.format(mre(pred, obs, n=1000)))
    with open(pred_file, 'w') as file:
        file.write('Observation\tPrediction\n')
        for idx, prd in enumerate(pred):
            file.write('{:0.3f}\t{:0.3f}\n'.format(obs[idx],prd))

          
end_time = time.time()
print('Time taken = {} sec'.format(end_time - start_time))