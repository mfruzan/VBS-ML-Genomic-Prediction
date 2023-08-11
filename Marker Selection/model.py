import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn.init as init
import numpy

class Linearvbd(nn.Module):
    def __init__(self,in_features, out_features, threshold=3, bias=True):
        super(Linearvbd, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold

        self.W = Parameter(torch.Tensor(out_features, in_features))
        self.log_sigma = Parameter(torch.Tensor(out_features, in_features))

        self.reset_parameters()



    def reset_parameters(self):
        self.W.data.normal_(0, 0.02)
        init.kaiming_normal_(self.W, a=0, mode='fan_in')
        self.log_sigma.data.fill_(-5)

    def forward(self, x):
        self.log_alpha = self.log_sigma * 2.0 - 2.0 * torch.log(1e-8 + torch.abs(self.W))
        self.log_alpha = torch.clamp(self.log_alpha, -10, 10)

        if self.training:
            lrt_mean = F.linear(x, self.W)
            lrt_std = torch.sqrt(F.linear(x * x, torch.exp(self.log_sigma * 2.0)) + 1e-8)
            eps = lrt_std.data.new(lrt_std.size()).normal_()
            return lrt_mean + lrt_std * eps
        mask = (self.log_alpha < self.threshold).float()
        #return F.linear(x, self.W * mask)
        return F.linear(x, self.W)

    def kl_reg(self):
        kl = 0.5 * torch.log1p(torch.exp(-self.log_alpha))
        a = torch.sum(kl)
        return a


class Linearvbd1(nn.Module):
    def __init__(self,in_features, out_features, threshold=3, bias=True):
        super(Linearvbd1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold

        self.W = Parameter(torch.Tensor(out_features, in_features))
        self.log_sigma = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(1, out_features))

        self.reset_parameters()



    def reset_parameters(self):
        self.bias.data.zero_()
        self.W.data.normal_(0, 0.001)
        #init.kaiming_normal_(self.W, a=0, mode='fan_in')
        self.log_sigma.data.fill_(-4)


    def forward(self, x):
        self.log_alpha = self.log_sigma * 2.0 - 2.0 * torch.log(1e-8 + torch.abs(self.W))
        self.log_alpha = torch.clamp(self.log_alpha, -10, 10)

        if self.training:
            lrt_mean = F.linear(x, self.W) + self.bias
            lrt_std = torch.sqrt(F.linear(x * x, torch.exp(self.log_sigma * 2.0)) + 1e-8)
            eps = lrt_std.data.new(lrt_std.size()).normal_()
            return lrt_mean + lrt_std * eps
        mask = (self.log_alpha < self.threshold).float()
        #return F.linear(x, self.W * mask) + self.bias
        return F.linear(x, self.W) + self.bias

    def kl_reg(self):
        kl = 0.5 * torch.log1p(torch.exp(-self.log_alpha))
        a = torch.sum(kl)
        return a




class SBP_layer(nn.Module):
    """
    Structured Bayesian Pruning layer
    #don't forget add kl to loss
    y, kl = sbp_layer(x)
    loss = loss + kl
    """
    def __init__(self, input_dim, init_logsigma2=9.0):
        super(SBP_layer, self).__init__()

        self.log_sigma2 = nn.Parameter(torch.Tensor(input_dim))
        self.mu = nn.Parameter(torch.Tensor(input_dim))

        self.mu.data.normal_(1.0,0.01)
        self.log_sigma2.data.fill_(-init_logsigma2)

    def forward(self, input):

        self.log_alpha = self.log_sigma2 - 2.0 * torch.log(abs(self.mu) + 1e-8)
        self.log_alpha = torch.clamp(self.log_alpha, -10.0, 10.0)


        #mask = (self.log_alpha < 1.0).float()
        self.mask = (self.log_alpha < 0.0).float()
        if self.training:
            si = (self.log_sigma2).mul(0.5).exp_()
            eps = si.data.new(si.size()).normal_()
            multiplicator = self.mu + si*eps
        else:
            multiplicator = self.mu*self.mask
        return multiplicator*input

    def kl_reg_input(self):
        kl = 0.5 * torch.log1p(torch.exp(-self.log_alpha))
        a = torch.sum(kl)
        return a
    def sparse_reg_input(self):
        s_ratio = torch.sum(self.mask.view(-1)==0.0).item() / self.mask.view(-1).size(0)
        return s_ratio



class Residual_Block(nn.Module):
    def __init__(self, i_channel, o_channel):
        super(Residual_Block, self).__init__()
        self.Linear1 = nn.Linear(in_features=i_channel, out_features=o_channel)
        self.bn1 = nn.BatchNorm1d(o_channel)
        self.relu = nn.ReLU(inplace=True)

        self.Linear2 = nn.Linear(in_features=i_channel, out_features=o_channel)
        self.bn2 = nn.BatchNorm1d(o_channel)

    def forward(self, x):
        residual = x

        out = self.Linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.Linear2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class testModel(nn.Module):
    def __init__(self, threshold=0):
        super(testModel, self).__init__()
        # self.fc1 = Linearvbd(17114, 128, threshold)
        # self.fc2 = Linearvbd(128, 64, threshold)
        # self.fc3 = Linearvbd(64, 32, threshold)
        # self.fc4 = Linearvbd(32, 1, threshold)
        # self.threshold = threshold
        #
        self.sbp1 = SBP_layer(17114)

        #self.sbp1 = SBP_layer(18101)
        # self.sbp2 = SBP_layer(128)
        # self.sbp3 = SBP_layer(64)
        # self.sbp4 = SBP_layer(32)


        # linear
        #self.fc1 = Linearvbd(17114, 1, threshold)

        # non-linear
        #self.fc2 = Linearvbd1(302, 302, threshold)
        self.fc3 = Linearvbd1(17114, 256, threshold)

        #self.fc3 = Linearvbd1(18101, 256, threshold)
        self.fc4 = Linearvbd1(256, 128)
        self.fc5 = Linearvbd1(128, 32)
        self.fc6 = Linearvbd1(32, 1)
        self.threshold = threshold
        #
        # self.sbp1 = SBP_layer(17114)



        #self.sbp1 = SBP_layer(17114)



    def forward(self, input):

        # output = self.sbp1(input)
        # output = F.relu(self.fc1(output))
        # output = self.sbp2(output)
        # output = F.relu(self.fc2(output))
        # output = self.sbp3(output)
        # output = F.relu(self.fc3(output))
        # output = self.sbp4(output)
        # output = (self.fc4(output))
        # return output

        #linearoutput = (self.fc1(input))

        # nonlinearoutput = self.sbp1(input)
        # nonlinearoutput = F.relu(self.fc2(nonlinearoutput))
        #
        # nonlinearoutput = F.relu(self.fc3(nonlinearoutput))
        #
        # nonlinearoutput = F.relu(self.fc4(nonlinearoutput))
        #
        # nonlinearoutput = (self.fc5(nonlinearoutput))
        # return nonlinearoutput

        # nonlinearoutput = (self.sbp1(input))

        nonlinearoutput = F.relu(self.sbp1(input))

        nonlinearoutput = F.relu(self.fc3(nonlinearoutput))

        nonlinearoutput = F.relu(self.fc4(nonlinearoutput))
        nonlinearoutput = F.relu(self.fc5(nonlinearoutput))
        nonlinearoutput = (self.fc6(nonlinearoutput))


        return nonlinearoutput

    def layerwise_sparsity(self):
         return [self.sbp1.sparse_reg_input()]


# class testModel(nn.Module):
#     def __init__(self, block):
#         super(testModel, self).__init__()
#         #self.sbp1 = SBP_layer(17114)
#         self.linear1 = Linearvbd1(302,100)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.layer1 = self.make_layer(block, 100, 10)
#
#
#         self.linear2 = Linearvbd1(100,1)
#
#     def make_layer(self, block, out_channels, blocks):
#         layers = []
#         layers.append(block(100, 100))
#         for i in range(1, blocks):
#             layers.append(block(out_channels, out_channels))
#         return nn.Sequential(*layers)  # add all of the residual block
#
#     def forward(self, x):
#      #   out = self.sbp1(x)
#         out = self.linear1(x)
#         out = self.relu1(out)
#         out = self.layer1(out)
#         out = self.linear2(out)
#
#         return out
#   #  def layerwise_sparsity(self):
#   #      return [self.sbp1.sparse_reg_input()]





class ELBO(nn.Module):

    def __init__(self, net, train_size):
        super(ELBO, self).__init__()
        self.train_size = train_size
        self.net = net

    def forward(self, input, target, kl_weight=1.0):
        assert not target.requires_grad
        kl = 0.0
        for module in self.net.children():
            if hasattr(module, 'kl_reg'):
               kl = kl + module.kl_reg()
            if hasattr(module, 'kl_reg_input'):
               kl = kl + module.kl_reg_input()

        sparsity_arr = self.net.layerwise_sparsity()

       # print('l1-Sparsity: %.4f, l2-Sparsity: %.4f, l3-Sparsity: %.4f' % (sparsity_arr[0], sparsity_arr[1], sparsity_arr[2]))
        loss = kl_weight * kl + 0.5 * self.train_size * F.mse_loss(input, target)
        return loss, kl, F.mse_loss(input, target), sparsity_arr