This code is used for data selection, based on our recent paper:
"Variational Bayesian Dropout with a Hierarchical Prior"
https://arxiv.org/abs/1811.07533v3

class SBP_layer in model.py is the main code for selection. In the SBP_layer, self.mask==0 mean that we can delete the correspond input data.

Any problems, please drop me an email (liuyuhang@whu.edu.cn)