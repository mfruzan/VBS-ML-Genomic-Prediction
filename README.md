# VBS-ML-Genomic-Prediction
Variational Bayes Select
## Dependencies

    Python 3.6+
    PyTorch 1.0+
    TorchVision
    numpy



## VBS-ML folder

This code is used for Marker selection. To run python code:
>python3  mre.py


class SBP_layer in model.py is the main code for selection. In the SBP_layer, where self.mask==0, the corresponding marker can be deleted .

Output of each split is written in a folder called run_output under that split. Yield observation vs prediction for test lines is written in a tab-separeted text file called predictions.txt.

## Naive-ML folder

This code is used for yield prediction using all markers. To run python code:
>python3  main.py


Output Yield observations vs predictions  for test lines is written to a tab-separated text file in result/predictions.txt.

For enquiries please contact Qingsen Yan <qingsenyan@gmail.com>
