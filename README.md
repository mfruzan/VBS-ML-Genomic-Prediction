# VBS-ML-Genomic-Prediction
Variational Bayes Select
## Dependencies

    Python 3.6+
    PyTorch 1.0+
    TorchVision
    numpy

## VBS-ML folder

This code is used for marker selection. To run python code:
>python3  mre.py

class SBP_layer in model.py is the main code for marker selection. In the SBP_layer, where self.mask==0, the corresponding marker can be deleted.

Output of each split is written in a folder called run_output in eachy split folder. Yield observations vs predictions for test lines are written in a tab-separeted text file called predictions.txt.

## Naive-ML folder

This code is for generating a yield prediction using all markers. To run python code:
>python3  main.py

Yield observations vs predictions for test lines are written in a tab-separeted text file in results/predictions.txt.

For enquiries please contact Qingsen Yan <qingsenyan@gmail.com>
