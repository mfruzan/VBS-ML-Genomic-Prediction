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

## Naive-ML folder

This code is used for yield prediction using all markers. To run python code:
>python3  main.py


For enquiries please contact Qingsen Yan <qingsenyan@gmail.com>
