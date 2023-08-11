# VBS-ML-Genomic-Prediction
Variational Bayes Select
## Dependencies

    Python 3.6+
    PyTorch 1.0+
    TorchVision
    numpy



## Data selection folder

This code is used for Marker selection, based on our recent paper:

class SBP_layer in model.py is the main code for selection. In the SBP_layer, where self.mask==0, the corresponding marker can be deleted .
