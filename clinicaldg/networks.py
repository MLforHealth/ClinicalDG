# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from clinicaldg.lib import misc
import clinicaldg.eicu.models as eICUModels
import clinicaldg.cxr.models as cxrModels

class MLP(nn.Module):
    """Just an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'],hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


def Featurizer(input_shape, hparams, dataset_name, dataset):
    """Auto-select an appropriate featurizer for the given dataset."""
    if dataset_name == 'ColoredMNIST':
        return MLP(input_shape[0], 128, hparams)
    elif dataset_name[:4] == 'eICU':        
        if hparams['eicu_architecture'] == 'MLP':
            return eICUModels.FlattenedDense(*dataset.d.get_num_levels(), hparams['emb_dim'], hparams['mlp_depth'], hparams['mlp_width'],
                                            dropout_p = hparams['mlp_dropout'])
        elif hparams['eicu_architecture'] == 'GRU':
            return eICUModels.GRUNet(*dataset.d.get_num_levels(), hparams['emb_dim'], hparams['gru_layers'], hparams['gru_hidden_dim'],
                                            dropout_p = hparams['mlp_dropout'])
    elif dataset_name[:3] == 'CXR':
        return cxrModels.EmbModel('densenet', pretrain = True, concat_features = 1 if dataset_name == 'CXRSubsampleObs' else 0)
    else:
        raise NotImplementedError
