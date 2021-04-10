# Datasets

The following table shows how the dataset parameter (which is passed to `--dataset` in `train.py` or `--datasets` in `sweep.py`) maps to the datasets described in the paper.

| Parameter Value           | Dataset         | Synthetic Shift |
|--------------------|-----------------|-----------------|
| `eICU`              | eICU            | `Base`          |
| `eICUCorrLabel`      | eICU            | `CorrLabel`     |
| `eICUCorrNoise`      | eICU            | `CorrNoise`     |
| `eICUSubsampleUnobs` | eICU            | `BiasSampUnobs` |
| `eICUSubsampleObs`   | eICU            | `BiasSampObs`   |
| `CXR`                | CXR (Multitask) | `Base`          |
| `CXRBinary`          | CXR (Binary)    | `Base`          |
| `CXRSubsampleUnobs`  | CXR (Binary)    | `BiasSampUnobs` |
| `CXRSubsampleObs`    | CXR (Binary)    | `BiasSampObs`   |
| `ColoredMNIST`    | Colored MNIST    |    |




# Data Hyperparameters

The following table shows the list of parameters that can be passed as part of the JSON encoded dictionary to the `--hparams` argument in `train.py` or `sweep.py`. Hyperparameters not explicitly specified during training will default to (one of) their paper values.


| hparam                        | Applicable Datasets | Possible Values | Description                                                                          | Paper Value(s)    |
|-------------------------------|---------------------|-----------------|--------------------------------------------------------------------------------------|-------------------|
| `eicu_architecture`             | `eICU*`               | {`MLP`, `GRU`}  | Model architecture for eICU tasks                                                   | `GRU`            |
| `corr_label_train_corrupt_dist` | `eICUCorrLabel`       |    [0, 0.5]             |  𝛿 from Section 4.2 of the paper                                                                                    |      0.1           |
| `corr_label_train_corrupt_mean` | `eICUCorrLabel`       |   [0, 1]              |     β from Section 4.2 of the paper                                                                               |       {0.1, 0.3, 0.5}               |
| `corr_label_val_corrupt`        | `eICUCorrLabel`       |   [0, 1]              |     p_{val} from Section 4.2 of the paper                                                                                 |     0.5             |
| `corr_label_test_corrupt`       | `eICUCorrLabel`       |    [0, 1]             |       p_{test} from Section 4.2 of the paper                                                                                |       0.9            |
| `corr_noise_train_corrupt_dist` | `eICUCorrNoise`       |   (-∞, +∞)             |    𝛿 from Section 4.3 of the paper                                                                                   |     {0.1, 0.5}              |
| `corr_noise_train_corrupt_mean` | `eICUCorrNoise`       |      (-∞, +∞)           |      β from Section 4.3 of the paper                                                                                 |     {1.0, 2.0}              |
| `corr_noise_val_corrupt`        | `eICUCorrNoise`       |  (-∞, +∞)               |   λ_{val} from Section 4.3 of the paper                                                                                   |   0.0                |
| `corr_noise_test_corrupt`       | `eICUCorrNoise`       |    (-∞, +∞)             |   λ_{test} from Section 4.3 of the paper                                                                                   |     -1.0              |
| `corr_noise_std`                | `eICUCorrNoise`       | [0, ∞)     |    σ from Section 4.3 of the paper                                                                                  |     0.5              |
| `corr_noise_feature`            | `eICUCorrNoise`       | column          | Feature in eICU dataset to add noise to.                                             | `admissionweight` |
| `subsample_g1_mean`        | `*Subsample*`      |      [0, 1]           |    Average μ_M on the training environments                                                                                   |   eICU: 0.7 <br> CXR: 0.15                |
| `subsample_g2_mean`        | `*Subsample*`      |    [0, 1]             |   Average μ_F on the training environments                                                                                   |     eICU: 0.1 <br> CXR: 0.025              |
| `subsample_g1_dist`        | `*Subsample*`      |      [0, 0.5]           |    Distance of μ_M between each training environment                                                                                   |  eICU: 0.1 <br> CXR: 0.1                 |
| `subsample_g2_dist`        | `*Subsample*`      |    [0, 0.5]             |   Distance of μ_F between each training environment                                                                                   |    eICU: 0.05 <br> CXR: 0.01              |
| `cxr_augment`                   | `CXR*`                | {0, 1}          | Whether to use simple geometric image augmentations during training                 | 1                 |
| `use_cache`                     | `CXR*`                | {0, 1}          | Whether to load images from pre-cached binaries, or directly from downloaded images | N/A               |
| `cmnist_eta`                    | `ColoredMNIST`        |    [0, 1]             |        η from Appendix B.1 of the paper                                                                                               |    [0, 0.5]               |
| `cmnist_beta`                   | `ColoredMNIST`        |   [0, 1]              |    β from Appendix B.1 of the paper                                                                                                  |   [0.05, 0.5]                |
| `cmnist_delta`                  | `ColoredMNIST`        |       [0, 1]          |    𝛿 from Appendix B.1 of the paper                                                                                                    |     [0, 0.3]              |


