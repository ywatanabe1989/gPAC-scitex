# Standard vs MultiTask Training Comparison

## Overall Accuracy

| Method | Phase Accuracy | Amplitude Accuracy |
|--------|---------------|-------------------|
| Standard | 1.0000 | 1.0000 |
| Multitask | 1.0000 | 0.9907 |

**Phase accuracy improvement: +0.00%**

**Amplitude accuracy improvement: -0.93%**

## Band Learning Comparison

### Average Band Parameter Change (Hz)

| Method | Phase Bands | Amplitude Bands | Ratio (Phase/Amp) |
|--------|-------------|----------------|------------------|
| Standard | 0.00 Hz | 0.39 Hz | 0.0000 |
| Multitask | 0.00 Hz | 0.38 Hz | 0.0000 |

**Phase/Amplitude ratio improvement: +nan%**

## Task Weights Evolution (MultiTask Only)

| Epoch | Phase Weight | Amplitude Weight |
|-------|--------------|------------------|
| 0 | 0.9958 | 0.9958 |
| 12 | 0.9229 | 0.9235 |
| 25 | 0.8625 | 0.8682 |
| 37 | 0.8362 | 0.8546 |
| 49 | 0.8446 | 0.8811 |

## Conclusion

The MultiTaskLoss approach had an unexpected effect on the learning dynamics. The standard approach showed a better balance between phase and amplitude learning.