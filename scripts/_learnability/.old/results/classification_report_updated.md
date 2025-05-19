# PAC Signal Classification Report

## Summary

| Model | Phase Accuracy | Amplitude Accuracy | Samples |
|-------|---------------|-------------------|--------|
| Trainable gPAC | 1.0000 | 1.0000 | 54 |
| Fixed gPAC | 1.0000 | 1.0000 | 54 |

## Performance Across Noise Levels

### Phase Frequency Classification

| Noise Level | Trainable gPAC | Fixed gPAC | Improvement |
|------------|---------------|-----------|-------------|
| 0.10 | 1.0000 | 1.0000 | +0.0000 |
| 0.20 | 1.0000 | 1.0000 | +0.0000 |
| 0.30 | 1.0000 | 1.0000 | +0.0000 |

### Amplitude Frequency Classification

| Noise Level | Trainable gPAC | Fixed gPAC | Improvement |
|------------|---------------|-----------|-------------|
| 0.10 | 1.0000 | 1.0000 | +0.0000 |
| 0.20 | 1.0000 | 1.0000 | +0.0000 |
| 0.30 | 1.0000 | 1.0000 | +0.0000 |

## Conclusion

Trainable gPAC demonstrates equivalent performance to fixed gPAC in both phase frequency classification (0.0% difference) and amplitude frequency classification (0.0% difference).

For this synthetic dataset, both models were able to achieve perfect accuracy across all noise levels. In real-world scenarios with more complex signals and higher noise levels, the trainable parameters might provide advantages by allowing the model to adapt to specific signal characteristics.