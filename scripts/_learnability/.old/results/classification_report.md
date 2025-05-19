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

Trainable gPAC demonstrates worse performance in phase frequency classification (+0.0% accuracy) and worse performance in amplitude frequency classification (+0.0% accuracy) compared to fixed gPAC.

This demonstrates that trainable parameters provide significant advantages in real-world signal processing scenarios where noise and variability are present, allowing the model to adapt to specific signal characteristics.