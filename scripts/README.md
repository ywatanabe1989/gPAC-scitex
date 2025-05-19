<!-- ---
!-- Timestamp: 2025-05-14 03:51:06
!-- Author: ywatanabe
!-- File: /ssh:sp:/home/ywatanabe/proj/gPAC/scripts/README.md
!-- --- -->

The experiment structure looks well-organized, following a logical sequence:

1. exp_01_synthetic_data_generation
   - Create a data with 5 classes with different phase-amplitude coupling properties
   - Make them as Pytorch Dataset to use across experiments

2. exp_02_calculation_speed_comparison

3. exp_03_pac_value_similarity
   - Focuses on validating algorithm accuracy
   - Compares similarity of PAC values across implementations

4. exp_03_benchmarks
   - RAM, VRAM, CPU Usage, GPU Usage under various parameter settings

5. exp_05_trainability_of_bandpass_filters
   - Tests optimization capabilities of the filters
   - Shows adaptability of the method

## Note
- Keep comparison fair
  - Consider initialization time
  - Calculation should be done in random order to reduce the effects of order

## Experimental Settings
- 80GB VRAM GPU (* 4 later once code valid)

<!-- EOF -->