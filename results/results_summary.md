# MPS-SSM Multivariate Prediction Results Summary

## Overview

Total experiments analyzed: 24

### Experiments by Dataset Type

- ETT: 16 experiments

- Traffic: 4 experiments

- Weather: 4 experiments



## Performance Summary by Dataset Type

| Dataset_Type   |   Num_Experiments |   Avg_MSE |   Avg_MAE | Avg_Impulse_Degrad   | Avg_Spurious_Degrad   | Avg_Robustness   |
|:---------------|------------------:|----------:|----------:|:---------------------|:----------------------|:-----------------|
| ETT            |                16 |    0.5603 |    0.5364 | 128.26%              | 99.55%                | 113.90%          |
| Traffic        |                 4 |    0.6718 |    0.386  | -0.00%               | 0.01%                 | 0.00%            |
| Weather        |                 4 |    0.3096 |    0.3608 | 232.43%              | 206.50%               | 219.47%          |


## Detailed Results by Dataset and Prediction Length

| Dataset   | Type    |   Pred_Len |   Lambda |   Test_MSE |   Test_MAE |   Impulse_MSE | Impulse_Degrad   |   Spurious_MSE | Spurious_Degrad   |
|:----------|:--------|-----------:|---------:|-----------:|-----------:|--------------:|:-----------------|---------------:|:------------------|
| ETTh1     | ETT     |         96 |   100    |     0.725  |     0.6077 |        0.8253 | 13.83%           |         1.1928 | 64.52%            |
| ETTh1     | ETT     |        192 |     0.5  |     0.7049 |     0.6156 |        0.7392 | 4.86%            |         0.9317 | 32.17%            |
| ETTh1     | ETT     |        336 |     0.05 |     0.7785 |     0.6421 |        0.8507 | 9.28%            |         1.1477 | 47.43%            |
| ETTh1     | ETT     |        720 |     0.01 |     0.8084 |     0.6768 |        0.918  | 13.56%           |         1.0886 | 34.67%            |
| ETTh2     | ETT     |         96 |    50    |     0.2848 |     0.3933 |        0.5605 | 96.78%           |         0.3187 | 11.89%            |
| ETTh2     | ETT     |        192 |     0.01 |     0.5901 |     0.5294 |        1.5099 | 155.88%          |         1.23   | 108.45%           |
| ETTh2     | ETT     |        336 |   100    |     0.4039 |     0.4741 |        0.8551 | 111.69%          |         0.5276 | 30.62%            |
| ETTh2     | ETT     |        720 |    50    |     0.5663 |     0.5661 |        0.7343 | 29.66%           |         0.9828 | 73.53%            |
| ETTm1     | ETT     |         96 |     1    |     0.5388 |     0.5138 |        0.7423 | 37.78%           |         0.7705 | 43.02%            |
| ETTm1     | ETT     |        192 |     1    |     0.6171 |     0.5489 |        0.7991 | 29.48%           |         1.0646 | 72.51%            |
| ETTm1     | ETT     |        336 |     0.01 |     0.707  |     0.5983 |        1.4372 | 103.28%          |         1.103  | 56.02%            |
| ETTm1     | ETT     |        720 |     5    |     0.7673 |     0.6396 |        1.073  | 39.85%           |         1.2658 | 64.97%            |
| ETTm2     | ETT     |         96 |     5    |     0.2138 |     0.3268 |        2.3019 | 976.52%          |         0.9927 | 364.25%           |
| ETTm2     | ETT     |        192 |     5    |     0.3734 |     0.4638 |        0.8411 | 125.26%          |         0.6916 | 85.22%            |
| ETTm2     | ETT     |        336 |   100    |     0.3785 |     0.4603 |        0.8434 | 122.81%          |         1.0753 | 184.09%           |
| ETTm2     | ETT     |        720 |     0.01 |     0.5066 |     0.5261 |        1.4266 | 181.62%          |         2.1247 | 319.43%           |
| traffic   | Traffic |         96 |     2    |     0.6366 |     0.3742 |        0.6366 | 0.00%            |         0.6365 | -0.00%            |
| traffic   | Traffic |        192 |     5    |     0.6605 |     0.3801 |        0.6605 | -0.00%           |         0.6605 | 0.00%             |
| traffic   | Traffic |        336 |     2    |     0.6841 |     0.3896 |        0.6841 | -0.00%           |         0.6841 | 0.01%             |
| traffic   | Traffic |        720 |     2    |     0.7059 |     0.4    |        0.7059 | 0.00%            |         0.7061 | 0.02%             |
| weather   | Weather |         96 |   100    |     0.2285 |     0.3036 |        1.287  | 463.24%          |         1.1864 | 419.21%           |
| weather   | Weather |        192 |   100    |     0.2927 |     0.3563 |        0.8482 | 189.77%          |         0.8219 | 180.77%           |
| weather   | Weather |        336 |    50    |     0.3189 |     0.3663 |        0.7533 | 136.22%          |         0.5204 | 63.19%            |
| weather   | Weather |        720 |    50    |     0.3983 |     0.417  |        0.9579 | 140.51%          |         1.0467 | 162.82%           |


## Average Performance Across Prediction Lengths

| Dataset   | Type    |   Avg_MSE |   Avg_MAE | Avg_Impulse_Degrad   | Avg_Spurious_Degrad   | Avg_Total_Degrad   |
|:----------|:--------|----------:|----------:|:---------------------|:----------------------|:-------------------|
| ETTh1     | ETT     |    0.7542 |    0.6355 | 10.38%               | 44.70%                | 27.54%             |
| ETTh2     | ETT     |    0.4613 |    0.4907 | 98.50%               | 56.12%                | 77.31%             |
| ETTm1     | ETT     |    0.6575 |    0.5752 | 52.60%               | 59.13%                | 55.86%             |
| ETTm2     | ETT     |    0.3681 |    0.4443 | 351.55%              | 238.25%               | 294.90%            |
| traffic   | Traffic |    0.6718 |    0.386  | -0.00%               | 0.01%                 | 0.00%              |
| weather   | Weather |    0.3096 |    0.3608 | 232.43%              | 206.50%               | 219.47%            |


## Optimal Lambda Values

| Dataset   |   96 |    192 |    336 |   720 |
|:----------|-----:|-------:|-------:|------:|
| ETTh1     |  100 |   0.5  |   0.05 |  0.01 |
| ETTh2     |   50 |   0.01 | 100    | 50    |
| ETTm1     |    1 |   1    |   0.01 |  5    |
| ETTm2     |    5 |   5    | 100    |  0.01 |
| traffic   |    2 |   5    |   2    |  2    |
| weather   |  100 | 100    |  50    | 50    |


## Key Findings

1. **Overall Robustness Performance**:
   - Average impulse noise degradation: 124.24%
   - Average spurious correlation degradation: 100.78%
   - Overall robustness score: 112.51%

2. **Performance by Dataset Type**:
   - ETT: 113.90% average degradation
   - Traffic: 0.00% average degradation
   - Weather: 219.47% average degradation

3. **Best and Worst Performers**:
   - Most robust: traffic (0.00% degradation)
   - Least robust: ETTm2 (294.90% degradation)

4. **Hyperparameter Insights**:
   - Lambda range: [0.01, 100.0]
   - Most common lambda: 100.0

5. **Dataset-Specific Insights**:
   - ETT: Average MSE = 0.5603
   - Weather: Average MSE = 0.3096
   - Traffic: Average MSE = 0.6718