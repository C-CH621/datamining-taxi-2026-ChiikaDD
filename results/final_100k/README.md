# 100k Data Experiment Results

All models retrained with `--nrows 100000` on the FHVHV taxi fare prediction dataset.

## Data
- Training: 100,000 rows from `data/processed/fhvhv_tripdata_2026-03_prediction_format/train.csv`
- Test: 10,000 rows from `test.csv`
- Evaluation: against `sample_submission.csv` (9,996 valid rows after filtering NaN/Inf)

## Comprehensive Results (sorted by RMSE)

| Rank | Model | RMSE | MAE | R² | vs Best Baseline |
|------|-------|------|-----|-----|------------------|
| 1 | **Core Model - LGB+RF Blend** | **1.9625** | 1.1169 | 0.9942 | **+5.6%** |
| 2 | Core Model - LGB Optimized | 2.0229 | 1.1378 | 0.9938 | +2.6% |
| 3 | Baseline - Random Forest | 2.0778 | 1.1671 | 0.9935 | baseline |
| 4 | Baseline - LightGBM | 2.0798 | 1.1596 | 0.9935 | -0.1% |
| 5 | Core Model - Original (100k) | 2.5068 | 1.1586 | 0.9905 | -20.6% |
| 6 | Baseline - Simple Linear | 2.6077 | 1.5344 | 0.9897 | -25.5% |
| 7 | Baseline - XGBoost | 3.9751 | 1.3879 | 0.9762 | -91.3% |

## Optimization Process

The core model was optimized through 4 rounds of experiments:

1. **Round 1** (8 configs): Found optimal hyperparameters (lr=0.04, num_leaves=63, max_depth=7, etc.) achieving RMSE=2.031
2. **Round 2** (8 configs): Tested enhanced features (20k+ cols) - found they cause overfitting with 100k data; baseline features (61 cols) work best
3. **Round 3** (5 configs + ensembles): Multi-seed training, XGBoost tuning, ensemble methods. Best single model: RMSE=2.023
4. **Round 4** (blending): LGB + RF blending. Best result: 50/50 LGB+RF blend achieving RMSE=1.963

## Best Model Configuration

### Single Model (LGB Optimized)
- Learning rate: 0.04
- Num leaves: 63
- Max depth: 7
- Subsample: 0.85, Colsample bytree: 0.85
- Reg alpha: 0.5, Reg lambda: 0.2
- Min child samples: 20
- Num boost rounds: 160 (full training, no validation split)

### Blended Model (LGB + RF)
- 50% LGB (above config) + 50% Random Forest (n_estimators=100, max_depth=20)

## Files
- `submission_core_model_optimized.csv` - Best single model predictions
- `submission_core_model_blended.csv` - Best blended model predictions
- `comprehensive_comparison.json` - Full metrics for all models
