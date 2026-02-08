# PyTorch Linear Regression

## Overview
This implementation uses PyTorch to build a linear regression model predicting used car prices. It demonstrates PyTorch's autograd system compared to manual gradient computation in No-Framework.

## Key PyTorch Concepts Demonstrated
- **Tensors**: GPU-compatible arrays with automatic gradient tracking
- **nn.Linear**: Encapsulates weights and bias in a single layer
- **nn.MSELoss**: Pre-built loss function
- **optim.SGD**: Optimizer handling parameter updates
- **autograd**: Automatic differentiation via `.backward()`

## Results

### Model Performance
| Metric | Value |
|--------|-------|
| Test RMSE | $10,105 |
| Test R² | 0.4986 |

### Framework Comparison
| Metric | No-Framework | Scikit-Learn | PyTorch |
|--------|--------------|--------------|---------|
| Training Time | 0.3799 sec | 0.0258 sec | 3.4400 sec |
| Peak Memory | 1.96 MB | 14.76 MB | 54.18 MB |
| Test RMSE | $10,058 | $10,105 | $10,105 |
| Test R² | 0.4983 | 0.4986 | 0.4986 |

### Key Insights
- **PyTorch is slower** for simple linear regression due to autograd overhead
- **PyTorch uses more memory** for computational graph storage
- **Same accuracy** confirms fair apples-to-apples comparison
- **PyTorch shines** on complex neural networks, not simple linear regression

## Files
- `pipeline.ipynb` - Main implementation notebook
- `results/` - Saved visualizations
  - `cost_curve.png`
  - `predictions_vs_actual.png`
  - `feature_importance.png`

## Usage
```bash
pip install -r requirements.txt
jupyter notebook pipeline.ipynb
```
