# Notebooks

This directory contains Jupyter notebooks for exploratory data analysis and visualization.

## Notebooks

- `01_eda_and_baselines.ipynb`: Initial EDA and baseline model exploration

## Usage

```bash
# Launch jupyter
jupyter notebook notebooks/01_eda_and_baselines.ipynb

# Or with voila for standalone HTML
voila notebooks/01_eda_and_baselines.ipynb
```

## Notes

- Notebooks are for EDA and visualization only
- All production logic is in the `src/` directory
- Notebooks can use features from `src/` via `sys.path.insert(0, "../src")`