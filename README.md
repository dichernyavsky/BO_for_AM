# Laser Powder Bed Fusion of Defect-Free AA2024 using Bayesian Machine Learning

This repository provides the implementation of the methods described in our paper [1].

---

## Repository Structure

### Single-Objective Optimization

The `single-objective` folder contains the single-objective Bayesian optimization approach:

- **`single_objective_BO_CT.ipynb`**: Jupyter notebook demonstrating the single-objective Bayesian optimization workflow.
- **`gauss_custom.py`**: Custom implementation of heteroscedastic Gaussian Process regression, extending scikit-learn's Gaussian Process Regression (GPR).

### Bi-Objective Optimization

The `bi-objective` folder contains the bi-objective Bayesian optimization approach:

- **`bi-objective.ipynb`**: Jupyter notebook demonstrating the bi-objective optimization workflow.
- **`data_handler.py`**: Utilities for data preprocessing, scaling, and management.
- **`gp_module.py`**: Gaussian Process models for predicting density and build-up rate.
- **`optimizer.py`**: Bi-objective Bayesian optimization using Expected Hypervolume Improvement.
- **`utils.py`**: Visualization and data processing helper functions.

---

## Requirements

This implementation uses the following Python packages:

- PyTorch
- GPyTorch
- BoTorch
- NumPy
- Pandas
- Matplotlib
- Plotly

---

## Citation

If you use this code in your research, please cite our paper:

```bibtex
[1] Dmitry Chernyavsky, Denys Y. Kononenko, Julia Kristin Hufenbach, Jeroen van den Brink, and Konrad Kosiba.  
Laser Powder Bed Fusion of defect-free AA2024 using Bayesian Machine Learning. *(to be published)*.

