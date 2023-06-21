# IOLS Method for Log-Linear Regression

## Overview

The IOLS (Iteratively Optimized Least Squares) method is a powerful approach for implementing log-linear regression models in datasets containing zero values in the target variable. This method provides an effective solution to handle such scenarios by incorporating suitable adjustments to the traditional log-linear regression framework.

This project, IOLSMP, is an open-source Python package that implements the IOLS method for log-linear regression analysis in datasets with zero-inflated target variables.

## Features

- Implements the IOLS method for log-linear regression analysis.
- Accommodates datasets with zero values in the target variable.
- Handles zero-inflated target variables by applying appropriate adjustments.
- Provides accurate predictions and model fitting for zero-inflated datasets.
- Supports customizable model configurations and parameters.

## Installation

You can install the package using pip

```shell

pip install iolsmp

```
``` python 
from iolsmp import regression

# Load your dataset
data = ...

# Preprocess your data
preprocessed_data = ...
X = ...
Y = ...

# Apply the IOLS method
optimal_coeffs, clustered_standard_errors = regession.linear_regression(X, Y,method = "iOLS",se = "clustered", cluster = cluster_variable)


# Additional functionalities
# ...

```
For more detailed examples and usage instructions, please refer to the documentation.
# Licence
This project is under the MIT Licence. For more details, please refer to the [LICENSE](LICENSE) file.
# Contributing
you are welcome! If you want to contribute to this project, please follow these steps:

1 - Fork the repository.
2 - Create a new branch for your feature or bug fix.
3 - Make your changes and commit them with descriptive messages.
4 - Push your changes to your forked repository.
5 - Submit a pull request detailing your changes.
Please ensure that you adhere to the project's code style and guidelines.

# Acknowledgements
We would like to express our sincere gratitude to Louis Pape for his invaluable guidance and supervision throughout the development of this project. His paper, "Dealing with Logs and Zeros," served as a significant reference.