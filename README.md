# NeoBayesian

This package contains 4 main Bayesian models and multiple routines to facilitate problem solving to anybody starting with Bayesian Inference. Functions docstrings contain descriptions and formulas. Also, multiple functions print out useful information to better understand steps taken before final results.

## Contents

### 1) NeoBayesian - Tools package:

- **variance:** module contains 2 functions to compute variance (within group variance and between group variance) using the *conditional variance formula* and the direct method.
- **others:** module contains 2 functions. 1 to calculate the probability of an event by computing all possible outcomes and 1 to calculate the Positive Predictive Value and Negative Predictive Value of a test.
- **routines:** 3 functions to apply the Bayesian algorithm. 

### 2) NeoBayesian - Models package:

- **discrete:** basic discrete models with “pdf” and “cdf” modes.
- **naive:** module to apply naive Bayes to a data set as a CSV file.
- **continuous:** package contains 4 main continuous Bayesian models with functions to estimate initial and posterior parameters. Models include:
  * Beta-binomial model
  * Gamma-Poisson model
  * Normal-inverse-gamma model
  * Normal-normal model
  
  ### 2) Notebooks:
  
  Jupyter Notebooks with multiple examples. 



