# Selecting piecewise linear fit

This Python script performs regression analysis of a data set that is expected of having a piecewise linear structure. The data are fit to five piecewise linear models and produce five measures to indicate quality of the fit. The goal is to provide the user with information regarding how well the data fit with each model to aid in model selection.

Features:

* Automatic segmentation
* Fit to five piecewise models
  * single linear line
  * two-segment linear line
  * three-segment linear line
  * interpolation+two-segment linear line (3 segments in total)
  * interpolation+single linear line (2 segments in total)
* Report slope, intercept and their corresponding uncertainty for each segment.
* Report adjusted R-squared, adjusted AIC, F-value and Mallows' Cp for each model.
* Produce fitted plots with data for visual inspection

Potential pitfalls:

* Except for the three-segment linear fit, segmentation algorithm is greedy and has low tolerant to noise in the data. Smarter algorithm may be desirable depending on the data.

## Copyright and license

This is a free software made available under the [GNU GENERAL PUBLIC LICENSE](http://www.gnu.org/licenses/gpl-3.0.html), which means you can share, modify, and redistribute this software. While we endeavor to make this software as useful and as error-free as possible, we cannot make any such guarantee, and the software is hence released **without any warranty**.

## Installation

The script uses piecewise regression library `pwlf.py' by Charles Jekel for 3-segment regression. To install this library, follow the instruction on [Charles's blog] (https://jekel.me/2017/Fit-a-piecewise-linear-function-to-data/) and copy the 'pwlf.py' file to the same folder as `piecewise_analysis.py'.

## Usage

Input assumption in this version:

* y data is assumed to be a column of double-precision floating-point 
    numerics.
* x data is generated within the code.

## References

1. C. Jekel, “Fitting a piecewise linear function to data,” https://jekel.me/2017/Fit-a-piecewise-linear-function-to-data/ (2017).

2. S. Chatterjee and J. S. Simonoff, Handbook of Regression Analysis (Wiley, Hoboken, NJ, 2013)

3. S. Chatterjee and A. S. Hadi, Regression  Analysis  by  Example,  Wiley  Series  in  Probablity and Statistics (Wiley, Hoboken, NJ, 2012)

4. D. C. Montgomery, E. A. Peck,  and G. G. Vining, Introduction to Linear Regression Analysis, Wiley Series in Probability and Statistics (Wiley,
Hoboken, NJ, 2012) 
