# Fix for RidgeClassifierCV's store_cv_values parameter

## Issue
The `RidgeClassifierCV` class in scikit-learn was missing the `store_cv_values` parameter in its constructor, despite documentation suggesting that this parameter was available. When users attempted to use the parameter, they received an error:

```python
TypeError: __init__() got an unexpected keyword argument 'store_cv_values'
```

However, the documentation mentioned the `cv_values_` attribute that would be created when `store_cv_values=True`:

> cv_values_ : array, shape = [n_samples, n_alphas] or shape = [n_samples, n_responses, n_alphas], optional
> Cross-validation values for each alpha (if `store_cv_values=True` and `cv=None`).

## Root Cause
The issue was in the class hierarchy and parameter passing:

1. `_BaseRidgeCV` defined the `store_cv_values` parameter in its constructor
2. `RidgeCV` properly inherited from `_BaseRidgeCV` and passed all parameters 
3. `RidgeClassifierCV` also inherited from `_BaseRidgeCV` but did not include the `store_cv_values` parameter in its constructor or pass it to the parent class

## Fix Implemented
The fix involves two changes to the `RidgeClassifierCV` class:

1. Added the `store_cv_values` parameter with a default value of `False` to the constructor signature:
```python
def __init__(self, alphas=(0.1, 1.0, 10.0), fit_intercept=True,
             normalize=False, scoring=None, cv=None, class_weight=None,
             store_cv_values=False):
```

2. Updated the call to the parent class constructor to pass this parameter:
```python
super(RidgeClassifierCV, self).__init__(
    alphas=alphas, fit_intercept=fit_intercept, normalize=normalize,
    scoring=scoring, cv=cv, store_cv_values=store_cv_values)
```

3. Added documentation for the `store_cv_values` parameter to match the documentation in `RidgeCV`

## Testing
A unit test was added to verify that:
1. `RidgeClassifierCV` accepts the `store_cv_values` parameter
2. When `store_cv_values=True`, the `cv_values_` attribute is present with the expected shape
3. When `store_cv_values=False`, the `cv_values_` attribute is not present

The test passes, confirming that the fix works correctly.

## Example Usage
The following code now works correctly:

```python
import numpy as np
from sklearn import linear_model as lm

# test database
n = 100
x = np.random.randn(n, 30)
y = np.random.normal(size=n)
y = (y > 0).astype(int)  # Convert to binary for classification

rr = lm.RidgeClassifierCV(alphas=np.arange(0.1, 1000, 0.1), normalize=True,
                         store_cv_values=True).fit(x, y)

# Access the cross-validation values
print(rr.cv_values_.shape)
```