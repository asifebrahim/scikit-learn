"""
Test to verify the fix for RidgeClassifierCV's store_cv_values parameter.
This reproduces the example from the issue description.
"""
import numpy as np
from sklearn import linear_model as lm

# Test database
n = 100
x = np.random.randn(n, 30)
y = np.random.normal(size=n)
y = (y > 0).astype(int)  # Convert to binary for classification

try:
    rr = lm.RidgeClassifierCV(alphas=np.arange(0.1, 1000, 0.1), 
                             normalize=True,
                             store_cv_values=True).fit(x, y)
    print('SUCCESS: RidgeClassifierCV accepted store_cv_values parameter')
    print('cv_values_ attribute exists:', hasattr(rr, 'cv_values_'))
    if hasattr(rr, 'cv_values_'):
        print('cv_values_ shape:', rr.cv_values_.shape)
except TypeError as e:
    print('FAILURE: Test produced an error:', str(e))