import numpy as np
from sklearn.linear_model import RidgeClassifierCV

def test_ridge_classifier_cv_store_cv_values():
    """
    Test that RidgeClassifierCV works with store_cv_values parameter.
    
    This test verifies that the store_cv_values parameter is properly 
    supported by RidgeClassifierCV and that it stores cross-validation
    values as expected when the parameter is set to True.
    """
    # Generate sample data for classification
    n_samples, n_features = 10, 5
    np.random.seed(0)
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, size=n_samples)  # Binary classification
    
    # Create classifier with store_cv_values=True
    alphas = np.logspace(-3, 3, 10)
    clf = RidgeClassifierCV(alphas=alphas, store_cv_values=True, cv=None)
    clf.fit(X, y)
    
    # Check that cv_values_ exists
    assert hasattr(clf, 'cv_values_'), "cv_values_ attribute not present"
    
    # Check cv_values_ shape: should be [n_samples, n_alphas] or
    # [n_samples, n_responses, n_alphas]
    if len(clf.cv_values_.shape) == 2:
        assert clf.cv_values_.shape == (n_samples, len(alphas)), \
            f"Expected shape {(n_samples, len(alphas))}, got {clf.cv_values_.shape}"
    else:
        # For binary classification, there's typically 1 response
        assert clf.cv_values_.shape[0] == n_samples, \
            f"First dimension should be n_samples={n_samples}"
        assert clf.cv_values_.shape[2] == len(alphas), \
            f"Third dimension should be n_alphas={len(alphas)}"
    
    # Now create a classifier without store_cv_values
    clf_no_cv = RidgeClassifierCV(alphas=alphas, store_cv_values=False, cv=None)
    clf_no_cv.fit(X, y)
    
    # Check that cv_values_ is not present
    assert not hasattr(clf_no_cv, 'cv_values_'), "cv_values_ should not be present"

if __name__ == "__main__":
    test_ridge_classifier_cv_store_cv_values()
    print("Test passed!")