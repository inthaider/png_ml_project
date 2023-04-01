#
# Created by Jibran Haider.
#
r"""Validation/accuracy analysis functions for 1D ICA.

Routine Listings
----------------
calculate_all_metrics(true_field, extracted_field, round=None, is_print=True, norm=True, relative=True)
    Calculate and print all metrics in validate_1d.py for a pair of fields.
calculate_residuals_ica(x, y, norm=True, relative=True)
    Compute both scalar & vector residuals between $x$ & $y$, where $x$ is the source field and $y$ is the estimated field.
test_calculate_residuals_ica()
    Test the calculate_residuals_ica() function.
test_calc_resid_ica()
    Test the calculate_residuals_ica() function.
calculate_residuals(x, y, norm=True, relative=True)
    Compute both scalar & vector residuals between $x$ & $y$.
test_calculate_residuals()
    Test the calculate_residuals() function.
calculate_pearson_coefficient(src_field, est_field)
    Calculate the Pearson correlation coefficient between the source field and the estimated field.
biweight_midcorrelation(x, y)
    Calculate the biweight midcorrelation between the source field and the estimated field.
rescale_extracted_field(true_field, extracted_field)
    Rescale the extracted field to match the true field.

See Also
--------


Notes
-----

"""

import numpy as np
from scipy.stats import pearsonr

def calculate_all_metrics(true_field, extracted_field, round=None, is_print=True, norm=True, relative=True):
    r"""Calculate and print all metrics in validate_1d.py for a pair of fields.

    Parameters
    ----------
    true_field : np.ndarray
        The true field.
    extracted_field : np.ndarray
        The extracted field.
    round : int, optional
        The number of decimal places to round the metrics to. If None, no rounding is performed.
    print : bool, optional
        If True, print all metrics. If False, do not print any metrics.
    norm : bool, optional
        If True, normalize the fields before calculating the residuals. If False, do not normalize the fields.
    relative : bool, optional
        If True, calculate the relative residuals. If False, calculate the absolute residuals.

    Returns
    -------
    metrics : dict
        A dictionary containing all metrics for the two fields.
    """
    if true_field is None or extracted_field is None:
        raise ValueError("Invalid input field(s): Field must be a vector/array, not None.")
    if len(true_field) == 0 or len(extracted_field) == 0:
        raise ValueError("Invalid input field(s): Field must have a nonzero size/length.")
    
    def print_metrics(metrics):        
        # Print all metrics. 
        # The tuple from calculate_residuals() is unpacked into two variables and printed separately
        # Note that we could use f-string formatting syntax to round the values (e.g. f"{v:.2f}")
        # instead of using np.around() above.
        print("Metrics:")
        for metric_name, metric_value in metrics.items():
            # Skip the "Residual Vector" metric, since it can be very long
            if metric_name == "Residual Vector":
                continue
            else:
                print(f"    {metric_name}: {metric_value:}")

    # Calculate and store all metrics in a dictionary
    # Note that calculate_residuals() returns a tuple of two values
    res_scalar, res_vector = calculate_residuals(true_field, extracted_field, norm=norm, relative=relative)
    metrics = {
        # "Mean Absolute Error": calculate_mean_absolute_error(true_field, extracted_field),
        # "Root Mean Squared Error": calculate_root_mean_squared_error(true_field, extracted_field),
        # "Mean Absolute Percentage Error": calculate_mean_absolute_percentage_error(true_field, extracted_field),
        "Residual Scalar": res_scalar,
        "Residual Vector": res_vector,
        "Projection Residual (for ICA scaling)": calculate_residuals_ica(true_field, extracted_field, norm=norm, relative=relative),
        "Pearson Correlation Coefficient": calculate_pearson_coefficient(true_field, extracted_field),
        "Biweight Midcorrelation": biweight_midcorrelation(true_field, extracted_field),
    }

    if round is not None:
        # Round all values in the dictionary to the specified number of decimal places
        metrics = {k: np.around(v, round) for k, v in metrics.items()}

        # # Account for the fact that calculate_residuals() returns a tuple of two values
        # # one of which is a scalar and the other of which is a ndarray
        # for metric_name, metric_value in metrics.items():
        #     if metric_name == "Residuals":
        #         metrics[metric_name] = (np.around(metric_value[0], round), np.around(metric_value[1], round))
        #     else:
        #         metrics[metric_name] = np.around(metric_value, round)

    if is_print:
        print_metrics(metrics)

    return metrics


############################################################
#
# ICA RESIDUALS 
#
############################################################
def calculate_residuals_ica(x, y, norm=True, relative=True):
    r"""Compute both scalar & vector residuals between $x$ & $y$, where $x$ is the source field and $y$ is the estimated field.

    Scalar residual $r$ is defined as:
        $r_s = 1 - \frac{\left\|a\right\|_2}{\left\|b\right\|_2}$
    Vector residual $r_v$ is defined as:
        $r_v = 1 - \frac{\left\|a\right\|_2}{\left\|b\right\|_2}$
    where $\left\|a\right\|_2$ is the 2-norm of $x$ and $\left\|b\right\|_2$ is the 2-norm of $y$.

    Parameters
    ----------
    x : np.ndarray
        True field $x$. Must be same shape as $y$.
    y : np.ndarray
        Extracted field $y$. Must be same shape as $x$.
        
    Returns
    -------
    rs : float
        Scalar residual of $x$ and $y$ (see below).
    rv : np.ndarray
        Vector residual of $x$ and $y$ (see below).

    Notes
    -----
        (1) Normalize vector y by mean-subtraction and standard deviation-division, then rescale it by the standard deviation of x. 
        This step ensures that y has the same mean and scale as x, which is essential for a fair comparison.
    
        (2) The vector residual (rv) is calculated by projecting the rescaled y vector onto the x vector, then dividing the result by the magnitude of the x vector. 
        This measures how well y aligns with x, considering both direction and magnitude.
        The vector residual ($rv$) is calculated as:
            rv = 1 - ( ( b.a / a.a ) * a ) / |a|

        (3) The scalar residual (rs) is calculated by finding the magnitude of the projection of y onto x, dividing it by the square of the magnitude of x, and subtracting the result from 1. 
        This measures the absolute difference between the magnitudes of: x AND the projection of y onto x.
        The scalar residual ($rr$) is calculated as:
            rs = 1 - | ( b.a / |a| ) / |a| |


        Note that the magnitude of x is:
            |x| = (x.x)^{1/2}
        where $x$ is a vector.

    TODO
    ----
    FILL IN WHY IT MAKES SENSE TO CHOOSE TO COMPUTE $RV$ & $RR$ THIS WAY!
    """
    if x is None or y is None:
        raise ValueError("Invalid input field(s): Field must be a vector/array, not None.")
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Invalid input field(s): Field must have a nonzero size/length.")

    if norm:
        # Normalize y by std-division and rescale by x's std
        y = rescale_extracted_field(x, y)
        
        # print("y_normd: ", y)

    # Calculate dot products and scalar projection
    ydotx = np.dot(y, x)
    xdotx = np.dot(x, x)
    mag_x = np.sqrt(xdotx)      # magnitude of x: |x| = (x.x)^{1/2}
    scalar_proj_xy = ydotx/mag_x
    
    # print("ydotx: ", ydotx)
    # print("xdotx: ", xdotx)
    # print("proj: ", scalar_proj_xy)
    # print("proj/|x|: ", scalar_proj_xy/mag_x)

    # Calculate scalar residuals
    if not relative:
        # Calculate absolute scalar residual
        rs = 1 - scalar_proj_xy
        
        # print("rs (1 - proj): ", rs)
    else:
        # Calculate relative scalar residual
        rs_test = 1 - (scalar_proj_xy / mag_x)        # rs = 1 - (b.a / |a|) / |a|
        rs = 1 - np.abs(scalar_proj_xy) / mag_x        # rs = 1 - |(b.a / |a|) / |a||; absolute value to ensure positive scalar projection

        # print("rs, not absolute ( 1 - proj/|x| ): ", rs_test)
        # print("rs ( 1 - |(proj/|x|)| ): ", rs)

    print("rs_ica: ", rs)

    return np.abs(rs)       # return absolute value of scalar residual since we want the relative scalar residual to be positive

##################
# TEST FUNCTIONS #
##################
def test_calculate_residuals_ica(tol=0.4, norm=True, relative=True):
    r"""Test calculate_residuals_ica() function.
    
    This test function checks three different cases:

    Identical vectors: The scalar residual should be 1, and the vector residual should be equal to the original vector a.
    
    Opposite vectors: The scalar residual should be -1, and the vector residual should be a zero vector.
    
    Orthogonal vectors: The scalar residual should be 0, and the vector residual should be a zero vector.
    
    You can run this test function to check if the calculate_residuals function works correctly. If all test cases pass, you will see the message "All test cases passed!" printed in the console.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    This function uses the assert keyword to check if the test cases pass. If the test cases fail, an AssertionError will be raised and the test function will terminate.

        np.allclose is a NumPy function that checks if all elements in two arrays are element-wise equal within a given tolerance. It returns True if all elements in the arrays are close, and False otherwise. The default tolerance is 1e-8.
    """
    # Helper function to assert that two numbers are close to each other
    def assert_close(a, b, tol=1e-6):
        assert np.abs(a - b) < tol, f"Expected {a} to be close to {b}"
    
    # Test case 1: Identical vectors, scalar residual should be 0, vector residual should be equal to the zero vector
    print("Test case 1: Identical vectors...")
    a1 = np.array([1., 2., 3.])
    b1 = np.array([1., 2., 3.])
    print("a1: ", a1); print("b1: ", b1)
    expected_rs1 = 0.0
    rs1 = calculate_residuals_ica(a1, b1, norm=norm, relative=relative)
    assert_close(rs1, expected_rs1)
    print("Test case 1 passed!")

    # Test case 2: Opposite vectors, scalar residual should be 1, vector residual should be a zero vector
    print("\nTest case 2: Opposite vectors...")
    a2 = np.array([1., 2., 3.])
    b2 = -a2
    print("a2: ", a2); print("b2: ", b2)
    expected_rs2 = 0.0
    rs2 = calculate_residuals_ica(a2, b2, norm=norm, relative=relative)
    assert_close(rs2, expected_rs2)
    print("Test case 2 passed!")

    # Test case 3: Orthogonal vectors, scalar residual should be 0, vector residual should be a zero vector
    print("\nTest case 3: Orthogonal vectors...")
    a3 = np.array([1., 0., 0.])
    b3 = np.array([0., 1., 0.])
    print("a3: ", a3); print("b3: ", b3)
    expected_rs3 = 1.0
    rs3 = calculate_residuals_ica(a3, b3, norm=norm, relative=relative)
    assert_close(rs3, expected_rs3)
    print("Test case 3 passed!")

    # Test case 4: Similar vectors, scalar residual should be ??, vector residual should be ??
    print("\nTest case 4: Similar vectors...")
    a4 = np.array([1., 2., 3.])
    b4 = np.array([1.1, 1.9, 3.2])
    print("a4: ", a4); print("b4: ", b4)
    expected_rs4 = 0.0
    rs4 = calculate_residuals_ica(a4, b4, norm=norm, relative=relative)
    tol = tol
    assert_close(rs4, expected_rs4, tol=tol)
    print("Test case 4 passed!")

    # Test case 5: Very different yet not orthogonal vectors, scalar residual should be ??, vector residual should be ??
    print("\nTest case 5: Very different yet not orthogonal vectors...")
    a5 = np.array([1., 2., 3.])
    b5 = np.array([350., 100., 150.])
    print("a5: ", a5); print("b5: ", b5)
    expected_rs5 = 0.0
    rs5 = calculate_residuals_ica(a5, b5, norm=norm, relative=relative)
    tol = 1-tol
    assert_close(rs5, expected_rs5, tol=tol)
    print("Test case 5 passed!")

    # print("\nAll test cases passed!")

def test_calc_resid_ica():
    """
    Test the calc_resid function with different cases of input vectors a and b.
    """
    def calc_resid(a, b):
        ab_projection = np.dot(b, a) / np.dot(a, a) * a
        return 1 - np.abs(np.dot(ab_projection, a)) / (np.linalg.norm(a) * np.linalg.norm(a))

    # Case 1: Perfect match
    a1 = np.array([1, 2, 3])
    b1 = np.array([1, 2, 3])
    resid1 = calc_resid(a1, b1)
    print("Case 1 (Perfect match):", resid1)

    # Case 2: Same amplitude, opposite direction
    a2 = np.array([1, 2, 3])
    b2 = np.array([-1, -2, -3])
    resid2 = calc_resid(a2, b2)
    print("Case 2 (Same amplitude, opposite direction):", resid2)

    # Case 3: Different amplitudes
    a3 = np.array([1, 2, 3])
    b3 = np.array([2, 4, 6])
    resid3 = calc_resid(a3, b3)
    print("Case 3 (Different amplitudes):", resid3)

    # Case 4: Orthogonal vectors
    a4 = np.array([1, 0, 0])
    b4 = np.array([0, 1, 0])
    resid4 = calc_resid(a4, b4)
    print("Case 4 (Orthogonal vectors):", resid4)





############################################################
#
# RESIDUAL CALCULATION
#
############################################################
def calculate_residuals(x, y, norm=True, relative=True):
    r"""Compute both scalar & vector residuals between $x$ & $y$, where $x$ is the source field and $y$ is the estimated field.

    The residual (b - <b\cdot a><a\cdot a>^-1 a) or its magnitude, an associated quadratic. Here a is the input nonG and b is the output nonG, and if it was perfect separation this would be zero.

    Parameters
    ----------
    x : np.ndarray
        True field $x$. Must be same shape as $y$.
    y : np.ndarray
        Extracted field $y$. Must be same shape as $x$.
        
    Returns
    -------
    rs : float
        Scalar residual of $x$ and $y$ (see below).
    rv : np.ndarray
        Vector residual of $x$ and $y$ (see below).

    Notes
    -----
    """
    if x is None or y is None:
        raise ValueError("Invalid input field(s): Field must be a vector/array, not None.")
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Invalid input field(s): Field must have a nonzero size/length.")

    if norm:
        # Normalize y by mean-subtraction and std-division and rescale by x's std
        y = rescale_extracted_field(x, y)
        
        # print("y_normd: ", y)

    # Calculate dot products and vector projection
    ydotx = np.dot(y, x)
    xdotx = np.dot(x, x)
    mag_x = np.sqrt(xdotx)      # magnitude of x: |x| = (x.x)^{1/2}
    vector_proj_xy = (ydotx / mag_x**2) * x     # (( y.x / |x|^2 ) * x) gives the projection of y onto x; |x|^2 is equivalent to x.x

    # print("ydotx: ", ydotx)
    # print("xdotx: ", xdotx)

    # Calculate residuals
    if not relative:        
        # Calculate absolute residuals
        rv = y - vector_proj_xy      # absolute vector residual; subtracting from y gives the vector pointing from the proj_{x}y (proj of y onto x) to y
        rs = np.sqrt(np.dot(rv, rv))        # absolute scalar residual; magnitude of vector residual
    else:       
        # Calculate relative residuals, where the vector projection is calculated in units of |x|
        rv = ( y - vector_proj_xy ) / mag_x     # relative vector residual, since we're dividing by |x|
        # rv = 1 - ( (ydotx / xdotx) * x ) / xmag         # different form of relative vector residual; here, rv = 1 - (( y.x / x.x ) * x ) / |x|; subtracting from 1 gives an inverted residual (i.e. higher values indicate a better match)
        rs = np.sqrt(np.dot(rv, rv))        # relative scalar residual (since rv is relative)
        
        # test_rv = y - vector_proj_xy
        # test_rs = np.sqrt(np.dot(test_rv, test_rv)) / mag_x
        # print("test_rs: ", test_rs)

    print("rs: ", rs)
    # print("rv: ", rv)

    return np.abs(rs), rv       # return absolute value of scalar residual since we want the scalar residual to be positive

#################
# TEST FUNCTION #
#################
def test_calculate_residuals(tol=0.4, norm=True, relative=True):
    r"""Test calculate_residuals() function.
    
    This test function checks three different cases:

    Identical vectors: The scalar residual should be 1, and the vector residual should be equal to the original vector a.
    
    Opposite vectors: The scalar residual should be -1, and the vector residual should be a zero vector.
    
    Orthogonal vectors: The scalar residual should be 0, and the vector residual should be a zero vector.
    
    You can run this test function to check if the calculate_residuals function works correctly. If all test cases pass, you will see the message "All test cases passed!" printed in the console.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    This function uses the assert keyword to check if the test cases pass. If the test cases fail, an AssertionError will be raised and the test function will terminate.

        np.allclose is a NumPy function that checks if all elements in two arrays are element-wise equal within a given tolerance. It returns True if all elements in the arrays are close, and False otherwise. The default absolute tolerance (atol) is 1e-8, and the default relative tolerance (rtol) is 1e-5.
    """
    # Helper function to assert that two numbers are close to each other
    def assert_close(a, b, tol=1e-6):
        assert np.abs(a - b) < tol, f"Expected {a} to be close to {b}"
    
    # Test case 1: Identical vectors, scalar residual should be 0, vector residual should be a zero vector
    print("Test case 1: Identical vectors...")
    a1 = np.array([1., 2., 3.])
    b1 = np.array([1., 2., 3.])
    print("a1: ", a1); print("b1: ", b1)
    expected_rs1 = 0.0
    expected_rv1 = np.zeros(3, dtype=float)
    rs1, rv1 = calculate_residuals(a1, b1, norm=norm, relative=relative)
    assert_close(rs1, expected_rs1)
    assert np.allclose(rv1, expected_rv1)
    print("Test case 1 passed!")

    # Test case 2: Opposite vectors, scalar residual should be 0, vector residual should be a zero vector
    print("\nTest case 2: Opposite vectors...")
    a2 = np.array([1., 2., 3.])
    b2 = -a2
    print("a2: ", a2); print("b2: ", b2)
    expected_rs2 = 0.0
    expected_rv2 = np.zeros(3, dtype=float)
    rs2, rv2 = calculate_residuals(a2, b2, norm=norm, relative=relative)
    assert_close(rs2, expected_rs2)
    assert np.allclose(rv2, expected_rv2)
    print("Test case 2 passed!")

    # Test case 3: Orthogonal vectors, scalar residual should be 1, vector residual should be equal to the vector b
    print("\nTest case 3: Orthogonal vectors...")
    a3 = np.array([1., 0., 0.])
    b3 = np.array([0., 1., 0.])
    print("a3: ", a3); print("b3: ", b3)
    expected_rs3 = 1.0
    expected_rv3 = b3
    rs3, rv3 = calculate_residuals(a3, b3, norm=norm, relative=relative)
    assert_close(rs3, expected_rs3)
    assert np.allclose(rv3, expected_rv3)
    print("Test case 3 passed!")

    # Test case 4: Similar vectors, scalar residual should be ??, vector residual should be ??
    print("\nTest case 4: Similar vectors...")
    a4 = np.array([1., 2., 3.])
    b4 = np.array([1.1, 1.9, 3.2])
    print("a4: ", a4); print("b4: ", b4)
    expected_rs4 = 0.0
    expected_rv4 = np.zeros(3)
    rs4, rv4 = calculate_residuals(a4, b4, norm=norm, relative=relative)
    tol = tol
    assert_close(rs4, expected_rs4, tol=tol)
    assert np.allclose(rv4, expected_rv4, atol=tol)
    print("Test case 4 passed!")

    # Test case 5: Very different yet not orthogonal vectors, scalar residual should be ??, vector residual should be ??
    print("\nTest case 5: Very different yet not orthogonal vectors...")
    a5 = np.array([1., 2., 3.])
    b5 = np.array([350., 100., 150.])
    print("a5: ", a5); print("b5: ", b5)
    expected_rs5 = 0.0
    expected_rv5 = np.zeros(3)
    rs5, rv5 = calculate_residuals(a5, b5, norm=norm, relative=relative)
    tol = 1-tol
    assert_close(rs5, expected_rs5, tol=tol)
    assert np.allclose(rv5, expected_rv5, atol=tol)
    print("Test case 5 passed!")

    print("\nAll test cases passed!")





############################################################
#
# OTHER VALIDATION METHODS
#
############################################################
def calculate_pearson_coefficient(x, y):
    r"""Calculate Pearson correlation coefficient between two fields, where $x$ is the source field and $y$ is the estimated field.

    The Pearson correlation coefficient is a measure of the linear correlation between two variables $X$ and $Y$.
    It has a value between -1 and 1, where 1 is total positive linear correlation, 0 is no linear correlation, and -1 is total negative linear correlation.
    The mathematical definition of the coefficient is:
        $r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2 \sum_{i=1}^{n} (y_i - \bar{y})^2}}$
    where $\bar{x}$ is the mean of the data in $X$ and $\bar{y}$ is the mean of the data in $Y$.
    The coefficient is also known as the Pearson product-moment correlation coefficient.
    
    Parameters
    ----------
    x : np.ndarray
        The true field.
    y : np.ndarray
        The extracted field.

    Returns
    -------
    correlation_coefficient : float

    Examples
    --------
        >>> true_field = np.random.randn(100)  # Replace with your true NG field
        >>> extracted_field = np.random.randn(100)  # Replace with your extracted NG field
        >>> correlation_coefficient = calculate_pearson_coefficient(true_field, extracted_field)
        >>> print("Pearson Correlation Coefficient:", correlation_coefficient)
    
    Notes
    -----
        Keep in mind that the Pearson Correlation Coefficient only measures the linear relationship between the two fields. If your problem requires assessing the performance of your component separation algorithm in recovering more complex relationships, you may need to consider alternative or additional metrics.
    """
    if x is None or y is None:
        raise ValueError("Invalid input field(s): Field must be a vector/array, not None.")
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Invalid input field(s): Field must have a nonzero size/length.")
    
    # Normalize y by mean-subtraction and std-division and rescale by x's std
    y = rescale_extracted_field(x, y)

    correlation_coefficient, _ = pearsonr(x, y)
    return correlation_coefficient

def biweight_midcorrelation(x, y):
    r"""Calculate the biweight midcorrelation between two fields, where $x$ is the source field and $y$ is the estimated field.
    
    Biweight midcorrelation is a robust correlation measur between two variables $X$ and $Y$ based on the biweight midvariance, which is a robust alternative to the classical variance. It calculates the correlation between two variables while down-weighting the contribution of outliers. The biweight midcorrelation ranges from -1 to 1, similar to the Pearson correlation coefficient, with 1 indicating a perfect positive relationship, -1 a perfect negative relationship, and 0 no relationship.
    The mathematical definition of the biweight midcorrelation is:
        $r_{bc} = \frac{\sum_{i=1}^{n} (x_i - \tilde{x})(y_i - \tilde{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \tilde{x})^2 \sum_{i=1}^{n} (y_i - \tilde{y})^2}}$

    Parameters
    ----------
    x : np.ndarray
        The true field.
    y : np.ndarray
        The extracted field.

    Returns
    -------
    correlation_coefficient : float
        The biweight midcorrelation between the two fields.

    Examples
    --------
        >>> x = np.array([1, 2, 3, 4, 5, 100])  # Original/reference field
        >>> y = np.array([1.2, 2.1, 2.9, 4.1, 5.2, 101])  # Extracted/estimated field
        
        >>> result = biweight_midcorrelation(x, y)
        >>> print(f"Biweight midcorrelation: {result:.2f}")

    Notes
    -----
        Keep in mind that this method assumes that your data is reasonably well-behaved (i.e., the majority of the data points are not outliers), as extreme cases with a large number of outliers can still affect the biweight midcorrelation. If you think your data might have a high percentage of outliers or other peculiarities, consider exploring other robust correlation methods as well.
    """
    if x is None or y is None:
        raise ValueError("Invalid input field(s): Field must be a vector/array, not None.")
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Invalid input field(s): Field must have a nonzero size/length.")
    
    # Normalize y by mean-subtraction and std-division and rescale by x's std
    y = rescale_extracted_field(x, y)
    
    # Calculate the medians of the input arrays
    # These are used as central values to compute the biweight midvariances
    median_x = np.median(x)
    median_y = np.median(y)

    # Calculate the scaling factors for each input array (x and y)
    # The scaling factor is based on the median absolute deviation (MAD)
    # The MAD is multiplied by 9 to obtain a more robust estimate of the spread
    scaling_factor_x = 9 * np.median(np.abs(x - median_x))
    scaling_factor_y = 9 * np.median(np.abs(y - median_y))

    # Compute the u values for each input array
    # u is the scaled distance of each data point from the median
    u_x = (x - median_x) / scaling_factor_x
    u_y = (y - median_y) / scaling_factor_y

    # Calculate the biweight midcorrelation
    # We only consider data points with |u| < 1 to reduce the impact of outliers
    mask = (np.abs(u_x) < 1) & (np.abs(u_y) < 1)
    
    # Calculate the numerator of the biweight midcorrelation formula
    # This is the sum of the product of the differences between each data point and the median
    # The product is weighted by (1 - u^2)^2 for both x and y to down-weight the contribution of outliers
    numerator = np.sum((x[mask] - median_x) * (y[mask] - median_y) * (1 - u_x[mask]**2)**2 * (1 - u_y[mask]**2)**2)
    
    # Calculate the denominator of the biweight midcorrelation formula
    # The denominator is the product of the sum of squared differences between each data point and the median
    # The sum is weighted by (1 - u^2)^2 for both x and y to down-weight the contribution of outliers
    denominator_x = np.sum((x[mask] - median_x)**2 * (1 - u_x[mask]**2)**2)
    denominator_y = np.sum((y[mask] - median_y)**2 * (1 - u_y[mask]**2)**2)

    # Calculate the biweight midcorrelation by dividing the numerator by the square root of the product of the denominators
    correlation = numerator / np.sqrt(denominator_x * denominator_y)
    return correlation







############################################################
#
# MISCELLANEOUS FUNCTIONS
#
############################################################
def rescale_extracted_field(true_field, extracted_field):
    """Rescale the extracted field to match the true field.

    Parameters
    ----------
    true_field : np.ndarray
        The true field.
    extracted_field : np.ndarray
        The extracted field.

    Returns
    -------
    extracted_field : np.ndarray
        The extracted field rescaled to match the true field.

    Notes
    -----
    Rescale the extracted field to match the true field by:
        1) Dividing the extracted field by its standard deviation.
        3) Multiplying the extracted field by the standard deviation of the true field.

    """
    if true_field is None or extracted_field is None:
        raise ValueError("Invalid input field(s): Field must be a vector/array, not None.")
    if len(true_field) == 0 or len(extracted_field) == 0:
        raise ValueError("Invalid input field(s): Field must have a nonzero size/length.")
    
    # Normalize y by mean-subtraction and std-division and rescale by x's std
    # x_std = np.std(x)
    # x_mean = np.mean(x)
    # y_std = np.std(y)
    # y_mean = np.mean(y)
    # y_normd = ((y - y_mean) / y_std) 
    # y_normd = y_normd * x_std
    
    # extracted_field = extracted_field - np.mean(extracted_field)
    extracted_field = extracted_field / np.std(extracted_field)
    extracted_field = extracted_field * np.std(true_field)
    # extracted_field = extracted_field + np.mean(true_field)
    return extracted_field


# if __name__ == "__main__":
#     # Run the test function
#     test_calculate_residuals()

#     # # Load the true and extracted fields
#     # true_field = np.load("true_field.npy")
#     # extracted_field = np.load("extracted_field.npy")

#     # # Calculate and print all metrics
#     # metrics = calculate_all_metrics(true_field, extracted_field)

#     # # Save the metrics to a file
#     # np.save("metrics.npy", metrics)