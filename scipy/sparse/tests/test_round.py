import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, isspmatrix_csr, isspmatrix_csc

def test_round():
    """Test rounding of sparse matrix elements"""
    # Test basic rounding
    data = np.array([1.1, 2.5, 3.9, 4.4])
    row = np.array([0, 0, 1, 1])
    col = np.array([0, 1, 0, 1])
    mat = csr_matrix((data, (row, col)), shape=(2, 2))
    
    rounded = mat.round()
    expected_data = np.array([1., 2., 4., 4.])
    expected = csr_matrix((expected_data, (row, col)), shape=(2, 2))
    
    assert isspmatrix_csr(rounded)
    assert np.array_equal(rounded.data, expected.data)
    assert np.array_equal(rounded.indices, expected.indices)
    assert np.array_equal(rounded.indptr, expected.indptr)

def test_round_with_decimals():
    """Test rounding with different decimal places"""
    data = np.array([1.123, 2.456, 3.789, 4.999])
    row = np.array([0, 0, 1, 1])
    col = np.array([0, 1, 0, 1])
    mat = csc_matrix((data, (row, col)), shape=(2, 2))
    
    # Test rounding to 1 decimal place
    rounded1 = mat.round(decimals=1)
    expected_data1 = np.array([1.1, 2.5, 3.8, 5.0])
    expected1 = csc_matrix((expected_data1, (row, col)), shape=(2, 2))
    
    assert isspmatrix_csc(rounded1)
    assert np.allclose(rounded1.data, expected1.data)
    
    # Test rounding to 2 decimal places
    rounded2 = mat.round(decimals=2)
    expected_data2 = np.array([1.12, 2.46, 3.79, 5.00])
    expected2 = csc_matrix((expected_data2, (row, col)), shape=(2, 2))
    
    assert isspmatrix_csc(rounded2)
    assert np.allclose(rounded2.data, expected2.data)

def test_round_empty_matrix():
    """Test rounding with empty matrix"""
    mat = csr_matrix((3, 4))
    rounded = mat.round()
    assert isspmatrix_csr(rounded)
    assert rounded.shape == (3, 4)
    assert rounded.nnz == 0

def test_round_integer_matrix():
    """Test rounding with integer matrix (should be unchanged)"""
    data = np.array([1, 2, 3, 4])
    row = np.array([0, 0, 1, 1])
    col = np.array([0, 1, 0, 1])
    mat = csr_matrix((data, (row, col)), shape=(2, 2))
    
    rounded = mat.round()
    assert isspmatrix_csr(rounded)
    assert np.array_equal(rounded.data, mat.data)