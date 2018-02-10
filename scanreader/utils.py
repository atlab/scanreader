"""Utility functions to check that the key and indices send to __getitem__ are valid."""
import numpy as np

def fill_key(key, num_dimensions):
    """ Fill key with slice(None) (':') until num_dimensions size.

    Args:
        key: tuple of indices or single index. key as received by __getitem__().
        num_dimensions: integer. Total number of dimensions needed.

    Raises:
        IndexError: Too many indices in key: len(key) > num_dimensions.
    """
    # Deal with single valued keys, e.g., scan[:] or scan[0]
    if not isinstance(key, tuple):
        key = (key,)

    # Check key is not larger than num_dimensions
    if len(key) > num_dimensions:
        raise IndexError('too many indices for scan: {}'.format(len(key)))

    # Add missing dimensions
    missing_dimensions = num_dimensions - len(key)
    full_key = tuple(list(key) + [slice(None)] * missing_dimensions)

    return full_key


def check_index_type(axis, index):
    """ Checks that index is an integer, slice or array/list/tuple of integers.

    Args:
        axis: An integer. Axis of the index.
        index: A single index (integer, slice or list/tuple/array of integers).

    Raises:
        TypeError: If index is not integer, slice, or array/list/tuple of integers.
    """
    if not _index_has_valid_type(index): # raise error
        error_msg = ('index {} in axis {} is not an integer, slice or array/list/tuple '
                     'of integers'.format(index, axis))
        raise TypeError(error_msg)


def _index_has_valid_type(index):
    if np.issubdtype(type(index), np.signedinteger): # integer
        return True
    if isinstance(index, slice): # slice
        return True
    if (isinstance(index, (list, tuple)) and
        all(np.issubdtype(type(x), np.signedinteger) for x in index)):  # list or tuple
        return True
    if (isinstance(index, np.ndarray) and np.issubdtype(index.dtype, np.signedinteger)
        and index.ndim == 1):  # array
        return True

    return False


def check_index_is_in_bounds(axis, index, dim_size):
    """ Check that an index is in bounds for the given dimension size. By python indexing
    rules, anything from -dim_size to dim_size-1 is valid.

    Args:
        axis: An integer. Axis of the index.
        index: A single index (integer, slice or list/tuple/array of integers).
        dim_size: Size of the dimension against which the index will be checked.

    Raises:
        TypeError: If index is not either integer, slice, or array.
        IndexError: If index is out of bounds for the specified axis.
    """
    if not _is_index_in_bounds(index, dim_size):
        error_msg = ('index {} is out of bounds for axis {} with size '
                     '{}'.format(index, axis, dim_size))
        raise IndexError(error_msg)


def _is_index_in_bounds(index, dim_size):
    if np.issubdtype(type(index), np.signedinteger):
        return (index in range(-dim_size, dim_size))
    elif isinstance(index, (list, tuple, np.ndarray)):
        return all(x in range(-dim_size, dim_size) for x in index)
    elif isinstance(index, slice):
        return True  # slices never go out of bounds, they are just cropped
    else:
        error_msg = ('index {} is not either integer, slice or array/list/tuple of '
                     'integers'.format(index))
        raise TypeError(error_msg)


def listify_index(index, dim_size):
    """ Generates the list representation of an index for the given dim_size.

    Args:
        index: A single index (integer, slice or list/tuple/array of integers).
        dim_size: Size of the dimension corresponding to the index.

    Returns:
        A list of positive integers. List of indices.

    Raises:
        TypeError: If index is not either integer, slice, or array.
    """
    if np.issubdtype(type(index), np.signedinteger):
        index_as_list = [index] if index >= 0 else [dim_size + index]
    elif isinstance(index, (list, tuple, np.ndarray)):
        index_as_list = [x if x >= 0 else (dim_size + x) for x in index]
    elif isinstance(index, slice):
        start, stop, step = index.indices(dim_size)  # transforms Nones and negative ints to valid slice
        index_as_list = list(range(start, stop, step))
    else:
        error_msg = ('index {} is not integer, slice or array/list/tuple of '
                     'integers'.format(index))
        raise TypeError(error_msg)

    return index_as_list