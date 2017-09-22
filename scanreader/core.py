"""
Reader for ScanImage 5 scans (including multiROI).

Example:
    import scanreader
    scan = scanreader.read_scan('my_scan_*.tif')
    for field in scan:
        #process field
"""
from tifffile import TiffFile
from glob import glob
from os import path
import numpy as np
import re
from .exceptions import ScanImageVersionError, PathnameError
from .scans import Scan5Point1,Scan5Point2, Scan2016b, ScanMultiROI
from .stacks import Stack5Point1,Stack5Point2, Stack2016b, StackMultiROI

def read_scan(pathnames, dtype=np.int16, join_contiguous=False):
    """ Reads a ScanImage scan.

    Args:
        pathnames: String or list of strings. Pathname(s) or pathname pattern(s) to read.
        dtype: Data-type. Data type of the output array.
        join_contiguous: Boolean. For multiROI scans (2016b and beyond) it will join
            contiguous scanfields in the same depth. No effect in non-multiROI scans. See
            help of ScanMultiROI._join_contiguous_fields for details.

    Returns:
        A Scan object (subclass of BaseScan) with metadata and data. See Readme for details.
    """
    # Expand wildcards
    filenames = expand_wildcard(pathnames)
    if len(filenames) == 0:
        error_msg = 'Pathname(s) {} do not match any files in disk.'.format(pathnames)
        raise PathnameError(error_msg)

    # Read version from one of the tiff files
    tiff_file = TiffFile(filenames[0], pages=[0])
    version = get_scanimage_version(tiff_file.info())

    # Select the appropriate scan object
    if version == '5.1':
        scan = Scan5Point1()
    elif version == '5.2':
        scan = Scan5Point2()
    elif version == '2016b':
        if is_scan_multiROI(tiff_file.info()):
            scan = ScanMultiROI(join_contiguous=join_contiguous)
        else:
            scan = Scan2016b()
    else:
        error_msg = 'Sorry, ScanImage version {} is not supported'.format(version)
        raise ScanImageVersionError(error_msg)

    # Read metadata and data (lazy operation)
    scan.read_data(filenames, dtype=dtype)

    # Close tiff_file
    tiff_file.close()

    return scan

def read_stack(pathnames, dtype=np.int16, join_contiguous=False):
    """ Reads a ScanImage stack.

    Stacks are similar to scans except that we record all frames for a single scanning
    depth before moving to the next.

    Args:
        pathnames: String or list of strings. Pathname(s) or pathname pattern(s) to read.
        dtype: Data-type. Data type of the output array.
        join_contiguous: Boolean. For multiROI stacks (2016b and beyond) it will join
            contiguous scanfields in the same depth. No effect in non-multiROI stacks.
            See help of ScanMultiROI._join_contiguous_fields for details.

    Returns:
        A Stack object (subclass of BaseStack) with metadata and data. See Readme for details.
    """
    # Expand wildcards
    filenames = expand_wildcard(pathnames)
    if len(filenames) == 0:
        error_msg = 'Pathname(s) {} do not match any files in disk.'.format(pathnames)
        raise PathnameError(error_msg)

    # Read version from one of the tiff files
    tiff_file = TiffFile(filenames[0], pages=[0])
    version = get_scanimage_version(tiff_file.info())

    # Select the appropriate scan object
    if version == '5.1':
        stack = Stack5Point1()
    elif version == '5.2':
        stack = Stack5Point2()
    elif version == '2016b':
        if is_scan_multiROI(tiff_file.info()):
            stack = StackMultiROI(join_contiguous=join_contiguous)
        else:
            stack = Stack2016b()
    else:
        error_msg = 'Sorry, ScanImage version {} is not supported'.format(version)
        raise ScanImageVersionError(error_msg)

    # Read metadata and data (lazy operation)
    stack.read_data(filenames, dtype=dtype)

    # Close tiff_file
    tiff_file.close()

    return stack

def expand_wildcard(wildcard):
    """ Expands a list of pathname patterns to form a sorted list of absolute filenames.

    Args:
        wildcard: String or list of strings. Pathname pattern(s) to be extended with glob.

    Returns:
        A list of string. Absolute filenames.
    """
    if isinstance(wildcard, str):
        wildcard_list = [wildcard]
    elif isinstance(wildcard, (tuple, list)):
        wildcard_list = wildcard
    else:
        error_msg = 'Expected string or list of strings, received {}'.format(wildcard)
        raise TypeError(error_msg)

    # Expand wildcards
    rel_filenames = [glob(wildcard) for wildcard in wildcard_list]
    rel_filenames = [item for sublist in rel_filenames for item in sublist] # flatten list

    # Make absolute filenames
    abs_filenames = [path.abspath(filename) for filename in rel_filenames]

    # Sort
    sorted_filenames = sorted(abs_filenames, key=path.basename)

    return sorted_filenames

def get_scanimage_version(info):
    """ Looks for the ScanImage version in the tiff file headers.

    Args:
        info: A string. All headers from tiff tags.

    Returns:
        A string. ScanImage version
    """
    pattern = re.compile(r"SI.?\.VERSION_MAJOR = '(?P<version>.*)'")
    match = re.search(pattern, info)
    if match:
        version = match.group('version')
    else:
        raise ScanImageVersionError('Could not find ScanImage version in the tiff header')

    return version

def is_scan_multiROI(info):
    """Looks whether the scan is multiROI in the tiff file headers.

    Args:
        info: A string. All headers from tiff tags.

    Returns:
        A bool. Whether scan is multiroi or not.
    """
    match = re.search(r'hRoiManager\.mroiEnable = (?P<is_multiROI>.)', info)
    is_multiROI = (match.group('is_multiROI') == '1') if match else None
    return is_multiROI