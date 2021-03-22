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
from . import scans

_scans = {'5.1': scans.Scan5Point1, '5.2': scans.Scan5Point2, '5.3': scans.Scan5Point3,
          '5.4': scans.Scan5Point4, '5.5': scans.Scan5Point5, 
          '5.6': scans.Scan5Point6, '5.7': scans.Scan5Point7, 
          '2016b': scans.Scan2016b, 
          '2017a': scans.Scan2017a, '2017b': scans.Scan2017b, 
          '2018a': scans.Scan2018a, '2018b': scans.Scan2018b,
          '2019a': scans.Scan2019a, '2019b': scans.Scan2019b,
          '2020': scans.Scan2020,}

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
    with TiffFile(filenames[0]) as tiff_file:
        file_info = tiff_file.pages[0].description + '\n' + tiff_file.pages[0].software
    version = get_scanimage_version(file_info)

    # Select the appropriate scan object
    
    if (version in ['2016b', '2017a', '2017b', '2018a', '2018b', '2019a', '2019b', '2020'] and
            is_scan_multiROI(file_info)):
        scan = scans.ScanMultiROI(join_contiguous=join_contiguous)
    elif version in _scans:
        scan = _scans[version]()
    else:
        error_msg = 'Sorry, ScanImage version {} is not supported'.format(version)
        raise ScanImageVersionError(error_msg)

    # Read metadata and data (lazy operation)
    scan.read_data(filenames, dtype=dtype)

    return scan

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
    pattern = re.compile(r"SI.?\.VERSION_MAJOR = '?(?P<version>[^\s']*)'?")
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