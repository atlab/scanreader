class ScanReaderException(Exception):
    """Base ScanReader exception. """
    pass

class ScanImageVersionError(ScanReaderException):
    """ Exception for unsupported ScanImage versions."""
    pass

class PathnameError(ScanReaderException):
    """ Exception for dealing with paths and pathname patterns (wildcards)."""
    pass

class FieldDimensionMismatch(ScanReaderException):
    """ Exception for trying to slice an array with fields of different dimensions."""
    pass