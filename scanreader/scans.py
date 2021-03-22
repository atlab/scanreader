"""
ScanImage scans. Each version handles things a little differently. Scan objects are
usually instantiated by a call to scanreader.read_scan().

Hierarchy:
BaseScan
    ScanLegacy
    BaseScan5
        Scan5Point1
        Scan5Point2
            Scan5Point3
                Scan5Point4
                Scan5Point5
                Scan5Point6
                Scan5Point7
                Scan2016b
                Scan2017a
                Scan2017b
                Scan2018a
                Scan2018b
                Scan2019a
                Scan2019b
                Scan2020
    ScanMultiRoi
"""
from tifffile import TiffFile
from tifffile.tifffile import matlabstr2py
import numpy as np
import re
import itertools
from . import utils
from .multiroi import ROI
from .exceptions import FieldDimensionMismatch

class BaseScan():
    """ Properties and methods shared among all scan versions.

    Scan objects are a collection of recording fields: rectangular planes at a given x, y,
    z position in the scan recorded in a number of channels during a preset amount of
    time. All fields have the same number of channels and number of frames.
    Scan objects are:
        indexable: scan[field, y, x, channel, frame] works as long as the fields' spatial
            dimensions (y, x) match.
        iterable: 'for field in scan:' iterates over all fields (4-d array) in the scan.

    Examples:
        scan.version                ScanImage version of the scan.
        scan[:, :, :3, :, :1000]    5-d numpy array with the first 1000 frames of the
            first 3 fields (if x, y dimensions match).
        for field in scan:          generates 4-d numpy arrays ([y, x, channels, frames]).

    Note:
        We use the word 'frames' as in video frames, i.e., number of timesteps the scan
        was recorded; ScanImage uses frames to refer to slices/scanning depths in the
        scan.
    """
    """
    Interface rules:
        If it (method or property) is shared among all subclasses it should be here,
            either implemented or as an abstract method. Even if private (e.g. _page_height)
        If one subclass needs to overwrite it, then erase it here and implement them in
            the subclasses (this applies for now that I only have two subclasses). If code
            needs to be shared add it as a private method here.
        If it is not in every subclass, it should not be here.
    """
    def __init__(self):
        self.filenames = None
        self.dtype = None
        self._tiff_files = None
        self.header = ''

    @property
    def tiff_files(self):
        if self._tiff_files is None:
            self._tiff_files = [TiffFile(filename) for filename in self.filenames]
        return self._tiff_files

    @tiff_files.deleter
    def tiff_files(self):
        if self._tiff_files is not None:
            for tiff_file in self._tiff_files:
                tiff_file.close()
            self._tiff_files = None

    @property
    def version(self):
        match = re.search(r"SI.?\.VERSION_MAJOR = '?(?P<version>[^\s']*)'?", self.header)
        version = match.group('version') if match else None
        return version

    @property
    def is_slow_stack(self):
        """ True if fastZ is disabled. All frames for one slice are recorded first before
        moving to the next slice."""
        match = re.search(r'hFastZ\.enable = (?P<is_slow>.*)', self.header)
        is_slow_stack = (match.group('is_slow') in ['false', '0']) if match else None
        return is_slow_stack

    @property
    def is_multiROI(self):
        """Only True if mroiEnable exists (2016b and up) and is set to True."""
        match = re.search(r'hRoiManager\.mroiEnable = (?P<is_multiROI>.)', self.header)
        is_multiROI = (match.group('is_multiROI') == '1') if match else False
        return is_multiROI

    @property
    def num_channels(self):
        match = re.search(r'hChannels\.channelSave = (?P<channels>.*)', self.header)
        if match:
            channels = matlabstr2py(match.group('channels'))
            num_channels = len(channels) if isinstance(channels, list) else 1
        else:
            num_channels = None
        return num_channels

    @property
    def requested_scanning_depths(self):
        match = re.search(r'hStackManager\.zs = (?P<zs>.*)', self.header)
        if match:
            zs = matlabstr2py(match.group('zs'))
            scanning_depths = zs if isinstance(zs, list) else [zs]
        else:
            scanning_depths = None
        return scanning_depths

    @property
    def num_scanning_depths(self):
        if self.is_slow_stack:
            """ Number of scanning depths actually recorded in this stack."""
            num_scanning_depths = self._num_pages / (self.num_channels * self.num_frames)
            num_scanning_depths = int(num_scanning_depths) # discard last slice if incomplete
        else:
            num_scanning_depths = len(self.requested_scanning_depths)
        return num_scanning_depths

    @property
    def scanning_depths(self):
        return self.requested_scanning_depths[:self.num_scanning_depths]

    @property
    def num_requested_frames(self):
        if self.is_slow_stack:
             match = re.search(r'hStackManager\.framesPerSlice = (?P<num_frames>.*)',
                              self.header)
        else:
            match = re.search(r'hFastZ\.numVolumes = (?P<num_frames>.*)', self.header)
        num_requested_frames = int(1e9 if match.group('num_frames')=='Inf' else
                                   float(match.group('num_frames'))) if match else None
        return num_requested_frames

    @property
    def num_frames(self):
        """ Each tiff page is an image at a given channel, scanning depth combination."""
        if self.is_slow_stack:
            num_frames = min(self.num_requested_frames / self._num_averaged_frames,
                             self._num_pages / self.num_channels) # finished in the first slice
        else:
            num_frames = self._num_pages / (self.num_channels * self.num_scanning_depths)
        num_frames = int(num_frames) # discard last frame if incomplete
        return num_frames

    @property
    def is_bidirectional(self):
        match = re.search(r'hScan2D\.bidirectional = (?P<is_bidirectional>.*)', self.header)
        is_bidirectional = (match.group('is_bidirectional') == 'true') if match else False
        return is_bidirectional

    @property
    def scanner_frequency(self):
        match = re.search(r'hScan2D\.scannerFrequency = (?P<scanner_freq>.*)', self.header)
        scanner_frequency = float(match.group('scanner_freq')) if match else None
        return scanner_frequency

    @property
    def seconds_per_line(self):
        if np.isnan(self.scanner_frequency):
            match = re.search(r'hRoiManager\.linePeriod = (?P<secs_per_line>.*)', self.header)
            seconds_per_line = float(match.group('secs_per_line')) if match else None
        else:
            scanner_period = 1 / self.scanner_frequency # secs for mirror to return to initial position
            seconds_per_line = scanner_period / 2 if self.is_bidirectional else scanner_period
        return seconds_per_line

    @property
    def _num_pages(self):
        num_pages = sum([len(tiff_file.pages) for tiff_file in self.tiff_files])
        return num_pages

    @property
    def _page_height(self):
        return self.tiff_files[0].pages[0].imagelength

    @property
    def _page_width(self):
        return self.tiff_files[0].pages[0].imagewidth

    @property
    def _num_averaged_frames(self):
        """ Number of requested frames are averaged to form one saved frame. """
        match = re.search(r'hScan2D\.logAverageFactor = (?P<num_avg_frames>.*)', self.header)
        num_averaged_frames = int(float(match.group('num_avg_frames'))) if match else None
        return num_averaged_frames

    @property
    def num_fields(self):
        raise NotImplementedError('Subclasses of BaseScan must implement this property')

    @property
    def field_depths(self):
        raise NotImplementedError('Subclasses of BaseScan must implement this property')

    # Properties from here on are not strictly necessary
    @property
    def fps(self):
        match = re.search(r'hRoiManager\.scanVolumeRate = (?P<fps>.*)',self.header)
        fps = float(match.group('fps')) if match else None
        return fps

    @property
    def spatial_fill_fraction(self):
        match = re.search(r'hScan2D\.fillFractionSpatial = (?P<spatial_ff>.*)', self.header)
        spatial_fill_fraction = float(match.group('spatial_ff')) if match else None
        return spatial_fill_fraction

    @property
    def temporal_fill_fraction(self):
        match = re.search(r'hScan2D\.fillFractionTemporal = (?P<temporal_ff>.*)', self.header)
        temporal_fill_fraction = float(match.group('temporal_ff')) if match else None
        return temporal_fill_fraction

    @property
    def scanner_type(self):
        match = re.search(r"hScan2D\.scannerType = '(?P<scanner_type>.*)'", self.header)
        scanner_type = match.group('scanner_type') if match else None
        return scanner_type

    @property
    def motor_position_at_zero(self):
        """ Motor position (x, y and z in microns) corresponding to the scan's (0, 0, 0)
        point. For non-multiroi scans, (x=0, y=0) marks the center of the FOV."""
        match = re.search(r'hMotors\.motorPosition = (?P<motor_position>.*)', self.header)
        motor_position = matlabstr2py(match.group('motor_position'))[:3] if match else None
        return motor_position

    @property
    def initial_secondary_z(self):
        """ Initial position in z (microns) of the secondary motor (if any)."""
        match = re.search(r'hMotors\.motorPosition = (?P<motor_position>.*)', self.header)
        if match:
            motor_position = matlabstr2py(match.group('motor_position'))
            secondary_z = motor_position[3] if len(motor_position) > 3 else None
        else:
            secondary_z = None
        return secondary_z

    @property
    def _initial_frame_number(self):
        match = re.search(r'\sframeNumbers = (?P<frame_number>.*)', self.header)
        initial_frame_number = int(match.group('frame_number')) if match else None
        return initial_frame_number

    @property
    def _num_fly_back_lines(self):
        """ Lines/mirror cycles that it takes to move from one depth to the next."""
        match = re.search(r'hScan2D\.flybackTimePerFrame = (?P<fly_back_seconds>.*)',
                          self.header)
        if match:
            fly_back_seconds = float(match.group('fly_back_seconds'))
            num_fly_back_lines = self._seconds_to_lines(fly_back_seconds)
        else:
            num_fly_back_lines = None
        return num_fly_back_lines

    @property
    def _num_lines_between_fields(self):
        """ Lines/mirror cycles scanned from the start of one field to the start of the
        next. """
        if self.is_slow_stack:
            num_lines_between_fields = ((self._page_height + self._num_fly_back_lines) *
                                        (self.num_frames * self._num_averaged_frames))
        else:
            num_lines_between_fields = self._page_height + self._num_fly_back_lines
        return num_lines_between_fields

    @property
    def is_slow_stack_with_fastZ(self):
        raise NotImplementedError('Subclasses of BaseScan must implement this property')

    @property
    def field_offsets(self):
        raise NotImplementedError('Subclasses of BaseScan must implement this property')

    def read_data(self, filenames, dtype):
        """ Set self.header, self.filenames and self.dtype. Data is read lazily when needed.

        Args:
            filenames: List of strings. Tiff filenames.
            dtype: Data type of the output array.
        """
        self.filenames = filenames # set filenames
        self.dtype=dtype # set dtype of read data
        self.header = '{}\n{}'.format(self.tiff_files[0].pages[0].description,
                                      self.tiff_files[0].pages[0].software) # set header (ScanImage metadata)

    def __array__(self):
        return self[:]

    def __str__(self):
        msg = '{}\n{}\n{}'.format(type(self), '*' * 80, self.header, '*' * 80)
        return msg

    def __len__(self):
        return 0 if self.num_fields is None else self.num_fields

    def __getitem__(self, key):
        """ Index scans by field, y, x, channels, frames. Supports integer, slice and
        array/tuple/list of integers as indices."""
        raise NotImplementedError('Subclasses of BaseScan must implement this method')

    def __iter__(self):
        class ScanIterator:
            """ Iterator for Scan objects."""
            def __init__(self, scan):
                self.scan = scan
                self.next_field = 0

            def __next__(self):
                if self.next_field < self.scan.num_fields:
                    field = self.scan[self.next_field]
                    self.next_field += 1
                else:
                    raise StopIteration
                return field

        return ScanIterator(self)

    def _read_pages(self, slice_list, channel_list, frame_list, yslice=slice(None),
                    xslice=slice(None)):
        """ Reads the tiff pages with the content of each slice, channel, frame
        combination and slices them in the y, x dimension.

        Each tiff page holds a single depth/channel/frame combination. For slow stacks,
        channels change first, timeframes change second and slices/depths change last.
        Example:
            For two channels, three slices, two frames.
                Page:       0   1   2   3   4   5   6   7   8   9   10  11
                Channel:    0   1   0   1   0   1   0   1   0   1   0   1
                Frame:      0   0   1   1   2   2   0   0   1   1   2   2
                Slice:      0   0   0   0   0   0   1   1   1   1   1   1

        For scans, channels change first, slices/depths change second and timeframes
        change last.
        Example:
            For two channels, three slices, two frames.
                Page:       0   1   2   3   4   5   6   7   8   9   10  11
                Channel:    0   1   0   1   0   1   0   1   0   1   0   1
                Slice:      0   0   1   1   2   2   0   0   1   1   2   2
                Frame:      0   0   0   0   0   0   1   1   1   1   1   1

        Args:
            slice_list: List of integers. Slices to read.
            channel_list: List of integers. Channels to read.
            frame_list: List of integers. Frames to read
            yslice: Slice object. How to slice the pages in the y axis.
            xslice: Slice object. How to slice the pages in the x axis.

        Returns:
            A 5-D array (num_slices, output_height, output_width, num_channels, num_frames).
                Required pages reshaped to have slice, channel and frame as different
                dimensions. Channel, slice and frame order received in the input lists are
                respected; for instance, if slice_list = [1, 0, 2, 0], then the first
                dimension will have four slices: [1, 0, 2, 0].

        Note:
            We use slices in y, x for memory efficiency, If lists were passed another copy
            of the pages will be needed coming up to 3x the amount of data we actually
            want to read (the output array, the read pages and the list-sliced pages).
            Slices limit this to 2x (output array and read pages which are sliced in place).
        """
        # Compute pages to load from tiff files
        if self.is_slow_stack:
            frame_step = self.num_channels
            slice_step = self.num_channels * self.num_frames
        else:
            slice_step = self.num_channels
            frame_step = self.num_channels * self.num_scanning_depths
        pages_to_read = []
        for frame in frame_list:
            for slice_ in slice_list:
                for channel in channel_list:
                    new_page = frame * frame_step + slice_ * slice_step + channel
                    pages_to_read.append(new_page)

        # Compute output dimensions
        out_height = len(utils.listify_index(yslice, self._page_height))
        out_width = len(utils.listify_index(xslice, self._page_width))

        # Read pages
        pages = np.empty([len(pages_to_read), out_height, out_width], dtype=self.dtype)
        start_page = 0
        for tiff_file in self.tiff_files:

            # Get indices in this tiff file and in output array
            final_page_in_file = start_page + len(tiff_file.pages)
            is_page_in_file = lambda page: page in range(start_page, final_page_in_file)
            pages_in_file = filter(is_page_in_file, pages_to_read)
            file_indices = [page - start_page for page in pages_in_file]
            global_indices = [is_page_in_file(page) for page in pages_to_read]

            # Read from this tiff file (if needed)
            if len(file_indices) > 0:
                # this line looks a bit ugly but is memory efficient. Do not separate
                pages[global_indices] = tiff_file.asarray(key=file_indices)[..., yslice, xslice]
            start_page += len(tiff_file.pages)

        # Reshape the pages into (slices, y, x, channels, frames)
        new_shape = [len(frame_list), len(slice_list), len(channel_list), out_height, out_width]
        pages = pages.reshape(new_shape).transpose([1, 3, 4, 2, 0])

        return pages

    def _seconds_to_lines(self, seconds):
        """ Compute how many lines would be scanned in the given amount of seconds."""
        num_lines = int(np.ceil(seconds / self.seconds_per_line))
        if self.is_bidirectional:
            # scanning starts at one end of the image so num_lines needs to be even
            num_lines += (num_lines % 2)

        return num_lines

    def _compute_offsets(self, field_height, start_line):
        """ Computes the time offsets at which a given field was recorded.

        Computes the time delay at which each pixel was recorded using the start of the
        scan as zero. It first creates an image with the number of lines scanned until
        that point and then uses self.seconds_per_line to  transform it into seconds.

        :param int field_height: Height of the field.
        :param int start_line: Line at which this field starts.

        :returns: A field_height x page_width mask with time offsets in seconds.
        """
        # Compute offsets within a line (negligible if seconds_per_line is small)
        max_angle = (np.pi / 2) * self.temporal_fill_fraction
        line_angles = np.linspace(-max_angle, max_angle, self._page_width + 2)[1:-1]
        line_offsets = (np.sin(line_angles) + 1) / 2

        # Compute offsets for entire field
        field_offsets = np.expand_dims(np.arange(0, field_height), -1) + line_offsets
        if self.is_bidirectional: # odd lines scanned from left to right
            field_offsets[1::2] = field_offsets[1::2] - line_offsets + (1 - line_offsets)

        # Transform offsets from line counts to seconds
        field_offsets = (field_offsets + start_line) * self.seconds_per_line

        return field_offsets


class ScanLegacy(BaseScan):
    """ Scan versions 4 and below. Not implemented."""

    def __init__(self):
        raise NotImplementedError('Legacy scans not supported')


class BaseScan5(BaseScan):
    """ ScanImage 5 scans: one field per scanning depth and all fields have the same
    height and width."""

    @property
    def num_fields(self):
        return self.num_scanning_depths # one field per scanning depth

    @property
    def field_depths(self):
        return self.scanning_depths

    @property
    def image_height(self):
        return self._page_height

    @property
    def image_width(self):
        return self._page_width

    @property
    def shape(self):
        return (self.num_fields, self.image_height, self.image_width, self.num_channels,
                self.num_frames)

    @property
    def zoom(self):
        match = re.search(r'hRoiManager\.scanZoomFactor = (?P<zoom>.*)', self.header)
        zoom = float(match.group('zoom')) if match else None
        return zoom

    @property
    def is_slow_stack_with_fastZ(self):
        match = re.search(r'hMotors\.motorSecondMotorZEnable = (?P<uses_fastZ>.*)',
                          self.header)
        uses_fastZ = (match.group('uses_fastZ') in ['true', '1']) if match else None
        return self.is_slow_stack and uses_fastZ

    @property
    def field_offsets(self):
        """ Seconds elapsed between start of frame scanning and each pixel."""
        next_line = 0
        field_offsets = []
        for i in range(self.num_fields):
            field_offsets.append(self._compute_offsets(self.image_height, next_line))
            next_line += self._num_lines_between_fields
        return field_offsets

    @property
    def _y_angle_scale_factor(self):
        """ Scan angles in y are scaled by this factor, shrinking the angle range."""
        match = re.search(r'hRoiManager\.scanAngleMultiplierSlow = (?P<angle_scaler>.*)',
                          self.header)
        y_angle_scaler = float(match.group('angle_scaler')) if match else None
        return y_angle_scaler

    @property
    def _x_angle_scale_factor(self):
        """ Scan angles in x are scaled by this factor, shrinking the angle range."""
        match = re.search(r'hRoiManager\.scanAngleMultiplierFast = (?P<angle_scaler>.*)',
                          self.header)
        x_angle_scaler = float(match.group('angle_scaler')) if match else None
        return x_angle_scaler

    def __getitem__(self, key):
        """ In non-multiROI, all fields have the same x, y dimensions. """
        # Fill key to size 5 (raises IndexError if more than 5)
        full_key = utils.fill_key(key, num_dimensions=5)

        # Check index types are valid
        for i, index in enumerate(full_key):
            utils.check_index_type(i, index)

        # Check each dimension is in bounds
        max_dimensions = self.shape
        for i, (index, dim_size) in enumerate(zip(full_key, max_dimensions)):
            utils.check_index_is_in_bounds(i, index, dim_size)

        # Get fields, channels and frames as lists
        field_list = utils.listify_index(full_key[0], self.num_fields)
        y_list = utils.listify_index(full_key[1], self.image_height)
        x_list = utils.listify_index(full_key[2], self.image_width)
        channel_list = utils.listify_index(full_key[3], self.num_channels)
        frame_list = utils.listify_index(full_key[4], self.num_frames)

        # Edge case when slice index gives 0 elements or index is empty list, e.g., scan[10:0], scan[[]]
        if [] in [field_list, y_list, x_list, channel_list, frame_list,]:
            return np.empty(0)

        # Read the required pages
        pages = self._read_pages(field_list, channel_list, frame_list)

        # Index in y, x using the original key (usually slices) for memory efficiency.
        if isinstance(full_key[1], list) and isinstance(full_key[2], list):
            # Our behaviour for lists is to take the submatrix defined by those indices.
            ys = [[y] for y in y_list] # ys as nested lists does the trick
            item = pages[:, ys, x_list, :, :]
        else:
            item = pages[:, full_key[1], full_key[2], :, :]
            item = item.reshape(len(field_list), len(y_list), len(x_list), len(channel_list),
                                len(frame_list)) # put back any dropped dimension

        # If original index was an integer, delete that axis (as in numpy indexing)
        squeeze_dims = [i for i, index in enumerate(full_key) if np.issubdtype(type(index),
                                                                               np.signedinteger)]
        item = np.squeeze(item, axis=tuple(squeeze_dims))

        return item


class Scan5Point1(BaseScan5):
    """ ScanImage 5.1. Basic."""
    pass


class Scan5Point2(BaseScan5):
    """ ScanImage 5.2. Addition of FOV measures in microns."""

    @property
    def image_height_in_microns(self):
        match = re.search(r'hRoiManager\.imagingFovUm = (?P<fov_corners>.*)', self.header)
        if match:
            fov_corners = matlabstr2py(match.group('fov_corners'))
            image_height_in_microns = fov_corners[2][1] - fov_corners[1][1]  # y1-y0
        else:
            image_height_in_microns = None
        return image_height_in_microns

    @property
    def image_width_in_microns(self):
        match = re.search(r'hRoiManager\.imagingFovUm = (?P<fov_corners>.*)', self.header)
        if match:
            fov_corners = matlabstr2py(match.group('fov_corners'))
            image_width_in_microns = fov_corners[1][0] - fov_corners[0][0] # x1-x0
        else:
            image_width_in_microns = None
        return image_width_in_microns


class NewerScan():
    """ Shared features among all newer scans. """
    @property
    def is_slow_stack_with_fastZ(self):
        match = re.search(r'hStackManager\.slowStackWithFastZ = (?P<slow_with_fastZ>.*)',
                          self.header)
        slow_with_fastZ = (match.group('slow_with_fastZ') in ['true', '1']) if match else None
        return slow_with_fastZ


class Scan5Point3(NewerScan, Scan5Point2): # NewerScan first to shadow Scan5Point2's properties
    """ScanImage 5.3"""
    pass

class Scan5Point4(Scan5Point3):
    """ScanImage 5.4"""
    pass

class Scan5Point5(Scan5Point3):
    """ScanImage 5.5"""
    pass

class Scan5Point6(Scan5Point3):
    """ScanImage 5.6"""
    pass

class Scan5Point7(Scan5Point3):
    """ScanImage 5.7"""
    pass


class Scan2016b(Scan5Point3):
    """ ScanImage 2016b"""
    pass


class Scan2017a(Scan5Point3):
    """ ScanImage 2017a"""
    pass


class Scan2017b(Scan5Point3):
    """ ScanImage 2017b"""
    pass


class Scan2018a(Scan5Point3):
    """ ScanImage 2018a"""
    pass


class Scan2018b(Scan5Point3):
    """ ScanImage 2018b"""
    pass


class Scan2019a(Scan5Point3):
    """ ScanImage 2019a"""
    pass


class Scan2019b(Scan5Point3):
    """ ScanImage 2019b"""
    pass

class Scan2020(Scan5Point3):
    """ ScanImage 2020"""
    pass


class ScanMultiROI(NewerScan, BaseScan):
    """An extension of ScanImage v5 that manages multiROI data (output from mesoscope).

     Attributes:
         join_contiguous: A bool. Whether contiguous fields are joined into one.
         rois: List of ROI objects (defined in multiroi.py)
         fields: List of Field objects (defined in multiroi.py)
     """

    def __init__(self, join_contiguous):
        super().__init__()
        self.join_contiguous = join_contiguous
        self.rois = None
        self.fields = None

    @property
    def num_fields(self):
        return len(self.fields)

    @property
    def num_rois(self):
        return len(self.rois)

    @property
    def field_heights(self):
        return [field.height for field in self.fields]

    @property
    def field_widths(self):
        return [field.width for field in self.fields]

    @property
    def field_depths(self):
        return [field.depth for field in self.fields]

    @property
    def field_slices(self):
        return [field.slice_id for field in self.fields]

    @property
    def field_rois(self):
        return [field.roi_ids for field in self.fields]

    @property
    def field_masks(self):
        return [field.roi_mask for field in self.fields]

    @property
    def field_offsets(self):
        return [field.offset_mask for field in self.fields]

    @property
    def field_heights_in_microns(self):
        field_heights_in_degrees = [field.height_in_degrees for field in self.fields]
        return [self._degrees_to_microns(deg) for deg in field_heights_in_degrees]

    @property
    def field_widths_in_microns(self):
        field_widths_in_degrees = [field.width_in_degrees for field in self.fields]
        return [self._degrees_to_microns(deg) for deg in field_widths_in_degrees]

    @property
    def _num_fly_to_lines(self):
        """ Number of lines recorded in the tiff page while flying to a different field,
        i.e., distance between fields in the tiff page."""
        match = re.search(r'hScan2D\.flytoTimePerScanfield = (?P<fly_to_seconds>.*)',
                          self.header)
        if match:
            fly_to_seconds = float(match.group('fly_to_seconds'))
            num_fly_to_lines = self._seconds_to_lines(fly_to_seconds)
        else:
            num_fly_to_lines = None
        return num_fly_to_lines

    def _degrees_to_microns(self, degrees):
        """ Convert scan angle degrees to microns using the objective resolution."""
        match = re.search(r'objectiveResolution = (?P<deg2um_factor>.*)', self.header)
        microns = (degrees * float(match.group('deg2um_factor'))) if match else None
        return microns

    def read_data(self, filenames, dtype):
        """ Set the header, create rois and fields (joining them if necessary)."""
        super().read_data(filenames, dtype)
        self.rois = self._create_rois()
        self.fields = self._create_fields()
        if self.join_contiguous:
            self._join_contiguous_fields()

    def _create_rois(self):
        """Create scan rois from the configuration file. """
        roi_infos = self.tiff_files[0].scanimage_metadata['RoiGroups']['imagingRoiGroup']['rois']
        roi_infos = roi_infos if isinstance(roi_infos, list) else [roi_infos]
        roi_infos = list(filter(lambda r: isinstance(r['zs'], (int, float, list)),
                                roi_infos)) # discard empty/malformed ROIs

        rois = [ROI(roi_info) for roi_info in roi_infos]
        return rois

    def _create_fields(self):
        """ Go over each slice depth and each roi generating the scanned fields. """
        fields = []
        previous_lines = 0
        for slice_id, scanning_depth in enumerate(self.scanning_depths):
            next_line_in_page = 0 # each slice is one tiff page
            for roi_id, roi in enumerate(self.rois):
                new_field = roi.get_field_at(scanning_depth)

                if new_field is not None:
                    if next_line_in_page + new_field.height > self._page_height:
                        error_msg = ('Overestimated number of fly to lines ({}) at '
                                     'scanning depth {}'.format(self._num_fly_to_lines,
                                                                scanning_depth))
                        raise RuntimeError(error_msg)

                    # Set xslice and yslice (from where in the page to cut it)
                    new_field.yslices = [slice(next_line_in_page, next_line_in_page
                                               + new_field.height)]
                    new_field.xslices = [slice(0, new_field.width)]

                    # Set output xslice and yslice (where to paste it in output)
                    new_field.output_yslices = [slice(0, new_field.height)]
                    new_field.output_xslices = [slice(0, new_field.width)]

                    # Set slice and roi id
                    new_field.slice_id = slice_id
                    new_field.roi_ids = [roi_id]

                    # Set timing offsets
                    offsets = self._compute_offsets(new_field.height, previous_lines +
                                                                      next_line_in_page)
                    new_field.offsets = [offsets]

                    # Compute next starting y
                    next_line_in_page += new_field.height + self._num_fly_to_lines

                    # Add field to fields
                    fields.append(new_field)

            # Accumulate overall number of scanned lines
            previous_lines += self._num_lines_between_fields

        return fields

    def _join_contiguous_fields(self):
        """ In each scanning depth, join fields that are contiguous.

        Fields are considered contiguous if they appear next to each other and have the
        same size in their touching axis. Process is iterative: it tries to join each
        field with the remaining ones (checked in order); at the first union it will break
        and restart the process at the first field. When two fields are joined, it deletes
        the one appearing last and modifies info such as field height, field width and
        slices in the one appearing first.

        Any rectangular area in the scan formed by the union of two or more fields which
        have been joined will be treated as a single field after this operation.
        """
        for scanning_depth in self.scanning_depths:
            two_fields_were_joined = True
            while two_fields_were_joined: # repeat until no fields were joined
                two_fields_were_joined = False

                fields = filter(lambda field: field.depth == scanning_depth, self.fields)
                for field1, field2 in itertools.combinations(fields, 2):

                    if field1.is_contiguous_to(field2):
                        # Change info in field 1 to reflect the union
                        field1.join_with(field2)

                        # Delete field 2 in self.fields
                        self.fields.remove(field2)

                        # Restart join contiguous search (at while)
                        two_fields_were_joined = True
                        break

    def __getitem__(self, key):
        # Fill key to size 5 (raises IndexError if more than 5)
        full_key = utils.fill_key(key, num_dimensions=5)

        # Check index types are valid
        for i, index in enumerate(full_key):
            utils.check_index_type(i, index)

        # Check each dimension is in bounds
        utils.check_index_is_in_bounds(0, full_key[0], self.num_fields)
        for field_id in utils.listify_index(full_key[0], self.num_fields):
            utils.check_index_is_in_bounds(1, full_key[1], self.field_heights[field_id])
            utils.check_index_is_in_bounds(2, full_key[2], self.field_widths[field_id])
        utils.check_index_is_in_bounds(3, full_key[3], self.num_channels)
        utils.check_index_is_in_bounds(4, full_key[4], self.num_frames)

        # Get fields, channels and frames as lists
        field_list = utils.listify_index(full_key[0], self.num_fields)
        y_lists = [utils.listify_index(full_key[1], self.field_heights[field_id]) for
                   field_id in field_list]
        x_lists = [utils.listify_index(full_key[2], self.field_widths[field_id]) for
                   field_id in field_list]
        channel_list = utils.listify_index(full_key[3], self.num_channels)
        frame_list = utils.listify_index(full_key[4], self.num_frames)

        # Edge case when slice index gives 0 elements or index is empty list, e.g., scan[10:0], scan[[]]
        if [] in [field_list, *y_lists, *x_lists, channel_list, frame_list]:
            return np.empty(0)

        # Check output heights and widths match for all fields
        if not all(len(y_list) == len(y_lists[0]) for y_list in y_lists):
            raise FieldDimensionMismatch('Image heights for all fields do not match')
        if not all(len(x_list) == len(x_lists[0]) for x_list in x_lists):
            raise FieldDimensionMismatch('Image widths for all fields do not match')

        # Over each field, read required pages and slice
        item = np.empty([len(field_list), len(y_lists[0]), len(x_lists[0]),
                        len(channel_list), len(frame_list)], dtype=self.dtype)
        for i, (field_id, y_list, x_list) in enumerate(zip(field_list, y_lists, x_lists)):
            field = self.fields[field_id]

            # Over each subfield in field (only one for non-contiguous fields)
            slices = zip(field.yslices, field.xslices, field.output_yslices, field.output_xslices)
            for yslice, xslice, output_yslice, output_xslice in slices:

                # Read the required pages (and slice out the subfield)
                pages = self._read_pages([field.slice_id], channel_list, frame_list,
                                         yslice, xslice)

                # Get x, y indices that need to be accessed in this subfield
                y_range = range(output_yslice.start, output_yslice.stop)
                x_range = range(output_xslice.start, output_xslice.stop)
                ys = [[y - output_yslice.start] for y in y_list if y in y_range]
                xs = [x - output_xslice.start for x in x_list if x in x_range]
                output_ys = [[index] for index, y in enumerate(y_list) if y in y_range]
                output_xs = [index for index, x in enumerate(x_list) if x in x_range]
                # ys as nested lists are needed for numpy to slice them correctly

                # Index pages in y, x
                item[i, output_ys, output_xs] = pages[0, ys, xs]

        # If original index was an integer, delete that axis (as in numpy indexing)
        squeeze_dims = [i for i, index in enumerate(full_key) if np.issubdtype(type(index),
                                                                               np.signedinteger)]
        item = np.squeeze(item, axis=tuple(squeeze_dims))

        return item