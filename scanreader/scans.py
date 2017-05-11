"""
ScanImage scans. Each version handles things a little differently. Scan objects are 
usually instantiated by a call to scanreader.read_scan().

Hierarchy:
BaseScan
    ScanLegacy
    BaseScan5
        Scan5Point1
        Scan5Point2
            Scan2016b
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
        We use frames as in video frames, i.e., number of timesteps the scan was recorded.
        ScanImage uses frames to refer to slices/scanning depths in the scan.
        
    """
    """
    Interface rules:
        If it (method or property) is shared among all subclasses it should be here, 
            either implemented or as an abstract method. Even if private (e.g. _num_pages)
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
        self._field_for_iter = 0

    @property
    def tiff_files(self):
        if self._tiff_files is None:
            self._tiff_files = [TiffFile(filename) for filename in self.filenames]
        return self._tiff_files

    @tiff_files.deleter
    def tiff_files(self):
        for tiff_file in self.tiff_files:
            tiff_file.close()
        self._tiff_files = None

    @property
    def version(self):
        match = re.search(r"SI.?\.VERSION_MAJOR = '(?P<version>.*)'", self.header)
        version = match.group('version') if match else None
        return version

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
    def scanning_depths(self):
        match = re.search(r'hStackManager\.zs = (?P<zs>.*)', self.header)
        if match:
            zs = matlabstr2py(match.group('zs'))
            scanning_depths = zs if isinstance(zs, list) else [zs]
        else:
            scanning_depths = None
        return scanning_depths

    @property
    def num_scanning_depths(self):
        return len(self.scanning_depths)

    @property
    def num_frames(self):
        # Each tiff page is an image at a given channel, scanning depth combination.
        num_pages = sum([len(tiff_file) for tiff_file in self.tiff_files])
        num_frames = num_pages / (self.num_scanning_depths * self.num_channels)
        if not num_frames.is_integer():
            error_msg = ('total number of pages {} not divisible by num_scanning_depths '
                         '* num_channels'.format(num_pages))
            raise ValueError(error_msg)
        return int(num_frames)

    @property
    def _page_height(self):
        match = re.search(r'image_length \([^)]*\) (?P<page_height>.*)', self.header)
        page_height = int(match.group('page_height')) if match else None
        return page_height

    @property
    def _page_width(self):
        match = re.search(r'image_width \([^)]*\) (?P<page_width>.*)', self.header)
        page_width = int(match.group('page_width')) if match else None
        return page_width

    @property
    def is_multiROI(self):
        """Only True if mroiEnable exists (2016b and up) and is set to True."""
        match = re.search(r'hRoiManager\.mroiEnable = (?P<is_multiROI>.)', self.header)
        is_multiROI = (match.group('is_multiROI') == '1') if match else False
        return is_multiROI

    @property
    def is_bidirectional(self):
        match = re.search(r'hScan2D\.bidirectional = (?P<is_bidirectional>.*)', self.header)
        is_bidirectional = (match.group('is_bidirectional') == 'true') if match else False
        return is_bidirectional

    @property
    def seconds_per_line(self):
        match = re.search(r'hRoiManager\.linePeriod = (?P<secs_per_line>.*)', self.header)
        seconds_per_line = float(match.group('secs_per_line')) if match else None
        return seconds_per_line

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
    def uses_fastZ(self):
        match = re.search(r'hFastZ\.enable = (?P<uses_fastZ>.*)', self.header)
        uses_fastZ = (match.group('uses_fastZ') in ['true', '1']) if match else None
        return uses_fastZ

    @property
    def num_requested_frames(self):
        if self.uses_fastZ:
            match = re.search(r'hFastZ\.numVolumes = (?P<num_frames>.*)', self.header)
        else:
            match = re.search(r'hStackManager\.framesPerSlice = (?P<num_frames>.*)',
                              self.header)
        num_requested_frames = int(match.group('num_frames')) if match else None
        return num_requested_frames

    @property
    def zstep_in_microns(self):
        match = re.search(r"hStackManager\.stackZStepSize = (?P<zstep>.*)", self.header)
        zstep_in_microns = float(match.group('zstep')) if match else None
        return zstep_in_microns

    @property
    def scanner_type(self):
        match = re.search(r"hScan2D\.scannerType = '(?P<scanner_type>.*)'", self.header)
        scanner_type = match.group('scanner_type') if match else None
        return scanner_type

    @property
    def scanner_frequency(self):
        match = re.search(r'hScan2D\.scannerFrequency = (?P<scanner_freq>.*)', self.header)
        scanner_frequency = float(match.group('scanner_freq')) if match else None
        return scanner_frequency

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

    def read_data(self, filenames, dtype):
        """ Set self.header, self.filenames and self.dtype. Data is read lazily when needed.
        
        Args:
            filenames: List of strings. Tiff filenames.
            dtype: Data type of the output array.
        """
        # Set header (used to read ScanImage metadata information).
        tiff_file = TiffFile(filenames[0], pages=[0])
        self.header = tiff_file.info()

        # Set dtype of readed data
        self.dtype=dtype

        # Set filenames
        self.filenames = filenames

    def __array__(self):
        return self[:]

    def __iter__(self):
        return self

    def __next__(self):
        if self._field_for_iter < self.num_fields:
            next_field = self[self._field_for_iter]
            self._field_for_iter += 1
            return next_field
        else:
            raise StopIteration

    def __str__(self):
        msg = '{}\n{}\n{}'.format(type(self), '*' * 80, self.header, '*' * 80)
        return msg

    def __len__(self):
        return 0 if self.num_fields is None else self.num_fields

    def __getitem__(self, key):
        """ Index scans by field, y, x, channels, frames. Supports integer, slice and 
        array/tuple/list of integers as indices."""
        raise NotImplementedError('Subclasses of BaseScan must implement this method')

    def _read_pages(self, slice_list ,channel_list, frame_list):
        """ Read the pages with the content of each slice, channel, frame combination.
        
        Each tiff page holds a single depth/channel/frame combination. Channels change 
        first, slices/depths change second and timeframes change last.
        
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
            
        Returns:
            A 5-D array (num_slices, page_height, page_width, num_channels, num_frames).
                Required pages reshaped to have slice, channel and frame as different
                dimensions. Channel, slice and frame order received in the input is 
                respected; for instance, if slice_list = [1, 0, 2, 0], then the first 
                dimension will have four slices: [1, 0, 2, 0]. 
        """
        # Compute pages to load from tiff files (a bit dirty but does the trick)
        slice_step = self.num_channels
        frame_step = self.num_channels * self.num_scanning_depths
        pages_to_read = []
        for frame in frame_list:
            for slice in slice_list:
                for channel in channel_list:
                    new_index = frame * frame_step + slice * slice_step + channel
                    pages_to_read.append(new_index)

        # Read pages
        pages = np.empty([len(pages_to_read), self._page_height, self._page_width],
                         dtype=self.dtype)
        start_page = 0
        for tiff_file in self.tiff_files:

            # Get indices in output array (pages) and in this tiff file
            final_page_in_file = start_page + len(tiff_file)
            is_page_in_file = lambda page: page in range(start_page, final_page_in_file)
            pages_in_file = filter(is_page_in_file, pages_to_read)
            global_indices = [is_page_in_file(page) for page in pages_to_read]
            file_indices = [page - start_page for page in pages_in_file]

            # Read from this tiff file
            if len(file_indices) > 0: # maybe we don't read anything in this file
                pages[global_indices] = tiff_file.asarray(key=file_indices)
            start_page += len(tiff_file)

        # Reshape the pages into slices, y, x, channels, frames
        new_shape = (len(frame_list), len(slice_list), len(channel_list),
                     self._page_height, self._page_width)
        pages = pages.reshape(new_shape).transpose([1, 3, 4, 2, 0])

        return pages

class ScanLegacy(BaseScan):
    """Scan versions 4 and below. Not implemented. """

    def __init__(self):
        raise NotImplementedError('Legacy scans not supported')

class BaseScan5(BaseScan):
    """ScanImage 5 scans. 
    Only one field per scanning depth and all fields have the same y, x dimensions."""

    def __init__(self):
        super().__init__()

    @property
    def num_fields(self):
        return len(self.scanning_depths) # one field per scanning depth

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

    def __getitem__(self, key):
        """ In non-multiROI, all fields have the same x, y dimensions. """
        # Fill key to size 5 (raises IndexError if more than 5)
        full_key = utils.fill_key(key, num_dimensions=5)

        # Check index types are valid
        for i, index in enumerate(full_key):
            utils.check_index_type(i, index)

        # Check each dimension is in bounds
        max_dimensions = self.shape
        for i, (index, max_dimension) in enumerate(zip(full_key, max_dimensions)):
            utils.check_index_is_in_bounds(i, index, max_dimension)

        # Get slices/scanning_depths, channels and frames as lists
        slice_list = utils.listify_index(full_key[0], self.num_scanning_depths)
        channel_list = utils.listify_index(full_key[3], self.num_channels)
        frame_list = utils.listify_index(full_key[4], self.num_frames)

        # Edge case when slice index gives 0 elements or index is empty list, e.g., scan[10:0], scan[[]]
        if [] in [slice_list, channel_list, frame_list]:
            return np.empty(0)

        # Read the required pages
        pages = self._read_pages(slice_list, channel_list, frame_list)

        # Index on y, x
        item = pages[:, full_key[1], full_key[2], :, :]

        # If original index was an integer, delete that axis (as in numpy indexing)
        int_indices = [i for i, index in enumerate(full_key) if isinstance(index, int)]
        int_indices = filter(lambda index: index not in [1, 2], int_indices) # ignore y, x which are applied as int already
        item = np.squeeze(item, axis=tuple(int_indices))

        return item


class Scan5Point1(BaseScan5):
    """ ScanImage 5.1. Basic."""
    pass


class Scan5Point2(BaseScan5):
    """ ScanImage 5.2. Addition of FOV measures in microns."""
    @property
    def image_height_in_microns(self):
        match = re.search(r'hRoiManager\.imagingFovUm = (?P<fov_corners>.*)', self.header)
        image_height_in_microns = None
        if match:
            fov_corners = matlabstr2py(match.group('fov_corners'))
            image_height_in_microns = fov_corners[2][1] - fov_corners[1][1]  # y1-y0
        return image_height_in_microns

    @property
    def image_width_in_microns(self):
        match = re.search(r'hRoiManager\.imagingFovUm = (?P<fov_corners>.*)', self.header)
        image_width_in_microns = None
        if match:
            fov_corners = matlabstr2py(match.group('fov_corners'))
            image_width_in_microns = fov_corners[1][0] - fov_corners[0][0] # x1-x0
        return image_width_in_microns


class Scan2016b(Scan5Point2):
    """ ScanImage 2016b. Same as 5.2"""
    pass


class ScanMultiROI(BaseScan):
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
    def field_widths(self):
        return [field.width for field in self.fields]

    @property
    def field_heights(self):
        return [field.height for field in self.fields]

    @property
    def field_depths(self):
        return [field.depth for field in self.fields]

    @property
    def _fly_to_seconds(self):
        match = re.search(r'hScan2D\.flytoTimePerScanfield = (?P<fly_to_seconds>.*)',
                          self.header)
        fly_to_seconds = float(match.group('fly_to_seconds')) if match else None
        return fly_to_seconds

    @property
    def num_fly_to_lines(self):
        """ Number of lines recorded in the tiff page while flying to a different field, 
        i.e., distance between fields in the tiff page."""
        num_fly_to_lines = self._fly_to_seconds / self.seconds_per_line
        num_fly_to_lines = int(np.ceil(num_fly_to_lines))
        if self.is_bidirectional:
            # line scanning always starts at one end of the image so this has to be even
            num_fly_to_lines += (num_fly_to_lines % 2)
        return num_fly_to_lines

    def _degrees_to_microns(self, degrees):
        """ Convert scan angle degrees to microns using the objective resolution."""
        match = re.search(r'objectiveResolution = (?P<deg2um_factor>.*)', self.header)
        microns = (degrees * float(match.group('deg2um_factor'))) if match else None
        return microns

    @property
    def field_heights_in_microns(self):
        field_heights_in_degrees = [field.height_in_degrees for field in self.fields]
        return [self._degrees_to_microns(deg) for deg in field_heights_in_degrees]

    @property
    def field_widths_in_microns(self):
        field_widths_in_degrees = [field.width_in_degrees for field in self.fields]
        return [self._degrees_to_microns(deg) for deg in field_widths_in_degrees]

    def read_data(self, filenames, dtype):
        """ Set the header, create rois and fields (joining them if necessary)"""
        super().read_data(filenames, dtype)
        self._create_rois()
        self._compute_fields()
        if self.join_contiguous:
            self._join_contiguous_fields()

    def _create_rois(self):
        """Create scan rois from the configuration file. """
        scanimage_metadata = self.tiff_files[0].scanimage_metadata
        roi_cfgs = scanimage_metadata['RoiGroups']['imagingRoiGroup']['rois']

        self.rois = []
        for roi_cfg in roi_cfgs:
            new_roi = ROI(roi_cfg)
            self.rois.append(new_roi)

    def _compute_fields(self):
        """ Go over each slice depth and each roi generating the scanned fields. """
        self.fields = []
        for scanning_depth in self.scanning_depths:
            starting_y = 0
            for roi in self.rois:
                new_field = roi.get_field_at(scanning_depth)

                if new_field is not None:
                    if starting_y + new_field.height > self._page_height:
                        error_msg = ('Overestimated number of fly to lines ({}) at '
                                     'scanning depth {}'.format(self.num_fly_to_lines,
                                                                scanning_depth))
                        raise RuntimeError(error_msg)

                    # Set xslice and yslice (from where in the page to cut it)
                    new_field.yslices = [slice(starting_y, starting_y + new_field.height)]
                    new_field.xslices = [slice(0, new_field.width)]

                    # Set output xslice and yslice (where to paste it in output)
                    new_field.output_yslices = [slice(0, new_field.height)]
                    new_field.output_xslices = [slice(0, new_field.width)]

                    # Compute next starting y
                    starting_y += new_field.height + self.num_fly_to_lines

                    # Add field to fields
                    self.fields.append(new_field)

    def _join_contiguous_fields(self):
        """In each scanning depth, go over all fields joining those that are contiguous.
        
        When two fields are joined, the one appearing last in the fields list is deleted 
        and info such as field height, field width and slices is changed in the first 
        field.
        
        Any rectangular area in the scan formed by the union of two or more fields will 
        be considered as a single field after this operation. 
        """
        for scanning_depth in self.scanning_depths:
            two_fields_were_joined = True
            while(two_fields_were_joined): # repeat until no fields were joined
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

    def field_to_slice(self, field_id):
        """Given a field id return the corresponding slice id. """
        field_depth =  self.fields[field_id].depth
        return self.scanning_depths.index(field_depth)

    def __getitem__(self, key):
        # Fill key to size 5 (raises IndexError if more than 5)
        full_key = utils.fill_key(key, num_dimensions=5)

        # Check index types are valid
        for i, index in enumerate(full_key):
            utils.check_index_type(i, index)

        # Check each dimension is in bounds
        utils.check_index_is_in_bounds(0, full_key[0], self.num_fields)
        for field_id in utils.listify_index(full_key[0], self.num_scanning_depths):
            utils.check_index_is_in_bounds(1, full_key[1], self.field_heights[field_id])
            utils.check_index_is_in_bounds(2, full_key[2], self.field_widths[field_id])
        utils.check_index_is_in_bounds(3, full_key[3], self.num_channels)
        utils.check_index_is_in_bounds(4, full_key[4], self.num_frames)

        # Get slices/scanning_depths, channels and frames as lists
        field_list = utils.listify_index(full_key[0], self.num_fields)
        slice_list = [self.field_to_slice(field_id) for field_id in field_list]
        channel_list = utils.listify_index(full_key[3], self.num_channels)
        frame_list = utils.listify_index(full_key[4], self.num_frames)
        ys_list = [utils.listify_index(full_key[1], self.field_heights[field_id])
                   for field_id in field_list]
        xs_list = [utils.listify_index(full_key[2], self.field_widths[field_id])
                   for field_id in field_list]

        # Edge case when slice index gives 0 elements or index is empty list, e.g., scan[10:0], scan[[]]
        if [] in [slice_list, channel_list, frame_list]:
            return np.empty(0)

        # Check output heights and widths match for all fields
        if not all(len(ys) == len(ys_list[0]) for ys in ys_list):
            raise FieldDimensionMismatch('Image heights for all fields do not match')
        if not all(len(xs) == len(xs_list[0]) for xs in xs_list):
            raise FieldDimensionMismatch('Image widths for all fields do not match')

        # Read the required pages
        pages = self._read_pages(slice_list, channel_list, frame_list)

        # Slice each field (this is not so memory efficient)
        output_shape = (pages.shape[0], len(ys_list[0]), len(xs_list[0]), pages.shape[3],
                        pages.shape[4])
        item = np.empty(output_shape)
        for i, (field_id, y_list, x_list) in enumerate(zip(field_list, ys_list, xs_list)):
            field = self.fields[field_id]

            # Over each subfield in field (only one for non-contiguous fields)
            slices = zip(field.xslices, field.yslices, field.output_xslices,
                         field.output_yslices)
            for xslice, yslice, output_xslice, output_yslice in slices:

                # Get x, ys indices that need to be accessed in this subfield
                x_range = range(output_xslice.start, output_xslice.stop)
                y_range = range(output_yslice.start, output_yslice.stop)
                xs = [x - output_xslice.start + xslice.start for x in x_list if x in x_range]
                ys = [[y - output_yslice.start + yslice.start] for y in y_list if y in y_range]
                output_xs = [index for index, x in enumerate(x_list) if x in x_range]
                output_ys = [[index] for index, y in enumerate(y_list) if y in y_range]
                # ys as nested lists are needed for numpy to slice them correctly

                item[i, output_ys, output_xs] = pages[i, ys, xs]

            # Old version: no contiguous fields
            #ys = [[field.yslice.start + y] for y in y_list] # nested list needed so numpy slices it as I want
            #xs = [field.xslice.start + x for x in x_list]
            #item[i] = pages[i, ys, xs]

        # If original index was an integer, delete that axis  (as in numpy indexing)
        int_indices = [i for i, index in enumerate(full_key) if isinstance(index, int)]
        item = np.squeeze(item, axis=tuple(int_indices))

        return item