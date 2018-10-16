""" Some classes used for MultiROI scan processing. """
import numpy as np


class ROI:
    """ Holds ROI info and computes an xy plane (scanfield) at a given z.

    ScanImage defines a ROI as the interpolation between a set of scanfields. See their
    docs for details.
    """
    def __init__(self, roi_info):
        """ Read the scanfields that define this ROI and other required info.

        Args:
            roi_info: A dictionary containing the definition of the roi extracted from the
                tiff header.
        """
        self.roi_info = roi_info
        self._scanfields = None

    @property
    def scanfields(self):
        if self._scanfields is None:
            self._scanfields = self._create_scanfields()
        return self._scanfields

    @property
    def is_discrete_plane_mode_on(self):
        return bool(self.roi_info['discretePlaneMode'])

    def _create_scanfields(self):
        """ Create all the scanfields that form this ROI. """
        # Get scanfield configuration info
        scanfield_infos = self.roi_info['scanfields']
        if not isinstance(scanfield_infos, list):
            scanfield_infos = [scanfield_infos] # make list if single scanfield

        # Get scanfield depths
        scanfield_depths = self.roi_info['zs']
        if not isinstance(scanfield_depths, list):
            scanfield_depths = [scanfield_depths]

        scanfields = []
        for scanfield_info, scanfield_depth in zip(scanfield_infos, scanfield_depths):
            # if scanfield_info['enable']: # this is always 1 even if ROI is disabled
            # Get scanfield info
            width, height = scanfield_info['pixelResolutionXY']
            xcenter, ycenter = scanfield_info['centerXY']
            size_in_x, size_in_y = scanfield_info['sizeXY']

            # Create scanfield
            new_scanfield = Scanfield(height=height, width=width, depth=scanfield_depth,
                                      y=ycenter, x=xcenter, height_in_degrees=size_in_y,
                                      width_in_degrees=size_in_x)
            scanfields.append(new_scanfield)

        # Sort them by depth (to ease interpolation)
        scanfields = sorted(scanfields, key=lambda scanfield: scanfield.depth)

        return scanfields

    def get_field_at(self, scanning_depth):
        """ Interpolates between the ROI scanfields to generate the 2-d field at the
        desired depth.

        Args:
            scanning_depth: An integer. Depth at which we want to obtain the field.

        Returns:
            field. A Field object, the desired field at scanning_depth

        Warning:
            Does not work for rotated ROIs.
            If there were more than one scanfield at the same depth, it will only consider
                the one defined last.
        """
        field = None

        if self.is_discrete_plane_mode_on: # only check at each scanfield depth
            for scanfield in self.scanfields:
                if scanning_depth == scanfield.depth:
                    field = scanfield.as_field()
        else:
            if len(self.scanfields) == 1: # single scanfield extending from -inf to inf
                field = self.scanfields[0].as_field()
                field.depth = scanning_depth

            else: # interpolate between scanfields
                scanfield_depths = [sf.depth for sf in self.scanfields]
                valid_range = range(min(scanfield_depths), max(scanfield_depths) + 1)
                if scanning_depth in valid_range:
                    field = Field()

                    scanfield_heights = [sf.height for sf in self.scanfields]
                    field.height = np.interp(scanning_depth, scanfield_depths,
                                             scanfield_heights)
                    field.height = int(round(field.height / 2)) * 2 # round to the closest even

                    scanfield_widths = [sf.width for sf in self.scanfields]
                    field.width = np.interp(scanning_depth, scanfield_depths,
                                            scanfield_widths)
                    field.width = int(round(field.width / 2)) * 2 # round to the closest even

                    field.depth = scanning_depth

                    scanfield_ys = [sf.y for sf in self.scanfields]
                    field.y = np.interp(scanning_depth, scanfield_depths, scanfield_ys)

                    scanfield_xs = [sf.x for sf in self.scanfields]
                    field.x = np.interp(scanning_depth, scanfield_depths, scanfield_xs)

                    scanfield_heights = [sf.height_in_degrees for sf in self.scanfields]
                    field.height_in_degrees = np.interp(scanning_depth, scanfield_depths,
                                                        scanfield_heights)

                    scanfield_widths = [sf.width_in_degrees for sf in self.scanfields]
                    field.width_in_degrees = np.interp(scanning_depth, scanfield_depths,
                                                       scanfield_widths)

        return field


class Scanfield:
    """ Small container for scanfield information. Used to define ROIs.

    Attributes:
        height: height of the field in pixels.
        width: width of the field in pixels.
        depth: depth at which this field was recorded (in microns relative to absolute z).
        y, x: Coordinates of the center of the field in the scan (in scan angle degrees).
        height_in_degrees: height of the field in degrees of the scan angle.
        width_in_degrees: width of the field in degrees of the scan angle.
    """
    def __init__(self, height=None, width=None, depth=None, y=None, x=None,
                 height_in_degrees=None, width_in_degrees=None):
        self.height = height
        self.width = width
        self.depth = depth
        self.y = y
        self.x = x
        self.height_in_degrees = height_in_degrees
        self.width_in_degrees = width_in_degrees

    def as_field(self):
        return Field(height=self.height, width=self.width, depth=self.depth, y=self.y,
                     x=self.x, height_in_degrees=self.height_in_degrees,
                     width_in_degrees=self.width_in_degrees)


class Field(Scanfield):
    """ Two-dimensional scanning plane. An extension of scanfield with some functionality.

    Attributes:
        height: height of the field in pixels.
        width: width of the field in pixels.
        depth: depth at which this field was recorded (in microns relative to absolute z).
        y, x: Coordinates of the center of the field in the scan (in scan angle degrees).
        height_in_degrees: height of the field in degrees of the scan angle.
        width_in_degrees: width of the field in degrees of the scan angle.
        yslices: list of slices. How to slice the page in the y axis to get this field.
        xslices: list of slices. How to slice the page in the x axis to get this field.
            For now, all fields have the same width so all xslices are slice(None).
        output_yslices: list of slices. Where to paste this field in the output field.
        output_xslices: list of slices. Where to paste this field in the output field.
        slice_id: index of the slice in the scan to which this field belongs.
        roi_ids: list of ROI indices to which each subfield belongs (one if single field).
        offsets: list of masks with time offsets per pixel (seconds, one if single field).

    Example:
        output_field[output_yslice, output_xslice] = page[yslice, xslice]

    When a field is formed by joining two or more subfields (via join_contiguous), the
    slice lists hold two or more slices representing where each subfield will be taken
    from the page and inserted in the (joint) output field. Attributes height, width, x,
    y, height_in_degrees and width_in_degrees are adjusted accordingly. For non-contiguous
    fields, each slice list has a single slice.

     Note:
        Slices in xslices, yslices, output_xslices and output_yslices hold two promises:
            step = 1 (fields are contiguous)
            stop = start + height|width
        In theory, we only need x_start and y_start but slices simplify operations.

    """
    def __init__(self, height=None, width=None, depth=None, y=None, x=None,
                 height_in_degrees=None, width_in_degrees=None, yslices=None,
                 xslices=None, output_yslices=None, output_xslices=None, slice_id=None,
                 roi_ids=None, offsets=None):
        self.height = height
        self.width = width
        self.depth = depth
        self.y = y
        self.x = x
        self.height_in_degrees = height_in_degrees
        self.width_in_degrees = width_in_degrees
        self.yslices = yslices
        self.xslices = xslices
        self.output_yslices = output_yslices
        self.output_xslices = output_xslices
        self.slice_id = slice_id
        self.roi_ids = roi_ids
        self.offsets = offsets

    @property
    def has_contiguous_subfields(self):
        """ Whether field is formed by many contiguous subfields. """
        return len(self.xslices) > 1

    @property
    def roi_mask(self):
        """ Mask of the size of the field. Each pixel shows the ROI from where it comes."""
        mask = np.full([self.height, self.width], -1, dtype=np.int8)
        for roi_id, output_yslice, output_xslice in zip(self.roi_ids, self.output_yslices,
                                                        self.output_xslices):
            mask[output_yslice, output_xslice] = roi_id
        return mask

    @property
    def offset_mask(self):
        """ Mask of the size of the field. Each pixel shows its time offset in seconds."""
        mask = np.full([self.height, self.width], -1, dtype=np.float32)
        for offsets, output_yslice, output_xslice in zip(self.offsets, self.output_yslices,
                                                        self.output_xslices):
            mask[output_yslice, output_xslice] = offsets
        return mask

    def _type_of_contiguity(self, field2):
        """ Compute how field 2 is contiguous to this one.

        Args:
            field2: A second field object.

        Returns:
            An integer {NONCONTIGUOUS = 0, ABOVE = 1, BELOW = 2, LEFT = 3, RIGHT = 4}.
               Whether field 2 is above, below, to the left or to the right of this field.
        """
        position = Position.NONCONTIGUOUS
        if np.isclose(self.width_in_degrees, field2.width_in_degrees):
            expected_distance = self.height_in_degrees / 2 + field2.height_in_degrees / 2
            if np.isclose(self.y, field2.y + expected_distance):
                position = Position.ABOVE
            if np.isclose(field2.y, self.y + expected_distance):
                position = Position.BELOW
        if np.isclose(self.height_in_degrees, field2.height_in_degrees):
            expected_distance = self.width_in_degrees / 2 + field2.width_in_degrees / 2
            if np.isclose(self.x, field2.x + expected_distance):
                position = Position.LEFT
            if np.isclose(field2.x, self.x + expected_distance):
                position = Position.RIGHT

        return position

    def is_contiguous_to(self, field2):
        """ Whether this field is contiguous to field2."""
        return not (self._type_of_contiguity(field2) == Position.NONCONTIGUOUS)

    def join_with(self, field2):
        """ Update attributes of this field to incorporate field2. Field2 is NOT changed.

        Args:
            field2: A second field object.
        """
        contiguity = self._type_of_contiguity(field2)
        if contiguity in [Position.ABOVE, Position.BELOW]:  # contiguous in y axis
            # Compute some specific attributes
            if contiguity == Position.ABOVE:  # field2 is above/atop self
                new_y = field2.y + (self.height_in_degrees / 2)
                output_yslices1 = [slice(s.start + field2.height, s.stop + field2.height)
                                   for s in self.output_yslices]
                output_yslices2 = field2.output_yslices
            else:  # field2 is below self
                new_y = self.y + (field2.height_in_degrees / 2)
                output_yslices1 = self.output_yslices
                output_yslices2 = [slice(s.start + self.height, s.stop + self.height) for
                                   s in field2.output_yslices]

            # Set new attributes
            self.y = new_y
            self.height += field2.height
            self.height_in_degrees += field2.height_in_degrees
            self.output_yslices = output_yslices1 + output_yslices2
            self.output_xslices = self.output_xslices + field2.output_xslices

        if contiguity in [Position.LEFT, Position.RIGHT]:  # contiguous in x axis
            # Compute some specific attributes
            if contiguity == Position.LEFT:  # field2 is to the left of self
                new_x = field2.x + (self.width_in_degrees / 2)
                output_xslices1 = [slice(s.start + field2.width, s.stop + field2.width)
                                   for s in self.output_xslices]
                output_xslices2 = field2.output_xslices
            else:  # field2 is to the right of self
                new_x = self.x + (field2.width_in_degrees / 2)
                output_xslices1 = self.output_xslices
                output_xslices2 = [slice(s.start + self.width, s.stop + self.width) for s
                                   in field2.output_xslices]

            # Set new attributes
            self.x = new_x
            self.width += field2.width
            self.width_in_degrees += field2.width_in_degrees
            self.output_yslices = self.output_yslices + field2.output_yslices
            self.output_xslices = output_xslices1 + output_xslices2

        # yslices and xslices just get appended regardless of the type of contiguity
        self.yslices = self.yslices + field2.yslices
        self.xslices = self.xslices + field2.xslices

        # Append roi ids and offsets
        self.roi_ids = self.roi_ids + field2.roi_ids
        self.offsets = self.offsets + field2.offsets


class Position:
    NONCONTIGUOUS = 0
    ABOVE = 1
    BELOW = 2
    LEFT = 3
    RIGHT = 4