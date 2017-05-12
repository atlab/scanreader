""" Some classes used for MultiROI scan processing. """
import numpy as np

class ROI:
    """ Holds ROI info and computes the xy plane at a given z.
    
    ScanImage defines a ROI as the interpolation between a set of scanfields. See their 
    docs for details."""

    def __init__(self, roi_info):
        """ Read the scanfields that define this ROI and other required info.
            
        Args:
            roi_info: A dictionary containing the definition of the roi extracted from the
                tiff header.
        """
        self.roi_info = roi_info
        self._scanfields = None

    @property
    def is_discrete_plane_mode_on(self):
        return bool(self.roi_info['discretePlaneMode'])

    @property
    def scanfields(self):
        if self._scanfields is None:
            self._create_scanfields()
        return self._scanfields

    def _create_scanfields(self):
        # Get scanfield configuration info
        scanfield_cfgs = self.roi_info['scanfields']
        if not isinstance(scanfield_cfgs, list):
            scanfield_cfgs = [scanfield_cfgs] # make list if single scanfield

        # Get scanfield depths
        scanfield_depths = self.roi_info['zs']
        if not isinstance(scanfield_depths, list):
            scanfield_depths = [scanfield_depths]

        self._scanfields = []
        for scanfield_info, scanfield_depth in zip(scanfield_cfgs, scanfield_depths):
            # if scanfield_info['enable']: # this is always 1 even if ROI is disabled
            # Get scanfield info
            width, height = scanfield_info['pixelResolutionXY']
            xcenter, ycenter = scanfield_info['centerXY']
            size_in_x, size_in_y = scanfield_info['sizeXY']

            # Create scanfield
            new_scanfield = Field(height=height, width=width, depth=scanfield_depth,
                                  x=xcenter, y=ycenter, width_in_degrees=size_in_x,
                                  height_in_degrees=size_in_y)
            self._scanfields.append(new_scanfield)

        # Sort them by depth (to ease interpolation)
        self._scanfields = sorted(self._scanfields, key=lambda field: field.depth)

    def get_field_at(self, scanning_depth):
        """ Interpolates between the ROI scanfields to generate the 2-d field at the 
        desired depth. 
        
        Will not work for rotated ROIs. If there is more than one scanfield at the same
        depth, it will only consider the one defined last. 
        
        Args:
            scanning_depth: An integer. Depth at which we want to obtain the field.
        """
        field = None

        if self.is_discrete_plane_mode_on: # only check at each scanfield depth
            for scanfield in self.scanfields:
                if scanning_depth == scanfield.depth:
                    field = scanfield.copy()
        else:
            if len(self.scanfields) == 1: # single scanfield extending from -inf to inf
                field = self.scanfields[0].copy()
                field.depth = round(scanning_depth)

            else: # interpolate between scanfield
                scanfield_depths = [s.depth for s in self.scanfields]
                valid_range = range(min(scanfield_depths), max(scanfield_depths) + 1)
                if scanning_depth in valid_range:
                    field = Field()

                    scanfield_heights = [s.height for s in self.scanfields]
                    field.height = np.interp(scanning_depth, scanfield_depths,
                                             scanfield_heights)
                    field.height = round(field.height / 2) * 2 # round to the closest even

                    scanfield_widths = [s.width for s in self.scanfields]
                    field.width = np.interp(scanning_depth, scanfield_depths,
                                            scanfield_widths)
                    field.width = round(field.width / 2) * 2 # round to the closest even

                    field.depth = round(scanning_depth)

                    scanfield_xs = [s.x for s in self.scanfields]
                    field.x = np.interp(scanning_depth, scanfield_depths, scanfield_xs)

                    scanfield_ys = [s.y for s in self.scanfields]
                    field.y = np.interp(scanning_depth, scanfield_depths, scanfield_ys)

                    scanfield_widths = [s.width_in_degrees for s in self.scanfields]
                    field.width_in_degrees = np.interp(scanning_depth, scanfield_depths,
                                                       scanfield_widths)

                    scanfield_heights = [s.height_in_degrees for s in self.scanfields]
                    field.height_in_degrees = np.interp(scanning_depth, scanfield_depths,
                                                 scanfield_heights)

        return field


class Field:
    """ Small container for field information. 
    
    Attributes:
        height: height of the field in pixels.
        width: width of the field in pixels.
        depth: depth at which this field was recorded (in microns relative to absolute z).
        yslices: list of slices. How to slice the page in the y axis to get this field.
        xslices: list of slices. How to slice the page in the x axis to get this field.
            For now, all fields have the same width so all xslices are slice(None).
        output_yslices: list of slices. Where to paste this field in the output field.
        output_xslices: list of slices. Where to paste this field in the output field.
        x, y: Coordinates of the center of the field in the scan (in scan angle degrees).
        height_in_degrees: height of the field in degrees of the scan angle.
        width_in_degrees: width of the field in degrees of the scan angle.
        
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
            stop = start + height/width 
        In theory, we only need x_start and y_start but slices simplify operations.
        
    """
    def __init__(self, height=None, width=None, depth=None, yslices=None, xslices=None,
                 output_yslices=None, output_xslices=None, x=None, y=None,
                 width_in_degrees=None, height_in_degrees=None):
        self.height = height
        self.width = width
        self.depth = depth
        self.xslices = xslices
        self.yslices = yslices
        self.output_xslices = output_xslices
        self.output_yslices = output_yslices
        self.x = x
        self.y = y
        self.width_in_degrees = width_in_degrees
        self.height_in_degrees = height_in_degrees

    @property
    def has_contiguous_subfields(self):
        """ Whether field is formed by many contiguous subfields. """
        return len(self.xslices) > 1

    def copy(self):
        return Field(height=self.height, width=self.width, depth=self.depth,
                     yslices=self.yslices, xslices=self.xslices,
                     output_yslices=self.output_yslices,
                     output_xslices=self.output_xslices, x=self.x, y=self.y,
                     width_in_degrees=self.width_in_degrees,
                     height_in_degrees=self.height_in_degrees)

    def _type_of_contiguity(self, field2):
        """ Compute how field 2 is contiguous to this one. 

        Args:
            field2: A second field object.
        
        Returns:
            An integer {NONCONTIGUOUS = 0, ABOVE = 1, BELOW = 2, LEFT = 3, RIGHT = 4}. 
               Whether field 2 is above, below, to the left or to the right of this field.
        """
        position = Position.NONCONTIGUOUS
        if self.height_in_degrees == field2.height_in_degrees:
            if self.y == field2.y + field2.height_in_degrees:
                position = Position.ABOVE
            if field2.y == self.y + self.height_in_degrees:
                position = Position.BELOW
        if self.width_in_degrees == field2.width_in_degrees:
            if self.x == field2.x + field2.width_in_degrees:
                position = Position.LEFT
            if field2.x == self.x + self.width_in_degrees:
                position = Position.RIGHT

        return position

    def is_contiguous_to(self, field2):
        """ Whether this field is contiguous to field2."""
        if self._type_of_contiguity(field2) == Position.NONCONTIGUOUS:
            return False
        else:
            return True

    def join_with(self, field2):
        """ Update attributes of this field to incorporate field2. Field2 is NOT changed.
        
        Args:
            field2: A second field object.
        """
        type_of_contiguity = self._type_of_contiguity(field2)
        if type_of_contiguity == Position.ABOVE:  # field2 is above/atop self
            # Update output slices
            self.output_xslices += field2.output_xslices
            updated_yslices = [slice(s.start + field2.height, s.stop + field2.height)
                               for s in self.output_yslices]
            self.output_yslices = updated_yslices + field2.output_yslices

            # Update other attributes
            self.y = field2.y
            self.height_in_degrees += field2.height_in_degrees
            self.height += field2.height

        if type_of_contiguity == Position.BELOW:  # field2 is below self
            # Update output slices
            self.output_xslices += field2.output_xslices
            updated_yslices = [slice(s.start + self.height, s.stop + self.height)
                               for s in field2.output_yslices]
            self.output_yslices += updated_yslices

            # Update other attributes
            self.height_in_degrees += field2.height_in_degrees
            self.height += field2.height

        if type_of_contiguity == Position.LEFT:  # field2 is to the left of self
            # Update output slices
            self.output_yslices += field2.output_yslices
            updated_xslices = [slice(s.start + field2.width, s.stop + field2.width)
                               for s in self.output_xslices]
            self.output_xslices = updated_xslices + field2.output_xslices

            # Update other attributes
            self.x = field2.x
            self.width_in_degrees += field2.width_in_degrees
            self.width += field2.width

        if type_of_contiguity == Position.RIGHT:  # field2 is to the right of self
            # Update output slices
            self.output_yslices += field2.output_yslices
            updated_xslices = [slice(s.start + self.width, s.stop + self.width)
                               for s in field2.output_xslices]
            self.output_xslices += updated_xslices

            # Update other attributes
            self.width_in_degrees += field2.width_in_degrees
            self.width += field2.width

        # These just get appended no matter the type of contiguity
        self.yslices.append(*field2.yslices)
        self.xslices.append(*field2.xslices)


class Position:
    NONCONTIGUOUS = 0
    ABOVE = 1
    BELOW = 2
    LEFT = 3
    RIGHT = 4
