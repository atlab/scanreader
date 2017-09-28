"""
ScanImage stacks. Each version handles things a little differently. Stack objects are
usually instantiated by a call to scanreader.read_stack().

Stacks are different from scan in that all frames for one slice are recorded first before
moving to the next slice.

Hierarchy:
BaseStack (MRO: BaseStack, BaseScan)
    Stack5Point1 (MRO: Stack5Point1, Scan5Point1, BaseScan5, BaseStack, BaseScan)
    Stack5Point2 (MRO: Stack5Point2, Scan5Point2, BaseScan5, BaseStack, BaseScan)
    Stack2016b (MRO: Stack2016b, Scan2016b, Scan5Point2, BaseScan5, BaseStack, BaseScan)
    StackMultiRoi (MRO: StackMultiROI, ScanMultiROI, BaseStack, BaseScan)
"""
import numpy as np
import re

from . import scans
from . import utils

class BaseStack(scans.BaseScan):
    """ Properties and methods shared among all stack versions."""
    @property
    def num_frames(self):
        return int(self.num_requested_frames / self._num_averaged_frames)

    @property
    def num_scanning_depths(self):
        """ Number of scanning depths actually recorded in this stack."""
        num_pages = sum([len(tiff_file) for tiff_file in self.tiff_files])
        num_scanning_depths = num_pages / (self.num_frames * self.num_channels)
        return int(num_scanning_depths) # discard last slice if incomplete

    @property
    def requested_scanning_depths(self):
        return super().scanning_depths

    @property
    def scanning_depths(self):
        return self.requested_scanning_depths[:self.num_scanning_depths]

    @property
    def _num_averaged_frames(self):
        """ How many requested frames are averaged to form one saved frame. """
        match = re.search(r'hScan2D\.logAverageFactor = (?P<num_avg_frames>.*)', self.header)
        num_averaged_frames = int(match.group('num_avg_frames')) if match else None
        return num_averaged_frames

    def _read_pages(self, slice_list, channel_list, frame_list, yslice=slice(None),
                    xslice=slice(None)):
        """ Reads the tiff pages with the content of each slice, channel, frame
        combination and slices them in the y, x dimension.

        Each tiff page holds a single depth/channel/frame combination. Channels change
        first, timeframes change second and slices/depths change last.
        Example:
            For two channels, three slices, two frames.
            Page:       0   1   2   3   4   5   6   7   8   9   10  11
            Channel:    0   1   0   1   0   1   0   1   0   1   0   1
            Frame:      0   0   1   1   2   2   0   0   1   1   2   2
            Slice:      0   0   0   0   0   0   1   1   1   1   1   1

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
        frame_step = self.num_channels
        slice_step = self.num_channels * self.num_frames
        pages_to_read = []
        for frame in frame_list:
            for slice_ in slice_list:
                for channel in channel_list:
                    new_page = slice_ * slice_step + frame * frame_step + channel
                    pages_to_read.append(new_page)

        # Compute output dimensions
        out_height = len(utils.listify_index(yslice, self._page_height))
        out_width = len(utils.listify_index(xslice, self._page_width))

        # Read pages
        pages = np.empty([len(pages_to_read), out_height, out_width], dtype=self.dtype)
        start_page = 0
        for tiff_file in self.tiff_files:

            # Get indices in this tiff file and in output array
            final_page_in_file = start_page + len(tiff_file)
            is_page_in_file = lambda page: page in range(start_page, final_page_in_file)
            pages_in_file = filter(is_page_in_file, pages_to_read)
            file_indices = [page - start_page for page in pages_in_file]
            global_indices = [is_page_in_file(page) for page in pages_to_read]

            # Read from this tiff file (if needed)
            if len(file_indices) > 0:
                # this line looks a bit ugly but is memory efficient. Do not separate
                pages[global_indices] = tiff_file.asarray(key=file_indices)[..., yslice, xslice]
            start_page += len(tiff_file)

        # Reshape the pages into (slices, y, x, channels, frames)
        new_shape = [len(frame_list), len(slice_list), len(channel_list), out_height, out_width]
        pages = pages.reshape(new_shape).transpose([1, 3, 4, 2, 0])

        return pages


class Stack5Point1(scans.Scan5Point1, BaseStack):
    pass


class Stack5Point2(scans.Scan5Point2, BaseStack):
    @property
    def field_offsets(self):
        raise AttributeError('Field offsets are not properly calculated for stacks.')


class Stack2016b(scans.Scan2016b, BaseStack):
    @property
    def field_offsets(self):
        raise AttributeError('Field offsets are not properly calculated for stacks.')


class StackMultiROI(scans.ScanMultiROI, BaseStack):
    @property
    def field_offsets(self):
        raise AttributeError('Field offsets are not properly calculated for stacks.')