"""Test suite for the different functionality of scanreader. It takes a minute.

Run from inside scanreader/ as: 
    `nose2 test_scanreader` (whole suite)
    `nose2 test_scanreader.ScanTest.test_2020` (a specific test)
"""

from unittest import TestCase
from os import path
import numpy as np
import scanreader
from scanreader.exceptions import ScanReaderException

# Get data directory
data_dir = path.join(path.dirname(path.abspath(__file__)), 'data')
#data_dir = '/mnt/lab/users/ecobost/data/' # 3x slower
print('Extracted data directory:', data_dir)

# These are big files so they are not available in Github
scan_file_5_1 = path.join(data_dir, 'scan_5_1_001.tif') # 2 channels, 3 slices
scan_file_5_2 = path.join(data_dir, 'scan_5_2.tif') # 2 channels, 3 slices
scan_file_5_3 = path.join(data_dir, 'scan_5_3.tif') # 2 channels, 1 slice
scan_file_2020 = path.join(data_dir, 'scan_2020.tif') # 2 channels, 1 slice
scan_file_2016b_multiroi = path.join(data_dir, 'scan_2016b_multiroi_001.tif') # all rois have same dimensions, 1 channel 5 slices
scan_file_2018a_multiroi = path.join(data_dir, 'scan_2018a_multiroi_001.tif') # all rois have same dimensions, 1 channel 3 slices, 5 fields per slice
scan_file_2016b_multiroi_hard = path.join(data_dir, 'scan_2016b_multiroi_hard.tif') # rois have diff dimensions and they are volumes, 2 channels, 3 slices, roi1 at depth1, roi1 and 2 at depth 2, roi 2 at depth 2, thus 4 fields
scan_file_5_1_multifiles = [path.join(data_dir, 'scan_5_1_001.tif'), path.join(data_dir, 'scan_5_1_002.tif')] # second file has less pages
scan_file_2016b_multiroi_multifiles = [path.join(data_dir, 'scan_2016b_multiroi_001.tif'), path.join(data_dir, 'scan_2016b_multiroi_002.tif')]
scan_file_join_contiguous = scan_file_2016b_multiroi

stack_file_5_1 = path.join(data_dir, 'stack_5_1_001.tif') # 2 channels, 60 slices
stack_file_2016b = path.join(data_dir, 'stack_2016b.tif') # 1 channel, 1 slice, mroiEnable=false
stack_file_5_1_multifiles = [path.join(data_dir, 'stack_5_1_001.tif'), path.join(data_dir, 'stack_5_1_002.tif')] # second has 10 slices
stack_file_2016b_multiroi = path.join(data_dir, 'stack_2016b_multiroi_001.tif') # 2 channels, 50 slices, 4 fields per slice


class ScanTest(TestCase):
    """ Test scans from different ScanImage versions. """

    def test_expand_wildcard(self):
        """Testing correct expansion of wildcard input."""
        from scanreader.core import expand_wildcard

        # Single filename
        self.assertEqual(expand_wildcard(scan_file_5_1), [scan_file_5_1])

        # Relative path
        self.assertEqual(expand_wildcard(path.relpath(scan_file_5_1)), [scan_file_5_1])

        # List of filenames
        self.assertEqual(expand_wildcard([scan_file_5_1, scan_file_5_2]),
                         [scan_file_5_1, scan_file_5_2])

        # Single wildcard
        self.assertEqual(expand_wildcard(path.join(data_dir, 'scan_5_1_*.tif')),
                         [scan_file_5_1_multifiles[0], scan_file_5_1_multifiles[1]])

        # List of wildcards
        self.assertEqual(expand_wildcard([path.join(data_dir, 'scan_2016b_multiroi_0*.tif'), path.join(data_dir, 'scan_5_1_0*.tif')]),
                         [scan_file_2016b_multiroi_multifiles[0], scan_file_2016b_multiroi_multifiles[1], scan_file_5_1_multifiles[0], scan_file_5_1_multifiles[1]])


    def test_attributes(self):

        # 5.1
        scan = scanreader.read_scan(scan_file_5_1)
        self.assertEqual(scan.version, '5.1')
        self.assertEqual(scan.is_slow_stack, False)
        self.assertEqual(scan.is_multiROI, False)
        self.assertEqual(scan.num_channels, 2)
        self.assertEqual(scan.requested_scanning_depths, [-5, 10, 25])
        self.assertEqual(scan.num_scanning_depths, 3)
        self.assertEqual(scan.scanning_depths, [-5, 10, 25])
        self.assertEqual(scan.num_requested_frames, 60500)
        self.assertEqual(scan.num_frames, 1000)
        self.assertEqual(scan.is_bidirectional, True)
        self.assertEqual(scan.scanner_frequency, 7920.62)
        self.assertAlmostEqual(scan.seconds_per_line, 6.31264e-05)
        self.assertEqual(scan.num_fields, 3)
        self.assertEqual(scan.field_depths, [-5, 10, 25])
        self.assertEqual(scan.fps, 11.0467)
        self.assertEqual(scan.spatial_fill_fraction, 0.9)
        self.assertEqual(scan.temporal_fill_fraction, 0.712867)
        self.assertEqual(scan.scanner_type, 'Resonant')
        self.assertEqual(scan.motor_position_at_zero, [-1025, -495.5, -202.8])
        self.assertEqual(scan.initial_secondary_z, 0)
        for i, max_offset in enumerate([0.01615707, 0.03421122, 0.05226535]):
            self.assertAlmostEqual(scan.field_offsets[i].max(), max_offset)

        self.assertEqual(scan.image_height, 256)
        self.assertEqual(scan.image_width, 256)
        self.assertEqual(scan.shape, (3, 256, 256, 2, 1000))
        self.assertEqual(scan.zoom, 1.8)
        self.assertEqual(scan.is_slow_stack_with_fastZ, False)

        # 2016b multiROI
        scan = scanreader.read_scan(scan_file_2016b_multiroi_hard)
        self.assertEqual(scan.version, '2016b')
        self.assertEqual(scan.is_slow_stack, False)
        self.assertEqual(scan.is_multiROI, True)
        self.assertEqual(scan.num_channels, 2)
        self.assertEqual(scan.requested_scanning_depths, [50, 100, 150])
        self.assertEqual(scan.num_scanning_depths, 3)
        self.assertEqual(scan.scanning_depths, [50, 100, 150])
        self.assertEqual(scan.num_requested_frames, 10)
        self.assertEqual(scan.num_frames, 10)
        self.assertEqual(scan.is_bidirectional, True)
        self.assertEqual(scan.scanner_frequency, 12045.5)
        self.assertAlmostEqual(scan.seconds_per_line, 4.15092e-05)
        self.assertEqual(scan.num_fields, 4)
        self.assertEqual(scan.field_depths, [50, 100, 100, 150])
        self.assertEqual(scan.fps, 5.00651)
        self.assertEqual(scan.spatial_fill_fraction, 0.9)
        self.assertEqual(scan.temporal_fill_fraction, 0.712867)
        self.assertEqual(scan.scanner_type, 'Resonant')
        self.assertEqual(scan.motor_position_at_zero, [0, 0, 0])
        self.assertEqual(scan.initial_secondary_z, None)
        for i, max_offset in enumerate([0.03320531, 0.09978618, 0.12709929, 0.1544124]):
            self.assertAlmostEqual(scan.field_offsets[i].max(), max_offset)

        self.assertEqual(scan.is_slow_stack_with_fastZ, False)
        self.assertEqual(scan.num_rois, 2)
        self.assertEqual(scan.field_heights, [800, 800, 512, 512])
        self.assertEqual(scan.field_widths, [512, 512, 512, 512])
        self.assertEqual(scan.field_slices, [0, 1, 1, 2])
        self.assertEqual(scan.field_rois, [[0], [0], [1], [1]])
        roi_masks = [np.full([800, 512], 0, dtype=np.int8), np.full([800, 512], 0, dtype=np.int8),
                     np.full([512, 512], 1, dtype=np.int8), np.full([512, 512], 1, dtype=np.int8)]
        heights_in_microns = [800, 800, 500, 613.21963]
        widths_in_microns = [400, 400, 400, 400]
        for i in range(4): # for each field
            self.assertEqual(scan.field_masks[i].tolist(), roi_masks[i].tolist())
            self.assertAlmostEqual(scan.field_heights_in_microns[i], heights_in_microns[i], places=4)
            self.assertAlmostEqual(scan.field_widths_in_microns[i], widths_in_microns[i], places=4)

        # 2018a multiROI
        scan = scanreader.read_scan(scan_file_2018a_multiroi)
        self.assertEqual(scan.version, '2018a')
        self.assertEqual(scan.is_slow_stack, False)
        self.assertEqual(scan.is_multiROI, True)
        self.assertEqual(scan.num_channels, 1)
        self.assertEqual(scan.requested_scanning_depths, [0, 10, 50])
        self.assertEqual(scan.num_scanning_depths, 3)
        self.assertEqual(scan.scanning_depths, [0, 10, 50])
        self.assertEqual(scan.num_requested_frames, 1000000000)
        self.assertEqual(scan.num_frames, 159)
        self.assertEqual(scan.is_bidirectional, True)
        self.assertEqual(scan.scanner_frequency, 12024.1)
        self.assertAlmostEqual(scan.seconds_per_line, 4.15832e-05)
        self.assertEqual(scan.num_fields, 15)
        self.assertEqual(scan.field_depths, [0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 50, 50, 50, 50, 50])
        self.assertEqual(scan.fps, 2.12628)
        self.assertEqual(scan.spatial_fill_fraction, 0.9)
        self.assertEqual(scan.temporal_fill_fraction, 0.712867)
        self.assertEqual(scan.scanner_type, 'Resonant')
        self.assertEqual(scan.motor_position_at_zero, [0, 0, 0])
        self.assertEqual(scan.initial_secondary_z, None)
        for i, max_offset in enumerate([0.02128845, 0.05264215, 0.08399585, 0.11534955,
                                        0.14670324, 0.17805694, 0.20941064, 0.24076433,
                                        0.27211803, 0.30347174, 0.33482543, 0.36617914,
                                        0.39753282, 0.42888653, 0.46024022]):
            self.assertAlmostEqual(scan.field_offsets[i].max(), max_offset)

        self.assertEqual(scan.is_slow_stack_with_fastZ, True)
        self.assertEqual(scan.num_rois, 5)
        self.assertEqual(scan.field_heights, [512, ] * 15)
        self.assertEqual(scan.field_widths, [512, ] * 15)
        self.assertEqual(scan.field_slices, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        self.assertEqual(scan.field_rois, [[0], [1], [2], [3], [4]] * 3)
        heights_in_microns = [2499.99999, ] * 15
        widths_in_microns = [555.55499, ] * 15
        for i in range(15): # for each field
            self.assertEqual(scan.field_masks[i].tolist(), np.full([512, 512], i % 5).tolist())
            self.assertAlmostEqual(scan.field_heights_in_microns[i], heights_in_microns[i], places=4)
            self.assertAlmostEqual(scan.field_widths_in_microns[i], widths_in_microns[i], places=4)


    def assertEqualShapeAndSum(self, array, expected_shape, expected_sum):
        self.assertEqual(array.shape, expected_shape)
        self.assertEqual(np.sum(array, dtype=int), expected_sum)


    def test_5_1(self):
        scan = scanreader.read_scan(scan_file_5_1)

        # Test it is iterable
        fields_sum =  [114187329049, 119703328706, 125845219838]
        for i, field in enumerate(scan):
            self.assertEqualShapeAndSum(field, (256, 256, 2, 1000), fields_sum[i])

        # Test it can be obtained as array
        scan_as_array = np.array(scan)
        self.assertEqualShapeAndSum(scan_as_array, (3, 256, 256, 2, 1000), 359735877593)

        # Test indexation
        first_field = scan[0, :, :, :, :]
        self.assertEqualShapeAndSum(first_field, (256, 256, 2, 1000), 114187329049)
        first_row = scan[:, 0,  :, :, :]
        self.assertEqualShapeAndSum(first_row, (3, 256, 2, 1000), 917519804)
        first_column = scan[:, :, 0, :, :]
        self.assertEqualShapeAndSum(first_column, (3, 256, 2, 1000), 498901499)
        first_channel = scan[:, :, :, 0, :]
        self.assertEqualShapeAndSum(first_channel, (3, 256, 256, 1000), 340492324453)
        first_frame = scan[:, :, :, :, 0]
        self.assertEqualShapeAndSum(first_frame, (3, 256, 256, 2), 337564522)


    def test_5_2(self):
        scan = scanreader.read_scan(scan_file_5_2)

        # Test it is iterable
        fields_sum =  [165077647124, 150775776929, 176081992915]
        for i, field in enumerate(scan):
            self.assertEqualShapeAndSum(field, (512, 512, 2, 366), fields_sum[i])

        # Test it can be obtained as array
        scan_as_array = np.array(scan)
        self.assertEqualShapeAndSum(scan_as_array, (3, 512, 512, 2, 366), 491935416968)

        # Test indexation
        first_field = scan[0, :, :, :, :]
        self.assertEqualShapeAndSum(first_field, (512, 512, 2, 366), 165077647124)
        first_row = scan[:, 0,  :, :, :]
        self.assertEqualShapeAndSum(first_row, (3, 512, 2, 366), 879446899)
        first_column = scan[:, :, 0, :, :]
        self.assertEqualShapeAndSum(first_column, (3, 512, 2, 366), 236836271)
        first_channel = scan[:, :, :, 0, :]
        self.assertEqualShapeAndSum(first_channel, (3, 512, 512, 366), 468225501096)
        first_frame = scan[:, :, :, :, 0]
        self.assertEqualShapeAndSum(first_frame, (3, 512, 512, 2), 1381773476)

        
    def test_5_3(self):
        scan = scanreader.read_scan(scan_file_5_3)

        # Test it is iterable
        fields_sum =  [1471837154]
        for i, field in enumerate(scan):
            self.assertEqualShapeAndSum(field, (256, 256, 2, 21), fields_sum[i])

        # Test it can be obtained as array
        scan_as_array = np.array(scan)
        self.assertEqualShapeAndSum(scan_as_array, (1, 256, 256, 2, 21), 1471837154)

        # Test indexation
        first_field = scan[0, :, :, :, :]
        self.assertEqualShapeAndSum(first_field, (256, 256, 2, 21), 1471837154)
        first_row = scan[:, 0,  :, :, :]
        self.assertEqualShapeAndSum(first_row, (1, 256, 2, 21), 5749516)
        first_column = scan[:, :, 0, :, :]
        self.assertEqualShapeAndSum(first_column, (1, 256, 2, 21), 5749625)
        first_channel = scan[:, :, :, 0, :]
        self.assertEqualShapeAndSum(first_channel, (1, 256, 256, 21), 923762774)
        first_frame = scan[:, :, :, :, 0]
        self.assertEqualShapeAndSum(first_frame, (1, 256, 256, 2), 70090210)

        
    def test_2020(self):
        scan = scanreader.read_scan(scan_file_2020)

        # Test it is iterable
        fields_sum =  [-24781107]
        for i, field in enumerate(scan):
            self.assertEqualShapeAndSum(field, (256, 256, 2, 50), fields_sum[i])

        # Test it can be obtained as array
        scan_as_array = np.array(scan)
        self.assertEqualShapeAndSum(scan_as_array, (1, 256, 256, 2, 50), -24781107)

        # Test indexation
        first_field = scan[0, :, :, :, :]
        self.assertEqualShapeAndSum(first_field, (256, 256, 2, 50), -24781107)
        first_row = scan[:, 0,  :, :, :]
        self.assertEqualShapeAndSum(first_row, (1, 256, 2, 50), -107604)
        first_column = scan[:, :, 0, :, :]
        self.assertEqualShapeAndSum(first_column, (1, 256, 2, 50), -94825)
        first_channel = scan[:, :, :, 0, :]
        self.assertEqualShapeAndSum(first_channel, (1, 256, 256, 50), -15753120)
        first_frame = scan[:, :, :, :, 0]
        self.assertEqualShapeAndSum(first_frame, (1, 256, 256, 2), -466022)


    def test_5_1_multifile(self):
        scan = scanreader.read_scan(scan_file_5_1_multifiles)

        # Test it is iterable
        fields_sum = [163553755531, 171473993442, 180238513125]
        for i, field in enumerate(scan):
            self.assertEqualShapeAndSum(field, (256, 256, 2, 1500), fields_sum[i])

        # Test it can be obtained as array
        scan_as_array = np.array(scan)
        self.assertEqualShapeAndSum(scan_as_array, (3, 256, 256, 2, 1500), 515266262098)

        # Test indexation
        first_field = scan[0, :, :, :, :]
        self.assertEqualShapeAndSum(first_field, (256, 256, 2, 1500), 163553755531)
        first_row = scan[:, 0, :, :, :]
        self.assertEqualShapeAndSum(first_row, (3, 256, 2, 1500), 1328396733)
        first_column = scan[:, :, 0, :, :]
        self.assertEqualShapeAndSum(first_column, (3, 256, 2, 1500), 734212945)
        first_channel = scan[:, :, :, 0, :]
        self.assertEqualShapeAndSum(first_channel, (3, 256, 256, 1500), 487380452100)
        first_frame = scan[:, :, :, :, 0]
        self.assertEqualShapeAndSum(first_frame, (3, 256, 256, 2), 337564522)


    def test_2016b_multiroi(self):
        scan = scanreader.read_scan(scan_file_2016b_multiroi)

        # Test it is iterable
        fields_sum = [10437019861, 8288826827, 8590264328, 6532028278, 7713680015,
                      6058542598, 7171244110, 5541391024, 6386669378, 4886799974]
        for i, field in enumerate(scan):
            self.assertEqualShapeAndSum(field, (500, 250, 1, 100), fields_sum[i])

        # Test it can be obtained as array
        scan_as_array = np.array(scan)
        self.assertEqualShapeAndSum(scan_as_array, (10, 500, 250, 1, 100), 71606466393)

        # Test indexation
        first_field = scan[0, :, :, :, :]
        self.assertEqualShapeAndSum(first_field, (500, 250, 1, 100), 10437019861)
        first_row = scan[:, 0, :, :, :]
        self.assertEqualShapeAndSum(first_row, (10, 250, 1, 100), 147185283)
        first_column = scan[:, :, 0, :, :]
        self.assertEqualShapeAndSum(first_column, (10, 500, 1, 100), 224378620)
        first_channel = scan[:, :, :, 0, :]
        self.assertEqualShapeAndSum(first_channel, (10, 500, 250, 100), 71606466393)
        first_frame = scan[:, :, :, :, 0]
        self.assertEqualShapeAndSum(first_frame, (10, 500, 250, 1), 663727054)


    def test_2016b_multiroi_multifile(self):
        scan = scanreader.read_scan(scan_file_2016b_multiroi_multifiles)

        # Test it is iterable
        fields_sum = [20522111917, 16488768331, 16895482228, 13022673521, 15193380706,
                      12066890926, 14094412675, 11043585631, 12549291755,  9747543988]
        for i, field in enumerate(scan):
            self.assertEqualShapeAndSum(field, (500, 250, 1, 200), fields_sum[i])

        # Test it can be obtained as array
        scan_as_array = np.array(scan)
        self.assertEqualShapeAndSum(scan_as_array, (10, 500, 250, 1, 200), 141624141678)

        # Test indexation
        first_field = scan[0, :, :, :, :]
        self.assertEqualShapeAndSum(first_field, (500, 250, 1, 200), 20522111917)
        first_row = scan[:, 0, :, :, :]
        self.assertEqualShapeAndSum(first_row, (10, 250, 1, 200), 291067934)
        first_column = scan[:, :, 0, :, :]
        self.assertEqualShapeAndSum(first_column, (10, 500, 1, 200), 442948597)
        first_channel = scan[:, :, :, 0, :]
        self.assertEqualShapeAndSum(first_channel, (10, 500, 250, 200), 141624141678)
        first_frame = scan[:, :, :, :, 0]
        self.assertEqualShapeAndSum(first_frame, (10, 500, 250, 1), 663727054)


    def test_2016b_multiroi_hard(self):
        scan = scanreader.read_scan(scan_file_2016b_multiroi_hard)

        # Test it is iterable
        fields_shapes = [(800, 512, 2, 10), (800, 512, 2, 10), (512, 512,2, 10), (512, 512, 2, 10)]
        fields_sum = [2248989268, 2238433858, 1496780320, 1444886093]
        for i, field in enumerate(scan):
            self.assertEqualShapeAndSum(field, fields_shapes[i], fields_sum[i])

        # Test it can NOT be obtained as array
        self.assertRaises(ScanReaderException, lambda: np.array(scan))

        # Test indexation
        first_field = scan[0, :, :, :, :]
        self.assertEqualShapeAndSum(first_field, (800, 512, 2, 10), 2248989268)
        first_row = scan[:, 0, :, :, :]
        self.assertEqualShapeAndSum(first_row, (4, 512, 2, 10), 10999488)
        self.assertRaises(ScanReaderException, lambda: scan[:, :, 0, :, :])
        self.assertRaises(ScanReaderException, lambda: scan[:, :, :, 0, :])
        self.assertRaises(ScanReaderException, lambda: scan[:, :, :, :, 0])

        # Test indexation for last two slices
        first_column = scan[-2:, :, 0, :, :]
        self.assertEqualShapeAndSum(first_column, (2, 512, 2, 10), 3436369)
        first_channel = scan[-2:, :, :, 0, :]
        self.assertEqualShapeAndSum(first_channel, (2, 512, 512, 10), 2944468254)
        first_frame = scan[-2:, :, :, :, 0]
        self.assertEqualShapeAndSum(first_frame, (2, 512, 512, 2), 290883684)


    def test_2018a_multiroi(self):
        scan = scanreader.read_scan(scan_file_2018a_multiroi)

        # Test it is iterable
        fields_sum = [958342536, 872406772, 565935414, 190064269, -313625001, 871668983,
                      803397461, 518333983, 132827834, -330521191, 537668374, 510901435,
                      295330281, -64869402, -381772187]
        for i, field in enumerate(scan):
            self.assertEqualShapeAndSum(field, (512, 512, 1, 159), fields_sum[i])

        # Test it can be obtained as array
        scan_as_array = np.array(scan)
        self.assertEqualShapeAndSum(scan_as_array, (15, 512, 512, 1, 159), 5166089561)

        # Test indexation
        first_field = scan[0, :, :, :, :]
        self.assertEqualShapeAndSum(first_field, (512, 512, 1, 159), 958342536)
        first_row = scan[:, 0, :, :, :]
        self.assertEqualShapeAndSum(first_row, (15, 512, 1, 159), -20472715)
        first_column = scan[:, :, 0, :, :]
        self.assertEqualShapeAndSum(first_column, (15, 512, 1, 159), -8517112)
        first_channel = scan[:, :, :, 0, :]
        self.assertEqualShapeAndSum(first_channel, (15, 512, 512, 159), 5166089561)
        first_frame = scan[:, :, :, :, 0]
        self.assertEqualShapeAndSum(first_frame, (15, 512, 512, 1), 23273401)


    def test_advanced_indexing(self):
        """ Testing advanced indexing functionality."""
        scan = scanreader.read_scan(scan_file_5_1)

        # Testing slices
        part = scan[:, :200, :, 0, -100:]
        self.assertEqualShapeAndSum(part, (3, 200, 256, 100),  22309059758)
        part = scan[::-2]
        self.assertEqualShapeAndSum(part, (2, 256, 256, 2, 1000), 240032548887)

        # Testing lists
        part = scan[:, :, :, [-1, 0, 0, 1], :]
        self.assertEqualShapeAndSum(part, (3, 256, 256, 4, 1000), 719471755186)

        # Testing empty indices
        part = scan[:, :, :, 2:1, :]
        self.assertEqual(part.size, 0)
        part = scan[:, :, [], :, :]
        self.assertEqual(part.size, 0)

        # One field from a page appears twice separated by a field in another page
        scan = scanreader.read_scan(scan_file_2016b_multiroi)
        part = scan[[9, 3, 8, 3, 9, 8]]
        self.assertEqualShapeAndSum(part, (6, 500, 250, 1, 100), 35610995260)


    def test_join_contiguous(self):
        """ Testing whether contiguous fields are joined together."""
        scan = scanreader.read_scan(scan_file_join_contiguous, join_contiguous=True)

        # Test attributes
        self.assertEqual(scan.version, '2016b')
        self.assertEqual(scan.is_slow_stack, False)
        self.assertEqual(scan.is_multiROI, True)
        self.assertEqual(scan.num_channels, 1)
        self.assertEqual(scan.requested_scanning_depths, [-40, -20, 0, 20, 40])
        self.assertEqual(scan.num_scanning_depths, 5)
        self.assertEqual(scan.scanning_depths, [-40, -20, 0, 20, 40])
        self.assertEqual(scan.num_requested_frames, 500)
        self.assertEqual(scan.num_frames, 100)
        self.assertEqual(scan.is_bidirectional, True)
        self.assertEqual(scan.scanner_frequency, 12045.4)
        self.assertAlmostEqual(scan.seconds_per_line, 4.15097e-05)
        self.assertEqual(scan.num_fields, 5)
        self.assertEqual(scan.field_depths, [-40, -20, 0, 20, 40])
        self.assertEqual(scan.fps, 3.72926)
        self.assertEqual(scan.spatial_fill_fraction, 0.9)
        self.assertEqual(scan.temporal_fill_fraction, 0.712867)
        self.assertEqual(scan.scanner_type, 'Resonant')
        self.assertEqual(scan.motor_position_at_zero, [0, 0, 0])
        self.assertEqual(scan.initial_secondary_z, None)
        for i, max_offset in enumerate([0.04756787, 0.1011983, 0.15482873, 0.20845917, 0.26208961]):
            self.assertAlmostEqual(scan.field_offsets[i].max(), max_offset)

        self.assertEqual(scan.is_slow_stack_with_fastZ, False)
        self.assertEqual(scan.num_rois, 2)
        self.assertEqual(scan.field_heights, [500, 500, 500, 500, 500])
        self.assertEqual(scan.field_widths, [500, 500, 500, 500, 500])
        self.assertEqual(scan.field_slices, [0, 1, 2, 3, 4])
        self.assertEqual(scan.field_rois, [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
        roi_mask = np.zeros([500, 500], dtype=np.int8)
        roi_mask[:, 250:] = 1
        heights_in_microns = [1000, 1000, 1000, 1000, 1000]
        widths_in_microns = [1000, 1000, 1000, 1000, 1000]
        for i in range(5):
            self.assertEqual(scan.field_masks[i].tolist(), roi_mask.tolist())
            self.assertAlmostEqual(scan.field_heights_in_microns[i], heights_in_microns[i], places=4)
            self.assertAlmostEqual(scan.field_widths_in_microns[i], widths_in_microns[i], places=4)

        # Test it is iterable
        fields_sum = [18725846688, 15122292606, 13772222613, 12712635134, 11273469352]
        for i, field in enumerate(scan):
            self.assertEqualShapeAndSum(field, (500, 500, 1, 100), fields_sum[i])

        # Test it can be obtained as array
        scan_as_array = np.array(scan)
        self.assertEqualShapeAndSum(scan_as_array, (5, 500, 500, 1, 100), 71606466393)

        # Test indexation
        first_field = scan[0, :, :, :, :]
        self.assertEqualShapeAndSum(first_field, (500, 500, 1, 100), 18725846688)
        first_row = scan[:, 0, :, :, :]
        self.assertEqualShapeAndSum(first_row, (5, 500, 1, 100), 147185283)
        first_column = scan[:, :, 0, :, :]
        self.assertEqualShapeAndSum(first_column, (5, 500, 1, 100), 116583780)
        first_channel = scan[:, :, :, 0, :]
        self.assertEqualShapeAndSum(first_channel, (5, 500, 500, 100), 71606466393)
        first_frame = scan[:, :, :, :, 0]
        self.assertEqualShapeAndSum(first_frame, (5, 500, 500, 1), 663727054)


    def test_exceptions(self):
        """ Tests some exceptions are raised correctly. """
        # Wrong type and inexistent file
        self.assertRaises(TypeError, lambda: scanreader.read_scan(None))
        self.assertRaises(ScanReaderException, lambda: scanreader.read_scan('inexistent_file.tif'))

        scan = scanreader.read_scan(scan_file_5_1)

        # Too many dimensions
        self.assertRaises(IndexError, lambda: scan[0, 1, 2, 3, 4, 5])

        # Out of bounds, shape is (3, 256, 256, 2, 1000)
        self.assertRaises(IndexError, lambda: scan[-4])
        self.assertRaises(IndexError, lambda: scan[:, -257])
        self.assertRaises(IndexError, lambda: scan[:, :, -257])
        self.assertRaises(IndexError, lambda: scan[:, :, :, -3])
        self.assertRaises(IndexError, lambda: scan[:, :, :, :, -1001])

        # Wrong index type
        self.assertRaises(TypeError, lambda: scan[1, 'sup!'])
        self.assertRaises(TypeError, lambda: scan[[True, False, True]])
        self.assertRaises(TypeError, lambda: scan[0.1])
        self.assertRaises(TypeError, lambda: scan[0, ...])





class StackTest(TestCase):
    """ Test reading stacks from different ScanImage versions. """

    def test_attributes(self):

        # 5.1
        scan = scanreader.read_scan(stack_file_5_1)
        self.assertEqual(scan.version, '5.1')
        self.assertEqual(scan.is_slow_stack, True)
        self.assertEqual(scan.is_multiROI, False)
        self.assertEqual(scan.num_channels, 2)
        self.assertEqual(scan.requested_scanning_depths, list(range(310)))
        self.assertEqual(scan.num_scanning_depths, 60)
        self.assertEqual(scan.scanning_depths, list(range(60)))
        self.assertEqual(scan.num_requested_frames, 25)
        self.assertEqual(scan.num_frames, 25)
        self.assertEqual(scan.is_bidirectional, False)
        self.assertEqual(scan.scanner_frequency, 7919.95)
        self.assertAlmostEqual(scan.seconds_per_line, 0.000126264)
        self.assertEqual(scan.num_fields, 60)
        self.assertEqual(scan.field_depths, list(range(60)))
        self.assertEqual(scan.fps, 0.0486657)
        self.assertEqual(scan.spatial_fill_fraction, 0.9)
        self.assertEqual(scan.temporal_fill_fraction, 0.712867)
        self.assertEqual(scan.scanner_type, 'Resonant')
        self.assertEqual(scan.motor_position_at_zero, [0.5, 0, -320.4])
        self.assertEqual(scan.initial_secondary_z, 0)

        self.assertEqual(scan.image_height, 512)
        self.assertEqual(scan.image_width, 512)
        self.assertEqual(scan.shape, (60, 512, 512, 2, 25))
        self.assertEqual(scan.zoom, 2.1)
        self.assertEqual(scan.is_slow_stack_with_fastZ, False)

        # 2016b
        scan = scanreader.read_scan(stack_file_2016b)
        self.assertEqual(scan.version, '2016b')
        self.assertEqual(scan.is_slow_stack, True)
        self.assertEqual(scan.is_multiROI, False)
        self.assertEqual(scan.num_channels, 1)
        self.assertEqual(scan.requested_scanning_depths, [0])
        self.assertEqual(scan.num_scanning_depths, 1)
        self.assertEqual(scan.scanning_depths, [0])
        self.assertEqual(scan.num_requested_frames, 4000)
        self.assertEqual(scan.num_frames, 200)
        self.assertEqual(scan.is_bidirectional, False)
        self.assertEqual(scan.scanner_frequency, 7926.87)
        self.assertAlmostEqual(scan.seconds_per_line, 0.000126153)
        self.assertEqual(scan.num_fields, 1)
        self.assertEqual(scan.field_depths, [0])
        self.assertEqual(scan.fps, 30.0255)
        self.assertEqual(scan.spatial_fill_fraction, 0.9)
        self.assertEqual(scan.temporal_fill_fraction, 0.712867)
        self.assertEqual(scan.scanner_type, 'Resonant')
        self.assertEqual(scan.motor_position_at_zero, [1359.5, 46710.5, -5323])
        self.assertEqual(scan.initial_secondary_z, None)

        self.assertEqual(scan.image_height, 256)
        self.assertEqual(scan.image_width, 256)
        self.assertEqual(scan.shape, (1, 256, 256, 1, 200))
        self.assertEqual(scan.zoom, 1.9)
        self.assertEqual(scan.is_slow_stack_with_fastZ, False)
        self.assertEqual(scan.image_height_in_microns, 307.08)
        self.assertEqual(scan.image_width_in_microns, 307.08)

        # 2016b multiROI
        scan = scanreader.read_scan(stack_file_2016b_multiroi)
        self.assertEqual(scan.version, '2016b')
        self.assertEqual(scan.is_slow_stack, True)
        self.assertEqual(scan.is_multiROI, True)
        self.assertEqual(scan.num_channels, 2)
        self.assertEqual(scan.requested_scanning_depths, list(range(150, 99, -1)))
        self.assertEqual(scan.num_scanning_depths, 51)
        self.assertEqual(scan.scanning_depths, list(range(150, 99, -1)))
        self.assertEqual(scan.num_requested_frames, 10)
        self.assertEqual(scan.num_frames, 10)
        self.assertEqual(scan.is_bidirectional, True)
        self.assertEqual(scan.scanner_frequency, 12039.1)
        self.assertAlmostEqual(scan.seconds_per_line, 4.15312e-05)
        self.assertEqual(scan.num_fields, 204)
        self.assertEqual(scan.field_depths, list(np.repeat(range(150, 99, -1), 4)))
        self.assertEqual(scan.fps, 0.244914)
        self.assertEqual(scan.spatial_fill_fraction, 0.9)
        self.assertEqual(scan.temporal_fill_fraction, 0.712867)
        self.assertEqual(scan.scanner_type, 'Resonant')
        self.assertEqual(scan.motor_position_at_zero, [0, 0, 0])
        self.assertEqual(scan.initial_secondary_z, None)

        self.assertEqual(scan.is_slow_stack_with_fastZ, True)
        self.assertEqual(scan.num_rois, 4)
        self.assertEqual(scan.field_heights, [360] * 204)
        self.assertEqual(scan.field_widths, [120] * 204)
        self.assertEqual(scan.field_slices, list(np.repeat(range(0, 51), 4)))
        self.assertEqual(scan.field_rois, [[0], [1], [2], [3]] * 51)
        roi_masks = [np.full([360, 120], i, dtype=np.int8) for i in range(4)]
        for i in range(204): # for each field
            self.assertEqual(scan.field_masks[i].tolist(), roi_masks[i % 4].tolist())
            self.assertAlmostEqual(scan.field_heights_in_microns[i], 1800, places=4)
            self.assertAlmostEqual(scan.field_widths_in_microns[i], 600, places=4)


    def assertEqualShapeAndSum(self, array, expected_shape, expected_sum):
        self.assertEqual(array.shape, expected_shape)
        self.assertEqual(np.sum(array, dtype=int), expected_sum)


    def test_5_1(self):
        scan = scanreader.read_scan(stack_file_5_1)

        # Test it is iterable
        for i, field in enumerate(scan):
            self.assertEqual(field.shape, (512, 512, 2, 25))
        self.assertEqual(i, 59) # 60 fields

        # Test it can be obtained as array
        scan_as_array = np.array(scan)
        self.assertEqualShapeAndSum(scan_as_array, (60, 512, 512, 2, 25), 1766199881650)

        # Test indexation
        first_field = scan[0, :, :, :, :]
        self.assertEqualShapeAndSum(first_field, (512, 512, 2, 25), 27836374986)
        first_row = scan[:, 0,  :, :, :]
        self.assertEqualShapeAndSum(first_row, (60, 512, 2, 25), 2838459027)
        first_column = scan[:, :, 0, :, :]
        self.assertEqualShapeAndSum(first_column, (60, 512, 2, 25), 721241569)
        first_channel = scan[:, :, :, 0, :]
        self.assertEqualShapeAndSum(first_channel, (60, 512, 512, 25), 1649546136958)
        first_frame = scan[:, :, :, :, 0]
        self.assertEqualShapeAndSum(first_frame, (60, 512, 512, 2), 69769537416)


    def test_5_1_multifile(self):
        scan = scanreader.read_scan(stack_file_5_1_multifiles)

        # Test it is iterable
        for i, field in enumerate(scan):
            self.assertEqual(field.shape, (512, 512, 2, 25))
        self.assertEqual(i, 69) # 70 fields

        # Test it can be obtained as array
        scan_as_array = np.array(scan)
        self.assertEqualShapeAndSum(scan_as_array, (70, 512, 512, 2, 25), 2021813090863)

        # Test indexation
        first_field = scan[0, :, :, :, :]
        self.assertEqualShapeAndSum(first_field, (512, 512, 2, 25), 27836374986)
        first_row = scan[:, 0, :, :, :]
        self.assertEqualShapeAndSum(first_row, (70, 512, 2, 25), 3294545077)
        first_column = scan[:, :, 0, :, :]
        self.assertEqualShapeAndSum(first_column, (70, 512, 2, 25), 885838245)
        first_channel = scan[:, :, :, 0, :]
        self.assertEqualShapeAndSum(first_channel, (70, 512, 512, 25), 1832276046863)
        first_frame = scan[:, :, :, :, 0]
        self.assertEqualShapeAndSum(first_frame, (70, 512, 512, 2), 79887927681)


    def test_2016b(self):
        scan = scanreader.read_scan(stack_file_2016b)

        # Test it is iterable
        fields_sum = [-7855587]
        for i, field in enumerate(scan):
            self.assertEqualShapeAndSum(field, (256, 256, 1, 200), fields_sum[i])

        # Test it can be obtained as array
        scan_as_array = np.array(scan)
        self.assertEqualShapeAndSum(scan_as_array, (1, 256, 256, 1, 200), -7855587)

        # Test indexation
        first_field = scan[0, :, :, :, :]
        self.assertEqualShapeAndSum(first_field, (256, 256, 1, 200), -7855587)
        first_row = scan[:, 0, :, :, :]
        self.assertEqualShapeAndSum(first_row, (1, 256, 1, 200), -30452)
        first_column = scan[:, :, 0, :, :]
        self.assertEqualShapeAndSum(first_column, (1, 256, 1, 200), -31680)
        first_channel = scan[:, :, :, 0, :]
        self.assertEqualShapeAndSum(first_channel, (1, 256, 256, 200), -7855587)
        first_frame = scan[:, :, :, :, 0]
        self.assertEqualShapeAndSum(first_frame, (1, 256, 256, 1), -42389)


    def test_2016b_multiroi(self):
        scan = scanreader.read_scan(stack_file_2016b_multiroi)

        # Test it is iterable
        for i, field in enumerate(scan):
            self.assertEqual(field.shape, (360, 120, 2, 10))
        self.assertEqual(i, 203)

        # Test it can be obtained as array
        scan_as_array = np.array(scan)
        self.assertEqualShapeAndSum(scan_as_array, (204, 360, 120, 2, 10), 30797502048)

        # Test indexation
        first_field = scan[0, :, :, :, :]
        self.assertEqualShapeAndSum(first_field, (360, 120, 2, 10), 148674123)
        first_row = scan[:, 0, :, :, :]
        self.assertEqualShapeAndSum(first_row, (204, 120, 2, 10), 70350224)
        first_column = scan[:, :, 0, :, :]
        self.assertEqualShapeAndSum(first_column, (204, 360, 2, 10), 160588726)
        first_channel = scan[:, :, :, 0, :]
        self.assertEqualShapeAndSum(first_channel, (204, 360, 120, 10), 26825949131)
        first_frame = scan[:, :, :, :, 0]
        self.assertEqualShapeAndSum(first_frame, (204, 360, 120, 2), 2952050950)