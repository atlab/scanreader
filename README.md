# scanreader
Python TIFF Stack Reader for ScanImage 5 scans (including multiROI).

We treat a scan as a collection of recording fields: rectangular planes at a given x, y, z position in the scan recorded in a number of channels during a preset amount of time. All fields have the same number of channels and number of frames.

We plan to support new versions of ScanImage scans as our lab starts using them. If you would like us to add support for an older (or different) version of ScanImage, send us a small sample scan.

### Installation
```shell
pip3 install git+https://github.com/atlab/scanreader.git
```

### Usage
```python
import scanreader
scan = scanreader.read_scan('/data/my_scan_*.tif')
print(scan.version)
print(scan.num_frames)
print(scan.num_channels)
print(scan.num_fields)

for field in scan:
    # process field (4-d array: [y, x, channels, frames])
    del field  # free memory before next iteration

x = scan[:]  # 5-d array [fields, y, x, channel, frames]
y = scan[:2, :, :, 0, -1000:]  # 5-d array: last 1000 frames of first 2 fields on the first channel
z = scan[1]  # 4-d array: the second field (over all channels and time)

scan = scanreader.read_scan('/data/my_scan_*.tif', dtype=np.float32, join_contiguous=True)
# scan loaded as np.float32 (default is np.int16) and adjacent fields at same depth will be joined.
```
Scan objects (returned by `read_scan()`) are iterable and indexable (as shown). Indexes can be integers, slice objects (:) or lists/tuples/arrays of integers. It should act like a numpy 5-d array---no boolean indexing, though.

This reader is based on a previous [version](https://github.com/atlab/tiffreader) developed by Fabian Sinz.

## Details on data loading (for future developers)
As of this version, `scanreader` relies on [`tifffile`](https://pypi.org/project/tifffile/) to read the underlying tiff files. Reading a scan happens in three stages:
1. `scan = scanreader.read_scan(filename)` will create a list of `tifffile.TiffFile`s, one per each tiff file in the scan. This entails opening a file handle and reading the tags of the first page of each; tags for the rest of pages are ignored (they have the same info).
2. `scan.num_frames`, `scan.shape` or another operation that requires the number of frames in the scan---which includes the first stage of any data loading operation---will need the number of pages in each tiff file. `tifffile` was designed for files with pages of varying shapes so it iterates over each page looking for its offset (number of bytes from the start of the file until the very first byte of the page), which it saves to use for reading. After this operation, it knows the number of pages per file.
3. Once the file has been opened and the offset to each page has been calculated we can load the actual data. We load each page sequentially and take care of reformatting them to match the desired output.
