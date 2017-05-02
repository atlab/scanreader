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
	# process field (4-d array: \[x, y, channels, frames\])

x = scan[:] # 5-d array \[fields, x, y, channel, frames\]
y = scan[:2, :, :, 0, -1000:] # 5-d array: last 1000 frames of first 2 fields on the first channel
z = scan[1] # 4-d array: the second field (over all channels and time)
```
Scan objects (returned by read_scan()) are iterable and indexable (as shown). Indexes can be integers, slice objects (:) or lists/tuples/arrays of integers. It should act like a numpy 5-d array---no boolean indexing, though.

This reader is based on a previous [version](https://github.com/atlab/tiffreader) developed by Fabian Sinz.
