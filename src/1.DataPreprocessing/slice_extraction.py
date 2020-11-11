""" Python script for extract slices from IXI volumes.
We will use DeepBrainSliceExtractor class
"""

__author__ = "Adrian Arnaiz-Rodriguez"
__email__ = "aarnaizr@uoc.edu"

# Path improvement configuration
from os.path import dirname
import sys
import nibabel as nib

script_path = dirname(__file__)
sys.path.append(script_path)

from deep_brain_slice_extractor import DeepBrainSliceExtractor

se = DeepBrainSliceExtractor(script_path+'/../IXI-T1/*.gz')

print(len(se.all_volume_files))

files = se.all_volume_files
file_vol_0 = files[0]
vol_0_np = nib.load(file_vol_0).get_fdata()