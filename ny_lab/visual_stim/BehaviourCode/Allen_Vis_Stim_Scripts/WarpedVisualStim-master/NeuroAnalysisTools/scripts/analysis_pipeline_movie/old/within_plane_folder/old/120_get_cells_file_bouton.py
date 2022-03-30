import os
import numpy as np
import h5py
import tifffile as tf
import NeuroAnalysisTools.core.ImageAnalysis as ia
import NeuroAnalysisTools.core.PlottingTools as pt
import scipy.ndimage as ni
import matplotlib.pyplot as plt


isSave = True
is_filter = True

filter_sigma = 0. # parameters only used if filter the rois
# dilation_iterations = 1. # parameters only used if filter the rois
cut_thr = 2.5 # low for more rois, high for less rois

bg_fn = "corrected_mean_projections.tif"
save_folder = 'figures'

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

data_f = h5py.File('caiman_segmentation_results.hdf5')
masks = data_f['masks'].value
data_f.close()

bg = ia.array_nor(np.max(tf.imread(bg_fn), axis=0))

final_roi_dict = {}

for i, mask in enumerate(masks):

    if is_filter:
        mask_nor = (mask - np.mean(mask.flatten())) / np.abs(np.std(mask.flatten()))
        mask_nor_f = ni.filters.gaussian_filter(mask_nor, filter_sigma)
        mask_bin = np.zeros(mask_nor_f.shape, dtype=np.uint8)
        mask_bin[mask_nor_f > cut_thr] = 1

    else:
        mask_bin = np.zeros(mask.shape, dtype=np.uint8)
        mask_bin[mask > 0] = 1

    mask_labeled, mask_num = ni.label(mask_bin)
    curr_mask_dict = ia.get_masks(labeled=mask_labeled, keyPrefix='caiman_mask_{:03d}'.format(i), labelLength=5)
    for roi_key, roi_mask in curr_mask_dict.items():
        final_roi_dict.update({roi_key: ia.WeightedROI(roi_mask * mask)})

print ('Total number of ROIs:',len(final_roi_dict))

f = plt.figure(figsize=(15, 8))
ax1 = f.add_subplot(121)
ax1.imshow(bg, vmin=0, vmax=0.5, cmap='gray', interpolation='nearest')
colors1 = pt.random_color(masks.shape[0])
for i, mask in enumerate(masks):
    pt.plot_mask_borders(mask, plotAxis=ax1, color=colors1[i])
ax1.set_title('original ROIs')
ax1.set_axis_off()
ax2 = f.add_subplot(122)
ax2.imshow(ia.array_nor(bg), vmin=0, vmax=0.5, cmap='gray', interpolation='nearest')
colors2 = pt.random_color(len(final_roi_dict))
i = 0
for roi in final_roi_dict.values():
    pt.plot_mask_borders(roi.get_binary_mask(), plotAxis=ax2, color=colors2[i])
    i = i + 1
ax2.set_title('filtered ROIs')
ax2.set_axis_off()
plt.show()

if isSave:

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    f.savefig(os.path.join(save_folder, 'caiman_segmentation_filtering.pdf'), dpi=300)

    cell_file = h5py.File('cells.hdf5', 'w')

    i = 0
    for key, value in sorted(final_roi_dict.iteritems()):
        curr_grp = cell_file.create_group('cell{:04d}'.format(i))
        curr_grp.attrs['name'] = key
        value.to_h5_group(curr_grp)
        i += 1

    cell_file.close()