#%% ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nptdms import TdmsFile
import dataloader

np.random.seed(1)

#%% Configuration ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Folder containing TDMS files
# data_folder = '/work/huy.leminh/code/Dr.Huy/PECT/Data/Duong/basic'
data_folder = '/work/huy.leminh/code/Dr.Huy/PECT/Data/Feedback'
# Output folder for saving plots
output_folder = os.path.join(data_folder, 'heatmap_plots')
os.makedirs(output_folder, exist_ok=True)

# Processing parameters
# slice_window = range(15, 30)  # Window for slice feature extraction
# loi_window = range(15, 22)    # Window for LOI feature extraction
# peak_window = range(0, 100)   # Window for peak feature extraction

slice_window = range(0, 250)  # Window for slice feature extraction
loi_window = range(20, 30)    # Window for LOI feature extraction
peak_window = range(0, 499)   # Window for peak feature extraction

# Filter parameters
lpf_cutoff = 0.04
lpf_order = 24

#%% Get all TDMS files ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Get all TDMS files (excluding index files and PECT_default.tdms)
tdms_files = glob.glob(os.path.join(data_folder, '*.tdms'))
tdms_files = [f for f in tdms_files if not f.endswith('_index') and not f.endswith('PECT_default.tdms')]
tdms_files.sort()

print(f"Found {len(tdms_files)} TDMS files:")
for i, file in enumerate(tdms_files):
    print(f"  {i+1}. {os.path.basename(file)}")

#%% Process each TDMS file ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

k = 4  # Number of feedback periods concatenated

plt.rc('font', size=24)  # Set default text size for all elements
for tdms_file in tdms_files[:1]:
    filename = os.path.basename(tdms_file)
    print(f"\nProcessing: {filename}")    
    try:
        # Load and preprocess scan data
        # data, samples, sX, sY = dataloader.read_scan_tdms(tdms_file, raster=True)
        data, samples, sX, sY = dataloader.read_scan_tdms_feedback(tdms_file,raster=False,k=k)
        data = data[:,:samples]
        data = dataloader.reshape_raster(data, sX).squeeze()
        data -= np.mean(data, axis=2, keepdims=True)
        
        # Apply Low-Pass Filter
        data = data - data[:, :, :1]
        X_filter = dataloader.LPF_2D(data, cutoff=lpf_cutoff, order=lpf_order)
        X_filter -= np.mean(X_filter, axis=2, keepdims=True)
        
        # ========================================
        # 1. SLICE - Extract features from raw filtered data
        # ========================================
        X_feat_slice = dataloader.extract_features(
            X_filter, 
            slice_window, 
            method='mean', 
            sub_bg_x=False, 
            sub_bg_y=False, 
            plot=False
        )
        
        # ========================================
        # 2. LOI - Extract features from gradient data
        # ========================================
        X_filter_grad = np.diff(X_filter, axis=2)
        X_feat_loi = dataloader.extract_features(
            X_filter_grad, 
            loi_window, 
            method='abs', 
            sub_bg_x=False, 
            sub_bg_y=False, 
            plot=False
        )
        
        # ========================================
        # 3. PEAK - Extract max features from gradient data
        # ========================================
        X_feat_peak = dataloader.extract_features(
            X_filter_grad, 
            peak_window, 
            method='max', 
            sub_bg_x=False, 
            sub_bg_y=False, 
            plot=False
        )
        
        # ========================================
        # Plot all three heatmaps in one figure
        # ========================================
        fig = plt.figure(figsize=(24, 8))
        
        # Slice
        plt.subplot(1, 3, 1)
        plt.imshow(X_feat_slice, cmap='jet', interpolation='bilinear')        
        plt.title('Slice Feature Map')
        plt.xlabel('X-direction [mm]')
        plt.ylabel('Y-direction [mm]')
        plt.colorbar()
        
        # LOI
        plt.subplot(1, 3, 2)
        plt.imshow(X_feat_loi, cmap='jet', interpolation='bilinear')
        plt.title('LOI Feature Map')
        plt.xlabel('X-direction [mm]')
        plt.ylabel('Y-direction [mm]')
        plt.colorbar()
        
        # Peak
        plt.subplot(1, 3, 3)
        plt.imshow(X_feat_peak, cmap='jet', interpolation='bilinear')
        plt.title('Peak Feature Map')
        plt.xlabel('X-direction [mm]')
        plt.ylabel('Y-direction [mm]')
        plt.colorbar()
        
        plt.tight_layout()
        
        # Save figure
        output_filename = os.path.splitext(filename)[0] + '_heatmaps.png'
        output_path = os.path.join(output_folder, output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_filename}")
        
        plt.show()
        plt.close()
        
    except Exception as e:
        print(f"  ERROR processing {filename}: {str(e)}")
        continue

print(f"\n✓ All plots saved to: {output_folder}")

# %%
