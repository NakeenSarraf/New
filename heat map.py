import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import SimpleITK as sitk

# Load DICOM series
dicom_dir = r"D:\opera\ORIG_3D_FSPGR_20_Average\ORIG_3D_FSPGR_20_Average"
dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
dicom_files.sort()

# Load DICOM images as a 3D numpy array
dicom_images = []
for dicom_file in dicom_files:
    dicom_path = os.path.join(dicom_dir, dicom_file)
    dicom_data = pydicom.dcmread(dicom_path)
    dicom_images.append(dicom_data.pixel_array)

dicom_images = np.stack(dicom_images, axis=-1)

# Convert to SimpleITK image (32-bit float) for bias field correction
image_sitk = sitk.GetImageFromArray(dicom_images.astype(np.float32))
corrector = sitk.N4BiasFieldCorrectionImageFilter()
corrected_image_sitk = corrector.Execute(image_sitk)
corrected_image = sitk.GetArrayFromImage(corrected_image_sitk)

# Compute MIPs
mip_original = np.max(dicom_images, axis=-1)
mip_corrected = np.max(corrected_image, axis=-1)

# Difference image
mip_difference = mip_corrected - mip_original

# Normalize for heatmap (diverging colormap)
vmax = np.percentile(mip_difference, 99)
vmin = np.percentile(mip_difference, 1)
abs_max = max(abs(vmin), abs(vmax))

# Plotting
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Original MIP
ax[0].imshow(mip_original, cmap='gray')
ax[0].set_title('Original MIP')
ax[0].axis('off')

# Corrected MIP
ax[1].imshow(mip_corrected, cmap='gray')
ax[1].set_title('Bias Field Corrected MIP')
ax[1].axis('off')

# Difference as heatmap
heatmap = ax[2].imshow(mip_difference, cmap='seismic', vmin=-abs_max, vmax=abs_max)
ax[2].set_title('Difference Heatmap')
ax[2].axis('off')

# Add colorbar to heatmap
fig.colorbar(heatmap, ax=ax[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
