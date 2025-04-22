import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage.filters import threshold_otsu

# Load DICOM series
dicom_dir = r"D:\opera\ORIG_3D_FSPGR_20_Average\ORIG_3D_FSPGR_20_Average"
dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
dicom_files.sort()  # Sort files to ensure correct loading order

# Load DICOM images as a 3D numpy array
dicom_images = []
for dicom_file in dicom_files:
    dicom_path = os.path.join(dicom_dir, dicom_file)
    dicom_data = pydicom.dcmread(dicom_path)
    dicom_images.append(dicom_data.pixel_array)

dicom_images = np.stack(dicom_images, axis=-1)

# Convert to SimpleITK image (32-bit float) for bias field correction
image_sitk = sitk.GetImageFromArray(dicom_images.astype(np.float32))

# Bias field correction using SimpleITK
corrector = sitk.N4BiasFieldCorrectionImageFilter()
corrected_image_sitk = corrector.Execute(image_sitk)
corrected_image = sitk.GetArrayFromImage(corrected_image_sitk)

# Compute MIP of the original DICOM images (maximum intensity projection along z-axis)
mip_original = np.max(dicom_images, axis=-1)

# Compute MIP of the bias field-corrected images
mip_corrected = np.max(corrected_image, axis=-1)

# Otsu Thresholding on the corrected MIP
thresh = threshold_otsu(mip_corrected)
otsu_image = mip_corrected > thresh

# Plotting the results
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Original MIP
ax[0].imshow(mip_original, cmap='gray')
ax[0].set_title('Original MIP')
ax[0].axis('off')

# Bias Field Corrected MIP
ax[1].imshow(mip_corrected, cmap='gray')
ax[1].set_title('Bias Field Corrected MIP')
ax[1].axis('off')

# Otsu Thresholded MIP
ax[2].imshow(otsu_image, cmap='gray')
ax[2].set_title('Otsu Thresholded MIP')
ax[2].axis('off')

plt.tight_layout()
plt.show()
