import numpy as np
import SimpleITK as sitk
import cv2
from skimage.metrics import structural_similarity as ssim

# Load DICOM series
dicom_folder = r"D:\opera\ORIG_3D_FSPGR_20_Average\ORIG_3D_FSPGR_20_Average"
reader = sitk.ImageSeriesReader()
series_ids = reader.GetGDCMSeriesIDs(dicom_folder)
dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder, series_ids[0])
reader.SetFileNames(dicom_names)
image_sitk = reader.Execute()

# Convert to float for N4
image_float = sitk.Cast(image_sitk, sitk.sitkFloat32)

# Apply N4 bias field correction
corrector = sitk.N4BiasFieldCorrectionImageFilter()
corrected_sitk = corrector.Execute(image_float)

# Convert to NumPy
original_np = sitk.GetArrayFromImage(image_float)
corrected_np = sitk.GetArrayFromImage(corrected_sitk)

# MIPs
mip_before = np.max(original_np, axis=0)
mip_after = np.max(corrected_np, axis=0)

# Normalize for SSIM and Otsu
mip_before_u8 = cv2.normalize(mip_before, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
mip_after_u8 = cv2.normalize(mip_after, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# SSIM
ssim_val = ssim(mip_before_u8, mip_after_u8)
print(f"SSIM: {ssim_val:.4f}")

# Otsu thresholding
_, mask_before = cv2.threshold(mip_before_u8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, mask_after = cv2.threshold(mip_after_u8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# DSC
def dsc(m1, m2):
    inter = np.sum((m1 == 1) & (m2 == 1))
    return 2 * inter / (np.sum(m1 == 1) + np.sum(m2 == 1))

dsc_val = dsc(mask_before, mask_after)
print(f"DSC: {dsc_val:.4f}")
