import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

# Path to your DICOM folder
dicom_path = r"C:\Users\sarra\Downloads\ORIG_3D_FSPGR_20_Average\ORIG_3D_FSPGR_20_Average"

# Step 1: Load DICOM series
reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
reader.SetFileNames(dicom_names)
image = reader.Execute()
image = sitk.Cast(image, sitk.sitkFloat32)

# Step 2: Apply Bias Field Correction
corrector = sitk.N4BiasFieldCorrectionImageFilter()
corrected_image = corrector.Execute(image)

# Step 3: Normalize both images to range [0, 255]
def normalize_to_255(img):
    arr = sitk.GetArrayFromImage(img)
    arr = arr - np.min(arr)
    arr = arr / np.max(arr) * 255.0
    norm_img = sitk.GetImageFromArray(arr.astype(np.float32))
    norm_img.CopyInformation(img)
    return norm_img

image_norm = normalize_to_255(image)
corrected_norm = normalize_to_255(corrected_image)

# Step 4: Convert to NumPy arrays
original_np = sitk.GetArrayFromImage(image_norm)
corrected_np = sitk.GetArrayFromImage(corrected_norm)

# Step 5: Compute absolute difference
difference_np = np.abs(original_np - corrected_np)

# Step 6: Compute Maximum Intensity Projections (MIPs)
mip_original = np.max(original_np, axis=0)
mip_corrected = np.max(corrected_np, axis=0)
mip_difference = np.max(difference_np, axis=0)

# Step 7: Visualize with intensity info
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Compute intensity ranges
orig_min, orig_max = mip_original.min(), mip_original.max()
corr_min, corr_max = mip_corrected.min(), mip_corrected.max()
diff_min, diff_max = mip_difference.min(), mip_difference.max()

# Plot Original MIP
axs[0].imshow(mip_original, cmap='gray')
axs[0].set_title(f"Original MIP\n[Min: {orig_min:.1f}, Max: {orig_max:.1f}]")
axs[0].axis("off")

# Plot Corrected MIP
axs[1].imshow(mip_corrected, cmap='gray')
axs[1].set_title(f"Corrected MIP\n[Min: {corr_min:.1f}, Max: {corr_max:.1f}]")
axs[1].axis("off")

# Plot Difference MIP
axs[2].imshow(mip_difference, cmap='gray')
axs[2].set_title(f"Difference MIP\n[Min: {diff_min:.1f}, Max: {diff_max:.1f}]")
axs[2].axis("off")

plt.tight_layout()
plt.show()
