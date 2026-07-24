from scipy.ndimage import zoom
import iso2mesh as i2m
import nibabel as nib
import numpy as np
import os
import pydicom
from pydicom import dcmread
from pydicom.errors import InvalidDicomError
import glob
import gzip
import shutil
import tempfile
import subprocess
import time


def extract_head_surface(img_data, threshold_list=None, target_faces=30000):
    """
    Extract a closed head surface from the image using iso2mesh.
    Tries smoothed and original volume, with multiple thresholds.
    Returns (nodes, faces) or raises RuntimeError.
    """
    if threshold_list is None:
        # Start with higher thresholds to avoid noisy background.
        # Low thresholds (0.10, 0.12) are tried later as fallback.
        threshold_list = [0.20, 0.25, 0.30, 0.35, 0.15]

    # Normalise and smooth
    img_data = np.nan_to_num(img_data)
    if img_data.max() > img_data.min():
        img_data = (img_data - img_data.min()) / \
            (img_data.max() - img_data.min())
    from scipy.ndimage import gaussian_filter
    img_smooth = gaussian_filter(img_data, sigma=1.0)

    for th in threshold_list:
        for img, label in [(img_smooth, "smoothed"), (img_data, "original")]:
            try:
                print(
                    f"  Trying head extraction: threshold={th}, {label} volume...")
                node, face, regions, centroids = i2m.v2s(
                    img,
                    isovalues=th,
                    opt={'distbound': 2, 'maxnode': 200000, 'radbound': 3}
                )
                if len(face) == 0:
                    print("  No faces generated.")
                    continue

                # Repair using CGAL if available, else standard repair
                try:
                    node, face = i2m.meshcheckrepair(node, face, 'cgal')
                except:
                    node, face = i2m.meshcheckrepair(node, face)

                # --- REMOVED the meshresample call that was causing hangs ---
                # The raw surface (usually ~20k-200k faces) is fine for the
                # 10‑20 electrode system; we skip simplification entirely.
                # If you still want to reduce faces, set target_faces to a
                # very large number or use a safer method.
                # if len(face) > target_faces:
                #     try:
                #         node, face = i2m.meshresample(node, face, target_faces)
                #     except:
                #         print("  Mesh resampling skipped.")

                print(
                    f"  ✓ Head surface: {len(node)} nodes, {len(face)} faces")
                return node, face

            except Exception as e:
                print(
                    f"  ✗ Failed with threshold {th} ({label}): {str(e)[:150]}")
                continue

    raise RuntimeError("All head extraction strategies failed.")


def unzip_nifti_gz(file_path):
    """
    Unzip a .nii.gz file and return the path to the temporary .nii file.
    Uses a temporary file that will be cleaned up automatically.
    """
    if not file_path.endswith('.nii.gz') and not file_path.endswith('.nii'):
        return file_path

    if file_path.endswith('.nii'):
        return file_path

    temp_nii = tempfile.NamedTemporaryFile(suffix='.nii', delete=False)
    temp_nii.close()

    try:
        print(f"Unzipping {file_path}...")
        with gzip.open(file_path, 'rb') as f_in:
            with open(temp_nii.name, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Unzipped to temporary file: {temp_nii.name}")
        return temp_nii.name
    except Exception as e:
        print(f"Error unzipping {file_path}: {e}")
        if os.path.exists(temp_nii.name):
            os.unlink(temp_nii.name)
        raise


def cleanup_temp_files(temp_files):
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
                print(f"Cleaned up temporary file: {temp_file}")
        except Exception as e:
            print(f"Warning: Could not delete {temp_file}: {e}")


def get_filename_without_extension(path):
    basename = os.path.basename(path)
    if basename.endswith('.nii.gz'):
        return basename[:-7]
    else:
        return os.path.splitext(basename)[0]


def load_dicom_series(dicom_folder):
    dicom_files = glob.glob(os.path.join(dicom_folder, "*.dcm"))
    if not dicom_files:
        dicom_files = glob.glob(os.path.join(dicom_folder, "*"))
        dicom_files = [f for f in dicom_files if os.path.isfile(f)]

    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_folder}")

    slices = []
    for file_path in dicom_files:
        try:
            ds = dcmread(file_path, force=True)
            if hasattr(ds, 'pixel_array') and ds.pixel_array is not None:
                slices.append(ds)
        except InvalidDicomError:
            continue
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")

    if not slices:
        raise ValueError("No valid DICOM image files found")

    try:
        slices.sort(key=lambda x: x.InstanceNumber)
    except:
        try:
            slices.sort(key=lambda x: float(x.SliceLocation))
        except:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    pixel_arrays = [ds.pixel_array for ds in slices]
    volume = np.stack(pixel_arrays, axis=0)
    volume = volume.astype(np.float32)
    return volume


def load_medical_image(path):
    temp_files = []
    try:
        if os.path.isdir(path):
            print(f"Loading DICOM series from folder: {path}")
            img_data = load_dicom_series(path)
            print(f"Loaded DICOM volume with shape: {img_data.shape}")
            folder_name = os.path.basename(path)
            return img_data, None, temp_files, folder_name

        elif os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            if path.endswith('.nii.gz'):
                print(f"Loading NIfTI .nii.gz file: {path}")
                unzipped_path = unzip_nifti_gz(path)
                temp_files.append(unzipped_path)
                img = nib.load(unzipped_path)
                img_data = img.get_fdata()
                print(f"Image shape: {img_data.shape}")
                file_id = get_filename_without_extension(path)
                return img_data, img, temp_files, file_id

            elif ext in ['.nii']:
                print(f"Loading NIfTI .nii file: {path}")
                img = nib.load(path)
                img_data = img.get_fdata()
                print(f"Image shape: {img_data.shape}")
                file_id = get_filename_without_extension(path)
                return img_data, img, temp_files, file_id

            elif ext == '.dcm':
                print(f"Loading single DICOM file: {path}")
                ds = dcmread(path, force=True)
                img_data = ds.pixel_array.astype(np.float32)
                if len(img_data.shape) == 2:
                    img_data = img_data[np.newaxis, :, :]
                print(f"Image shape: {img_data.shape}")
                file_id = get_filename_without_extension(path)
                return img_data, None, temp_files, file_id

            else:
                raise ValueError(f"Unsupported file format: {ext}")
        else:
            raise ValueError(f"Path does not exist: {path}")

    except Exception as e:
        cleanup_temp_files(temp_files)
        raise


def generate_surface_mesh_robust(img_data, threshold, max_attempts=5):
    thresholds_to_try = [0.3, 0.25, 0.35, 0.4, 0.2, 0.45, 0.15, 0.5]
    thresholds_to_try = sorted(set(thresholds_to_try))
    print(
        f"Attempting surface extraction with thresholds: {thresholds_to_try}")

    for thresh in thresholds_to_try:
        try:
            print(f"  Trying threshold = {thresh}...")
            node, face, regions, centroids = i2m.v2s(
                img_data,
                isovalues=thresh,
                opt={'distbound': 2, 'maxnode': 20000}
            )
            if len(face) > 0 and len(node) > 0:
                print(
                    f"  ✓ Success with threshold {thresh}: {len(node)} nodes, {len(face)} faces")
                return node, face, regions, centroids, thresh
            else:
                print(f"  ✗ No surface found with threshold {thresh}")
        except Exception as e:
            print(f"  ✗ Failed with threshold {thresh}: {str(e)[:100]}...")
            continue

    print("\nAll threshold attempts failed. Trying with smoothing...")
    try:
        from scipy.ndimage import gaussian_filter
        smoothed_img = gaussian_filter(img_data, sigma=1.0)
        for thresh in [0.3, 0.25, 0.35, 0.4]:
            try:
                print(f"  Trying smoothed with threshold = {thresh}...")
                node, face, regions, centroids = i2m.v2s(
                    smoothed_img,
                    isovalues=thresh,
                    opt={'distbound': 2, 'maxnode': 20000}
                )
                if len(face) > 0 and len(node) > 0:
                    print(
                        f"  ✓ Success with smoothed threshold {thresh}: {len(node)} nodes, {len(face)} faces")
                    return node, face, regions, centroids, thresh
            except:
                continue
    except Exception as e:
        print(f"Smoothing approach failed: {e}")

    print("\nTrying with different CGAL parameters...")
    try:
        node, face, regions, centroids = i2m.v2s(
            img_data,
            isovalues=0.3,
            opt={'distbound': 4, 'maxnode': 30000,
                 'radbound': 5, 'maxsurfnode': 50000}
        )
        if len(face) > 0 and len(node) > 0:
            print(
                f"✓ Success with modified parameters: {len(node)} nodes, {len(face)} faces")
            return node, face, regions, centroids, 0.3
    except Exception as e:
        print(f"Modified parameters failed: {e}")

    print("\nTrying adaptive threshold based on image statistics...")
    try:
        flat_img = img_data.flatten()
        for percentile in [75, 80, 85, 90, 95]:
            thresh = np.percentile(flat_img, percentile)
            if thresh < 0.1 or thresh > 0.9:
                continue
            try:
                print(
                    f"  Trying percentile {percentile}% (threshold = {thresh:.3f})...")
                node, face, regions, centroids = i2m.v2s(
                    img_data,
                    isovalues=thresh,
                    opt={'distbound': 2, 'maxnode': 20000}
                )
                if len(face) > 0 and len(node) > 0:
                    print(
                        f"  ✓ Success with adaptive threshold {thresh}: {len(node)} nodes, {len(face)} faces")
                    return node, face, regions, centroids, thresh
            except:
                continue
    except Exception as e:
        print(f"Adaptive threshold failed: {e}")

    raise RuntimeError("All surface mesh generation strategies failed.")


def create_head_mesh_from_brain(node, face, scale_factor=1.2):
    """
    Create a head (scalp) mesh by scaling the brain surface outward from its centroid.
    Returns (head_nodes, head_faces) with the same topology.
    """
    centroid = np.mean(node, axis=0)
    head_nodes = centroid + scale_factor * (node - centroid)
    return head_nodes, face


# =============================================================================
# MAIN SCRIPT
# =============================================================================

# <-- Replace with your path
medical_path = r"replace_with_your_path_to_nifti_or_dicom"

img_data = None
img = None
temp_files = []
file_id = None

try:
    img_data, img, temp_files, file_id = load_medical_image(medical_path)
except Exception as e:
    print(f"Error loading image: {e}")
    print("Please check the path and ensure it's a valid NIfTI file or DICOM folder")
    exit()

if file_id is None:
    file_id = "brain"

print(f"Processing file: {file_id}")

# Normalize image
img_data = np.nan_to_num(img_data)
if img_data.max() > img_data.min():
    img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
else:
    print("WARNING: Image has constant values. Normalization skipped.")
    img_data = img_data - img_data.min()

print(f"Image intensity range: [{img_data.min():.3f}, {img_data.max():.3f}]")
print(
    f"Image statistics: mean={np.mean(img_data):.3f}, std={np.std(img_data):.3f}")

# --------------------------------------------------------------------------
# 1. EXTRACT BRAIN SURFACE
# --------------------------------------------------------------------------
print("\n=== Extracting brain surface ===")
try:
    brain_nodes, brain_faces, regions, centroids, used_threshold = generate_surface_mesh_robust(
        img_data, threshold=0.3)
    print(
        f"Successfully generated brain surface with threshold {used_threshold}")
except Exception as e:
    print(f"ERROR: {e}")
    cleanup_temp_files(temp_files)
    exit()

print(f"Brain surface: {len(brain_nodes)} nodes, {len(brain_faces)} faces")

# Clean brain surface
print("Cleaning brain surface...")
try:
    brain_nodes, brain_faces = i2m.meshcheckrepair(brain_nodes, brain_faces)
    print(f"After repair: {len(brain_nodes)} nodes, {len(brain_faces)} faces")
except Exception as e:
    print(f"Warning: Brain mesh repair failed: {e}")

# --------------------------------------------------------------------------
# 2. GENERATE BRAIN TETRAHEDRAL MESH (volumetric)
# --------------------------------------------------------------------------
print("\n=== Generating brain tetrahedral mesh ===")
if len(brain_faces) == 0 or len(brain_nodes) == 0:
    print("ERROR: Invalid surface mesh, cannot generate tetrahedral mesh. Skipping.")
    brain_nodes_tet = brain_nodes
    brain_elem = brain_faces
else:
    try:
        maxvol_values = [0.03, 0.05, 0.02, 0.01]
        for maxvol in maxvol_values:
            try:
                print(f"  Trying surf2mesh with maxvol = {maxvol}...")
                brain_nodes_tet, brain_elem, _ = i2m.surf2mesh(
                    brain_nodes, brain_faces, [], [], 0.8, maxvol, [], [], 0, 'tetgen1.5'
                )
                if len(brain_elem) > 0 and len(brain_nodes_tet) > 0:
                    print(
                        f"  ✓ Success with maxvol = {maxvol}: {len(brain_nodes_tet)} nodes, {len(brain_elem)} elements")
                    break
            except Exception as e:
                print(f"  ✗ Failed with maxvol {maxvol}: {str(e)[:100]}")
                continue
        else:
            raise RuntimeError("All maxvol values failed")
        print(
            f"Brain tetrahedral mesh: {len(brain_nodes_tet)} nodes, {len(brain_elem)} elements")
    except Exception as e:
        print(f"Error with surf2mesh: {e}")
        print("Trying alternative method (v2m)...")
        try:
            mask = (img_data > used_threshold).astype(np.uint8)
            for maxvol in [5, 3, 8]:
                try:
                    print(f"  Trying v2m with maxvol = {maxvol}...")
                    brain_nodes_tet, brain_elem, _ = i2m.v2m(
                        mask, [], maxvol, 100, 'cgalmesh')
                    if len(brain_elem) > 0 and len(brain_nodes_tet) > 0:
                        print(f"  ✓ Success with v2m maxvol = {maxvol}")
                        break
                except Exception as e2:
                    print(f"  ✗ Failed: {str(e2)[:100]}")
                    continue
            else:
                raise RuntimeError("All v2m attempts failed")
            print(
                f"Using v2m: {len(brain_nodes_tet)} nodes, {len(brain_elem)} elements")
        except Exception as e2:
            print(f"Error with v2m: {e2}")
            print("Only brain surface will be saved.")
            brain_nodes_tet = brain_nodes
            brain_elem = brain_faces

# --------------------------------------------------------------------------
# 3. CREATE HEAD (SCALP) MESH FROM IMAGE (or fallback to scaling)
# --------------------------------------------------------------------------
print("\n=== Creating head (scalp) mesh ===")
head_nodes = head_faces = None
try:
    # The updated extract_head_surface now avoids the problematic meshresample
    head_nodes, head_faces = extract_head_surface(img_data)
except Exception as e:
    print(f"Extraction from image failed: {e}")
    print("Falling back to scaled brain surface...")
    scale_factor = 1.2
    head_nodes, head_faces = create_head_mesh_from_brain(
        brain_nodes, brain_faces, scale_factor)
    try:
        head_nodes, head_faces = i2m.meshcheckrepair(
            head_nodes, head_faces, 'cgal')
    except:
        head_nodes, head_faces = i2m.meshcheckrepair(head_nodes, head_faces)

if head_nodes is None or len(head_faces) == 0:
    print("ERROR: Could not generate head surface. Exiting.")
    cleanup_temp_files(temp_files)
    exit()

print(f"Final head surface: {len(head_nodes)} nodes, {len(head_faces)} faces")

# We skip head tetrahedral mesh generation – not needed for 10-20 electrode system
# and it was causing hangs. We just use the surface mesh.
head_nodes_tet = head_nodes
head_elem = head_faces

# --------------------------------------------------------------------------
# 4. VISUALIZATION (optional)
# --------------------------------------------------------------------------
print("\n=== Visualizing meshes ===")
try:
    if len(brain_elem) > 0 and len(brain_nodes_tet) > 0:
        i2m.plotmesh(brain_nodes_tet, brain_elem, alpha=0.3,
                     title="Brain tetrahedral mesh")
    else:
        i2m.plotmesh(brain_nodes, brain_faces,
                     alpha=0.3, title="Brain surface")
except:
    print("Could not plot brain mesh.")

try:
    if len(head_elem) > 0 and len(head_nodes_tet) > 0:
        i2m.plotmesh(head_nodes_tet, head_elem, alpha=0.3,
                     title="Head surface (10-20)")
    else:
        i2m.plotmesh(head_nodes, head_faces, alpha=0.3,
                     title="Head surface (10-20)")
except:
    print("Could not plot head mesh.")

# --------------------------------------------------------------------------
# 5. SAVE ALL MESHES
# --------------------------------------------------------------------------
print("\n=== Saving mesh files ===")
output_dir = os.path.dirname(medical_path) if os.path.isfile(
    medical_path) else medical_path

# ----- Brain meshes -----
try:
    out_brain_msh = os.path.join(output_dir, f"{file_id}_brain_mesh.msh")
    i2m.savemsh(brain_nodes_tet, brain_elem, out_brain_msh)
    print(f"Saved brain tetrahedral mesh: {out_brain_msh}")
except Exception as e:
    print(f"Error saving brain .msh: {e}")

try:
    out_brain_stl = os.path.join(output_dir, f"{file_id}_brain_surface.stl")
    face0_brain = brain_faces - 1  # iso2mesh expects 0‑based faces for STL
    i2m.savestl(brain_nodes, face0_brain, out_brain_stl)
    print(f"Saved brain surface: {out_brain_stl}")
except Exception as e:
    print(f"Error saving brain .stl: {e}")

# ----- Head meshes (10-20 electrode head model) -----
try:
    out_head_msh = os.path.join(output_dir, f"{file_id}_10-20_headmesh.msh")
    i2m.savemsh(head_nodes_tet, head_elem, out_head_msh)
    print(f"Saved 10-20 head surface mesh: {out_head_msh}")
except Exception as e:
    print(f"Error saving head .msh: {e}")

try:
    out_head_stl = os.path.join(output_dir, f"{file_id}_10-20_headmesh.stl")
    face0_head = head_faces - 1
    i2m.savestl(head_nodes, face0_head, out_head_stl)
    print(f"Saved 10-20 head surface: {out_head_stl}")
except Exception as e:
    print(f"Error saving head .stl: {e}")

# --------------------------------------------------------------------------
# 6. SUMMARY
# --------------------------------------------------------------------------
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Brain surface : {len(brain_nodes)} nodes, {len(brain_faces)} faces")
if len(brain_elem) > 0 and len(brain_nodes_tet) > 0:
    print(
        f"Brain tetra   : {len(brain_nodes_tet)} nodes, {len(brain_elem)} elements")
else:
    print("Brain tetra   : not generated")
print(f"Head surface  : {len(head_nodes)} nodes, {len(head_faces)} faces")
print("Head tetra    : skipped (not needed for 10-20 system)")
print("\nFiles saved:")
print(f"  Brain: {file_id}_brain_surface.stl, {file_id}_brain_mesh.msh")
print(f"  Head : {file_id}_10-20_headmesh.stl, {file_id}_10-20_headmesh.msh")
print("="*60)

cleanup_temp_files(temp_files)
print("Processing complete!")
