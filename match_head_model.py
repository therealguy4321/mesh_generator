#!/usr/bin/env python3
"""
Find the best matching scalp mesh from a library of head models
by comparing bounding box extents.

Usage:
    python match_head_model.py <nifti_file.nii.gz> [--models_dir <path>] [--top <N>]
"""

import sys
import argparse
import numpy as np
import nibabel as nib
import jdata as jd
from pathlib import Path


def load_bmsh_extents(bmsh_path):
    """Load a .bmsh file and return its bounding box extents (x, y, z)."""
    data = jd.load(str(bmsh_path))
    # Look for vertices
    vertices = data.get("MeshVertex3", data.get("_MeshVertex3_", None))
    if vertices is None:
        raise ValueError(f"No MeshVertex3 found in {bmsh_path}")
    vertices = np.array(vertices)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    extents = maxs - mins
    return extents


def scan_models_folder(folder_path):
    """Scan folder for .bmsh files and return list of (name, extents, full_path)."""
    folder = Path(folder_path)
    bmsh_files = folder.glob("*.bmsh")
    models = []
    for f in bmsh_files:
        try:
            ext = load_bmsh_extents(f)
            models.append((f.stem, ext, f))
        except Exception as e:
            print(f"Warning: Could not load {f.name}: {e}")
    return models


def compute_head_extents(nii_path, threshold_frac=0.05):
    """Load NIfTI and compute bounding box extents of the head (foreground)."""
    img = nib.load(nii_path)
    data = img.get_fdata()
    # Use a fraction of max intensity to separate head from background
    thresh = data.max() * threshold_frac
    mask = data > thresh
    if not np.any(mask):
        raise ValueError("No foreground found – try adjusting the threshold.")
    coords = np.argwhere(mask)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    extents = maxs - mins
    return extents.astype(float)


def find_best_match(head_extents, models, top_n=1):
    """Find closest models by Euclidean distance of extents."""
    distances = []
    for name, ext, path in models:
        dist = np.linalg.norm(head_extents - ext)
        distances.append((dist, name, ext, path))
    distances.sort(key=lambda x: x[0])
    return distances[:top_n]


def main():
    parser = argparse.ArgumentParser(
        description="Match a NIfTI head volume to the closest scalp mesh."
    )
    parser.add_argument(
        "nifti", help="Path to the NIfTI file (.nii or .nii.gz)")
    parser.add_argument("--models_dir", default="replace with_your_path_to_head_models",
                        help="Folder containing .bmsh head models")
    parser.add_argument("--top", type=int, default=3,
                        help="Show top N matches (default: 3)")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Fraction of max intensity to separate head from background (default: 0.05)")
    args = parser.parse_args()

    # 1. Load head extents from NIfTI
    nii_path = Path(args.nifti)
    if not nii_path.exists():
        print(f"Error: File not found: {nii_path}")
        sys.exit(1)
    print(f"Computing head extents from: {nii_path}")
    try:
        head_ext = compute_head_extents(nii_path, args.threshold)
    except Exception as e:
        print(f"Error processing NIfTI: {e}")
        sys.exit(1)
    print(
        f"Head extents (voxels): X={head_ext[0]:.1f}, Y={head_ext[1]:.1f}, Z={head_ext[2]:.1f}")

    # 2. Scan models folder
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"Error: Models folder not found: {models_dir}")
        sys.exit(1)
    print(f"Scanning models folder: {models_dir}")
    models = scan_models_folder(models_dir)
    if not models:
        print("No .bmsh files found.")
        sys.exit(1)
    print(f"Found {len(models)} models.")

    # 3. Find best matches
    matches = find_best_match(head_ext, models, top_n=args.top)
    print("\nBest matching model(s):")
    for rank, (dist, name, ext, path) in enumerate(matches, 1):
        print(f"\n{rank}. {name}")
        print(f"   Extents: X={ext[0]:.1f}, Y={ext[1]:.1f}, Z={ext[2]:.1f}")
        print(f"   Distance: {dist:.2f}")
        print(f"   File: {path}")


if __name__ == "__main__":
    main()
