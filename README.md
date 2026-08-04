```markdown
# 3D Mesh Generator for Neuroimaging

This project provides a set of Python scripts to generate 3D surface and volumetric meshes from raw medical imaging data (NIfTI or DICOM). It is specifically designed for **EEG/MEG source localization** and **finite element simulations**, producing head and brain models compatible with the **10‑20 electrode system**.

---

## Features

- Loads **NIfTI** (`.nii`, `.nii.gz`) and **DICOM** (single file or folder) images.
- Extracts **brain surface** and **head (scalp) surface** using iso2mesh.
- Generates a **tetrahedral volume mesh** for the brain (useful for FEM).
- Creates a **scalp surface** (no volumetric tetra mesh – not required for 10‑20 systems).
- **Repairs** and decimates surfaces, then exports to **.bmsh** (BrainSuite Mesh) format.
- **Matches** a given head volume to the closest scalp mesh from a library by comparing bounding box extents.

---

## Requirements

- **Python** 3.8 or higher
- Required Python packages:
  - `numpy`
  - `scipy`
  - `nibabel`
  - `pydicom`
  - `pyvista`
  - `jdata`
  - `iso2mesh` – for mesh generation (see installation notes below)
  - `trimesh` (optional, used for repair if iso2mesh fails)

Install all with:
```bash
pip install numpy scipy nibabel pydicom pyvista jdata trimesh
```

### Installing iso2mesh

`iso2mesh` is a Python wrapper around the [iso2mesh](https://github.com/fangq/iso2mesh) toolbox. It can be installed via:

```bash
pip install iso2mesh
```

The toolbox relies on external binaries (`tetgen`, `cgal`) for robust meshing. On Linux/macOS, these are often installed automatically. On Windows, you may need to place the binaries in your `PATH` or in the iso2mesh installation folder. See the [official documentation](https://iso2mesh.readthedocs.io/) for details.

---

## Script Descriptions

### 1. `mesh_gen_mixed.py`

**Purpose:**  
Loads a NIfTI or DICOM image, extracts brain and head surfaces, and generates a tetrahedral mesh for the brain. The head is kept as a surface only (no volumetric mesh). All meshes are saved as `.stl` (surface) and `.msh` (Gmsh format).

**Important:**  
The script currently has a **hard‑coded input path**. You must edit the line:
```python
medical_path = r"replace_with_your_path_to_nifti_or_dicom"
```
to point to your NIfTI file or DICOM folder. Future versions may accept command‑line arguments.

**What it does:**
1. Normalizes image intensity.
2. Extracts brain surface using `iso2mesh.v2s` with multiple thresholds and smoothing.
3. Generates a tetrahedral mesh from the brain surface using `iso2mesh.surf2mesh` (or fallback to `v2m`).
4. Extracts head surface from the same image (using `extract_head_surface`).
5. Visualizes both meshes interactively (requires `matplotlib`).
6. Saves:
   - Brain surface: `{file_id}_brain_surface.stl`
   - Brain tetra mesh: `{file_id}_brain_mesh.msh`
   - Head surface: `{file_id}_10-20_headmesh.stl`
   - Head surface (also in `.msh`): `{file_id}_10-20_headmesh.msh`

**Usage (after editing the path):**
```bash
python mesh_gen_mixed.py
```

---

### 2. `repair_and_convert_to_bmsh.py`

**Purpose:**  
Takes a **repaired surface mesh** (e.g., an `.stl` file cleaned in Blender or other tool), decimates it to a target number of faces, and exports it to the **`.bmsh`** format (binary JSON) used by BrainSuite.

**Important:**  
This script also contains hard‑coded paths. You must edit:
```python
out_dir = Path(r'replace_with_your_path_to_output_directory')
stl_file = out_dir / 'replace_with_your_stl_file.stl'
```
and the output `.bmsh` filename near the end.

**What it does:**
- Loads an `.stl` file with `pyvista`.
- Decimates to **20,000 faces** (configurable by changing `target_faces`).
- Extracts triangle indices and converts to 1‑based indexing.
- Builds a `jdata` dictionary with required `JMesh` metadata.
- Saves as a binary `.bmsh` file using the **BJData** format.

**Usage (after editing paths):**
```bash
python repair_and_convert_to_bmsh.py
```

> **Note:** This script does **not** perform automatic repair; it assumes the input mesh is already manifold and closed. If you need repair, use iso2mesh's `meshcheckrepair` or a tool like Blender before running this script.

---

### 3. `match_head_model.py`

**Purpose:**  
Given a NIfTI volume of a head, this script finds the **best matching scalp mesh** from a library of pre‑existing `.bmsh` head models by comparing their bounding box extents (x, y, z dimensions). This is useful for selecting a template head model that closely matches the subject’s head size.

**Usage:**
```bash
python match_head_model.py <nifti_file.nii.gz> [--models_dir <path>] [--top <N>] [--threshold <frac>]
```

**Arguments:**
- `nifti` : Path to the NIfTI file (`.nii` or `.nii.gz`).
- `--models_dir` : Folder containing `.bmsh` files to search (default: `"replace with_your_path_to_head_models"` – you **must** set this to your own models folder).
- `--top` : Number of best matches to show (default: `3`).
- `--threshold` : Fraction of maximum intensity to separate head from background (default: `0.05`). Adjust if the head is not correctly segmented.

**What it does:**
1. Loads the NIfTI and computes the bounding box of the foreground (head).
2. Scans the models folder for all `.bmsh` files, loads each, and extracts its vertex coordinates.
3. Computes the Euclidean distance between the NIfTI extents and each model’s extents.
4. Prints the top N closest models with their extents and file paths.

**Example:**
```bash
python match_head_model.py subject_t1.nii.gz --models_dir ./head_models/ --top 5
```

---

## Example Workflow

1. **Generate meshes** from your subject’s NIfTI:
   - Edit `mesh_gen_mixed.py` to set `medical_path`.
   - Run `python mesh_gen_mixed.py`.
   - This will create a brain tetra mesh and a head surface mesh (`.stl` and `.msh`).

2. **Repair and convert the head surface** to `.bmsh` (optional):
   - Open the generated head `.stl` in a mesh repair tool (e.g., Blender, MeshLab) to fix any holes or non‑manifold edges.
   - Edit `repair_and_convert_to_bmsh.py` to point to the repaired `.stl` and set the output folder.
   - Run `python repair_and_convert_to_bmsh.py` to get a `.bmsh` file.

3. **Match to a library** (optional):
   - If you have a collection of template head `.bmsh` files, use `match_head_model.py` to find the best match for your subject based on head size.

---

## Output Files

| Script | Output Files |
|--------|--------------|
| `mesh_gen_mixed.py` | `{file_id}_brain_surface.stl`, `{file_id}_brain_mesh.msh`, `{file_id}_10-20_headmesh.stl`, `{file_id}_10-20_headmesh.msh` |
| `repair_and_convert_to_bmsh.py` | `{filename}.bmsh` |
| `match_head_model.py` | (prints matches to console) |

---

## Notes and Limitations

- The mesh generation uses **iso2mesh**, which relies on external binaries (`tetgen`, `cgal`). Ensure they are installed and accessible.
- The `extract_head_surface` function in `mesh_gen_mixed.py` attempts multiple thresholds and smoothing strategies but may fail on very noisy or low‑contrast images.
- Head tetrahedral mesh is **not generated** by default, as it is not needed for the 10‑20 electrode system; if you need it, you can modify the script.
- The BMSH conversion script expects a **repaired** surface; it does not perform repair itself.
- The matching script only compares **bounding box extents**; it does not perform non‑rigid registration or shape matching.

---

## Customization

- To change the decimation target in `repair_and_convert_to_bmsh.py`, modify the `target_faces` variable.
- To generate a tetrahedral head mesh, uncomment or adapt the corresponding section in `mesh_gen_mixed.py` (currently skipped with comment).
- For more robust surface extraction, adjust the `threshold_list` in `extract_head_surface`.

---

## License

This project is provided as‑is under the **MIT License**. See the `LICENSE` file (if included) for details.

---

## Acknowledgements

- [iso2mesh](https://github.com/fangq/iso2mesh) for mesh generation.
- [NeuroJSON](https://neurojson.org) for the JMesh/BMSH format and `jdata` library.
- [pyvista](https://pyvista.org) for mesh processing.

---

## Contact

For questions or issues, please open an issue in the repository or contact the project maintainer.

---

**Happy meshing!**
```
