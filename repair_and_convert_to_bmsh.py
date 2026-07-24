import pyvista as pv
import jdata as jd
import numpy as np
from pathlib import Path

out_dir = Path(
    r'replace_with_your_path_to_output_directory')
stl_file = out_dir / 'replace_with_your_stl_file.stl'

# Load the repaired surface
surf = pv.read(stl_file)
print(f"Initial: {surf.n_points} pts, {surf.n_faces} faces")

# Decimate to ~20k faces (like the reference)
target_faces = 20000
if surf.n_faces > target_faces:
    reduction = 1 - (target_faces / surf.n_faces)
    surf = surf.decimate(target_reduction=reduction)
    print(f"Decimated: {surf.n_points} pts, {surf.n_faces} faces")

# Extract triangle indices (0‑based)
triangles = surf.faces.reshape(-1, 4)[:, 1:]   # shape (n,3)

# Convert to 1‑based (MATLAB style)
triangles_1based = triangles + 1

# Build the BMSH with all required metadata
jmesh_data = {
    "MeshVertex3": surf.points.tolist(),
    "MeshTri3": triangles_1based.tolist(),
    "JMeshVersion": "0.5",
    "Dimension": 3,
    "CreationTime": "07-Jun-2023 11:46:02",
    "Comment": "Scalp repaired in Blender, decimated to 20k faces",
    "AnnotationFormat": "https://github.com/NeuroJSON/jmesh/blob/master/JMesh_specification.md",
    "SerialFormat": "http://json.org",
    "Parser": {
        "Python": ["https://pypi.org/project/jdata", "https://pypi.org/project/bjdata"],
        "MATLAB": "https://github.com/NeuroJSON/jsonlab",
        "JavaScript": ["https://github.com/NeuroJSON/jsdata", "https://github.com/NeuroJSON/js-bjdata"],
        "CPP": "https://github.com/NeuroJSON/json",
        "C": ["https://github.com/DaveGamble/cJSON", "https://github.com/NeuroJSON/ubj"]
    }
}

# Save as binary JMesh
bmsh_file = out_dir / 'replace_with_your_bmsh_file.bmsh'
jd.save(jmesh_data, str(bmsh_file), fmt='bjdata')
print(f"✅ Working BMSH saved: {bmsh_file}")
