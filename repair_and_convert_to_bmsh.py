import pyvista as pv
import jdata as jd
import numpy as np
from pathlib import Path
import argparse   # <-- added


def main():
    parser = argparse.ArgumentParser(
        description="Repair and convert a head STL mesh to BMSH format."
    )
    parser.add_argument(
        "--input_stl", required=True,
        help="Path to the input STL file (e.g., *_10-20_headmesh.stl)"
    )
    parser.add_argument(
        "--output_bmsh", required=True,
        help="Path where the output BMSH file will be saved"
    )
    parser.add_argument(
        "--target_faces", type=int, default=20000,
        help="Target number of faces after decimation (default: 20000)"
    )
    args = parser.parse_args()

    stl_file = Path(args.input_stl)
    if not stl_file.exists():
        print(f"❌ Error: STL file not found: {stl_file}")
        return

    # Load the surface
    surf = pv.read(stl_file)
    print(f"Initial: {surf.n_points} pts, {surf.n_faces} faces")

    # Decimate if needed
    target_faces = args.target_faces
    if surf.n_faces > target_faces:
        reduction = 1 - (target_faces / surf.n_faces)
        surf = surf.decimate(target_reduction=reduction)
        print(f"Decimated: {surf.n_points} pts, {surf.n_faces} faces")

    # Extract triangle indices (0‑based) and convert to 1‑based
    triangles = surf.faces.reshape(-1, 4)[:, 1:] + 1

    # Build the BMSH metadata
    jmesh_data = {
        "MeshVertex3": surf.points.tolist(),
        "MeshTri3": triangles.tolist(),
        "JMeshVersion": "0.5",
        "Dimension": 3,
        "Comment": "Repaired and decimated head mesh",
        "AnnotationFormat": "https://github.com/NeuroJSON/jmesh/blob/master/JMesh_specification.md",
        "SerialFormat": "http://json.org"
    }

    # Save as binary JMesh (BMSH)
    bmsh_path = Path(args.output_bmsh)
    jd.save(jmesh_data, str(bmsh_path), fmt='bjdata')
    print(f"✅ BMSH saved: {bmsh_path}")


if __name__ == "__main__":
    main()
