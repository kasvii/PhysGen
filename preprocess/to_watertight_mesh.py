import torch
import numpy as np
import trimesh
import argparse
from tqdm import tqdm
import os
import json

def generate_dense_grid_points(
    bbox_min = np.array((-1.05, -1.05, -1.05)),
    bbox_max = np.array((1.05, 1.05, 1.05)),
    resolution = 512,
    indexing = "ij"
):
    x = np.linspace(bbox_min[0], bbox_max[0], resolution + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], resolution + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], resolution + 1, dtype=np.float32)
    xs, ys, zs = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    xyz = xyz.reshape(-1, 3)
    grid_size = [resolution + 1, resolution + 1, resolution + 1]

    return xyz, grid_size


def remesh(grid_xyz, grid_size, mesh_path, remesh_path, resolution, use_pcu):
    eps = 2 / resolution
    mesh = trimesh.load(mesh_path, force='mesh')
    mesh.fix_normals()
    
    try:
        mesh = trimesh.util.concatenate(mesh.dump())
    except Exception:
        mesh = trimesh.util.concatenate(mesh)

    vertices = mesh.vertices
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) / 2
    scale = 2.0 / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale

    mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
    mesh.export(remesh_path)

def main(resolution, json_file_path, remesh_target_path, use_pcu) -> None:
    grid_xyz = None
    grid_size = None
    with open(json_file_path, 'r') as f:
        meshes_paths = json.load(f)
    for mesh_path in tqdm(meshes_paths, desc="Processing meshes"):
        part_dir = remesh_target_path
        os.makedirs(part_dir, exist_ok=True)
        basename = os.path.basename(mesh_path)
        remesh_path = part_dir + '/' + basename.replace('stl','obj')
        print('process: '+remesh_path)
        if os.path.exists(remesh_path)==False:
            try:
                remesh(grid_xyz, grid_size, mesh_path, remesh_path, resolution, use_pcu)
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"ERROR: in processing path: {remesh_path}. Error: {e}")
                torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resolution",
        default="512",
        type= int,
        help=".",
    )
    parser.add_argument(
        "--json_file_path",
        type= str,
        help="Specify the JSON file to be traversed.",
    )
    parser.add_argument(
        "--remesh_target_path",
        type= str,
        help="Specify the remesh directory to be saved.",
    )
    parser.add_argument(
        "--use_pcu",
        action='store_true',
        help="If set to False, use cubvh (GPU-required). \
            It's fast for meshes with a moderate number of faces \
            but becomes extremely slow or causes GPU memory leakage for large number of faces. \
            If set to True, use pcu (CPU-based).\
            It's generally slower than cubvh but avoids GPU leakage overflow issues.",
    )
    args, extras = parser.parse_known_args()
    main(args.resolution, args.json_file_path, args.remesh_target_path, args.use_pcu)
