import trimesh
import numpy as np

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.
 
    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh
 
 
def recenter_mesh(mesh):
  mesh.vertices -= mesh.vertices.mean(0)
  mesh.vertices /= np.max(np.abs(mesh.vertices))
  mesh.vertices = .5 * (mesh.vertices + 1.)

