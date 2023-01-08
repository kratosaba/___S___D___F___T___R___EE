import trimesh
import numpy as np

def as_mesh(scene_or_mesh):
    """
    This function takes in an object scene_or_mesh, which may be either a trimesh.Scene or trimesh.Trimesh object, and returns a trimesh.Trimesh object. If the input is a trimesh.Scene object, the function concatenates the meshes in the scene into a single mesh. If the input is already a trimesh.Trimesh object, it is returned as is.
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
    """
    This function takes in a trimesh.Trimesh object and recenters it such that the mean of its vertices is at the origin, and the maximum absolute value of its vertices is 1. The vertices are then transformed to the range [-0.5, 0.5] and added to 1 to map them to the range [0.5, 1.5].
    """
    mesh.vertices -= mesh.vertices.mean(0)
    mesh.vertices /= np.max(np.abs(mesh.vertices))
    mesh.vertices = .5 * (mesh.vertices + 1.)

