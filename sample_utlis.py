import numpy as np

def uniform_bary(u):
  su0 = np.sqrt(u[..., 0])
  b0 = 1. - su0
  b1 = u[..., 1] * su0
  return np.stack([b0, b1, 1. - b0 - b1], -1)


def get_normal_batch(mesh, bsize):

  batch_face_inds = np.array(np.random.randint(0, mesh.faces.shape[0], [bsize]))
  batch_barys = np.array(uniform_bary(np.random.uniform(size=[bsize, 2])))
  batch_faces = mesh.faces[batch_face_inds]
  batch_normals = mesh.face_normals[batch_face_inds]
  batch_pts = np.sum(mesh.vertices[batch_faces] * batch_barys[...,None], 1)

  return batch_pts, batch_normals