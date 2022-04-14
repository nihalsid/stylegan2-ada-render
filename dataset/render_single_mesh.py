import pickle

import torch
from pathlib import Path
import random
import math
import numpy as np
from PIL import Image

from model.differentiable_renderer import transform_pos_mvp
from util.camera import spherical_coord_to_cam
from util.misc import vertex_to_normals, load_laplacian, load_inverse_laplacian


def get_random_view():
    azimuth = random.random() * 2 * math.pi
    elevation = math.pi / 8 + random.random() * (math.pi / 2 - math.pi / 8)
    perspective_cam = spherical_coord_to_cam(50, azimuth, elevation, cam_dist=1.35)
    projection_matrix = torch.from_numpy(perspective_cam.projection_mat()).float()
    view_matrix = torch.from_numpy(perspective_cam.view_mat()).float()
    return projection_matrix, view_matrix


class RenderedSingleMeshDataset(torch.utils.data.Dataset):

    def __init__(self, mesh_path, texture_image_path, size, renderer='texture'):
        super().__init__()
        self.size = size
        self.renderer = None
        self.device = None
        vertices, faces, uvs, uv_indices = load_mesh(mesh_path)
        vertex_bounds = (vertices.min(axis=0), vertices.max(axis=0))
        self.vertices = vertices - (vertex_bounds[0] + vertex_bounds[1]) / 2
        self.vertices = self.vertices / (vertex_bounds[1] - vertex_bounds[0]).max()
        self.vertices = torch.from_numpy(self.vertices).float()
        self.faces = torch.from_numpy(faces).int()
        self.uvs = torch.from_numpy(uvs).float()
        self.uvs = torch.cat([self.uvs[:, 0:1], 1 - self.uvs[:, 1:2]], dim=1)
        self.uv_indices = torch.from_numpy(uv_indices).int()
        self.texture_image = torch.from_numpy(np.array(Image.open(texture_image_path))).float() / 127.5 - 1
        self.get_render = self.get_textured_render if renderer == 'texture' else self.get_normal_render
        self.normals = vertex_to_normals(self.vertices, self.faces.long())

    def set_renderer(self, renderer):
        self.renderer = renderer

    def set_device(self, device):
        self.device = device
        self.vertices = self.vertices.to(device)
        self.uvs = self.uvs.to(device)
        self.faces = self.faces.to(device)
        self.uv_indices = self.uv_indices.to(device)
        self.texture_image = self.texture_image.to(device)
        self.normals = self.normals.to(device)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        projection_matrix, view_matrix = get_random_view()
        vertices = transform_pos_mvp(self.vertices, torch.matmul(projection_matrix, view_matrix).to(self.device).unsqueeze(0))
        return self.get_render(index, vertices)

    def get_textured_render(self, index, vertices):
        rendered_image = self.renderer.render_with_texture_map(vertices, self.faces, self.uvs, self.uv_indices, self.texture_image).permute((0, 3, 1, 2))
        return {
            "name": f"{index:06d}",
            "image": rendered_image.squeeze(0),
        }

    def get_normal_render(self, index, vertices):
        rendered_image = self.renderer.render(vertices, self.faces, self.normals).permute((0, 3, 1, 2))
        return {
            "name": f"{index:06d}",
            "image": rendered_image.squeeze(0),
        }


class RenderSMPLOffsetDataset(torch.utils.data.Dataset):

    def __init__(self, mesh_path_smpl, mesh_path_offset, laplacian_path, inverse_laplacian_path, extended_path, size):
        super().__init__()
        self.size = size
        self.renderer = None
        self.device = None
        vertices_smpl, faces, _, _ = load_mesh(mesh_path_smpl)
        vertices_offset, _, _, _ = load_mesh(mesh_path_offset)
        vertex_bounds = (vertices_smpl.min(axis=0), vertices_smpl.max(axis=0))

        with open(extended_path, 'rb') as file:
            extended = pickle.load(file)

        self.vertices_smpl = vertices_smpl - (vertex_bounds[0] + vertex_bounds[1]) / 2
        self.vertices_smpl = self.vertices_smpl / (vertex_bounds[1] - vertex_bounds[0]).max()
        self.vertices_offset = vertices_offset - (vertex_bounds[0] + vertex_bounds[1]) / 2
        self.vertices_offset = self.vertices_offset / (vertex_bounds[1] - vertex_bounds[0]).max()
        self.vertices_smpl = torch.from_numpy(self.vertices_smpl).float()
        self.vertices_offset = torch.from_numpy(self.vertices_offset).float()
        self.faces = torch.from_numpy(faces).int()
        self.uv = torch.zeros([self.vertices_smpl.shape[0], 2])
        self.uv[extended['indices'], :] = torch.tensor(extended['uv'], dtype=torch.float32)
        self.normals_offset = vertex_to_normals(self.vertices_offset, self.faces.long())
        self.laplacian = torch.tensor(load_laplacian(laplacian_path, lambda_key='L_32.0')).float()
        self.inverse_laplacian = torch.tensor(load_inverse_laplacian(inverse_laplacian_path, lambda_key='L_32.0')).float()

    def set_renderer(self, renderer):
        self.renderer = renderer

    def set_device(self, device):
        self.device = device
        self.vertices_smpl = self.vertices_smpl.to(device)
        self.vertices_offset = self.vertices_offset.to(device)
        self.faces = self.faces.to(device)
        self.normals_offset = self.normals_offset.to(device)
        self.laplacian = self.laplacian.to(device)
        self.inverse_laplacian = self.inverse_laplacian.to(device)
        self.uv = self.uv.to(device)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        projection_matrix, view_matrix = get_random_view()
        vertices = transform_pos_mvp(self.vertices_offset, torch.matmul(projection_matrix, view_matrix).to(self.device).unsqueeze(0))
        rendered_image = self.renderer.render(vertices, self.faces, self.normals_offset).permute((0, 3, 1, 2))
        return {
            "name": f"{index:06d}",
            "image": rendered_image.squeeze(0),
        }


def load_mesh(mesh_path):
    cvt = lambda x, t: [t(y) for y in x]
    mesh_text = Path(mesh_path).read_text().splitlines()
    vertices, indices, uvs, uv_indices = [], [], [], []
    for line in mesh_text:
        if line.startswith("v "):
            vertices.append(cvt(line.split(" ")[1:4], float))
        if line.startswith("vt "):
            uvs.append(cvt(line.split(" ")[1:], float))
        if line.startswith("f "):
            if '/' in line:
                indices.append([int(x.split('/')[0]) - 1 for x in line.split(' ')[1:]])
                uv_indices.append([int(x.split('/')[1]) - 1 for x in line.split(' ')[1:]])
            else:
                indices.append([int(x) - 1 for x in line.split(' ')[1:]])
    return np.array(vertices), np.array(indices), np.array(uvs), np.array(uv_indices)
