import torch
from pathlib import Path
import random
import math
import numpy as np
from PIL import Image

from model.differentiable_renderer import transform_pos_mvp
from util.camera import spherical_coord_to_cam


class RenderedSingleMeshDataset(torch.utils.data.Dataset):

    def __init__(self, mesh_path, texture_image_path, size):
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

    def set_renderer(self, renderer):
        self.renderer = renderer

    def set_device(self, device):
        self.device = device
        self.vertices = self.vertices.to(device)
        self.uvs = self.uvs.to(device)
        self.faces = self.faces.to(device)
        self.uv_indices = self.uv_indices.to(device)
        self.texture_image = self.texture_image.to(device)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        azimuth = random.random() * 2 * math.pi
        elevation = math.pi / 8 + random.random() * (math.pi / 2 - math.pi / 8)
        perspective_cam = spherical_coord_to_cam(50, azimuth, elevation, cam_dist=1.35)
        projection_matrix = torch.from_numpy(perspective_cam.projection_mat()).float()
        view_matrix = torch.from_numpy(perspective_cam.view_mat()).float()
        vertices = transform_pos_mvp(self.vertices, torch.matmul(projection_matrix, view_matrix).to(self.device).unsqueeze(0))
        rendered_image = self.renderer.render_with_texture_map(vertices, self.faces, self.uvs, self.uv_indices, self.texture_image).permute((0, 3, 1, 2))
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
            vertices.append(cvt(line.split(" ")[1:], float))
        if line.startswith("vt "):
            uvs.append(cvt(line.split(" ")[1:], float))
        if line.startswith("f "):
            indices.append([int(x.split('/')[0]) - 1 for x in line.split(' ')[1:]])
            uv_indices.append([int(x.split('/')[1]) - 1 for x in line.split(' ')[1:]])
    return np.array(vertices), np.array(indices), np.array(uvs), np.array(uv_indices)
