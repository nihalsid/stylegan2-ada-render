from pathlib import Path
from model.differentiable_renderer import DifferentiableRenderer
from torchvision.utils import save_image

from torch.utils.data import DataLoader

from dataset import render_single_mesh
import torch


def test_render_dataset():
    Path("runs/render_dataset/").mkdir(exist_ok=True, parents=True)
    _renderer = DifferentiableRenderer(256, mode='bounds', color_space='rgb', num_channels=3)
    dataset = render_single_mesh.RenderedSingleMeshDataset("data/render_people_0/000006_tri.obj", "data/render_people_0/tex/rp_philip_animated_006_dif.jpg", 50, renderer="normal")
    dataset.set_renderer(_renderer)
    dataset.set_device(torch.device("cuda:0"))
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=False)
    for idx, batch in enumerate(loader):
        save_image(batch['image'], f"runs/render_dataset/{idx:04d}.jpg", value_range=(-1, 1), normalize=True)


def test_smpl_render_dataset():
    Path("runs/render_dataset/").mkdir(exist_ok=True, parents=True)
    _renderer = DifferentiableRenderer(256, mode='bounds', color_space='rgb', num_channels=3)
    dataset = render_single_mesh.RenderSMPLOffsetDataset("data/smpl_0/smpl.obj", "data/smpl_0/offset.obj", "data/smpl_0/laplacian.npz", "data/smpl_0/inverse_laplacian.npz", "data/smpl_0/extended.pkl", 50)
    dataset.set_renderer(_renderer)
    dataset.set_device(torch.device("cuda:0"))
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=False)
    for idx, batch in enumerate(loader):
        save_image(batch['image'], f"runs/render_dataset/{idx:04d}.jpg", value_range=(-1, 1), normalize=True)


if __name__ == "__main__":
    test_smpl_render_dataset()
