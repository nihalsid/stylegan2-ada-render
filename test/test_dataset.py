from pathlib import Path

from torch.utils.data import DataLoader

from dataset import render_single_mesh
import torch


def test_render_dataset():
    from model.differentiable_renderer import DifferentiableRenderer, transform_pos_mvp
    from torchvision.utils import save_image
    Path("runs/render_dataset/").mkdir(exist_ok=True, parents=True)
    _renderer = DifferentiableRenderer(256, mode='bounds', color_space='rgb', num_channels=3)
    dataset = render_single_mesh.RenderedSingleMeshDataset("data/render_people_0/000006_tri.obj", "data/render_people_0/tex/rp_philip_animated_006_dif.jpg", 50, _renderer, torch.device("cuda:0"))
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=False)
    for idx, batch in enumerate(loader):
        save_image(batch['real'].permute((0, 3, 1, 2)), f"runs/render_dataset/{idx:04d}.jpg", value_range=(-1, 1), normalize=True)


if __name__ == "__main__":
    test_render_dataset()
