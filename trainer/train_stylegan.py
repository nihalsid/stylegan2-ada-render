import torch.multiprocessing

import math
import shutil
from pathlib import Path

import torch
import hydra
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from torchvision.utils import save_image
from cleanfid import fid

from dataset.render_single_mesh import RenderedSingleMeshDataset
from model.augment import AugmentPipe
from model.differentiable_renderer import DifferentiableRenderer
from model.generator import Generator
from model.discriminator import Discriminator
from model.loss import PathLengthPenalty, compute_gradient_penalty
from trainer import create_trainer

torch.multiprocessing.set_sharing_strategy('file_system')  # a fix for the "OSError: too many files" exception

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False


class StyleGAN2Trainer(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.image_size, 3, synthesis_layer=config.generator)
        self.D = Discriminator(config.image_size, 3)
        self.augment_pipe = AugmentPipe(config.ada_start_p, config.ada_target, config.ada_interval, config.ada_fixed, config.batch_size)
        self.grid_z = torch.randn(config.num_eval_images, self.config.latent_dim)
        self.R = None
        self.train_set = RenderedSingleMeshDataset("data/render_people_0/000006_tri.obj", "data/render_people_0/tex/rp_philip_animated_006_dif.jpg", 8192)
        self.val_set = RenderedSingleMeshDataset("data/render_people_0/000006_tri.obj", "data/render_people_0/tex/rp_philip_animated_006_dif.jpg", config.num_eval_images)
        self.automatic_optimization = False
        self.path_length_penalty = PathLengthPenalty(0.01, 2)
        self.ema = None

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(list(self.G.parameters()), lr=self.config.lr_g, betas=(0.0, 0.99), eps=1e-8)
        d_opt = torch.optim.Adam(self.D.parameters(), lr=self.config.lr_d, betas=(0.0, 0.99), eps=1e-8)
        return g_opt, d_opt

    def forward(self, limit_batch_size=False):
        z = self.latent(limit_batch_size)
        w = self.get_mapped_latent(z, 0.9)
        fake = self.G.synthesis(w)
        return fake, w

    def g_step(self):
        g_opt = self.optimizers()[0]
        g_opt.zero_grad(set_to_none=True)
        fake, w = self.forward()
        p_fake = self.D(self.augment_pipe(fake))
        gen_loss = torch.nn.functional.softplus(-p_fake).mean()
        self.manual_backward(gen_loss)
        step(g_opt, self.G)
        self.log("G", gen_loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)

    def g_regularizer(self):
        g_opt = self.optimizers()[0]
        g_opt.zero_grad(set_to_none=True)
        fake, w = self.forward()
        plp = self.path_length_penalty(fake, w)
        if not torch.isnan(plp):
            plp_loss = self.config.lambda_plp * plp * self.config.lazy_path_penalty_interval
            self.manual_backward(plp_loss)
            step(g_opt, self.G)
            self.log("rPLP", plp.item(), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

    def d_step(self, batch):
        d_opt = self.optimizers()[1]
        d_opt.zero_grad(set_to_none=True)
        fake, _ = self.forward()
        p_fake = self.D(self.augment_pipe(fake.detach()))
        fake_loss = torch.nn.functional.softplus(p_fake).mean()
        self.manual_backward(fake_loss)

        p_real = self.D(self.augment_pipe(batch["image"]))
        self.augment_pipe.accumulate_real_sign(p_real.sign().detach())
        real_loss = torch.nn.functional.softplus(-p_real).mean()
        self.manual_backward(real_loss)

        step(d_opt, self.D)
        self.log("D_real", real_loss.item(), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log("D_fake", fake_loss.item(), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        disc_loss = (real_loss + fake_loss).item()
        self.log("D", disc_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)

    def d_regularizer(self, batch):
        d_opt = self.optimizers()[1]
        d_opt.zero_grad(set_to_none=True)
        batch["image"].requires_grad_(True)
        p_real = self.D(self.augment_pipe(batch["image"], disable_grid_sampling=True))
        gp = compute_gradient_penalty(batch["image"], p_real)
        gp_loss = self.config.lambda_gp * gp * self.config.lazy_gradient_penalty_interval
        self.manual_backward(gp_loss)
        step(d_opt, self.D)
        self.log("rGP", gp.item(), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        self.g_step()
        if self.global_step > self.config.lazy_path_penalty_after and (self.global_step + 1) % self.config.lazy_path_penalty_interval == 0:
            self.g_regularizer()

        self.d_step(batch)

        if (self.global_step + 1) % self.config.lazy_gradient_penalty_interval == 0:
            self.d_regularizer(batch)

        self.execute_ada_heuristics()
        self.ema.update(self.G.parameters())

    def execute_ada_heuristics(self):
        if (self.global_step + 1) % self.config.ada_interval == 0:
            self.augment_pipe.heuristic_update()
        self.log("aug_p", self.augment_pipe.p.item(), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        pass

    @rank_zero_only
    def validation_epoch_end(self, _val_step_outputs):
        odir_real, odir_fake, odir_samples, odir_grid = self.create_directories()
        Path("runs", self.config.experiment, "checkpoints").mkdir(exist_ok=True)
        torch.save(self.ema, Path("runs") / self.config.experiment / "checkpoints" / f"ema_{self.global_step:09d}.pth")
        self.export_images("", odir_grid, None, None)
        self.ema.store(self.G.parameters())
        self.ema.copy_to([p for p in self.G.parameters() if p.requires_grad])
        self.export_images("ema_", odir_grid, odir_samples, odir_fake)
        self.ema.restore([p for p in self.G.parameters() if p.requires_grad])
        for iter_idx, batch in enumerate(self.val_dataloader()):
            save_image(batch['image'], odir_samples / f"real_{iter_idx}.jpg", value_range=(-1, 1), normalize=True)
            for batch_idx in range(batch['image'].shape[0]):
                save_image(batch['image'][batch_idx], odir_real / f"{iter_idx}_{batch_idx}.jpg", value_range=(-1, 1), normalize=True)
        fid_score = fid.compute_fid(odir_real, odir_fake, device=self.device)
        kid_score = fid.compute_kid(odir_real, odir_fake, device=self.device)
        self.log(f"fid", fid_score, on_step=False, on_epoch=True, prog_bar=False, logger=True, rank_zero_only=True)
        self.log(f"kid", kid_score, on_step=False, on_epoch=True, prog_bar=False, logger=True, rank_zero_only=True)
        print(f'FID: {fid_score:.3f} , KID: {kid_score:.3f}')
        shutil.rmtree(odir_real.parent)

    def get_mapped_latent(self, z, style_mixing_prob):
        if torch.rand(()).item() < style_mixing_prob:
            cross_over_point = int(torch.rand(()).item() * self.G.mapping.num_ws)
            w1 = self.G.mapping(z[0])[:, :cross_over_point, :]
            w2 = self.G.mapping(z[1], skip_w_avg_update=True)[:, cross_over_point:, :]
            return torch.cat((w1, w2), dim=1)
        else:
            w = self.G.mapping(z[0])
            return w

    def latent(self, limit_batch_size=False):
        batch_size = self.config.batch_size if not limit_batch_size else self.config.batch_size // self.path_length_penalty.pl_batch_shrink
        z1 = torch.randn(batch_size, self.config.latent_dim).to(self.device)
        z2 = torch.randn(batch_size, self.config.latent_dim).to(self.device)
        return z1, z2

    def train_dataloader(self):
        return DataLoader(self.train_set, self.config.batch_size, shuffle=True, pin_memory=False, drop_last=True, num_workers=self.config.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.config.batch_size, shuffle=True, drop_last=True, num_workers=self.config.num_workers)

    def export_images(self, prefix, output_dir_grid, output_dir_samples, output_dir_fid):
        vis_generated_images = []
        for iter_idx, latent in enumerate(self.grid_z.split(self.config.batch_size)):
            latent = latent.to(self.device)
            fake = self.G(latent, noise_mode='const').cpu()
            if output_dir_fid is not None:
                for batch_idx in range(fake.shape[0]):
                    save_image(fake[batch_idx], output_dir_fid / f"{iter_idx}_{batch_idx}.jpg", value_range=(-1, 1), normalize=True)
            if iter_idx < self.config.num_vis_images // self.config.batch_size:
                if output_dir_samples is not None:
                    save_image(fake, output_dir_samples / f"fake_{iter_idx}.jpg", value_range=(-1, 1), normalize=True)
                vis_generated_images.append(fake)
        torch.cuda.empty_cache()
        vis_generated_images = torch.cat(vis_generated_images, dim=0)
        save_image(vis_generated_images, output_dir_grid / f"{prefix}{self.global_step:06d}.png", nrow=int(math.sqrt(vis_generated_images.shape[0])), value_range=(-1, 1), normalize=True)

    def create_directories(self):
        output_dir_fid_real = Path(f'runs/{self.config.experiment}/fid/real')
        output_dir_fid_fake = Path(f'runs/{self.config.experiment}/fid/fake')
        output_dir_fid_grid = Path(f'runs/{self.config.experiment}/grid/')
        output_dir_fid_samples = Path(f'runs/{self.config.experiment}/images/{self.global_step:06d}')
        for odir in [output_dir_fid_real, output_dir_fid_fake, output_dir_fid_grid, output_dir_fid_samples]:
            odir.mkdir(exist_ok=True, parents=True)
        return output_dir_fid_real, output_dir_fid_fake, output_dir_fid_samples, output_dir_fid_grid

    def on_train_start(self):
        self.run_post_device_setup()

    def on_validation_start(self):
        self.run_post_device_setup()

    def run_post_device_setup(self):
        if self.ema is None:
            self.ema = ExponentialMovingAverage(self.G.parameters(), 0.995)
        if self.R is None:
            self.R = DifferentiableRenderer(self.config.image_size, "bounds")
            self.train_set.set_renderer(self.R)
            self.train_set.set_device(self.device)
            self.val_set.set_renderer(self.R)
            self.val_set.set_device(self.device)
        if self.config.resume_ema is not None:
            self.ema = torch.load(self.config.resume_ema, map_location=self.device)


def step(opt, module):
    for param in module.parameters():
        if param.grad is not None:
            torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
    # torch.nn.utils.clip_grad_norm_(module.parameters(), 1)
    opt.step()


@hydra.main(config_path='../config', config_name='stylegan2')
def main(config):
    trainer = create_trainer("RenderGAN", config)
    model = StyleGAN2Trainer(config)
    trainer.fit(model)


if __name__ == '__main__':
    main()
