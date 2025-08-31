#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# # Code to convert this notebook to .py if you want to run it via command line or with Slurm
# from subprocess import call
# command = "jupyter nbconvert Train_MindEye.ipynb --to python"
# call(command,shell=True)


# # Import packages & functions

# In[2]:


import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom models and functions #
import utils
from models import Clipper, BrainNetwork, VersatileDiffusionPriorNetwork

# Multi-GPU config #
from accelerate import Accelerator
from torch.utils.data import DataLoader
from datasets.RoiNsdDataset import RoiNsdDataset
from utils import *
import sys  # <-- 추가

TQDM_FILE = sys.stdout



accelerator = Accelerator(split_batches=False, mixed_precision='fp16')
print("PID of this process =", os.getpid())
print = accelerator.print  # only print if local_rank=0
device = accelerator.device
print("device:", device)
num_devices = torch.cuda.device_count()
if num_devices == 0: num_devices = 1
num_workers = num_devices
print(accelerator.state)
local_rank = accelerator.state.local_process_index
world_size = accelerator.state.num_processes
distributed = not accelerator.state.distributed_type == 'NO'
print("distributed =", distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =",
      world_size)

# # Configurations

# In[3]:


# if running this interactively, can specify jupyter_args here for argparser to use
if utils.is_interactive():
    # Example use
    jupyter_args = "--data_path=/fsx/proj-medarc/fmri/natural-scenes-dataset \
                    --model_name=test \
                    --subj=1 --hidden --clip_variant=ViT-L/14 --batch_size=32 --n_samples_save=0 \
                    --max_lr=3e-4 --mixup_pct=.33 --num_epochs=240 --ckpt_interval=5 --use_image_aug"

    jupyter_args = jupyter_args.split()
    print(jupyter_args)

    from IPython.display import clear_output  # function to clear print outputs in cell

    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload',
                                 '2 # this allows you to change functions in models.py or utils.py and have this notebook automatically update with your revisions')

# In[4]:


parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="testing",
    help="name of model, used for ckpt saving and wandb logging (if enabled)",
)
parser.add_argument(
    "--data_path", type=str, default="/fsx/proj-medarc/fmri/natural-scenes-dataset",
    help="Path to where NSD data is stored / where to download it to",
)
parser.add_argument(
    "--subj", type=int, default=1, choices=[1, 2, 5, 7],
)
parser.add_argument(
    "--batch_size", type=int, default=32,
    help="Batch size can be increased by 10x if only training v2c and not diffusion prior",
)
parser.add_argument(
    "--hidden", action=argparse.BooleanOptionalAction, default=True,
    help="if True, CLIP embeddings will come from last hidden layer (e.g., 257x768 - Versatile Diffusion), rather than final layer",
)
parser.add_argument(
    "--clip_variant", type=str, default="ViT-L/14", choices=["RN50", "ViT-L/14", "ViT-B/32", "RN50x64"],
    help='OpenAI clip variant',
)
parser.add_argument(
    "--wandb_log", action=argparse.BooleanOptionalAction, default=False,
    help="whether to log to wandb",
)
parser.add_argument(
    "--resume_from_ckpt", action=argparse.BooleanOptionalAction, default=False,
    help="if not using wandb and want to resume from a ckpt",
)
parser.add_argument(
    "--wandb_project", type=str, default="stability",
    help="wandb project name",
)
parser.add_argument(
    "--mixup_pct", type=float, default=.33,
    help="proportion of way through training when to switch from BiMixCo to SoftCLIP",
)
parser.add_argument(
    "--norm_embs", action=argparse.BooleanOptionalAction, default=True,
    help="Do l2-norming of CLIP embeddings",
)
parser.add_argument(
    "--use_image_aug", action=argparse.BooleanOptionalAction, default=True,
    help="whether to use image augmentation",
)
parser.add_argument(
    "--num_epochs", type=int, default=240,
    help="number of epochs of training",
)
parser.add_argument(
    "--prior", action=argparse.BooleanOptionalAction, default=False,
    help="if False, will only use CLIP loss and ignore diffusion prior",
)
parser.add_argument(
    "--v2c", action=argparse.BooleanOptionalAction, default=True,
    help="if False, will only use diffusion prior loss",
)
parser.add_argument(
    "--plot_umap", action=argparse.BooleanOptionalAction, default=False,
    help="Plot UMAP plots alongside reconstructions",
)
parser.add_argument(
    "--lr_scheduler_type", type=str, default='cycle', choices=['cycle', 'linear'],
)
parser.add_argument(
    "--ckpt_saving", action=argparse.BooleanOptionalAction, default=True,
)
parser.add_argument(
    "--ckpt_interval", type=int, default=5,
    help="save backup ckpt and reconstruct every x epochs",
)
parser.add_argument(
    "--save_at_end", action=argparse.BooleanOptionalAction, default=False,
    help="if True, saves best.ckpt at end of training. if False and ckpt_saving==True, will save best.ckpt whenever epoch shows best validation score",
)
parser.add_argument(
    "--seed", type=int, default=42,
)
parser.add_argument(
    "--max_lr", type=float, default=3e-4,
)
parser.add_argument(
    "--n_samples_save", type=int, default=0, choices=[0, 1],
    help="Number of reconstructions for monitoring progress, 0 will speed up training",
)
parser.add_argument(
    "--use_projector", action=argparse.BooleanOptionalAction, default=True,
    help="Additional MLP after the main MLP so model can separately learn a way to minimize NCE from prior loss (BYOL)",
)
parser.add_argument(
    "--vd_cache_dir", type=str,
    default='/fsx/proj-medarc/fmri/cache/models--shi-labs--versatile-diffusion/snapshots/2926f8e11ea526b562cd592b099fcf9c2985d0b7',
    help="Where is cached Versatile Diffusion model; if not cached will download to this path",
)

# ==== [ADD] NV2L ROI 입력 옵션 ====
parser.add_argument("--use_nv2l_roi", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--nv2l_root", type=str, default=os.environ.get("NV2L_OUT", "/NSD/nv2l_outputs/datasets"))
parser.add_argument("--roi_name", type=str, default="streams")  # make_subjfmri.py에서 atlasname='streams'
parser.add_argument("--roi_use_avg", action=argparse.BooleanOptionalAction, default=True)  # ave 파일 쓸지


if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()

for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)

# 여기에서!
if use_nv2l_roi:
    resume_from_ckpt = False

if args.prior:
    from models import BrainDiffusionPrior, BrainDiffusionPriorOld


# need non-deterministic CuDNN for conv3D to work
utils.seed_everything(seed, cudnn_deterministic=False)

# change learning rate based on number of devices
max_lr *= accelerator.num_processes

# change batch size based on number of devices if using multi-gpu
# batch_size *= accelerator.num_processes

# change num_epochs based on number of devices if using multi-gpu
num_epochs *= accelerator.num_processes

# In[5]:


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
outdir = os.path.join(ROOT_DIR, 'train_logs', model_name)
if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)
if use_image_aug:
    import kornia
    from kornia.augmentation.container import AugmentationSequential

    img_augment = AugmentationSequential(
        kornia.augmentation.RandomResizedCrop((224, 224), (0.6, 1), p=0.3),
        kornia.augmentation.Resize((224, 224)),
        kornia.augmentation.RandomHorizontalFlip(p=0.5),
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.3),
        kornia.augmentation.RandomGrayscale(p=0.3),
        data_keys=["input"],
    )

# # Prep models and data loaders

# In[6]:


print('Pulling NSD webdataset data...')


print('Building ROI+NSD datasets (no WebDataset)...')
subj_tag = f"subj0{subj}"
roi_dir  = os.path.join(nv2l_root, "nsd", "fmris", subj_tag, "area")

train_ds = RoiNsdDataset(
    nsd_root=data_path, subj_tag=subj_tag,
    roi_dir=roi_dir, roi_name=roi_name,
    split="train", use_avg=roi_use_avg, image_size=224
)
val_ds = RoiNsdDataset(
    nsd_root=data_path, subj_tag=subj_tag,
    roi_dir=roi_dir, roi_name=roi_name,
    split="val", use_avg=roi_use_avg, image_size=224
)

num_train, num_val = len(train_ds), len(val_ds)
print(f"num_train={num_train}, num_val={num_val}")

from torch.utils.data import DataLoader
train_dl = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True,
    persistent_workers=(num_workers>0), drop_last=True
)
val_dl = DataLoader(
    val_ds, batch_size=64, shuffle=False,
    num_workers=max(1, num_workers//2), pin_memory=True,
    persistent_workers=(num_workers>0)
)



# In[7]:

# ==== [ADD] NV2L ROI .npy 로드 및 trial → row 매핑 ====
def _first_pos_map(lst):
    d = {}
    for i, t in enumerate(lst):
        t = int(t)
        if t not in d:
            d[t] = i  # 첫 등장 위치
    return d

if use_nv2l_roi:
    subj_tag = f"subj0{subj}"
    roi_dir   = os.path.join(nv2l_root, "nsd", "fmris", subj_tag, "area")
    index_json = os.path.join(nv2l_root, "nsd", "fmris", subj_tag, "nsd_fmri2image.json")

    if roi_use_avg:
        # averaged (unique 73KID별 평균) 사용
        roi_tr_np = np.load(os.path.join(roi_dir, f"nsd_{roi_name}_betas_ave_tr.npy"))  # (N_train_unique, V)
        roi_te_np = np.load(os.path.join(roi_dir, f"nsd_{roi_name}_betas_ave_te.npy"))  # (N_val_unique,   V)
        ave_train_ids = np.load(os.path.join(roi_dir, f"nsd_{roi_name}_ave_train_ids.npy")).astype(int)
        ave_val_ids   = np.load(os.path.join(roi_dir, f"nsd_{roi_name}_ave_val_ids.npy")).astype(int)
        train_id2pos  = {int(t): i for i, t in enumerate(ave_train_ids.tolist())}
        val_id2pos    = {int(t): i for i, t in enumerate(ave_val_ids.tolist())}
    else:
        # raw-trial 배열 사용 (행 순서 = nsd_fmri2image.json의 train / val 순서)
        with open(index_json, "r") as f:
            idxjson = json.load(f)
        roi_tr_np = np.load(os.path.join(roi_dir, f"nsd_{roi_name}_betas_tr.npy"))  # (N_train_trials, V)
        roi_te_np = np.load(os.path.join(roi_dir, f"nsd_{roi_name}_betas_te.npy"))  # (N_val_trials,   V)
        train_id2pos = _first_pos_map(idxjson["train"])
        val_id2pos   = _first_pos_map(idxjson["val"])

    in_dim_roi = int(roi_tr_np.shape[1])


    def roi_from_trial_any(trial_tensor):
        ids = trial_tensor.view(-1).detach().cpu().numpy().astype(int)
        rows = []
        for t in ids:
            if t in train_id2pos:
                rows.append(roi_tr_np[train_id2pos[t]])
            elif t in val_id2pos:
                rows.append(roi_te_np[val_id2pos[t]])
            else:
                raise RuntimeError(f"[ROI] id {t} not found in train/val maps")
        x = np.stack(rows, axis=0)
        return torch.from_numpy(x).to(device=device, dtype=torch.float32)



print('Creating Clipper...')
clip_sizes = {"RN50": 1024, "ViT-L/14": 768, "ViT-B/32": 512, "ViT-H-14": 1024}
clip_size = clip_sizes[clip_variant]      # = 768 (ViT-L/14)

# 1) CLIP 추출기 생성
if hidden:
    clip_extractor = Clipper(clip_variant, device=device, hidden_state=True, norm_embs=norm_embs)
    tokens = 257                           # hidden state 토큰 수
else:
    clip_extractor = Clipper(clip_variant, device=device, hidden_state=False, norm_embs=norm_embs)
    tokens = 1

# 2) voxel2clip의 출력 차원(= NCE에서 쓸 임베딩 길이)
out_dim_v2c = tokens * clip_size          # hidden=True면 257*768, 아니면 768

if prior:
    if hidden:
        prior_network = VersatileDiffusionPriorNetwork(
            dim=out_dim, depth=depth, dim_head=dim_head, heads=heads,
            causal=False, num_tokens=257, learned_query_mode="pos_emb"
        ).to(device)
        diffusion_prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=out_dim,
            condition_on_text_encodings=False,
            timesteps=timesteps,
            cond_drop_prob=0.2,
            image_embed_scale=None,
            voxel2clip=voxel2clip,
        ).to(device)
    else:
        diffusion_prior = BrainDiffusionPriorOld.from_pretrained(
            dict(), dict(condition_on_text_encodings=False, timesteps=timesteps, voxel2clip=voxel2clip)
        ).to(device)

print('Creating voxel2clip...')

if use_nv2l_roi:
    num_voxels = in_dim_roi
else:
    if subj == 1:   num_voxels = 15724
    elif subj == 2: num_voxels = 14278
    elif subj == 3: num_voxels = 15226
    elif subj == 4: num_voxels = 13153
    elif subj == 5: num_voxels = 13039
    elif subj == 6: num_voxels = 17907
    elif subj == 7: num_voxels = 12682
    elif subj == 8: num_voxels = 14386

# voxel2clip_kwargs = dict(in_dim=num_voxels, out_dim=out_dim, clip_size=clip_size, use_projector=use_projector)
voxel2clip_kwargs = dict(
    in_dim=num_voxels,
    out_dim=out_dim_v2c,
    clip_size=clip_size,
    use_projector=use_projector
)
voxel2clip = BrainNetwork(**voxel2clip_kwargs)


# load from ckpt
voxel2clip_path = "None"
if voxel2clip_path != "None":
    checkpoint = torch.load(voxel2clip_path, map_location='cpu')
    voxel2clip.load_state_dict(checkpoint['model_state_dict'], strict=False)
    del checkpoint

print("params of voxel2clip:")
if local_rank == 0:
    utils.count_params(voxel2clip)


if prior:
    out_dim = clip_size
    depth = 6
    dim_head = 64
    heads = clip_size // 64
    if hidden:
        guidance_scale = 3.5
        timesteps = 100
        prior_network = VersatileDiffusionPriorNetwork(
            dim=out_dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            causal=False,
            num_tokens=257,
            learned_query_mode="pos_emb"
        ).to(device)
        diffusion_prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=clip_size,
            condition_on_text_encodings=False,
            timesteps=timesteps,
            cond_drop_prob=0.2,
            image_embed_scale=None,
            voxel2clip=voxel2clip,
        ).to(device)
    else:
        guidance_scale = 7.5
        timesteps = 1000
        diffusion_prior = BrainDiffusionPriorOld.from_pretrained(
            dict(),
            dict(condition_on_text_encodings=False, timesteps=timesteps, voxel2clip=voxel2clip),
            voxel2clip_path=None,
        )
else:
    import torch.nn as nn
    class _V2CWrapper(nn.Module):
        def __init__(self, v2c):
            super().__init__()
            self.voxel2clip = v2c
    diffusion_prior = _V2CWrapper(voxel2clip).to(device)


if not prior:
    diffusion_prior = diffusion_prior.requires_grad_(False)
    diffusion_prior.voxel2clip.requires_grad_(True)

print("params of diffusion prior:")
if local_rank == 0:
    utils.count_params(diffusion_prior)

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
opt_grouped_parameters = [
    # {'params': [p for n, p in diffusion_prior.net.named_parameters() if not any(nd in n for nd in no_decay)],
    #  'weight_decay': 1e-2},
    # {'params': [p for n, p in diffusion_prior.net.named_parameters() if any(nd in n for nd in no_decay)],
    #  'weight_decay': 0.0},
    {'params': [p for n, p in diffusion_prior.voxel2clip.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 1e-2},
    {'params': [p for n, p in diffusion_prior.voxel2clip.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=max_lr)

global_batch_size = batch_size * num_devices
if lr_scheduler_type == 'linear':
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        total_iters=int(num_epochs * (num_train // global_batch_size)),
        last_epoch=-1
    )
elif lr_scheduler_type == 'cycle':
    total_steps = int(num_epochs * (num_train // global_batch_size))
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        final_div_factor=1000,
        last_epoch=-1, pct_start=2 / num_epochs
    )

if plot_umap:
    import umap


# ===== InfoNCE loss (symmetric) =====
def info_nce_loss(q, k, temperature=0.07):
    """
    q, k: [B, D] (이미 L2-normalized 상태가 이상적)
    대칭 InfoNCE = 평균( q->k CE, k->q CE )
    """
    # [B, B] 유사도 행렬 (cosine ≈ dot, 정규화되어 있으면 dot=cos)
    logits = q @ k.t() / temperature
    labels = torch.arange(q.size(0), device=q.device)
    loss_qk = nn.functional.cross_entropy(logits, labels)
    loss_kq = nn.functional.cross_entropy(logits.t(), labels)
    loss = 0.5 * (loss_qk + loss_kq)
    return loss, logits

def save_ckpt(tag):
    ckpt_path = outdir + f'/{tag}.pth'
    print(f'saving {ckpt_path}', flush=True)
    unwrapped_model = accelerator.unwrap_model(diffusion_prior)
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'train_losses': losses,
            'val_losses': val_losses,
            'lrs': lrs,
        }, ckpt_path)
    except:
        print("Couldn't save... moving on to prevent crashing.")
    del unwrapped_model


print("\nDone with model preparations!")

# # Weights and Biases

# In[8]:


# params for wandb
if local_rank == 0 and wandb_log:  # only use main process for wandb logging
    import wandb

    wandb_project = 'stability'
    wandb_run = model_name
    wandb_notes = ''

    print(f"wandb {wandb_project} run {wandb_run}")
    wandb.login(host='https://stability.wandb.io')  # , relogin=True)
    wandb_config = {
        "model_name": model_name,
        "clip_variant": clip_variant,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "use_image_aug": use_image_aug,
        "max_lr": max_lr,
        "lr_scheduler_type": lr_scheduler_type,
        "mixup_pct": mixup_pct,
        "num_train": num_train,
        "num_val": num_val,
        "seed": seed,
        "distributed": distributed,
        "num_devices": num_devices,
        "world_size": world_size,
        "train_url": train_url,
        "val_url": val_url,
    }
    print("wandb_config:\n", wandb_config)
    if True:  # wandb_auto_resume
        print("wandb_id:", model_name)
        wandb.init(
            id=model_name,
            project=wandb_project,
            name=wandb_run,
            config=wandb_config,
            notes=wandb_notes,
            resume="allow",
        )
    else:
        wandb.init(
            project=wandb_project,
            name=wandb_run,
            config=wandb_config,
            notes=wandb_notes,
        )
else:
    wandb_log = False

# # Main

# In[9]:


epoch = 0
losses, val_losses, lrs = [], [], []
nce_losses, val_nce_losses = [], []
sim_losses, val_sim_losses = [], []
best_val_loss = 1e9
soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, num_epochs - int(mixup_pct * num_epochs))
if hidden:
    prior_mult = 30
else:
    prior_mult = .03
val_voxel0 = val_image0 = None

# Optionally resume from checkpoint #
if resume_from_ckpt:
    print("\n---resuming from last.pth ckpt---\n")
    try:
        checkpoint = torch.load(outdir + '/last.pth', map_location='cpu')
    except:
        print('last.pth failed... trying last_backup.pth')
        checkpoint = torch.load(outdir + '/last_backup.pth', map_location='cpu')
    epoch = checkpoint['epoch']
    print("Epoch", epoch)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    diffusion_prior.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint
elif wandb_log:
    if wandb.run.resumed:
        print("\n---resuming from last.pth ckpt---\n")
        try:
            checkpoint = torch.load(outdir + '/last.pth', map_location='cpu')
        except:
            print('last.pth failed... trying last_backup.pth')
            checkpoint = torch.load(outdir + '/last_backup.pth', map_location='cpu')
        epoch = checkpoint['epoch']
        print("Epoch", epoch)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        diffusion_prior.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint
torch.cuda.empty_cache()

# In[10]:


diffusion_prior, optimizer, train_dl, val_dl, lr_scheduler = accelerator.prepare(
    diffusion_prior, optimizer, train_dl, val_dl, lr_scheduler
)

# In[11]:


print(f"{model_name} starting with epoch {epoch} / {num_epochs}")

progress_bar = tqdm(
    range(epoch, num_epochs),
    ncols=120,                         # 과하게 넓지 않게
    file=TQDM_FILE,                    # stdout으로
    disable=(local_rank != 0),
    leave=True,
    bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
)


for epoch in progress_bar:
    diffusion_prior.train()

    sims_base = 0.
    val_sims_base = 0.
    fwd_percent_correct = 0.
    bwd_percent_correct = 0.
    val_fwd_percent_correct = 0.
    val_bwd_percent_correct = 0.
    loss_nce_sum = 0.
    loss_prior_sum = 0.
    val_loss_nce_sum = 0.
    val_loss_prior_sum = 0.

    for train_i, (voxel, image, trial) in enumerate(train_dl):
        with torch.amp.autocast('cuda'):
            optimizer.zero_grad()

            if use_image_aug:
                image = img_augment(image)

            if voxel.ndim == 2:
                voxel = voxel.float()
            elif voxel.ndim == 3:
                repeat_index = train_i % voxel.shape[1]
                voxel = voxel[:, repeat_index].float()
            else:
                raise RuntimeError(f"Unexpected voxel shape: {voxel.shape}")

            if epoch < int(mixup_pct * num_epochs):
                voxel, perm, betas, select = utils.mixco(voxel)

            clip_target = clip_extractor.embed_image(image).float()

            with torch.autocast('cuda', enabled=False):
                clip_voxels, clip_voxels_proj = (
                    diffusion_prior.module.voxel2clip(voxel.float())
                    if distributed else diffusion_prior.voxel2clip(voxel.float())
                )
            clip_voxels = clip_voxels.float()
            clip_voxels_proj = clip_voxels_proj.float()

            if hidden:
                clip_voxels = clip_voxels.view(len(voxel), -1, clip_size)

            aligned_clip_voxels = clip_voxels

            clip_voxels_norm = nn.functional.normalize(clip_voxels_proj.flatten(1), dim=-1)
            clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

            if epoch < int(mixup_pct * num_epochs):
                logits = (clip_voxels_norm @ clip_target_norm.T) / 0.06
                loss_nce = utils.mixco_nce(
                    logits,  # <- [B,B] 로짓
                    perm=perm, betas=betas, select=select
                )
            else:
                epoch_temp = soft_loss_temps[epoch - int(mixup_pct * num_epochs)]
                loss_nce = utils.soft_clip_loss(
                    clip_voxels_norm,
                    clip_target_norm,
                    temp=epoch_temp)


            loss = loss_nce
            utils.check_loss(loss)

            accelerator.backward(loss)
            optimizer.step()

            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])


            sims_base += nn.functional.cosine_similarity(clip_target_norm, clip_voxels_norm).mean().item()

            # forward and backward top 1 accuracy
            labels = torch.arange(len(clip_target_norm)).to(device)
            fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm),
                                              labels, k=1)
            bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm),
                                              labels, k=1)

            if lr_scheduler_type is not None:
                lr_scheduler.step()

    diffusion_prior.eval()
    for val_i, (voxel, image, trial) in enumerate(val_dl):
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                if use_image_aug:
                    image = img_augment(image)

                if use_nv2l_roi:
                    voxel = roi_from_trial_any(trial)
                else:
                    # 기존 경로
                    voxel = torch.mean(voxel, axis=1).float()

                if val_image0 is None:
                    val_image0 = image.detach().clone()
                    val_voxel0 = voxel.detach().clone()

                clip_target = clip_extractor.embed_image(image).float()

                with torch.autocast('cuda', enabled=False):
                    clip_voxels, clip_voxels_proj = (
                        diffusion_prior.module.voxel2clip(voxel.float())
                        if distributed else diffusion_prior.voxel2clip(voxel.float())
                    )
                clip_voxels = clip_voxels.float()
                clip_voxels_proj = clip_voxels_proj.float()

                if hidden:
                    clip_voxels = clip_voxels.view(len(voxel), -1, clip_size)


                aligned_clip_voxels = clip_voxels

                clip_voxels_norm = nn.functional.normalize(clip_voxels_proj.flatten(1), dim=-1)
                clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

                if epoch < int(mixup_pct * num_epochs):
                    logits = (clip_voxels_norm @ clip_target_norm.T) / 0.06
                    labels = torch.arange(logits.size(0), device=logits.device)
                    val_loss_nce = 0.5 * (
                            torch.nn.functional.cross_entropy(logits, labels) +
                            torch.nn.functional.cross_entropy(logits.t(), labels)
                    )
                else:
                    val_loss_nce = utils.soft_clip_loss(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=epoch_temp)


                val_loss_nce_sum += val_loss_nce.item()
                val_loss = val_loss_nce

                utils.check_loss(val_loss)

                val_losses.append(val_loss.item())


                val_sims_base += nn.functional.cosine_similarity(clip_target_norm, clip_voxels_norm).mean().item()

                labels = torch.arange(len(clip_target_norm)).to(device)
                val_fwd_percent_correct += utils.topk(
                    utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1)
                val_bwd_percent_correct += utils.topk(
                    utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1)

    if local_rank == 0:
        if (not save_at_end and ckpt_saving) or (save_at_end and epoch == num_epochs - 1):
            # save best model
            val_loss = np.mean(val_losses[-(val_i + 1):])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_ckpt('best')
            else:
                print(f'not best - val_loss: {val_loss:.3f}, best_val_loss: {best_val_loss:.3f}')

        if utils.is_interactive():
            clear_output(wait=True)

        logs = {"train/loss": np.mean(losses[-(train_i + 1):]),
                "val/loss": np.mean(val_losses[-(val_i + 1):]),
                "train/lr": lrs[-1],
                "train/num_steps": len(losses),
                "val/num_steps": len(val_losses),
                "train/cosine_sim_base": sims_base / (train_i + 1),
                "val/cosine_sim_base": val_sims_base / (val_i + 1),
                "train/fwd_pct_correct": fwd_percent_correct / (train_i + 1),
                "train/bwd_pct_correct": bwd_percent_correct / (train_i + 1),
                "val/val_fwd_pct_correct": val_fwd_percent_correct / (val_i + 1),
                "val/val_bwd_pct_correct": val_bwd_percent_correct / (val_i + 1),
                "train/loss_nce": loss_nce_sum / (train_i + 1),
                "train/loss_prior": loss_prior_sum / (train_i + 1),
                "val/loss_nce": val_loss_nce_sum / (val_i + 1),
                "val/loss_prior": val_loss_prior_sum / (val_i + 1)}
        progress_bar.set_postfix(**logs)

        print(
            "[epoch {}/{}] tr_loss={:.4f} val_loss={:.4f} "
            "lr={:.2e} | cos={:.4f}/{:.4f} | "
            "top1={:.3f}/{:.3f} → {:.3f}/{:.3f}".format(
                epoch + 1, num_epochs,
                logs["train/loss"], logs["val/loss"],
                logs["train/lr"],
                logs["train/cosine_sim_base"], logs["val/cosine_sim_base"],
                logs["train/fwd_pct_correct"], logs["train/bwd_pct_correct"],
                logs["val/val_fwd_pct_correct"], logs["val/val_bwd_pct_correct"],
            ),
            flush=True
        )

        # Save model checkpoint and reconstruct
        save_ckpt(f'last')
        if epoch % ckpt_interval == 0:
            save_ckpt(f'last_backup')

        if wandb_log: wandb.log(logs)

    # wait for other GPUs to catch up if needed
    accelerator.wait_for_everyone()

print("\n===Finished!===\n")
if not utils.is_interactive():
    sys.exit(0)

