# # Import packages & functions

import os
import shutil
import sys
import json
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from info_nce import InfoNCE
from tqdm import tqdm
from collections import OrderedDict

from torchvision.utils import make_grid
from PIL import Image
import kornia
from kornia.augmentation.container import AugmentationSequential
from pytorch_msssim import ssim

import utils
from models import Voxel2StableDiffusionModel
# from convnext import ConvnextXL
from torch.utils.data import DataLoader
from datasets.RoiNsdDataset import RoiNsdDataset
import argparse
from torch import autocast




# roi_name만 받는다. 하이픈/언더스코어 둘 다 허용
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--roi-name", "--roi_name", dest="cli_roi_name", type=str, default=None)
_cli_args, _ = _parser.parse_known_args()

try:
    sys.stdout.reconfigure(line_buffering=True)  # Py>=3.7
except Exception:
    os.environ["PYTHONUNBUFFERED"] = "1"

# 진행바 포맷 (짧고 안정적)
BAR_FMT = "{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
# step 로그를 N 스텝마다 출력하고 싶으면 환경변수로 조절 (0이면 끔)
LOG_EVERY_N = int(os.getenv("LOG_EVERY_N", "0"))  # e.g., export LOG_EVERY_N=50



def set_ddp():
    import torch.distributed as dist
    env_dict = {
        key: os.environ.get(key)
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK",
                    "LOCAL_RANK", "WORLD_SIZE", "NUM_GPUS")
    }
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    n = int(os.environ.get("NUM_GPUS", 8))
    device_ids = list(
        range(local_rank * n, (local_rank + 1) * n)
    )

    if local_rank == 0:
        print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
    print(
        f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )

    print(
        f"[{os.getpid()}] rank = {dist.get_rank()} ({rank}), "
        + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {device_ids}"
    )
    device = torch.device("cuda", local_rank)
    return local_rank, device, n


local_rank = 0
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
num_devices = 1
distributed = False
autocast_device = "cuda" if use_cuda else "cpu"

from diffusers.models import AutoencoderKL

# 스크립트 파일 기준 절대경로
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
AE_CKPT = os.path.join(ROOT_DIR, 'train_logs', 'models', 'sd_image_var_autoenc.pth')

autoenc = AutoencoderKL(
    down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
    up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
    block_out_channels=[128, 256, 512, 512],
    layers_per_block=2,
    sample_size=256
)
# === add this helper near the top (after imports) ===
def _remap_vae_attn_keys(state_dict: dict) -> dict:
    """Map old attention keys (query/key/value/proj_attn) to new names (to_q/to_k/to_v/to_out.0)."""
    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        # common attention key renames (encoder & decoder mid_block.attentions.0.*)
        nk = nk.replace(".query.", ".to_q.")
        nk = nk.replace(".key.", ".to_k.")
        nk = nk.replace(".value.", ".to_v.")
        nk = nk.replace(".proj_attn.", ".to_out.0.")
        new_sd[nk] = v
    return new_sd

try:
    state = torch.load(AE_CKPT, map_location="cpu", weights_only=True)  # torch>=2.4에서만 동작
except TypeError:
    state = torch.load(AE_CKPT, map_location="cpu")

# 키가 구형이면 매핑
if any((".query." in k) or (".proj_attn." in k) for k in state.keys()):
    state = _remap_vae_attn_keys(state)

# 가능하면 strict=True로 먼저 시도
missing, unexpected = [], []
try:
    info = autoenc.load_state_dict(state, strict=True)
    missing, unexpected = info.missing_keys, info.unexpected_keys
except RuntimeError:
    info = autoenc.load_state_dict(state, strict=False)
    missing, unexpected = info.missing_keys, info.unexpected_keys

if missing or unexpected:
    print("[VAE] load_state_dict: missing:", missing)
    print("[VAE] load_state_dict: unexpected:", unexpected)

autoenc.requires_grad_(False)
autoenc.eval()



# # Configurations
model_name = "autoencoder"
modality = "image"  # ("image", "text")
image_var = 'images' if modality == 'image' else None  # trial
clamp_embs = False  # clamp embeddings to (-1.5, 1.5)

voxel_dims = 1  # 1 for flattened 3 for 3d
n_samples_save = 4  # how many SD samples from train and val to save

use_reconst = False
batch_size = 8
num_epochs = 120
lr_scheduler = 'cycle'
initial_lr = 1e-3
max_lr = 5e-4
first_batch = False
ckpt_saving = True
ckpt_interval = 24
save_at_end = False
use_mp = False
remote_data = False
data_commit = "avg"  # '9947586218b6b7c8cab804009ddca5045249a38d'
mixup_pct = -1
use_cont = False
use_sobel_loss = False
use_blurred_training = False

use_full_trainset = True
subj_id = "01"
seed = 0
# ckpt_path = "../train_logs/models/autoencoder_final/test/ckpt-epoch015.pth"
ckpt_path = None
cont_model = 'cnx'
resume_from_ckpt = True
ups_mode = '4x'

# ==== [ADD] NV2L-ROI 입력 설정 ====
use_nv2l_roi = True               # NV2L ROI .npy를 입력으로 쓸지 여부
nv2l_root = os.environ.get("NV2L_OUT", "/NSD/nv2l_outputs/datasets")
roi_name = "streams"
roi_use_avg = True

if use_nv2l_roi:
    resume_from_ckpt = False

use_mp = use_mp and use_cuda
scaler = torch.cuda.amp.GradScaler(enabled=use_mp)

# need non-deterministic CuDNN for conv3D to work
utils.seed_everything(seed + local_rank, cudnn_deterministic=False)
torch.backends.cuda.matmul.allow_tf32 = True

# if running command line, read in args or config file values and override above params
try:
    config_keys = [k for k, v in globals().items() if not k.startswith('_') \
                   and isinstance(v, (int, float, bool, str))]
    exec(open('configurator.py').read())  # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys}  # will be useful for logging
except:
    pass

if _cli_args.cli_roi_name is not None:
    roi_name = _cli_args.cli_roi_name
    if local_rank == 0:
        print(f"[cfg] roi_name set by CLI: {roi_name}", flush=True)

if distributed:
    local_rank, device, num_devices = set_ddp()
autoenc.to(device)

if use_cont and cont_model == 'cnx':
    try:
        from convnext import ConvnextXL
    except ModuleNotFoundError:
        raise ImportError(
            "ConvnextXL 모듈이 없습니다. convnext.py를 프로젝트에 추가하거나 --use_cont=False로 실행하세요."
        )
    mixup_pct = -1
    if cont_model == 'cnx':
        cnx = ConvnextXL('../train_logs/models/convnext_xlarge_alpha0.75_fullckpt.pth')
        cnx.requires_grad_(False)
        cnx.eval()
        cnx.to(device)
    train_augs = AugmentationSequential(
        # kornia.augmentation.RandomCrop((480, 480), p=0.3),
        # kornia.augmentation.Resize((512, 512)),
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        kornia.augmentation.RandomGrayscale(p=0.2),
        kornia.augmentation.RandomSolarize(p=0.2),
        kornia.augmentation.RandomGaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0), p=0.1),
        kornia.augmentation.RandomResizedCrop((512, 512), scale=(0.5, 1.0)),
        data_keys=["input"],
    )

outdir = os.path.join(ROOT_DIR, 'train_logs', 'low', model_name)
if local_rank == 0:
    os.makedirs(outdir, exist_ok=True)

# auto resume
if not use_nv2l_roi and os.path.exists(os.path.join(outdir, 'last.pth')):
    ckpt_path = os.path.join(outdir, 'last.pth')
    resume_from_ckpt = True

# num_devices = torch.cuda.device_count()
if num_devices == 0: num_devices = 1
num_workers = num_devices

cache_dir = 'cache'
n_cache_recs = 0

# ==== [ADD] NV2L ROI 데이터 로드 & trial→row 매핑 ====
if use_nv2l_roi:
    subj_tag = f"subj{subj_id}"
    roi_dir = os.path.join(nv2l_root, "nsd", "fmris", subj_tag, "area")
    index_path = os.path.join(nv2l_root, "nsd", "fmris", subj_tag, "nsd_fmri2image.json")

    if roi_use_avg:
        # 평균된(unique image) 배열 사용
        roi_tr_np = np.load(os.path.join(roi_dir, f"nsd_{roi_name}_betas_ave_tr.npy"))  # (N_train_unique, V)
        roi_te_np = np.load(os.path.join(roi_dir, f"nsd_{roi_name}_betas_ave_te.npy"))  # (N_val_unique,   V)
        ave_train_ids = np.load(os.path.join(roi_dir, f"nsd_{roi_name}_ave_train_ids.npy")).astype(int)
        ave_val_ids   = np.load(os.path.join(roi_dir, f"nsd_{roi_name}_ave_val_ids.npy")).astype(int)
        train_id2pos = {int(t): i for i, t in enumerate(ave_train_ids.tolist())}
        val_id2pos   = {int(t): i for i, t in enumerate(ave_val_ids.tolist())}
    else:
        with open(index_path, "r") as f:
            idxjson = json.load(f)
        roi_tr_np = np.load(os.path.join(roi_dir, f"nsd_{roi_name}_betas_tr.npy"))   # (N_train_trials, V)
        roi_te_np = np.load(os.path.join(roi_dir, f"nsd_{roi_name}_betas_te.npy"))   # (N_val_trials,   V)

        def first_pos_map(lst):
            d = {}
            for i, tid in enumerate(lst):
                tid = int(tid)
                if tid not in d:
                    d[tid] = i
            return d

        train_id2pos = first_pos_map(idxjson["train"])
        val_id2pos   = first_pos_map(idxjson["val"])

    # ROI 차원으로 in_dim 지정
    in_dim_roi = int(roi_tr_np.shape[1])


# # Prep models and data loader
if local_rank == 0: print('Creating voxel2sd...')

if use_nv2l_roi:
    in_dim_roi = int(roi_tr_np.shape[1])
    voxel2sd = Voxel2StableDiffusionModel(in_dim=in_dim_roi, ups_mode=ups_mode, use_cont=use_cont)
else:
    in_dims = {'01': 15724, '02': 14278, '05': 13039, '07': 12682}
    voxel2sd = Voxel2StableDiffusionModel(use_cont=use_cont, in_dim=in_dims[subj_id], ups_mode=ups_mode)


voxel2sd.to(device)

if distributed and use_cuda:
    voxel2sd = torch.nn.SyncBatchNorm.convert_sync_batchnorm(voxel2sd)
    voxel2sd = DDP(voxel2sd)

try:
    utils.count_params(voxel2sd)
except:
    if local_rank == 0: print('Cannot count params for voxel2sd (probably because it has Lazy layers)')

# ==== [REPLACE] WebDataset → RoiNsdDataset (no WebDataset) ====
if local_rank == 0: print('Building ROI+NSD datasets (no WebDataset)...')

# NSD 루트 (이미 환경변수 사용 중이면 그대로 쓰고, 없으면 기본값)
NSD_ROOT = os.environ.get("NSD_ROOT", "/NSD")  # nsd_access가 읽을 최상위 루트

# NV2L ROI 출력 루트 (이미 파일 상단에 있으면 재사용)
nv2l_root = os.environ.get("NV2L_OUT", "/NSD/nv2l_outputs/datasets")

# subj_id가 "01","02","05","07" 형태이므로 그대로 사용
subj_tag = f"subj{subj_id}"
roi_dir  = os.path.join(nv2l_root, "nsd", "fmris", subj_tag, "area")
roi_name = roi_name if 'roi_name' in globals() else "streams"  # 기존 변수 유지

# 평균 ROI 사용할지 여부 (기존 변수 재사용)
roi_use_avg = roi_use_avg if 'roi_use_avg' in globals() else True

# Dataset
train_ds = RoiNsdDataset(
    nsd_root=NSD_ROOT, subj_tag=subj_tag,
    roi_dir=roi_dir, roi_name=roi_name,
    split="train", use_avg=roi_use_avg, image_size=224
)
val_ds = RoiNsdDataset(
    nsd_root=NSD_ROOT, subj_tag=subj_tag,
    roi_dir=roi_dir, roi_name=roi_name,
    split="val", use_avg=roi_use_avg, image_size=224
)

num_train, num_val = len(train_ds), len(val_ds)
if local_rank == 0: print(f"num_train={num_train}, num_val={num_val}")

# DataLoader
train_dl = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True,
    persistent_workers=(num_workers>0), drop_last=True
)
val_dl = DataLoader(
    val_ds, batch_size=max(16, batch_size), shuffle=False,
    num_workers=max(1, num_workers//2), pin_memory=True,
    persistent_workers=(num_workers>0)
)


no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
opt_grouped_parameters = [
    {'params': [p for n, p in voxel2sd.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 1e-2},
    {'params': [p for n, p in voxel2sd.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=1e-3)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr,
                                                   total_steps=num_epochs * ((num_train // batch_size) // num_devices),
                                                   final_div_factor=1000,
                                                   last_epoch=-1, pct_start=2 / num_epochs)


def save_ckpt(tag):
    ckpt_path = os.path.join(outdir, f'{tag}.pth')
    if tag == "last":
        if os.path.exists(ckpt_path):
            shutil.copyfile(ckpt_path, os.path.join(outdir, f'{tag}_old.pth'))
    tqdm.write(f"[ckpt] saving {ckpt_path}", file=sys.stdout)
    if local_rank == 0:
        state_dict = voxel2sd.state_dict()
        for key in list(state_dict.keys()):
            if 'module.' in key:
                state_dict[key.replace('module.', '')] = state_dict[key]
                del state_dict[key]
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': losses,
                'val_losses': val_losses,
                'lrs': lrs,
            }, ckpt_path)
        except:
            print('Failed to save weights')
            print(traceback.format_exc())
    if tag == "last":
        if os.path.exists(os.path.join(outdir, f'{tag}_old.pth')):
            os.remove(os.path.join(outdir, f'{tag}_old.pth'))

        # if wandb_log:
        #     wandb.save(ckpt_path)


# Optionally resume from checkpoint #
if resume_from_ckpt:
    print("\n---resuming from ckpt_path---\n", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    epoch = checkpoint['epoch'] + 1
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if hasattr(voxel2sd, 'module'):
        voxel2sd.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        voxel2sd.load_state_dict(checkpoint['model_state_dict'])
    total_steps_done = epoch * ((num_train // batch_size) // num_devices)
    for _ in range(total_steps_done):
        lr_scheduler.step()
    del checkpoint
    torch.cuda.empty_cache()
else:
    epoch = 0

if local_rank == 0: print("\nDone with model preparations!")


losses = []
val_losses = []
lrs = []
best_val_loss = 1e10
best_ssim = 0
mean = torch.tensor([0.485, 0.456, 0.406]).to(device).reshape(1, 3, 1, 1)
std = torch.tensor([0.228, 0.224, 0.225]).to(device).reshape(1, 3, 1, 1)
epoch = 0

if ckpt_path is not None:
    print("\n---resuming from ckpt_path---\n", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    epoch = checkpoint['epoch'] + 1
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    voxel2sd.module.load_state_dict(checkpoint['model_state_dict'])
    global_batch_size = batch_size * num_devices
    total_steps_done = epoch * (num_train // global_batch_size)
    for _ in range(total_steps_done):
        lr_scheduler.step()
    del checkpoint
    torch.cuda.empty_cache()


progress_bar = tqdm(
    range(epoch, num_epochs),
    ncols=120,
    bar_format=BAR_FMT,
    file=sys.stdout,        # ★ tee가 받는 stdout으로 고정
    disable=(local_rank != 0),
    leave=True
)


for epoch in progress_bar:
    voxel2sd.train()

    loss_mse_sum = 0
    loss_reconst_sum = 0
    loss_cont_sum = 0
    loss_sobel_sum = 0
    val_loss_mse_sum = 0
    val_loss_reconst_sum = 0
    val_ssim_score_sum = 0

    reconst_fails = []

    for train_i, (voxel, image, trial) in enumerate(train_dl):  # ← trial 받아오기(언더스코어 말고)
        optimizer.zero_grad(set_to_none=True)

        image = image.to(device).float()
        image_512 = F.interpolate(image, (512, 512), mode='bilinear', align_corners=False, antialias=True)


        if voxel.ndim == 2:
            voxel = voxel.to(device).float()
        elif voxel.ndim == 3:
            repeat_index = train_i % voxel.shape[1]
            voxel = voxel[:, repeat_index].to(device).float()
        else:
            raise RuntimeError(f"Unexpected voxel shape: {voxel.shape}")

        if epoch <= mixup_pct * num_epochs:
            voxel, perm, betas, select = utils.mixco(voxel)
        else:
            select = None

        with torch.cuda.amp.autocast(enabled=use_mp):
            autoenc_image = kornia.filters.median_blur(image_512, (15, 15)) if use_blurred_training else image_512
            image_enc = autoenc.encode(2 * autoenc_image - 1).latent_dist.mode() * 0.18215
            if use_cont:
                image_enc_pred, transformer_feats = voxel2sd(voxel, return_transformer_feats=True)
            else:
                image_enc_pred = voxel2sd(voxel)

            if epoch <= mixup_pct * num_epochs:
                image_enc_shuf = image_enc[perm]
                betas_shape = [-1] + [1] * (len(image_enc.shape) - 1)
                image_enc[select] = image_enc[select] * betas[select].reshape(*betas_shape) + \
                                    image_enc_shuf[select] * (1 - betas[select]).reshape(*betas_shape)

            if use_cont:
                image_norm = (image_512 - mean) / std
                image_aug = (train_augs(image_512) - mean) / std
                _, cnx_embeds = cnx(image_norm)
                _, cnx_aug_embeds = cnx(image_aug)

                cont_loss = utils.soft_cont_loss(
                    F.normalize(transformer_feats.reshape(-1, transformer_feats.shape[-1]), dim=-1),
                    F.normalize(cnx_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                    F.normalize(cnx_aug_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                    temp=0.075,
                    distributed=distributed
                )
                del image_aug, cnx_embeds, transformer_feats
            else:
                cont_loss = torch.tensor(0)

            # mse_loss = F.mse_loss(image_enc_pred, image_enc)/0.18215
            mse_loss = F.l1_loss(image_enc_pred, image_enc)
            del image_512, voxel

            if use_reconst:  # epoch >= 0.1 * num_epochs:
                # decode only non-mixed images
                if select is not None:
                    selected_inds = torch.where(~select)[0]
                    reconst_select = selected_inds[torch.randperm(len(selected_inds))][:4]
                else:
                    reconst_select = torch.arange(len(image_enc_pred))
                image_enc_pred = F.interpolate(image_enc_pred[reconst_select], scale_factor=0.5, mode='bilinear',
                                               align_corners=False)
                reconst = autoenc.decode(image_enc_pred / 0.18215).sample
                # reconst_loss = F.mse_loss(reconst, 2*image[reconst_select]-1)
                reconst_image = kornia.filters.median_blur(image[reconst_select], (7, 7)) if use_blurred_training else \
                image[reconst_select]
                reconst_loss = F.l1_loss(reconst, 2 * reconst_image - 1)
                if reconst_loss != reconst_loss:
                    reconst_loss = torch.tensor(0)
                    reconst_fails.append(train_i)
                if use_sobel_loss:
                    sobel_targ = kornia.filters.sobel(kornia.filters.median_blur(image[reconst_select], (3, 3)))
                    sobel_pred = kornia.filters.sobel(reconst / 2 + 0.5)
                    sobel_loss = F.l1_loss(sobel_pred, sobel_targ)
                else:
                    sobel_loss = torch.tensor(0)
            else:
                reconst_loss = torch.tensor(0)
                sobel_loss = torch.tensor(0)

            loss = mse_loss / 0.18215 + 2 * reconst_loss + 0.1 * cont_loss + 16 * sobel_loss
            # utils.check_loss(loss)

            loss_mse_sum += mse_loss.item()
            loss_reconst_sum += reconst_loss.item()
            loss_cont_sum += cont_loss.item()
            loss_sobel_sum += sobel_loss.item()

            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])

            if local_rank == 0:
                logs = OrderedDict(
                    train_loss=np.mean(losses[-(train_i + 1):]),
                    val_loss=np.nan,
                    lr=lrs[-1],
                )
                progress_bar.set_postfix(**logs)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        if local_rank == 0 and LOG_EVERY_N and ((train_i + 1) % LOG_EVERY_N == 0):
            # 최근 N개 평균(또는 마지막 값)을 한 줄로
            if LOG_EVERY_N <= len(losses):
                tr_recent = float(np.mean(losses[-LOG_EVERY_N:]))
            else:
                tr_recent = float(losses[-1])
            lr_cur = float(optimizer.param_groups[0]['lr'])
            print(f"[epoch {epoch + 1}/{num_epochs} step {train_i + 1}/{len(train_dl)}] "
                  f"train/loss={tr_recent:.4f} lr={lr_cur:.2e}", flush=True)

        if lr_scheduler is not None:
            lr_scheduler.step()

    if local_rank == 0:
        voxel2sd.eval()
        for val_i, (voxel, image, trial) in enumerate(val_dl):  # trial 받기
            with torch.inference_mode():
                image = image.to(device).float()
                image = F.interpolate(image, (512, 512), mode='bilinear', align_corners=False, antialias=True)


                if voxel.ndim == 2:
                    voxel = voxel.to(device).float()
                elif voxel.ndim == 3:
                    repeat_index = val_i % voxel.shape[1]
                    voxel = voxel[:, repeat_index].to(device).float()
                else:
                    raise RuntimeError(f"Unexpected voxel shape: {voxel.shape}")

                with torch.amp.autocast('cuda', enabled=use_mp):
                    image_enc = autoenc.encode(2 * image - 1).latent_dist.mode() * 0.18215
                    image_enc_pred = voxel2sd.module(voxel) if hasattr(voxel2sd, 'module') else voxel2sd(voxel)
                    mse_loss = F.l1_loss(image_enc_pred, image_enc)
                    val_loss_step = mse_loss / 0.18215

                    if use_reconst:
                        # z는 latent (image_enc_pred에서 일부 슬라이스)
                        z = image_enc_pred[-16:].detach()
                        z = z.to(dtype=autoenc.post_quant_conv.weight.dtype)  # 보통 torch.float32

                        with torch.cuda.amp.autocast(enabled=False):  # 디코드만 FP32
                            reconst = autoenc.decode(z / 0.18215).sample

                        reconst_loss = F.mse_loss(reconst, 2 * image - 1)
                        ssim_score = ssim((reconst / 2 + 0.5).clamp(0, 1), image, data_range=1, size_average=True,
                                          nonnegative_ssim=True)
                    else:
                        reconst = None
                        reconst_loss = torch.tensor(0)
                        ssim_score = torch.tensor(0)

                    val_loss_mse_sum += mse_loss.item()
                    val_loss_reconst_sum += reconst_loss.item()
                    val_ssim_score_sum += ssim_score.item()

                    val_losses.append(val_loss_step.item() + reconst_loss.item())

            logs = OrderedDict(
                train_loss=np.mean(losses[-(train_i + 1):]),
                val_loss=np.mean(val_losses[-(val_i + 1):]),
                lr=lrs[-1],
            )
            progress_bar.set_postfix(**logs)

        if (not save_at_end and ckpt_saving) or (save_at_end and epoch == num_epochs - 1):
            # save best model
            val_loss = np.mean(val_losses[-(val_i + 1):])
            val_ssim = val_ssim_score_sum / (val_i + 1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_ckpt('best')
            else:
                print(f'not best - val_loss: {val_loss:.3f}, best_val_loss: {best_val_loss:.3f}')
            if val_ssim > best_ssim:
                best_ssim = val_ssim
                save_ckpt('best_ssim')
            else:
                print(f'not best - val_ssim: {val_ssim:.3f}, best_ssim: {best_ssim:.3f}')

            save_ckpt('last')
            # Save model checkpoint every `ckpt_interval`` epochs or on the last epoch
            if (ckpt_interval is not None and (epoch + 1) % ckpt_interval == 0) or epoch == num_epochs - 1:
                save_ckpt(f'epoch{(epoch + 1):03d}')
            try:
                orig = image
                if reconst is None:
                    z = image_enc_pred[-16:].detach()
                    z = z.to(dtype=autoenc.post_quant_conv.weight.dtype)  # 보통 torch.float32

                    with torch.cuda.amp.autocast(enabled=False):  # 디코드만  FP32
                        reconst = autoenc.decode(z / 0.18215).sample
                    orig = image[-16:]
                pred_grid = make_grid(((reconst / 2 + 0.5).clamp(0, 1) * 255).byte(),
                                      nrow=int(len(reconst) ** 0.5)).permute(1, 2, 0).cpu().numpy()
                orig_grid = make_grid((orig * 255).byte(), nrow=int(len(orig) ** 0.5)).permute(1, 2, 0).cpu().numpy()
                comb_grid = np.concatenate([orig_grid, pred_grid], axis=1)
                del pred_grid, orig_grid
                Image.fromarray(comb_grid).save(f'{outdir}/reconst_epoch{(epoch + 1):03d}.png')
            except:
                print(traceback.format_exc())

        logs = {
            "train/loss": np.mean(losses[-(train_i + 1):]),
            "val/loss": np.mean(val_losses[-(val_i + 1):]),
            "train/lr": lrs[-1],
            "train/num_steps": len(losses),
            "train/loss_mse": loss_mse_sum / (train_i + 1),
            "train/loss_reconst": loss_reconst_sum / (train_i + 1),
            "train/loss_cont": loss_cont_sum / (train_i + 1),
            "train/loss_sobel": loss_sobel_sum / (train_i + 1),
            "val/loss_mse": val_loss_mse_sum / (val_i + 1),
            "val/loss_reconst": val_loss_reconst_sum / (val_i + 1),
            "val/ssim": val_ssim_score_sum / (val_i + 1),
        }
        if local_rank == 0: print(logs)
        if len(reconst_fails) > 0 and local_rank == 0:
            print(f'Reconst fails {len(reconst_fails)}/{train_i}: {reconst_fails}')

    if distributed:
        dist.barrier()







