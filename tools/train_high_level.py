#!/usr/bin/env python
# coding: utf-8
"""
train_high_level.py

ROI별 fMRI → (MLP backbone → BERT-style masking) → 2층 MLP projector → 텍스트 잠복벡터(z) 예측
타깃 z는 Optimus-BERT 인코더(가능하면)로 COCO 캡션을 인코딩해서 얻음.
Optimus 사용이 불가하면 안전한 폴백으로 HuggingFace BERT의 [CLS] 풀드 임베딩을 사용.

필수 경로:
- NSD_ROOT:    /NSD  (기본값)  └ nsddata/experiments/nsd/nsd_expdesign.mat (73KID→COCO image id 매핑)
- COCO captions: /NSD/nsddata_stimuli/stimuli/nsd/annotations/captions_{train,val}2017.json

선택:
- OPTIMUS_HOME: Optimus 레포 로컬 경로 (예: /path/to/Optimus)
- OPTIMUS_CKPT: Optimus 사전학습 체크포인트(.bin/.pt 등) 경로/디렉토리
"""

import os
import sys
import json
import math
import time
import random
import argparse
import traceback
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

# ---- 프로젝트 내부 모듈 (이미 제공된 것과 동일) ----
from datasets.RoiNsdDataset import RoiNsdDataset
import utils

try:
    sys.stdout.reconfigure(line_buffering=True)  # Py>=3.7
except Exception:
    os.environ["PYTHONUNBUFFERED"] = "1"

BAR_FMT = "{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
LOG_EVERY_N = int(os.getenv("LOG_EVERY_N", "0"))

class OptimusMuEncoder(nn.Module):
    """
    - BERT(base-uncased) pooled output -> Linear(2*latent) -> (mu, logvar)
    - encode_texts()는 posterior mean(mu)를 반환
    - optimus_ckpt가 있으면 가능한 키를 매핑해 로드(느슨한 로딩)
    """
    def __init__(
        self,
        device: torch.device,
        preferred_latent_dim: int = 768,
        max_len: int = 64,
        optimus_ckpt: Optional[str] = None,
    ):
        super().__init__()
        self.device = device
        self.max_len = max_len
        self.latent_dim = int(preferred_latent_dim)
        self.use_pooled_as_mu = False  # 기본값

        # tokenizer (신버전 우선)
        try:
            from transformers import BertTokenizer, BertModel
            self._use_hf = True
        except Exception:
            from pytorch_transformers import BertTokenizer, BertModel
            self._use_hf = False

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # BERT backbone
        if self._use_hf:
            from transformers import BertModel
            self.bert = BertModel.from_pretrained("bert-base-uncased")
        else:
            from pytorch_transformers import BertModel, BertConfig
            self.bert = BertModel.from_pretrained("bert-base-uncased")

        hidden = self.bert.config.hidden_size  # 보통 768
        # Optimus는 bert_fea -> linear(2*latent)로 (mu, logvar) 생성
        self.linear = nn.Linear(hidden, 2 * self.latent_dim, bias=True)

        # ckpt가 있으면 가능한 만큼 로드
        if optimus_ckpt is not None and os.path.exists(optimus_ckpt):
            self._load_loose_state_dict(optimus_ckpt)

        self.to(self.device).eval()
        for p in self.parameters():
            p.requires_grad_(False)

        print(f"[OptimusMuEncoder] ready (latent_dim={self.latent_dim}, hf={self._use_hf})", flush=True)

    def _load_loose_state_dict(self, ckpt_path: str):
        print(f"[OptimusMuEncoder] loading ckpt (loose): {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # 접두사 정리
        def strip_prefix(d, prefix):
            return { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in d.items() }

        state = strip_prefix(state, "module.")
        # Optimus 계열에서 흔한 접두사
        for pre in ["encoder.", "model.", "vae.", "text_vae.", "bert_vae.", "vae_module."]:
            state = strip_prefix(state, pre)

        # bert.*와 linear.*만 뽑기
        bert_sd = {k: v for k, v in state.items() if k.startswith("bert.")}
        head_sd = {}
        # 다양한 이름 후보: linear, encoder.linear, bert_connector.linear 등
        for k, v in state.items():
            if k.startswith("linear.") or "encoder.linear." in k or "bert_connector.linear." in k:
                # 키를 linear.* 형태로 정규화
                nk = k.split("linear.", 1)[-1] if "linear." in k else k.split("encoder.linear.", 1)[-1]
                nk = nk if nk and (nk.startswith("weight") or nk.startswith("bias")) else os.path.basename(k)
                head_sd[f"linear.{nk}"] = v

        miss_bert, unexp_bert = self.bert.load_state_dict(bert_sd, strict=False)
        miss_head, unexp_head = self.linear.load_state_dict(head_sd, strict=False)
        print(f"  - bert:    loaded={len(bert_sd)}  missing={len(miss_bert)}  unexpected={len(unexp_bert)}")
        print(f"  - linear:  loaded={len(head_sd)}  missing={len(miss_head)} unexpected={len(unexp_head)}")

        head_ok = ("linear.weight" in head_sd and "linear.bias" in head_sd)
        if not head_ok:
            self.use_pooled_as_mu = True
            self.latent_dim = self.bert.config.hidden_size  # 보통 768
            print("[OptimusMuEncoder] No usable VAE head found -> using BERT pooled output as mu.")

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        if len(texts) == 0:
            return torch.empty(0, self.latent_dim, device=self.device, dtype=torch.float32)

        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        out = self.bert(**enc, return_dict=True)
        pooled = out.pooler_output if (hasattr(out, "pooler_output") and out.pooler_output is not None) \
            else out.last_hidden_state[:, 0, :]

        if getattr(self, "use_pooled_as_mu", False):
            return pooled.float()

        mean_logvar = self.linear(pooled)  # [B, 2*latent]
        mu, logvar = mean_logvar.chunk(2, dim=-1)
        return mu.float()



# ============================================================
# 2) fMRI → 임베딩: MLP backbone + BERT masking + 2-layer projector
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h = self.fc(h)
        h = self.act(h)
        h = self.dp(h)
        return x + h


class BertStyleMask(nn.Module):
    """
    BERT의 MLM 마스킹 아이디어를 '토큰화된 fMRI 피처'에 적용.
    - 입력 x: [B, D]
    - 설정한 token_dim으로 D를 쪼개서 T개 토큰([B, T, token_dim])으로 보고,
      확률 p로 마스크 토큰(학습가능 파라미터)으로 치환 (train일 때만).
    """
    def __init__(self, feature_dim: int, token_dim: int = 32, mask_prob: float = 0.15):
        super().__init__()
        assert token_dim > 0
        self.feature_dim = feature_dim
        self.token_dim = token_dim
        self.mask_prob = mask_prob

        # 나누어떨어지지 않으면 오른쪽에 zero pad
        self.pad = 0
        if feature_dim % token_dim != 0:
            self.pad = token_dim - (feature_dim % token_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, token_dim))  # [1,1,token_dim]
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        if self.pad:
            x = F.pad(x, (0, self.pad), value=0.0)  # [B, D+pad]
            D = D + self.pad

        T = D // self.token_dim
        x_tok = x.view(B, T, self.token_dim)  # [B,T,d]

        if self.training and self.mask_prob > 0.0:
            mask = torch.rand(B, T, device=x.device) < self.mask_prob  # [B,T] bool
            if mask.any():
                # (단순화) 100% 마스크 토큰으로 치환
                x_tok = torch.where(mask[..., None], self.mask_token.expand_as(x_tok), x_tok)

        x_out = x_tok.reshape(B, T * self.token_dim)
        # 패딩 제거
        if self.pad:
            x_out = x_out[:, :self.feature_dim]
        return x_out


class Voxel2TextLatent(nn.Module):
    """
    인풋: [B, V(ROI)]
    백본: Linear → 4×ResBlock → Linear (hidden_dim 유지)
    마스킹: BertStyleMask
    프로젝터: 2-layer MLP → [B, latent_dim]
    """

    def __init__(
            self,
            in_dim: int,
            hidden_dim: int = 2048,
            token_dim: int = 32,
            mask_prob: float = 0.15,
            proj_hidden: int = 1024,
            latent_dim: int = 768,
            input_layernorm: bool = True,
            bottleneck_dim: int = 0,
    ):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim) if input_layernorm else nn.Identity()

        self.use_bottleneck = bottleneck_dim and bottleneck_dim > 0
        first_dim = bottleneck_dim if self.use_bottleneck else in_dim
        if self.use_bottleneck:
            self.in_bottleneck = nn.Linear(in_dim, bottleneck_dim)

        self.backbone_in = nn.Linear(first_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout=0.0) for _ in range(4)])
        self.backbone_out = nn.Linear(hidden_dim, hidden_dim)

        self.masker = BertStyleMask(feature_dim=hidden_dim, token_dim=token_dim, mask_prob=mask_prob)

        self.projector = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,V]
        x = self.in_norm(x)
        if self.use_bottleneck:
            x = self.in_bottleneck(x)

        h = self.backbone_in(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.backbone_out(h)
        h = self.masker(h)
        z_pred = self.projector(h)
        return z_pred


# ============================================================
# 3) COCO 캡션 로딩 & NSD 73KID → COCO image_id 매핑
# ============================================================

def _load_coco_captions(captions_json_paths: List[str]) -> Dict[int, List[str]]:
    """
    COCO captions json들을 읽어 image_id → [captions] 딕셔너리 생성
    """
    img2caps: Dict[int, List[str]] = {}
    for p in captions_json_paths:
        if not os.path.exists(p):
            continue
        with open(p, "r") as f:
            data = json.load(f)
        anns = data.get("annotations", [])
        for a in anns:
            img_id = int(a["image_id"])
            cap = a.get("caption", "")
            if cap is None or len(cap) == 0:
                continue
            img2caps.setdefault(img_id, []).append(cap)
    return img2caps

def _load_coco_id_vec_from_csv(csv_path: str) -> Optional[np.ndarray]:
    """
    nsd_stim_info_merged.csv에서 73k 순서의 cocoId 벡터 (int64, -1=없음)를 로드
    """
    import pandas as pd
    if not os.path.exists(csv_path):
        print(f"[coco csv] not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    # nsdId가 있으면 안전하게 정렬
    if "nsdId" in df.columns:
        df = df.sort_values("nsdId")
    # cocoId 같은 이름 찾기
    coco_col = None
    for col in ("cocoId", "coco_id", "COCO_ID", "cocoID"):
        if col in df.columns:
            coco_col = col
            break
    if coco_col is None:
        print(f"[coco csv] no cocoId-like column in {csv_path}")
        return None

    coco = df[coco_col].to_numpy()
    # float + NaN → -1로, 최종 int64
    coco = np.array(coco, dtype=np.float64)
    n_nan = int(np.isnan(coco).sum())
    coco = np.nan_to_num(coco, nan=-1.0).astype(np.int64)
    print(f"[coco csv] loaded cocoId: shape={coco.shape}, NaNs(before cast)={n_nan}, path={csv_path}")
    return coco


class CaptionProvider:
    """
    trial(=73KID, 0-based) → 캡션 하나 선택
    - coco_id_vec: np.ndarray [73000]  (-1=없음)
    - COCO 캡션 json에서 image_id→captions 리스트
    """
    def __init__(self, captions_dir: str, coco_id_vec: Optional[np.ndarray]):
        self.captions_dir = captions_dir

        train_json = os.path.join(captions_dir, "captions_train2017.json")
        val_json   = os.path.join(captions_dir, "captions_val2017.json")
        self.img2caps = _load_coco_captions([train_json, val_json])  # {image_id: [caps]}

        self.map73k = None
        if coco_id_vec is not None:
            arr = np.asarray(coco_id_vec)
            if arr.dtype != np.int64:
                arr = arr.astype(np.int64)
            self.map73k = arr
            print(f"[CaptionProvider] coco_id_vec loaded: shape={arr.shape}, dtype={arr.dtype}")
        else:
            print("[CaptionProvider] coco_id_vec is None. No 73k→COCO mapping.")

        total_caps = sum(len(v) for v in self.img2caps.values())
        print(f"[CaptionProvider] COCO captions loaded: {len(self.img2caps)} images, {total_caps} captions", flush=True)

        if self.map73k is not None:
            valid = (self.map73k > 0)
            covered = sum((int(cid) in self.img2caps) for cid in self.map73k[valid])
            cover_rate = covered / max(1, valid.sum())
            print(f"[CaptionProvider] coverage: {covered}/{valid.sum()} = {cover_rate:.3f}")
            for t in np.random.choice(np.where(valid)[0], size=min(5, valid.sum()), replace=False):
                cid = int(self.map73k[t])
                caps = self.img2caps.get(cid, [])
                if caps:
                    print(f"[CaptionProvider] sample trial={t} coco_id={cid} cap='{caps[0]}'")
                    break

    def caption_for_trial(self, trial_id: int) -> Optional[str]:
        if self.map73k is None:
            return None
        if not (0 <= trial_id < len(self.map73k)):
            return None
        coco_id = int(self.map73k[trial_id])
        if coco_id <= 0:
            return None
        caps = self.img2caps.get(coco_id, [])
        if not caps:
            return None
        return random.choice(caps)

    # ---------- Caption target cache (trial-wise h_GT) ----------
    @staticmethod
    def collect_used_trials(dataloader) -> List[int]:
        used = set()
        for vox, img, trial in dataloader:
            t = trial.view(-1).tolist()
            used.update(t)
        return sorted(used)

    @staticmethod
    @torch.no_grad()
    def build_or_load_text_targets(cap_provider: "CaptionProvider",
                                   text_enc: "OptimusMuEncoder",
                                   used_trials: List[int],
                                   cache_path: Optional[str],
                                   device: torch.device,
                                   batch_cap: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          z_cache: (N_trials, D) float32, missing=nan
          valid:   (N_trials,) bool
        If cache_path exists, loads from npz; else computes for used_trials and saves.
        """
        N = len(cap_provider.map73k) if cap_provider.map73k is not None else (max(used_trials) + 1)
        D = text_enc.latent_dim
        if cache_path is not None and os.path.exists(cache_path):
            npz = np.load(cache_path)
            z_cache = npz["z_cache"].astype(np.float32)
            valid = npz["valid"].astype(bool)
            assert z_cache.shape == (N, D), f"cache shape mismatch: {z_cache.shape} vs {(N, D)}"
            return z_cache, valid

        # init
        z_cache = np.full((N, D), np.nan, dtype=np.float32)
        valid = np.zeros((N,), dtype=bool)

        # compute only for used_trials
        for t in used_trials:
            if cap_provider.map73k is None or not (0 <= t < len(cap_provider.map73k)):
                continue
            cid = int(cap_provider.map73k[t])
            if cid <= 0:
                continue
            caps = cap_provider.img2caps.get(cid, [])
            if not caps:
                continue

            # encode all caps, average -> h_GT
            z = text_enc.encode_texts(caps)  # [K, D] on device
            z = z.mean(dim=0, keepdim=False)  # [D]
            z_cache[t] = z.detach().cpu().float().numpy()
            valid[t] = True

        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(cache_path, z_cache=z_cache, valid=valid)
            print(f"[cap-cache] saved: {cache_path} (shape={z_cache.shape}, valid={valid.sum()})", flush=True)
        return z_cache, valid


# ============================================================
# 4) 학습 루프
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    # --- 런타임/학습 설정 ---
    parser.add_argument("--subj-id", default="01", type=str, help="NSD subject id like 01,02,05,07")
    parser.add_argument("--roi-name", "--roi_name", dest="roi_name", default="streams", type=str)
    parser.add_argument("--use-avg", dest="roi_use_avg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str, default="onecycle", choices=["onecycle", "linear", "none"])
    parser.add_argument("--seed", type=int, default=42)

    # --- 경로 ---
    parser.add_argument("--nsd-root", default=os.environ.get("NSD_ROOT", "/NSD"), type=str)
    parser.add_argument("--nv2l-root", default=os.environ.get("NV2L_OUT", "/NSD/nv2l_outputs/datasets"), type=str)
    parser.add_argument("--coco-ann-root",
                        default="/NSD/nsddata_stimuli/stimuli/nsd/annotations", type=str)

    # --- Optimus ---
    parser.add_argument("--optimus-home", default=os.environ.get("OPTIMUS_HOME", None), type=str)
    parser.add_argument("--optimus-ckpt", default=os.environ.get("OPTIMUS_CKPT", None), type=str)
    parser.add_argument("--latent-dim", type=int, default=768)
    parser.add_argument("--max-text-len", type=int, default=64)

    # --- 모델 하이퍼파라미터 ---
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--token-dim", type=int, default=32)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--proj-hidden", type=int, default=1024)
    parser.add_argument("--input-layernorm", action=argparse.BooleanOptionalAction, default=True)

    # --- 로깅/저장 ---
    parser.add_argument("--model-name", type=str, default="high_text_latent")
    parser.add_argument("--ckpt-interval", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default=None)


    parser.add_argument("--coco-id-csv", type=str,
                        default="/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.csv",
                        help="nsd_stim_info_merged.csv 경로(73k→COCO 매핑을 여기서 읽음)")

    # --- 타깃 캡션 캐시/보조지표/정규화/입력 보틀넥 ---
    parser.add_argument("--cap-cache", type=str, default=None,
                        help="npz path to cache averaged caption targets (trial-wise h_GT).")
    parser.add_argument("--eval-cosine", action=argparse.BooleanOptionalAction, default=True,
                        help="Log cosine distance (1-cos) on validation.")
    parser.add_argument("--zscore-stats", type=str, default=None,
                        help="npz file containing 'mean' and 'std' arrays for ROI z-scoring (shape=[V]).")
    parser.add_argument("--bottleneck-dim", type=int, default=0,
                        help="If >0, add a learned Linear bottleneck (in_dim->bottleneck_dim) before backbone.")

    args = parser.parse_args()

    # ---- 73k→COCO id 매핑 로드 ----
    coco_id_vec = None
    if args.coco_id_csv and os.path.exists(args.coco_id_csv):
        coco_id_vec = _load_coco_id_vec_from_csv(args.coco_id_csv)
        print("[main] loaded COCO id vector from CSV")

    # 장치/시드
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    utils.seed_everything(args.seed, cudnn_deterministic=False)
    torch.backends.cuda.matmul.allow_tf32 = True

    # 출력 디렉터리
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    outdir = args.save_dir or os.path.join(ROOT_DIR, "train_logs", "high", args.model_name)
    os.makedirs(outdir, exist_ok=True)

    # ----- 데이터셋 (ROI + NSD 이미지) -----
    subj_tag = f"subj{args.subj_id}"
    roi_dir  = os.path.join(args.nv2l_root, "nsd", "fmris", subj_tag, "area")

    train_ds = RoiNsdDataset(
        nsd_root=args.nsd_root, subj_tag=subj_tag,
        roi_dir=roi_dir, roi_name=args.roi_name,
        split="train", use_avg=args.roi_use_avg, image_size=224
    )
    val_ds = RoiNsdDataset(
        nsd_root=args.nsd_root, subj_tag=subj_tag,
        roi_dir=roi_dir, roi_name=args.roi_name,
        split="val", use_avg=args.roi_use_avg, image_size=224
    )
    num_train, num_val = len(train_ds), len(val_ds)
    print(f"[data] num_train={num_train}, num_val={num_val}", flush=True)

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=max(1, torch.cuda.device_count()), pin_memory=True, drop_last=True, persistent_workers=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=max(16, args.batch_size), shuffle=False,
        num_workers=max(1, torch.cuda.device_count() // 2), pin_memory=True, persistent_workers=True
    )

    # ----- 캡션 로더 -----
    cap_provider = CaptionProvider(
        captions_dir=args.coco_ann_root,
        coco_id_vec=coco_id_vec
    )

    # ----- 텍스트 인코더(Optimus 전용) -----
    text_enc = OptimusMuEncoder(
        device=device,
        preferred_latent_dim=args.latent_dim,
        max_len=args.max_text_len,
        optimus_ckpt=args.optimus_ckpt,
    )

    # ===== 캡션 타깃 캐시  =====
    print("[cap-cache] collecting used trials (train/val) ...", flush=True)
    used_trials = sorted(
        set(CaptionProvider.collect_used_trials(train_dl)) |
        set(CaptionProvider.collect_used_trials(val_dl))
    )
    print(f"[cap-cache] #used_trials={len(used_trials)}", flush=True)

    z_cache, z_valid = CaptionProvider.build_or_load_text_targets(
        cap_provider=cap_provider,
        text_enc=text_enc,
        used_trials=used_trials,
        cache_path=args.cap_cache,
        device=device,
    )
    z_cache_t = torch.from_numpy(z_cache).to(device)  # [N_trials, D]
    z_valid_t = torch.from_numpy(z_valid)  # CPU bool ok
    print(f"[cap-cache] ready: z_cache={tuple(z_cache_t.shape)}, valid={z_valid.sum()}", flush=True)

    # ----- fMRI → 텍스트 잠복 -----
    # 입력 차원은 ROI npy의 두 번째 축 길이로 얻는다
    if args.roi_use_avg:
        roi_path = os.path.join(roi_dir, f"nsd_{args.roi_name}_betas_ave_tr.npy")
        roi_ave_tr = np.load(roi_path)
        print(f"[roi] loaded: {roi_path} shape={roi_ave_tr.shape}", flush=True)
        in_dim = int(roi_ave_tr.shape[1])
        del roi_ave_tr
    else:
        roi_path = os.path.join(roi_dir, f"nsd_{args.roi_name}_betas_tr.npy")
        roi_tr = np.load(roi_path)
        print(f"[roi] loaded: {roi_path} shape={roi_tr.shape}", flush=True)
        in_dim = int(roi_tr.shape[1])
        del roi_tr

    # 전역 z-score 통계 로드
    zscore_tensors = None
    if args.zscore_stats is not None and os.path.exists(args.zscore_stats):
        zs = np.load(args.zscore_stats)
        mean = torch.from_numpy(zs["mean"]).float().to(device)  # shape=[V]
        std = torch.from_numpy(zs["std"]).float().to(device)  # shape=[V]
        assert mean.shape[0] == in_dim and std.shape[0] == in_dim, \
            f"zscore mean/std shape mismatch: {mean.shape} {std.shape} vs in_dim={in_dim}"
        zscore_tensors = (mean, std)
        print(f"[roi] using zscore_stats: {args.zscore_stats} (shape={mean.shape})", flush=True)
    else:
        if args.zscore_stats is not None:
            print(f"[roi] zscore_stats not found: {args.zscore_stats} -> skip.", flush=True)

    model = Voxel2TextLatent(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        token_dim=args.token_dim,
        mask_prob=args.mask_prob,
        proj_hidden=args.proj_hidden,
        latent_dim=text_enc.latent_dim,
        input_layernorm=args.input_layernorm,
        bottleneck_dim=args.bottleneck_dim,
    ).to(device)

    utils.count_params(model)

    # ----- 옵티마이저/스케줄러 -----
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    opt_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=args.lr)

    if args.sched == "onecycle":
        total_steps = args.epochs * max(1, (num_train // args.batch_size))
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.max_lr, total_steps=total_steps,
            final_div_factor=1000, last_epoch=-1, pct_start=2/ max(args.epochs, 2)
        )
    elif args.sched == "linear":
        total_steps = args.epochs * max(1, (num_train // args.batch_size))
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=total_steps)
    else:
        lr_scheduler = None

    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    # ----- 체크포인트 -----
    def save_ckpt(tag: str, epoch_i: int, losses_tr: List[float], losses_val: List[float], lrs: List[float]):
        ckpt_path = os.path.join(outdir, f"{tag}.pth")
        print(f"[ckpt] saving {ckpt_path}", flush=True)
        try:
            torch.save({
                "epoch": epoch_i,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": losses_tr,
                "val_losses": losses_val,
                "lrs": lrs,
            }, ckpt_path)
        except Exception as e:
            print(f"[ckpt] save failed: {e}", flush=True)

    # ----- 학습 루프 -----
    best_val = 1e9
    train_losses: List[float] = []
    val_losses: List[float] = []
    lrs_hist: List[float] = []

    progress_bar = tqdm(range(args.epochs), ncols=120, bar_format=BAR_FMT, file=sys.stdout, leave=True)
    for epoch in progress_bar:
        model.train()
        tr_loss_sum = 0.0

        for step, (vox, img, trial) in enumerate(train_dl):
            optimizer.zero_grad(set_to_none=True)

            # voxel 텐서 정리
            if vox.ndim == 2:
                vox = vox.to(device).float()
            elif vox.ndim == 3:
                # 반복 차원 있으면 step 기준으로 하나 선택
                rep = step % vox.shape[1]
                vox = vox[:, rep].to(device).float()
            else:
                raise RuntimeError(f"Unexpected voxel shape: {vox.shape}")

            if zscore_tensors is not None:
                mean, std = zscore_tensors
                vox = (vox - mean) / (std + 1e-6)

            trial_ids = trial.view(-1).tolist()
            keep_idx = [i for i, t in enumerate(trial_ids) if (0 <= t < z_valid_t.numel()) and bool(z_valid_t[t])]
            if len(keep_idx) == 0:
                continue
            if len(keep_idx) < vox.size(0):
                vox = vox[keep_idx]

            tid_tensor = torch.tensor([trial_ids[i] for i in keep_idx], device=device, dtype=torch.long)
            z_targ = z_cache_t.index_select(dim=0, index=tid_tensor)  # [B', D]

            # ---- fMRI → z_pred ----
            with torch.cuda.amp.autocast(enabled=use_cuda, dtype=torch.float16):
                # voxel 경로는 fp32가 안전 → 모델 내부에서 자동 처리하고 싶다면 아래처럼 강제 캐스팅
                vox_fp32 = vox.float()
                z_pred = model(vox_fp32)  # [B', latent_dim]
                loss = F.mse_loss(z_pred, z_targ)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tr_loss_sum += float(loss.item())
            train_losses.append(float(loss.item()))
            lrs_hist.append(float(optimizer.param_groups[0]["lr"]))

            if lr_scheduler is not None:
                lr_scheduler.step()

            if LOG_EVERY_N and ((step + 1) % LOG_EVERY_N == 0):
                recent = float(np.mean(train_losses[-LOG_EVERY_N:])) if len(train_losses) >= LOG_EVERY_N else train_losses[-1]
                print(f"[epoch {epoch+1}/{args.epochs} step {step+1}/{len(train_dl)}] "
                      f"train/mse={recent:.4f} lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)

        # ---- 검증 ----
        model.eval()
        val_sum = 0.0
        n_val = 0
        val_cos_losses: List[float] = []
        val_cos_acc = 0.0  # 초기화 (locals() 트릭 제거)

        with torch.inference_mode():
            for val_i, (vox, img, trial) in enumerate(val_dl):
                if vox.ndim == 2:
                    vox = vox.to(device).float()
                elif vox.ndim == 3:
                    rep = val_i % vox.shape[1]
                    vox = vox[:, rep].to(device).float()
                else:
                    raise RuntimeError(f"Unexpected voxel shape: {vox.shape}")

                if zscore_tensors is not None:
                    mean, std = zscore_tensors
                    vox = (vox - mean) / (std + 1e-6)

                # ====== 캐시에서 타깃 로드 ======
                trial_ids = trial.view(-1).tolist()
                keep_idx = [i for i, t in enumerate(trial_ids)
                            if (0 <= t < z_valid_t.numel()) and bool(z_valid_t[t])]
                if len(keep_idx) == 0:
                    continue
                if len(keep_idx) < vox.size(0):
                    vox = vox[keep_idx]

                tid_tensor = torch.tensor([trial_ids[i] for i in keep_idx], device=device, dtype=torch.long)
                z_targ = z_cache_t.index_select(dim=0, index=tid_tensor)  # [B', D]

                z_pred = model(vox.float())
                vloss = F.mse_loss(z_pred, z_targ)
                val_sum += float(vloss.item())
                n_val += 1

                if args.eval_cosine:
                    z_pred_n = F.normalize(z_pred, dim=-1)
                    z_targ_n = F.normalize(z_targ, dim=-1)
                    cos = F.cosine_similarity(z_pred_n, z_targ_n, dim=-1)  # [B']
                    val_cos_acc += float((1.0 - cos.mean()).item())

        val_mean = (val_sum / max(1, n_val))
        val_losses.append(val_mean)

        if args.eval_cosine:
            val_cos_mean = (val_cos_acc / max(1, n_val))
            val_cos_losses.append(val_cos_mean)
        else:
            val_cos_mean = float('nan')

        logs = OrderedDict(
            train_loss=tr_loss_sum / max(1, len(train_dl)),
            val_loss=val_mean,
            val_cos=val_cos_mean if args.eval_cosine else None,
            lr=optimizer.param_groups[0]["lr"],
        )
        progress_bar.set_postfix(**{k: v for k, v in logs.items() if v is not None})
        print(f"[epoch {epoch + 1}/{args.epochs}] train/mse={logs['train_loss']:.4f}  "
              f"val/mse={logs['val_loss']:.4f}  "
              f"{('val/1-cos=%.4f  ' % logs['val_cos']) if args.eval_cosine else ''}"
              f"lr={logs['lr']:.2e}",
              flush=True)

        def save_ckpt(tag: str, epoch_i: int, losses_tr: List[float], losses_val: List[float], lrs: List[float]):
            ckpt_path = os.path.join(outdir, f"{tag}.pth")
            print(f"[ckpt] saving {ckpt_path}", flush=True)
            try:
                torch.save({
                    "epoch": epoch_i,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_losses": losses_tr,
                    "val_losses": losses_val,
                    "val_cos_losses": val_cos_losses if args.eval_cosine else None,
                    "lrs": lrs,
                }, ckpt_path)
            except Exception as e:
                print(f"[ckpt] save failed: {e}", flush=True)

    print("\n=== Finished! ===\n", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc(), flush=True)
        sys.exit(1)
