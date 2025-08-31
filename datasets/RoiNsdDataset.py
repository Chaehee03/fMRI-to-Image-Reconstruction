import os, json, numpy as np, torch
from torch.utils.data import Dataset
from nsd_access import NSDAccess

class RoiNsdDataset(Dataset):
    """
    ROI npy와 73KID(0-based) → NSD 이미지 매칭으로 (voxel, image, trial) 튜플을 반환.
    - use_avg=True  : (평균/유니크) 배열 + *_ave_{train,val}_ids.npy 사용
    - use_avg=False : (trial 단위) 배열 + nsd_fmri2image.json 사용
    이미지는 [0,1] float32, 224x224 로 고정해 배치가 잘 쌓이도록 함.
    """
    def __init__(self, nsd_root, subj_tag, roi_dir, roi_name, split="train",
                 use_avg=True, image_size=224, cache_size=256):
        assert split in ("train", "val")
        self.nsda = NSDAccess(nsd_root)
        self.image_size = image_size
        self.use_avg = use_avg
        self.roi_dir = roi_dir
        self.roi_name = roi_name
        self.split = split
        self._cache = {}    # 매우 단순한 LRU-ish
        self._cache_size = cache_size
        self._img_index_base = None  # 1 -> ids are 1-based (NSD "nsdId"), 0 -> 0-based

        if use_avg:
            # ROI: 평균(유니크) 배열 + 그 행에 대응하는 73KID-1 id 리스트
            npy = f"nsd_{roi_name}_betas_ave_{'tr' if split=='train' else 'te'}.npy"
            ids = f"nsd_{roi_name}_ave_{'train' if split=='train' else 'val'}_ids.npy"
            self.roi = np.load(os.path.join(roi_dir, npy)).astype(np.float32, copy=False)
            self.ids = np.load(os.path.join(roi_dir, ids)).astype(int).tolist()
            self.id2row = {int(t): i for i, t in enumerate(self.ids)}
            if len(self.ids):
                mn, mx = min(self.ids), max(self.ids)
                self._img_index_base = 1 if mn >= 1 and mx <= 73000 else 0
        else:
            # ROI: trial 배열 + nsd_fmri2image.json 의 순서와 1:1
            idxjson = json.load(open(os.path.join(os.path.dirname(roi_dir), "nsd_fmri2image.json")))
            self.ids = [int(t) for t in idxjson["train" if split=="train" else "val"]]
            npy = f"nsd_{roi_name}_betas_{'tr' if split=='train' else 'te'}.npy"
            self.roi = np.load(os.path.join(roi_dir, npy)).astype(np.float32, copy=False)
            assert len(self.ids) == self.roi.shape[0], "ids와 ROI 행수가 불일치"
            if len(self.ids):
                mn, mx = min(self.ids), max(self.ids)
                self._img_index_base = 1 if mn >= 1 and mx <= 73000 else 0

    def __len__(self):
        return len(self.ids)

    def _get_image(self, kid0: int) -> torch.Tensor:
        # NSDAccess는 1-based, 우리는 0-based로 들고 있으니 +1
        if kid0 in self._cache:
            img = self._cache[kid0]
        else:
            if self._img_index_base == 1:
                # ids가 1-based면 그대로 넘김 (nsd_access 내부에서 -1함)
                candidates = [kid0]
            elif self._img_index_base == 0:
                # ids가 0-based면 +1 해서 넘김
                candidates = [kid0 + 1]
            else:
                # 모르면 둘 다 시도
                candidates = [kid0, kid0 + 1]

            last_exc = None
            img = None
            for val in candidates:
                try:
                    # 일부 버전은 키워드 인자를 받지 않음 → 위치 인자로 호출
                    img = self.nsda.read_images([val])[0]  # HxWx3 uint8
                    # 성공했으면 베이스 확정
                    self._img_index_base = 1 if val == kid0 else 0
                    break
                except TypeError:
                    # 혹시 키워드만 받는 경우 대비
                    try:
                        img = self.nsda.read_images(image_ids=[val])[0]
                        self._img_index_base = 1 if val == kid0 else 0
                        break
                    except Exception as e:
                        last_exc = e
                except Exception as e:
                    last_exc = e
            if img is None:
                # 두 방식 모두 실패 → 원인 메시지 그대로 올림
                raise last_exc

            self._cache[kid0] = img
            if len(self._cache) > self._cache_size:
                # 가장 먼저 들어온 키 하나 제거
                self._cache.pop(next(iter(self._cache)))
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # [3,H,W], 0..1
        if self.image_size:
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0), (self.image_size, self.image_size),
                mode="bilinear", align_corners=False, antialias=True
            ).squeeze(0)
        return t

    def __getitem__(self, i):
        kid0 = int(self.ids[i])
        if self.use_avg:
            r = self.roi[self.id2row[kid0]]
        else:
            r = self.roi[i]
        voxel = torch.from_numpy(r)              # [V]
        image = self._get_image(kid0)            # [3,224,224], float32 in [0,1]
        trial = torch.tensor(kid0, dtype=torch.long)  # 73KID-1
        return voxel, image, trial
