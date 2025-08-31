import argparse
import json
import os, sys
import numpy as np
import pandas as pd
import scipy.io



sys.path.append("../../")

NSD_ROOT = "/NSD"

from nsd_access import NSDAccess
nsda = NSDAccess(NSD_ROOT)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )

    opt = parser.parse_args()
    subject = opt.subject

    OUTPUT_ROOT = os.environ.get("NV2L_OUT", "/NSD/nv2l_outputs/datasets")

    output_dir = os.path.join(OUTPUT_ROOT, "nsd", "fmris", subject)
    os.makedirs(os.path.join(output_dir, "whole"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "area"), exist_ok=True)


    atlasname = 'streams'

    nsda = NSDAccess(NSD_ROOT)

    exp_mat = os.path.join(NSD_ROOT, "nsddata/experiments/nsd/nsd_expdesign.mat")
    nsd_expdesign = scipy.io.loadmat(exp_mat)

    sharedix = nsd_expdesign['sharedix'] - 1

    atlas = nsda.read_atlas_results(subject=subject, atlas=atlasname, data_format='func1pt8mm')

    atlas_to_json = [atlas[0].tolist(), {k: int(v) for k, v in atlas[1].items()}]

    json.dump(atlas_to_json, open(f'{output_dir}/atlas.json', 'w'), indent=4)

    atlas_general = nsda.read_atlas_results(subject=subject, atlas='nsdgeneral', data_format='func1pt8mm')
    atlas_to_json = [atlas_general[0].tolist(), {k: int(v) for k, v in atlas_general[1].items()}]
    json.dump(atlas_to_json, open(f'{output_dir}/atlas_general.json', 'w'), indent=4)

    # exit(0)

    behs = pd.DataFrame()
    for i in range(1, 38):
        beh = nsda.read_behavior(subject=subject,
                                 session_index=i)
        behs = pd.concat((behs, beh))

    # Caution: 73KID is 1-based! https://cvnlab.slite.page/p/fRv4lz5V2F/Behavioral-data
    stims_unique = behs['73KID'].unique() - 1  # 9841 unique images
    stims_all = behs['73KID'] - 1  # 27750 trials

    mask = np.isin(stims_all, sharedix[0])
    train_index = stims_all[~mask]
    val_index = stims_all[mask]


    print("[Train/Val Split (INDEX)]")
    print(train_index.shape, val_index.shape)

    with open(f'{output_dir}/nsd_fmri2image.json', 'w') as f:
        index = {'train': train_index.tolist(), 'val': val_index.tolist()}
        json.dump(index, f, indent=4)


    def _move_trial_to_axis0(arr):
        TRIAL_SIZES = {750, 738, 735, 372, 300, 280}
        # 이미 trial이 앞축이면 그대로
        if arr.shape[0] in TRIAL_SIZES:
            return arr
        # trial이 끝축이면 앞으로 이동
        if arr.shape[-1] in TRIAL_SIZES:
            return np.moveaxis(arr, -1, 0)
        # 그 외는 그대로 (예외적으로 [X,Y,Z]만 온 경우 등)
        return arr

    if 'SESSION' in behs.columns:
        counts_by_sess = behs.groupby('SESSION').size().sort_index()  # 세션 번호 기준 정렬

    print("1")
    sess = 1
    beta_trial = nsda.read_betas(
        subject=subject, session_index=sess, trial_index=[],
        data_type='betas_fithrf_GLMdenoise_RR', data_format='func1pt8mm'
    )


    beta_trial = _move_trial_to_axis0(beta_trial).astype(np.float32, copy=False)


    if 'SESSION' in behs.columns:
        n_trials_beta = beta_trial.shape[0]
        n_trials_beh = int(counts_by_sess.loc[sess])
        assert n_trials_beta == n_trials_beh, \
            f"Session {sess}: betas({n_trials_beta}) vs behav({n_trials_beh}) 불일치"

    assert beta_trial.ndim in (3, 4), f"Unexpected betas ndim={beta_trial.ndim}"
    first_target = beta_trial.shape[1:]  # (X,Y,Z) 또는 (X,Y)


    A_streams = atlas[0]
    if A_streams.shape != first_target:
        for perm in [(0, 1, 2), (2, 1, 0), (2, 0, 1), (1, 2, 0), (1, 0, 2), (0, 2, 1)]:
            if A_streams.transpose(perm).shape == first_target:
                A_streams = A_streams.transpose(perm)
                break
        else:
            raise RuntimeError(f"streams atlas shape {A_streams.shape} cannot match betas {first_target}")

    A_gen = atlas_general[0]
    if A_gen.shape != first_target:
        for perm in [(0, 1, 2), (2, 1, 0), (2, 0, 1), (1, 2, 0), (1, 0, 2), (0, 2, 1)]:
            if A_gen.transpose(perm).shape == first_target:
                A_gen = A_gen.transpose(perm)
                break
        else:
            raise RuntimeError(f"nsdgeneral atlas shape {A_gen.shape} cannot match betas {first_target}")

    # ===== ROI 마스크 사전 준비 =====
    SKIP_WHOLE = True  # 메모리 폭발 방지: whole은 매우 큼
    roi_masks = {}
    for roi_label, val in atlas[1].items():
        if val == 0:
            if SKIP_WHOLE:
                continue
            name = 'whole'
            roi_masks[name] = np.ones_like(A_streams, dtype=bool)
        else:
            name = roi_label
            roi_masks[name] = (A_streams == val)

    mask_gen = (A_gen > 0)

    # ===== ROI별로 세션 조각을 모아둘 리스트 준비 =====
    roi_chunks = {name: [] for name in roi_masks.keys()}
    gen_chunks = []  # nsdgeneral

    # =====  i=1은 이미 읽었으니 먼저 누적
    for name, mask in roi_masks.items():
        # beta_trial shape: [S, X, Y, Z] 또는 [S, X, Y]
        roi_2d = beta_trial[:, mask]  # -> [S, V_roi]
        roi_chunks[name].append(roi_2d.astype(np.float32, copy=False))
    gen_chunks.append(beta_trial[:, mask_gen])


    for sess in range(2, 38):
        print(sess)
        beta_trial = nsda.read_betas(
            subject=subject, session_index=sess, trial_index=[],
            data_type='betas_fithrf_GLMdenoise_RR', data_format='func1pt8mm'
        )
        beta_trial = _move_trial_to_axis0(beta_trial).astype(np.float32, copy=False)

        if 'SESSION' in behs.columns:
            n_trials_beta = beta_trial.shape[0]
            n_trials_beh = int(counts_by_sess.loc[sess])
            assert n_trials_beta == n_trials_beh, \
                f"Session {sess}: betas({n_trials_beta}) vs behav({n_trials_beh}) 불일치"

        assert beta_trial.shape[1:] == first_target, \
            f"Session {i} shape mismatch: {beta_trial.shape[1:]} vs {first_target}"

        for name, mask in roi_masks.items():
            roi_2d = beta_trial[:, mask]
            roi_chunks[name].append(roi_2d.astype(np.float32, copy=False))

        gen_chunks.append(beta_trial[:, mask_gen])


    # ===== 루프 종료 후에만 concatenate
    for name in roi_chunks.keys():
        roi_chunks[name] = np.concatenate(roi_chunks[name], axis=0)  # [N_trials, V_roi]

    betas_gen = np.concatenate(gen_chunks, axis=0).astype(np.float16, copy=False)

    # betas_gen = np.concatenate(gen_chunks, axis=0)  # [N_trials, V_gen]
    print("[nsdgeneral] betas shape:", betas_gen.shape)

    def split_and_save(roi_name, betas_roi, outdir_area):
        """
        betas_roi: shape (N_trials, V)  - trial 단위
        """
        # 1) 유니크 이미지(73KID-1) 평균
        ave_list = []
        for stim in stims_unique_np:
            ave_list.append(np.mean(betas_roi[stims_all_np == stim, :], axis=0))
        betas_roi_ave = np.stack(ave_list)  # (N_unique, V)

        # 2) trial 단위 train/test split
        tr_list, te_list = [], []
        for idx, stim in enumerate(stims_all_np):
            (te_list if stim in sharedix_set else tr_list).append(betas_roi[idx, :])
        betas_tr = np.stack(tr_list)
        betas_te = np.stack(te_list)

        # 3) 평균(유니크) 단위 train/test split
        ave_tr_list, ave_te_list = [], []
        ave_train_ids, ave_val_ids = [], []
        for idx, stim in enumerate(stims_unique_np):
            if stim in sharedix_set:
                ave_te_list.append(betas_roi_ave[idx, :])
                ave_val_ids.append(int(stim))
            else:
                ave_tr_list.append(betas_roi_ave[idx, :])
                ave_train_ids.append(int(stim))
        betas_ave_tr = np.stack(ave_tr_list)
        betas_ave_te = np.stack(ave_te_list)

        np.save(f'{outdir_area}/nsd_{roi_name}_betas_tr.npy', betas_tr)
        np.save(f'{outdir_area}/nsd_{roi_name}_betas_te.npy', betas_te)
        np.save(f'{outdir_area}/nsd_{roi_name}_betas_ave_tr.npy', betas_ave_tr)
        np.save(f'{outdir_area}/nsd_{roi_name}_betas_ave_te.npy', betas_ave_te)
        # avg id도 함께 저장 (매핑 안정성 ↑)
        np.save(f'{outdir_area}/nsd_{roi_name}_ave_train_ids.npy', np.array(ave_train_ids, dtype=np.int32))
        np.save(f'{outdir_area}/nsd_{roi_name}_ave_val_ids.npy', np.array(ave_val_ids, dtype=np.int32))

    stims_all_np = np.asarray(stims_all)
    stims_unique_np = np.asarray(stims_unique)
    sharedix_set = set(np.asarray(sharedix).ravel().tolist())

    # ---- streams 아틀라스 ROI들 저장  ----
    out_area = os.path.join(output_dir, "area")
    for name, betas_roi in roi_chunks.items():
        print(f"[{name}] betas shape:", betas_roi.shape)
        split_and_save(name, betas_roi, out_area)

    # ---- nsdgeneral 저장 ----
    split_and_save('nsdgeneral', betas_gen, out_area)


if __name__ == "__main__":
    main()
