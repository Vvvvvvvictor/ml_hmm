# -*- coding: utf-8 -*-
# Optimization binning with Data-Driven Sideband Normalization
# 10,000 Toys for Robust Significance Estimation (Vectorized)

import math
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os
import uproot
import time

# 设置CMS风格
plt.style.use(hep.style.CMS)

# ------------------------
# ----- CONFIGURATION ----
# ------------------------
base_dir = "/eos/home-j/jiehan/root_mumu/outputs_VBF_2223_bdt_1123/VBF/"

sig_file   = f"{base_dir}/sig.root"
bkg_file   = f"{base_dir}/bkg.root"

sig_proc = ["GluGluHToMuMu_M125","VBFHToMuMu_M125"]
bkg_proc = ["EWK_LLJJ_M50", "ST_tW_antitop", "TTTo2L2Nu", "WWTo2L2Nu", "WWZ", "WZTo3LNu", "ZZTo2L2Nu", "ZZTo4L", "DY_105To160", "ST_tW_top", "WWW", "WZTo2L2Q", "WZZ", "ZZTo2L2Q", "ZZZ"]

# -------------------------------------------------------
# [NEW] TOY CONFIGURATION
# -------------------------------------------------------
N_TOYS = 10000      # 产生 Asimov Data 的次数
RANDOM_SEED = 12345 # 固定种子以保证结果可复现

tree_name  = "test"
score_var  = "bdt_score"
weight_var = "eventWeight"
mass_var   = "diMufsr_rc_mass"

# Region Definitions
sb_low_min, sb_low_max = 110.0, 115.0
sb_high_min, sb_high_max = 135.0, 150.0
sr_min, sr_max = 115.0, 135.0

num_thresholds = 200
max_categories = 10  # 建议根据时间调整，计算量较大

# ------------------------
# ----- UTILS ------------
# ------------------------

def check_and_merge(target, sources):
    if not os.path.exists(target):
        print(f"[INFO] Merging to {target}...")
        src_files = [f"{base_dir}/{p}.root" for p in sources]
        existing_srcs = [f for f in src_files if os.path.exists(f)]
        if existing_srcs:
            os.system(f"hadd -f {target} {' '.join(existing_srcs)}")
        else:
            print(f"[WARNING] No source files found for {target}")

check_and_merge(sig_file, sig_proc)
check_and_merge(bkg_file, bkg_proc)

def load_filtered_data(file, proc_filter=None):
    """
    加载数据，并根据 process name 进行过滤（如果是背景）
    但由于你的代码是 hadd 后的 root 文件，无法直接区分 process。
    
    【重要】：你需要重新生成一个只包含 DY (和 EWK) 的 bkg_for_opt.root，
    或者在读取时如果有 process_id 分支进行筛选。
    
    假设你已经有了只包含主要背景的文件，或者在这里简单处理加载全量（如果无法区分）。
    为了演示，这里假设 bkg_file 已经是剔除了 Top/Diboson 的文件。
    """
    with uproot.open(file) as f:
        arrays = f[tree_name].arrays(["bdt_score", "eventWeight", "diMufsr_rc_mass"], library="np")
    return arrays["bdt_score"], arrays["eventWeight"], arrays["diMufsr_rc_mass"]

def get_region_masks(masses):
    mask_sr = (masses >= sr_min) & (masses <= sr_max)
    mask_sb = ((masses >= sb_low_min) & (masses <= sb_low_max)) | \
              ((masses >= sb_high_min) & (masses <= sb_high_max))
    return mask_sr, mask_sb

def nll_term(n_obs, n_exp):
    # NLL = n_exp - n_obs * log(n_exp)
    # 对于 Asimov， n_obs = n_exp，所以项变成 n_exp - n_exp * log(n_exp)
    n_exp = np.maximum(n_exp, 1e-12)
    return n_exp - n_obs * np.log(n_exp)

def fit_asimov_analytic(B_sb, B_sr, S_sr):
    """
    对 Asimov 数据集进行拟合。
    Asimov 假设: 
      Observed Data (SB) = Expected Background (SB) = B_sb
      Observed Data (SR) = Expected Background (SR) + Signal (SR) * mu_true (通常设为1或0)
    
    文本中 "fitted data is Asimov... from pre-fit prediction"，
    通常意味着我们用 S+B 的假设生成 Asimov Data，然后去 Fit 它。
    或者用 B-only 生成 Asimov Data。
    
    计算 Significance 通常是：
    生成 Asimov (mu=1, S+B) -> Fit (mu=1) 得到 L1 -> Fit (mu=0) 得到 L0 -> Z = sqrt(-2ln(L0/L1))
    
    在 Asimov 数据集下（mu=1 generated），Fit mu=1 的结果必然是 theta=1, mu=1, NLL_min。
    我们只需要计算 mu=0 时的 NLL 差值。
    """
    
    # --- 1. Construct Asimov Data (Under Hypothesis mu=1) ---
    # 这完全基于 MC 预测，不引入 Data 的统计涨落
    asimov_D_sb = B_sb          # 期望 Sideband
    asimov_D_sr = B_sr + S_sr   # 期望 SR (S+B)
    
    # --- 2. Fit Hypothesis mu = 1 (S+B) ---
    # 对于 Asimov 数据，MLE 必然在输入值处取到 (theta=1)
    # nll_1 = nll_term(asimov_D_sb, B_sb) + nll_term(asimov_D_sr, B_sr + S_sr)
    # 实际上由于完美匹配，这一项是定值，但为了计算 delta NLL，我们保留公式
    theta_1 = 1.0
    exp_sb_1 = theta_1 * B_sb
    exp_sr_1 = 1.0 * S_sr + theta_1 * B_sr
    nll_1 = nll_term(asimov_D_sb, exp_sb_1) + nll_term(asimov_D_sr, exp_sr_1)

    # --- 3. Fit Hypothesis mu = 0 (B-only) ---
    # 我们需要找到最佳的 theta 来让 (theta*B) 去拟合 (S+B) 的 Asimov 数据
    # 似然函数 L = Pois(B_sb | theta*B_sb) * Pois(B_sr+S_sr | theta*B_sr)
    # 解析解: theta_0 = (Obs_SB + Obs_SR) / (Exp_B_SB + Exp_B_SR)
    #               = (B_sb + B_sr + S_sr) / (B_sb + B_sr)
    #               = 1 + S_sr / (B_sb + B_sr)
    
    K = B_sb + B_sr
    if K <= 0: return 0.0 # 保护
    
    theta_0 = (asimov_D_sb + asimov_D_sr) / K
    
    exp_sb_0 = theta_0 * B_sb
    exp_sr_0 = theta_0 * B_sr # mu=0
    
    nll_0 = nll_term(asimov_D_sb, exp_sb_0) + nll_term(asimov_D_sr, exp_sr_0)
    
    # --- 4. Significance ---
    delta_nll = nll_0 - nll_1
    # 数值保护，理论上 delta_nll >= 0
    if delta_nll < 0: delta_nll = 0
    return np.sqrt(2 * delta_nll)

def compute_significance_asimov(sig_s, sig_w, sig_sr_mask, 
                              bkg_s, bkg_w, bkg_sr_mask, bkg_sb_mask, 
                              boundaries):
    total_sig_sq = 0
    
    lower = -np.inf
    for boundary in list(boundaries) + [np.inf]:
        # Cuts
        s_cut = (sig_s > lower) & (sig_s <= boundary)
        b_cut = (bkg_s > lower) & (bkg_s <= boundary)
        
        # MC Yields (Prediction)
        S_sr = np.sum(sig_w[s_cut & sig_sr_mask])
        B_sr = np.sum(bkg_w[b_cut & bkg_sr_mask])
        B_sb = np.sum(bkg_w[b_cut & bkg_sb_mask])
        
        # 计算该 Bin 的 Asimov Significance
        # 注意：Sig Z ~ sqrt(sum(Zi^2)) 近似成立，
        # 但严谨的做法是把所有 Bin 的 NLL 加起来再算 Z。
        # 你的之前代码是 sum(Z^2)，这里为了保持 Profile Likelihood 的严谨性，
        # 我们应该累加 NLL。
        
        # 但为了适配你的迭代逻辑（需要单Bin数值），
        # 只要Bins之间是独立的，总 LogLikelihood = Sum(LogLikelihood_i)。
        # 所以 delta_NLL_total = Sum(delta_NLL_i)。
        # Z_total = sqrt(2 * Sum(delta_NLL_i)) = sqrt(Sum(2*delta_NLL_i)) = sqrt(Sum(Z_i^2))
        # 结论：计算单 Bin Z，然后平方和开根号是数学等价的。
        
        Z_bin = fit_asimov_analytic(B_sb, B_sr, S_sr)
        total_sig_sq += Z_bin**2
        
        lower = boundary
        
    return np.sqrt(total_sig_sq)

# ------------------------
# ----- OPTIMIZATION -----
# ------------------------
# 这里的主循环逻辑和之前一样，只是调用 compute_significance_asimov
# 且不需要 Data 输入，不需要 np.random

def iterative_binning_asimov(sig_s, sig_w, sig_m, bkg_s, bkg_w, bkg_m, 
                             num_thresholds=100, max_categories=10):
    
    print("[INFO] Starting optimization using Exact Asimov (No Toys)...")
    
    mask_sig_sr, _ = get_region_masks(sig_m)
    mask_bkg_sr, mask_bkg_sb = get_region_masks(bkg_m)
    
    # ... (省略：生成 thresholds 的代码，与之前相同) ...
    # 简写生成 thresholds
    sig_scores_sr = sig_s[mask_sig_sr]
    sig_weights_sr = sig_w[mask_sig_sr]
    total_sig_sr = np.sum(sig_weights_sr)
    idx_sorted = np.argsort(sig_scores_sr)[::-1]
    cum_w = np.cumsum(sig_weights_sr[idx_sorted])
    scores_sorted = sig_scores_sr[idx_sorted]
    thresholds = []
    for val in np.linspace(0, 1, num_thresholds+2)[1:-1]:
        idx = np.searchsorted(cum_w, val*total_sig_sr)
        if idx < len(scores_sorted): thresholds.append(scores_sorted[idx])
    thresholds = sorted(list(set(thresholds)))

    category_bounds = []
    history = []
    
    # Init
    Z0 = compute_significance_asimov(sig_s, sig_w, mask_sig_sr,
                                     bkg_s, bkg_w, mask_bkg_sr, mask_bkg_sb,
                                     category_bounds)
    print(f"Init Z: {Z0:.4f}")
    history.append({"bestZ": Z0, "eff_opt": None, "bdt_opt": None, "x_eff":[], "Z_values":[]})

    for n_cat in range(2, max_categories+1):
        print(f"Scanning {n_cat} categories...")
        best_Z = -1
        best_b = None
        
        x_eff = []
        Z_vals = []
        
        current_bins = [-np.inf] + category_bounds + [np.inf]
        
        for i in range(len(current_bins)-1):
            low, high = current_bins[i], current_bins[i+1]
            cands = [t for t in thresholds if t > low and t < high]
            
            for thr in cands:
                test_bounds = sorted(category_bounds + [thr])
                Z = compute_significance_asimov(sig_s, sig_w, mask_sig_sr,
                                                bkg_s, bkg_w, mask_bkg_sr, mask_bkg_sb,
                                                test_bounds)
                
                # Eff
                eff = np.sum(sig_weights_sr[sig_scores_sr > thr]) / total_sig_sr
                x_eff.append(1-eff)
                Z_vals.append(Z)
                
                if Z > best_Z:
                    best_Z = Z
                    best_b = thr
        
        if best_b:
            category_bounds.append(best_b)
            category_bounds.sort()
            opt_eff = np.sum(sig_weights_sr[sig_scores_sr > best_b]) / total_sig_sr
            history.append({
                "bestZ": best_Z, 
                "bdt_opt": best_b, 
                "eff_opt": 1-opt_eff,
                "x_eff": x_eff, 
                "Z_values": Z_vals
            })
            print(f"  Best Z: {best_Z:.4f} at BDT {best_b:.4f}")
            
    return history

# ------------------------
# ----- PLOTTING ---------
# ------------------------
def plot_results(scan_history):
    out_dir = "check_significance_toys"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    for i, entry in enumerate(scan_history):
        if i == 0: continue
        n_cat = i + 1
        
        x = np.array(entry["x_eff"])
        y_mean = np.array(entry["y_mean_Z"])
        y_std  = np.array(entry["y_std_Z"])
        y_first = np.array(entry["y_first_Z"])
        
        # Sorting
        idx = np.argsort(x)
        x = x[idx]
        y_mean = y_mean[idx]
        y_std = y_std[idx]
        y_first = y_first[idx]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 1. 绘制平均显著度曲线
        ax.plot(x, y_mean, 'b-', lw=2, label=f'Mean Z ({N_TOYS} Toys)')
        
        # 2. 绘制标准差阴影
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, color='blue', alpha=0.2, label=r'Mean $\pm$ 1 Std Dev')
        
        # 3. [REQUIREMENT] 绘制第一个 Asimov Toy 的结果
        ax.plot(x, y_first, 'r--', lw=1, alpha=0.8, label='First Toy Result')
        
        # 4. 绘制之前的边界线
        for pe in entry["prev_eff"]:
            ax.axvline(pe, color='gray', ls='-', alpha=0.5)
            
        # 5. 绘制当前最优选择
        ax.axvline(entry["eff_opt"], color='green', ls='--', lw=2, label='Optimal Cut')
        
        ax.set_xlabel(r'$1 - \epsilon_{sig}$ (SR)', fontsize=14)
        ax.set_ylabel(r'Significance $Z = \sqrt{-2\Delta NLL}$', fontsize=14)
        ax.set_title(f'Optimization: {n_cat} Categories (Vectorized Toys)', fontsize=16)
        
        hep.cms.label("Preliminary", data=True, loc=0, ax=ax)
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{out_dir}/scan_cat_{n_cat}_toys.png", dpi=300)
        plt.close()

# ------------------------
# ----- MAIN -------------
# ------------------------
if __name__ == "__main__":
    start_time = time.time()
    
    # Load
    sig_s, sig_w, sig_m = load_filtered_data(sig_file)
    bkg_s, bkg_w, bkg_m = load_filtered_data(bkg_file)
        
    # Run
    history = iterative_binning_asimov(
        sig_s, sig_w, sig_m,
        bkg_s, bkg_w, bkg_m,
        dat_s, dat_w, dat_m,
        num_thresholds=num_thresholds,
        max_categories=max_categories
    )
    
    # Plot
    plot_results(history)
    
    # Save Text Results
    with open("check_significance_toys/results_toys.txt", "w") as f:
        f.write(f"Optimization Results ({N_TOYS} Toys)\n")
        f.write("="*60 + "\n\n")
        
        for i, h in enumerate(history):
            if i == 0: continue
            f.write(f"--- {i+1} Categories ---\n")
            f.write(f"Best Boundary:  {h['bdt_opt']:.5f}\n")
            f.write(f"Mean Z:         {h['best_mean_Z']:.4f} +/- {h['best_std_Z']:.4f}\n")
            f.write(f"First Toy Z:    {h['best_first_Z']:.4f}\n\n")

    print(f"\n[INFO] Total time: {time.time() - start_time:.2f} s")
    print("[INFO] Done.")