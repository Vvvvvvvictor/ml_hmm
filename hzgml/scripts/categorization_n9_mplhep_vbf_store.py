# -*- coding: utf-8 -*-
# Optimization binning with per-category HWHM (for optimization) and
# a smooth "fixed-window" significance curve for plotting.

import math
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os
import subprocess
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

# 设置CMS风格
plt.style.use(hep.style.CMS)

# ------------------------
# ----- CONFIGURATION ----
# ------------------------
# trained with sig=ggH+VBF
#sig_file   = "/eos/user/q/qguo/vbfhmm/ml/RunIII/skimmed_ntuples_ggH_v1/bdt_0717_bdt_0716_2223/ggH/sig.root"
#bkg_file   = "/eos/user/q/qguo/vbfhmm/ml/RunIII/skimmed_ntuples_ggH_v1/bdt_0717_bdt_0716_2223/ggH/bkg.root"

# base_dir = "/eos/home-j/jiehan/root_mumu/outputs_ggH_reweight_24_bdt_1118/ggH"
base_dir = "/eos/user/j/jiehan/root_mumu/outputs_VBF_2223_bdt_1202/VBF"

sig_file   = f"{base_dir}/sig.root"
bkg_file   = f"{base_dir}/bkg.root"

# sig_proc = ["GluGluHToMuMu_M125","VBFHToMuMu_M125"]
# bkg_proc = ["EWK_LLJJ_M50", "ST_tW_antitop", "TTTo2L2Nu", "WWTo2L2Nu", "WWZ", "WZTo3LNu", "ZZTo2L2Nu", "ZZTo4L", "DY_105To160", "ST_tW_top", "WWW", "WZTo2L2Q", "WZZ", "ZZTo2L2Q", "ZZZ"]

# print(f"[INFO] Merging signal files: hadd -f {sig_file} {' '.join([f'{base_dir}/{proc}.root' for proc in sig_proc])}")
# os.system(f"hadd -f {sig_file} {' '.join([f'{base_dir}/{proc}.root' for proc in sig_proc])}")
# print(f"[INFO] Merging background files: hadd -f {bkg_file} {' '.join([f'{base_dir}/{proc}.root' for proc in bkg_proc])}")
# os.system(f"hadd -f {bkg_file} {' '.join([f'{base_dir}/{proc}.root' for proc in bkg_proc])}")

tree_name  = "test"
score_var  = "bdt_score_t"
weight_var = "eventWeight"
mass_var   = "diMufsr_rc_mass"

num_thresholds = 200    # denser -> smoother曲线
max_categories = 16

# Fixed mass window for signal/background counting
mass_window_min = 115.0
mass_window_max = 135.0

random_seed = 12345
np.random.seed(random_seed)

# ------------------------
# ----- UTILITIES --------
# ------------------------
# No utility functions needed for simple fixed window approach

# ------------------------
# ----- I/O --------------
# ------------------------
def load_data(file, tree=tree_name, score_var=score_var, weight_var=weight_var, mass_var=mass_var):
    """使用uproot分块加载数据"""
    print(f"[INFO] Loading data from: {file}")
    
    try:
        import uproot
    except ImportError:
        raise ImportError("Please install uproot: pip install uproot awkward")
    
    try:
        with uproot.open(file) as f:
            tree_obj = f[tree]
            n_entries = tree_obj.num_entries
            print(f"[INFO] Found {n_entries} entries in tree '{tree}'")
            
            # 使用分块读取，每次读取500000条记录
            chunksize = 500000
            scores_list = []
            weights_list = []
            masses_list = []
            
            n_chunks = (n_entries + chunksize - 1) // chunksize
            print(f"[INFO] Reading data in {n_chunks} chunks (chunksize={chunksize})...")
            
            for i, batch in enumerate(tree_obj.iterate([score_var, weight_var, mass_var], 
                                                       library="np", 
                                                       step_size=chunksize)):
                if (i + 1) % 5 == 0 or (i + 1) == n_chunks:
                    print(f"[INFO] Processing chunk {i+1}/{n_chunks} ({100*(i+1)/n_chunks:.1f}%)")
                
                scores_list.append(batch[score_var].astype(np.float64))
                weights_list.append(batch[weight_var].astype(np.float64))
                masses_list.append(batch[mass_var].astype(np.float64))
            
            # 合并所有块
            scores = np.concatenate(scores_list)
            weights = np.concatenate(weights_list)
            masses = np.concatenate(masses_list)
            
            print(f"[INFO] Loaded {len(scores)} entries successfully")
            return scores, weights, masses
    
    except Exception as e:
        print(f"[ERROR] Failed to load with uproot: {e}")
        raise RuntimeError(f"Failed to load data from {file}: {e}")

# ------------------------
# ----- METRICS ----------
# ------------------------
def asimov_sig_vectorized(S, B):
    """矢量化的Asimov显著度计算"""
    S = np.asarray(S, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    
    # 避免除零和log(0)
    B = np.maximum(B, 1e-300)
    
    # Asimov公式
    result = np.zeros_like(S)
    valid = B > 0
    result[valid] = np.sqrt(np.maximum(0.0, 2.0 * ((S[valid] + B[valid]) * np.log(1.0 + S[valid] / B[valid]) - S[valid])))
    
    return result

def compute_significance(sig_scores, sig_weights, sig_masses,
                         bkg_scores, bkg_weights, bkg_masses,
                         boundaries,
                         return_per_bin=False):
    """
    使用固定质量窗口[115, 135] GeV计算significance
    return_per_bin: 如果为True，返回(总significance, per_bin_info列表)
                    per_bin_info包含每个bin的S, B, Z信息
    """
    S_list = []
    B_list = []
    lower = -np.inf
    
    lo, hi = mass_window_min, mass_window_max
    
    for boundary in list(boundaries) + [np.inf]:
        # 选择当前BDT score bin的事件
        mask_sig_cat = (sig_scores > lower) & (sig_scores <= boundary)
        mask_bkg_cat = (bkg_scores > lower) & (bkg_scores <= boundary)

        # 在固定质量窗口内统计S和B
        mask_sig = mask_sig_cat & (sig_masses >= lo) & (sig_masses <= hi)
        mask_bkg = mask_bkg_cat & (bkg_masses >= lo) & (bkg_masses <= hi)

        S = float(sig_weights[mask_sig].sum())
        B = float(bkg_weights[mask_bkg].sum())
        if B < 4:
            B = 10000000
        S_list.append(S)
        B_list.append(B)
        lower = boundary

    # 矢量化计算所有类别的significance，然后平方求和
    sig_per_cat = asimov_sig_vectorized(S_list, B_list)
    total_sig = float(np.sqrt(np.sum(sig_per_cat ** 2)))
    
    if return_per_bin:
        per_bin_info = []
        for i, (S, B, Z) in enumerate(zip(S_list, B_list, sig_per_cat)):
            per_bin_info.append({
                'bin_idx': i,
                'S': S,
                'B': B,
                'Z': float(Z)
            })
        return total_sig, per_bin_info
    else:
        return total_sig

# ------------------------
# ----- SCAN -------------
# ------------------------
def iterative_binning(sig_scores, sig_weights, sig_masses,
                      bkg_scores, bkg_weights, bkg_masses,
                      num_thresholds=100, max_categories=5):
    """
    迭代建类，使用固定质量窗口[115, 135] GeV计算S和B
    按 1-eff(sig) 均匀取点进行扫描
    """
    print(f"\n[INFO] Starting iterative binning with {num_thresholds} thresholds, max {max_categories} categories")
    print(f"[INFO] Using fixed mass window: [{mass_window_min:.1f}, {mass_window_max:.1f}] GeV")
    
    category_bounds = []
    
    # 计算总的信号权重，用于效率计算
    total_sig_weight = sig_weights.sum()
    
    # 按 1-eff(sig) 均匀取点（从0到1）
    eff_rejection_points = np.linspace(0, 1, num_thresholds+2)[1:-1]  # 1-eff(sig)
    
    # 对信号按BDT score排序，用于快速查找对应的BDT阈值
    sorted_indices = np.argsort(sig_scores)[::-1]  # 降序排列
    sorted_sig_scores = sig_scores[sorted_indices]
    sorted_sig_weights = sig_weights[sorted_indices]
    cumsum_sig_weights = np.cumsum(sorted_sig_weights)
    
    # 对于每个 1-eff 点，找到对应的 BDT threshold
    thresholds = []
    for rej in eff_rejection_points:
        target_weight = rej * total_sig_weight
        # 找到累积权重刚好超过target的位置
        idx = np.searchsorted(cumsum_sig_weights, target_weight)
        if idx < len(sorted_sig_scores):
            thresholds.append(sorted_sig_scores[idx])
        elif len(sorted_sig_scores) > 0:
            thresholds.append(sorted_sig_scores[-1] - 0.01)  # 边界情况
    
    thresholds = np.array(thresholds)
    print(f"[INFO] Efficiency rejection range: [0, 1]")
    print(f"[INFO] Corresponding BDT range: [{thresholds.min():.4f}, {thresholds.max():.4f}]")
    
    scan_history = []

    # 起始：1个类别
    print("[INFO] Computing initial significance (1 category)...")
    Z0 = compute_significance(sig_scores, sig_weights, sig_masses,
                              bkg_scores, bkg_weights, bkg_masses,
                              category_bounds)
    print(f"[INFO] Initial significance: Z={Z0:.3f}")
    scan_history.append({
        "x_eff":[], "x_bdt":[],
        "Z_values":[],
        "prev_eff":[], "prev_bdt":[],
        "bestZ":Z0, "eff_opt":None, "bdt_opt":None
    })

    for n_cat in range(2, max_categories+1):
        print(f"\n[INFO] === Optimizing for {n_cat} categories ===")
        best_Z = -np.inf
        best_new_boundary = None
        best_split_idx = None

        x_eff = []; x_bdt = []
        Z_values = []
        split_idx = []

        all_bins = [-np.inf] + category_bounds + [np.inf]
        print(f"[INFO] Current boundaries: {category_bounds}")
        print(f"[INFO] Scanning {len(all_bins)-1} bins...")
        
        total_thresholds = 0
        for i_bin in range(len(all_bins)-1):
            low, up = all_bins[i_bin], all_bins[i_bin+1]
            t_in = thresholds[(thresholds > low) & (thresholds < up)]
            total_thresholds += len(t_in)
        print(f"[INFO] Total thresholds to scan: {total_thresholds}")
        
        scanned = 0
        for i_bin in range(len(all_bins)-1):
            low, up = all_bins[i_bin], all_bins[i_bin+1]
            t_in = thresholds[(thresholds > low) & (thresholds < up)]
            for j, thr in enumerate(t_in):
                scanned += 1
                if scanned % 20 == 0:
                    print(f"[INFO] Scanned {scanned}/{total_thresholds} thresholds ({100*scanned/total_thresholds:.1f}%)")

                new_bounds = sorted(category_bounds + [float(thr)])

                # 计算significance
                Z = compute_significance(sig_scores, sig_weights, sig_masses,
                                        bkg_scores, bkg_weights, bkg_masses,
                                        new_bounds)

                # 矢量化效率计算
                eff = sig_weights[sig_scores > thr].sum() / max(total_sig_weight, 1e-300)
                x_eff.append(1.0 - eff)
                x_bdt.append(float(thr))
                Z_values.append(float(Z))
                split_idx.append(i_bin)

                if Z > best_Z:
                    best_Z = Z
                    best_new_boundary = float(thr)
                    best_split_idx = i_bin

        eff_opt = 1.0 - sig_weights[sig_scores > best_new_boundary].sum() / max(total_sig_weight, 1e-300)
        category_bounds.insert(best_split_idx, best_new_boundary)
        category_bounds = sorted(set(category_bounds))

        print(f"[INFO] Best boundary for {n_cat} categories: {best_new_boundary:.6f}")
        print(f"[INFO] Best Z: {best_Z:.3f}, eff_opt: {eff_opt:.6f}")

        # 计算当前配置的per-bin significance
        _, per_bin_info = compute_significance(
            sig_scores, sig_weights, sig_masses,
            bkg_scores, bkg_weights, bkg_masses,
            category_bounds,
            return_per_bin=True
        )

        prev_eff_bounds = [1.0 - sig_weights[sig_scores > b].sum() / max(total_sig_weight, 1e-300)
                           for b in category_bounds if b != best_new_boundary]
        prev_bdt_bounds = [b for b in category_bounds if b != best_new_boundary]

        scan_history.append({
            "x_eff":x_eff, "x_bdt":x_bdt,
            "Z_values":Z_values,
            "prev_eff":prev_eff_bounds, "prev_bdt":prev_bdt_bounds,
            "bestZ":best_Z, "eff_opt":eff_opt, "bdt_opt":best_new_boundary,
            "per_bin_info":per_bin_info  # 存储per-bin信息
        })

    return scan_history

# ------------------------
# ----- PLOTS ------------
# ------------------------
def plot_iterative_scan(scan_history, max_categories=5):
    print("\n[INFO] Generating plots...")
    for n_cat in range(2, max_categories+1):
        print(f"[INFO] Plotting scan for {n_cat} categories...")
        entry = scan_history[n_cat-1]
        x_eff = np.array(entry["x_eff"])
        Z_values = np.array(entry["Z_values"])
        prev_eff = entry["prev_eff"]
        eff_opt  = entry["eff_opt"]

        # 排序后绘制（确保横轴单调 -> 连续曲线）
        idx = np.argsort(x_eff)
        xe = x_eff[idx]
        Zv = Z_values[idx]

        # 绘制significance扫描曲线
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(xe, Zv, 'k.', markersize=3, label=f'Scan (mass window [{mass_window_min:.0f}, {mass_window_max:.0f}] GeV)')
        
        # 绘制之前的边界
        for b in prev_eff:
            ax.axvline(b, color='gray', linestyle='-', linewidth=1, alpha=0.6)
        
        # 绘制最优边界
        if eff_opt is not None:
            ax.axvline(eff_opt, color='red', linestyle='--', linewidth=2, label='Optimal boundary')
        
        ax.set_xlabel(r'$1 - \epsilon_{\mathrm{sig}}$', fontsize=14)
        ax.set_ylabel(r'Significance ($\sigma$)', fontsize=14)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
        
        # 添加CMS标签
        hep.cms.label("Preliminary", data=False, lumi=None, year=None, ax=ax, loc=0, fontsize=13)
        ax.text(0.05, 0.88, f'{n_cat-1} → {n_cat} categories', 
                transform=ax.transAxes, fontsize=11, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(f'check_signifcance/check_significance_scan_eff_N_{n_cat}_bdt.png', dpi=300, bbox_inches='tight')
        plt.close()

# ------------------------
# ----- MAIN -------------
# ------------------------
if __name__ == "__main__":
    import time
    start_time = time.time()
    print("="*60)
    print("Starting categorization optimization")
    print("="*60)
    
    # Load full samples (no mass cut)
    sig_scores, sig_weights, sig_masses = load_data(sig_file)
    bkg_scores, bkg_weights, bkg_masses = load_data(bkg_file)
    
    print(f"\n[INFO] Signal: {len(sig_scores)} events, total weight: {sig_weights.sum():.2f}")
    print(f"[INFO] Background: {len(bkg_scores)} events, total weight: {bkg_weights.sum():.2f}")

    # Build categories iteratively using fixed mass window [115, 135] GeV
    scan_history = iterative_binning(
        sig_scores, sig_weights, sig_masses,
        bkg_scores, bkg_weights, bkg_masses,
        num_thresholds=num_thresholds,
        max_categories=max_categories
    )

    # Make diagnostic plots
    plot_iterative_scan(scan_history, max_categories=max_categories)

    elapsed_time = time.time() - start_time
    print(f"\n[INFO] Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    # 保存边界到文件
    output_dir = os.path.dirname(os.path.abspath(__file__))
    boundaries_file = os.path.join(output_dir, 'check_signifcance/optimal_boundaries.txt')
    print(f"\n[INFO] Saving boundaries to: {boundaries_file}")
    
    with open(boundaries_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Optimal Category Boundaries\n")
        f.write("="*60 + "\n\n")
        f.write(f"Fixed mass window: [{mass_window_min:.1f}, {mass_window_max:.1f}] GeV\n\n")
        
        # 收集所有最佳边界
        all_boundaries = []
        for i, entry in enumerate(scan_history):
            n_categories = i + 1
            f.write("-"*60 + "\n")
            if i < 1:
                f.write(f"After {n_categories} category:\n")
            else:
                f.write(f"After {n_categories} categories:\n")
            
            bdt_opt = entry["bdt_opt"]
            eff_opt = entry["eff_opt"]
            bestZ = entry["bestZ"]
            
            f.write(f"  BDT optimal cut     = {None if bdt_opt is None else round(bdt_opt, 6)}\n")
            f.write(f"  1 - epsilon_sig     = {None if eff_opt is None else round(eff_opt, 6)}\n")
            f.write(f"  Best Z              = {bestZ:.3f}\n")
            
            # 收集当前配置的所有边界
            if i > 0:
                current_bounds = entry["prev_bdt"] + [bdt_opt] if bdt_opt is not None else entry["prev_bdt"]
                current_bounds = sorted(current_bounds)
                all_boundaries.append((n_categories, current_bounds, bestZ))
                f.write(f"  All boundaries      = {[round(b, 6) for b in current_bounds]}\n")
                
                # 输出每个bin的significance（使用已存储的信息）
                f.write(f"\n  Per-bin significance (using fixed window [{mass_window_min:.0f}, {mass_window_max:.0f}] GeV):\n")
                per_bin_info = entry.get("per_bin_info", [])
                for bin_info in per_bin_info:
                    f.write(f"    Bin {bin_info['bin_idx']}: S={bin_info['S']:.3f}, B={bin_info['B']:.3f}, "
                           f"Z={bin_info['Z']:.3f}\n")
                f.write(f"    Total Z (quadrature): {bestZ:.3f}\n")
        
        # 写入最终推荐的边界配置
        f.write("\n" + "="*60 + "\n")
        f.write("Summary of All Configurations\n")
        f.write("="*60 + "\n\n")
        
        for n_cat, bounds, z_val in all_boundaries:
            f.write(f"{n_cat} categories (Z={z_val:.3f}): {[round(b, 6) for b in bounds]}\n")
        
        # 找到最佳配置
        if all_boundaries:
            best_config = max(all_boundaries, key=lambda x: x[2])
            f.write(f"\nBest configuration: {best_config[0]} categories with Z={best_config[2]:.3f}\n")
            f.write(f"Best boundaries: {[round(b, 6) for b in best_config[1]]}\n")
    
    # Report results to console
    print('\n' + "="*60)
    print('Final category boundaries (iterative history):')
    for i, entry in enumerate(scan_history):
        if i < 1:
            print(f"After {i+1} category:")
        else:
            print(f"After {i+1} categories:")
        bdt_opt = entry["bdt_opt"]
        eff_opt = entry["eff_opt"]
        print(f"  BDT optimal cut     = {None if bdt_opt is None else round(bdt_opt, 6)}")
        print(f"  1 - epsilon_sig     = {None if eff_opt is None else round(eff_opt, 6)}")
        print(f"  Best Z              = {entry['bestZ']:.3f}")
    print(f"\nFixed mass window: [{mass_window_min:.1f}, {mass_window_max:.1f}] GeV")
    print(f"\n[INFO] Boundaries saved to: {boundaries_file}")
