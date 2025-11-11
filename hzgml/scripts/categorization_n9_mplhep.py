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
sig_file   = "/eos/user/j/jiehan/root_mumu/outputs_ggH_reweight_bdt_1111/ggH/sig.root"
bkg_file   = "/eos/user/j/jiehan/root_mumu/outputs_ggH_reweight_bdt_1111/ggH/bkg.root"

tree_name  = "test"
score_var  = "bdt_score"
weight_var = "eventWeight"
mass_var   = "diMufsr_rc_mass"

num_thresholds = 200    # denser -> smoother曲线
max_categories = 6

# mass histogram settings used to find FWHM/HWHM
mass_nbins = 160
mass_min   = 110.0
mass_max   = 150.0

# FWHM护栏，避免离谱宽度（可按你的模型细化）
MIN_FWHM = 1.5  # GeV
MAX_FWHM = 8.0  # GeV

random_seed = 12345
np.random.seed(random_seed)

# ------------------------
# ----- UTILITIES --------
# ------------------------
def build_mass_hist_numpy(masses, weights, nbins=mass_nbins, xmin=mass_min, xmax=mass_max):
    """使用numpy快速构建质量直方图"""
    mask = np.isfinite(masses) & np.isfinite(weights)
    masses = masses[mask]
    weights = weights[mask]
    
    hist, edges = np.histogram(masses, bins=nbins, range=(xmin, xmax), weights=weights)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    
    return hist, bin_centers, edges

def _gauss_fit_mu_fwhm(hist, bin_centers, m0=None, halfwin=5.0):
    """使用scipy进行高斯拟合"""
    if hist.sum() <= 0 or len(hist) < 10:
        return None, None
    
    if m0 is None:
        m0 = bin_centers[np.argmax(hist)]
    
    lo = max(mass_min, m0 - halfwin)
    hi = min(mass_max, m0 + halfwin)
    if hi <= lo:
        return None, None
    
    # 选择拟合区域
    mask = (bin_centers >= lo) & (bin_centers <= hi)
    if mask.sum() < 5:
        return None, None
    
    x_fit = bin_centers[mask]
    y_fit = hist[mask]
    
    # 高斯函数
    def gaussian(x, amp, mu, sigma):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    try:
        # 初始参数估计
        p0 = [hist.max(), m0, 1.5]
        popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=p0, maxfev=1000)
        
        mu = popt[1]
        sigma = abs(popt[2])
        
        if not (0.2 <= sigma <= 6.0):
            return None, None
        
        fwhm = 2.354820045 * sigma
        return mu, fwhm
    except:
        return None, None

def _halfmax_mu_fwhm(hist, bin_centers, smooth_times=3):
    """使用numpy和scipy进行半高宽计算"""
    if hist.sum() <= 0 or len(hist) < 10:
        return None, None
    
    # 使用高斯滤波平滑
    hs = hist.copy()
    if smooth_times > 0:
        hs = gaussian_filter1d(hs, sigma=smooth_times * 0.5)
    
    ib = np.argmax(hs)
    ymax = hs[ib]
    if ymax <= 0:
        return None, None
    
    target = 0.5 * ymax
    mu = bin_centers[ib]
    
    # 寻找左侧交叉点
    i = ib
    while i > 0 and hs[i] >= target:
        i -= 1
    
    if i == 0 and hs[i] >= target:
        m_left = mass_min
    else:
        if i + 1 < len(hs):
            x1, y1 = bin_centers[i], hs[i]
            x2, y2 = bin_centers[i + 1], hs[i + 1]
            m_left = x1 if y2 == y1 else x1 + (target - y1) * (x2 - x1) / (y2 - y1)
        else:
            m_left = bin_centers[i]
    
    # 寻找右侧交叉点
    i = ib
    nb = len(hs)
    while i < nb - 1 and hs[i] >= target:
        i += 1
    
    if i == nb - 1 and hs[i] >= target:
        m_right = mass_max
    else:
        if i > 0:
            x1, y1 = bin_centers[i - 1], hs[i - 1]
            x2, y2 = bin_centers[i], hs[i]
            m_right = x2 if y2 == y1 else x1 + (target - y1) * (x2 - x1) / (y2 - y1)
        else:
            m_right = bin_centers[i]
    
    fwhm = max(0.0, m_right - m_left)
    if fwhm <= 0:
        return None, None
    return mu, fwhm

def _clamp_fwhm(fwhm):
    return max(MIN_FWHM, min(MAX_FWHM, fwhm))

def robust_mu_hwhm_from_hist(hist, bin_centers, mu_fallback=None, fwhm_fallback=None):
    """从numpy直方图获取mu和HWHM"""
    mu, fwhm = _gauss_fit_mu_fwhm(hist, bin_centers)
    if mu is None or fwhm is None:
        mu2, fwhm2 = _halfmax_mu_fwhm(hist, bin_centers, smooth_times=3)
        mu = mu if mu is not None else mu2
        fwhm = fwhm if fwhm is not None else fwhm2
    if (mu is None or fwhm is None) and (mu_fallback is not None and fwhm_fallback is not None):
        mu, fwhm = mu_fallback, fwhm_fallback
    if mu is None or fwhm is None:
        mu, fwhm = 125.0, 4.0
    fwhm = _clamp_fwhm(fwhm)
    hwhm = 0.5 * fwhm
    lo = max(mass_min, mu - hwhm)
    hi = min(mass_max, mu + hwhm)
    if hi <= lo:  # 极端回退
        lo, hi = max(mass_min, mu - 2.0), min(mass_max, mu + 2.0)
    return mu, hwhm, lo, hi, fwhm

def compute_global_window(sig_masses, sig_weights):
    print("[INFO] Computing global window for fixed-width significance...")
    hist, bin_centers, edges = build_mass_hist_numpy(sig_masses, sig_weights)
    mu, hwhm, lo, hi, fwhm = robust_mu_hwhm_from_hist(hist, bin_centers)
    print(f"[INFO] Global window: [{lo:.3f}, {hi:.3f}] GeV (mu={mu:.3f}, FWHM={fwhm:.3f})")
    return (lo, hi), mu, fwhm

# ------------------------
# ----- I/O --------------
# ------------------------
def load_data(file, tree=tree_name, score_var=score_var, weight_var=weight_var, mass_var=mass_var):
    """使用uproot快速加载数据"""
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
            
            # 一次性读取所有需要的分支
            arrays = tree_obj.arrays([score_var, weight_var, mass_var], library="np")
            
            scores = arrays[score_var].astype(np.float64)
            weights = arrays[weight_var].astype(np.float64)
            masses = arrays[mass_var].astype(np.float64)
            
            print(f"[INFO] Loaded {len(scores)} entries successfully")
            return scores, weights, masses
    
    except Exception as e:
        print(f"[ERROR] Failed to load with uproot: {e}")
        print("[INFO] Falling back to ROOT (slower)...")
        
        # 回退到ROOT
        import ROOT
        f = ROOT.TFile.Open(file)
        if not f or f.IsZombie():
            raise RuntimeError(f"Cannot open ROOT file: {file}")
        t = f.Get(tree)
        if not t:
            raise RuntimeError(f"Cannot find tree '{tree}' in {file}")
        
        n_entries = t.GetEntries()
        print(f"[INFO] Found {n_entries} entries in tree '{tree}'")
        
        scores = np.empty(n_entries, dtype=np.float64)
        weights = np.empty(n_entries, dtype=np.float64)
        masses = np.empty(n_entries, dtype=np.float64)
        
        for i, ev in enumerate(t):
            if i % 100000 == 0 and i > 0:
                print(f"[INFO] Processing entry {i}/{n_entries} ({100*i/n_entries:.1f}%)")
            scores[i] = getattr(ev, score_var)
            weights[i] = getattr(ev, weight_var)
            masses[i] = getattr(ev, mass_var)
        
        print(f"[INFO] Loaded {n_entries} entries successfully")
        return scores, weights, masses

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
                         mode="per_category",
                         global_window=None,
                         global_mu=None, global_fwhm=None):
    """
    矢量化的significance计算
    mode:
      - "per_category": 每个类别用自身信号得到的 HWHM 作为窗口（用于优化）
      - "global": 所有类别用同一个固定窗口 global_window=(lo,hi)（用于平滑画图）
    """
    S_list = []
    B_list = []
    lower = -np.inf
    
    for boundary in list(boundaries) + [np.inf]:
        mask_sig_cat = (sig_scores > lower) & (sig_scores <= boundary)
        mask_bkg_cat = (bkg_scores > lower) & (bkg_scores <= boundary)

        if mode == "global":
            lo, hi = global_window
        else:  # per_category
            hist, bin_centers, edges = build_mass_hist_numpy(
                sig_masses[mask_sig_cat], 
                sig_weights[mask_sig_cat]
            )
            mu, hwhm, lo, hi, _ = robust_mu_hwhm_from_hist(
                hist, bin_centers,
                mu_fallback=global_mu, 
                fwhm_fallback=global_fwhm
            )

        mask_sig = mask_sig_cat & (sig_masses >= lo) & (sig_masses <= hi)
        mask_bkg = mask_bkg_cat & (bkg_masses >= lo) & (bkg_masses <= hi)

        S = float(sig_weights[mask_sig].sum())
        B = float(bkg_weights[mask_bkg].sum())
        S_list.append(S)
        B_list.append(B)
        lower = boundary

    # 矢量化计算所有类别的significance，然后平方求和
    sig_per_cat = asimov_sig_vectorized(S_list, B_list)
    return float(np.sqrt(np.sum(sig_per_cat ** 2)))

# ------------------------
# ----- SCAN -------------
# ------------------------
def iterative_binning(sig_scores, sig_weights, sig_masses,
                      bkg_scores, bkg_weights, bkg_masses,
                      num_thresholds=100, max_categories=5):
    """
    迭代建类（用于决定最佳边界），优化目标用 per_category 窗口的显著度。
    同时，为画图准备：在相同阈值上再计算一份 global 窗口的显著度（平滑）。
    
    优化：按 1-eff(sig) 均匀取点，而不是按 BDT score 取点
    """
    print(f"\n[INFO] Starting iterative binning with {num_thresholds} thresholds, max {max_categories} categories")
    
    # 准备全局窗口（平滑曲线用）
    (g_lo, g_hi), g_mu, g_fwhm = compute_global_window(sig_masses, sig_weights)

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
    Z0_dyn = compute_significance(sig_scores, sig_weights, sig_masses,
                                  bkg_scores, bkg_weights, bkg_masses,
                                  category_bounds, mode="per_category",
                                  global_window=(g_lo, g_hi), global_mu=g_mu, global_fwhm=g_fwhm)
    Z0_fix = compute_significance(sig_scores, sig_weights, sig_masses,
                                  bkg_scores, bkg_weights, bkg_masses,
                                  category_bounds, mode="global",
                                  global_window=(g_lo, g_hi))
    print(f"[INFO] Initial significance: Z_dyn={Z0_dyn:.3f}, Z_fix={Z0_fix:.3f}")
    scan_history.append({
        "x_eff":[], "x_bdt":[],
        "Z_dyn":[], "Z_fix":[],
        "prev_eff":[], "prev_bdt":[],
        "bestZ_dyn":Z0_dyn, "eff_opt":None, "bdt_opt":None
    })

    for n_cat in range(2, max_categories+1):
        print(f"\n[INFO] === Optimizing for {n_cat} categories ===")
        best_Z_dyn = -np.inf
        best_new_boundary = None
        best_split_idx = None

        x_eff = []; x_bdt = []
        Z_dyn = []; Z_fix = []
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

                # per-category 窗口（用于优化）
                Zd = compute_significance(sig_scores, sig_weights, sig_masses,
                                          bkg_scores, bkg_weights, bkg_masses,
                                          new_bounds, mode="per_category",
                                          global_window=(g_lo, g_hi), global_mu=g_mu, global_fwhm=g_fwhm)
                # global 固定窗口（用于平滑画图）
                Zg = compute_significance(sig_scores, sig_weights, sig_masses,
                                          bkg_scores, bkg_weights, bkg_masses,
                                          new_bounds, mode="global",
                                          global_window=(g_lo, g_hi))

                # 矢量化效率计算
                eff = sig_weights[sig_scores > thr].sum() / max(total_sig_weight, 1e-300)
                x_eff.append(1.0 - eff)
                x_bdt.append(float(thr))
                Z_dyn.append(float(Zd))
                Z_fix.append(float(Zg))
                split_idx.append(i_bin)

                if Zd > best_Z_dyn:
                    best_Z_dyn = Zd
                    best_new_boundary = float(thr)
                    best_split_idx = i_bin

        eff_opt = 1.0 - sig_weights[sig_scores > best_new_boundary].sum() / max(total_sig_weight, 1e-300)
        category_bounds.insert(best_split_idx, best_new_boundary)
        category_bounds = sorted(set(category_bounds))

        print(f"[INFO] Best boundary for {n_cat} categories: {best_new_boundary:.6f}")
        print(f"[INFO] Best Z_dyn: {best_Z_dyn:.3f}, eff_opt: {eff_opt:.6f}")

        prev_eff_bounds = [1.0 - sig_weights[sig_scores > b].sum() / max(total_sig_weight, 1e-300)
                           for b in category_bounds if b != best_new_boundary]
        prev_bdt_bounds = [b for b in category_bounds if b != best_new_boundary]

        scan_history.append({
            "x_eff":x_eff, "x_bdt":x_bdt,
            "Z_dyn":Z_dyn, "Z_fix":Z_fix,
            "prev_eff":prev_eff_bounds, "prev_bdt":prev_bdt_bounds,
            "bestZ_dyn":best_Z_dyn, "eff_opt":eff_opt, "bdt_opt":best_new_boundary
        })

    # 把全局窗口也返回，便于打印/记录
    return scan_history, (g_lo, g_hi), g_mu, g_fwhm

# ------------------------
# ----- PLOTS ------------
# ------------------------
def plot_iterative_scan(scan_history, max_categories=5):
    print("\n[INFO] Generating plots...")
    for n_cat in range(2, max_categories+1):
        print(f"[INFO] Plotting scan for {n_cat} categories...")
        entry = scan_history[n_cat-1]
        x_eff = np.array(entry["x_eff"])
        Z_fix = np.array(entry["Z_fix"])
        Z_dyn = np.array(entry["Z_dyn"])
        prev_eff = entry["prev_eff"]
        eff_opt  = entry["eff_opt"]

        # 排序后绘制（确保横轴单调 -> 连续曲线）
        idx = np.argsort(x_eff)
        xe = x_eff[idx]; Zf = Z_fix[idx]; Zd = Z_dyn[idx]

        # 平滑曲线（固定窗口）——主图
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(xe, Zf, 'k.', markersize=3, label='Scan (fixed window)')
        
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
        ax.text(0.05, 0.88, f'{n_cat-1} → {n_cat} categories\n(fixed HWHM)', 
                transform=ax.transAxes, fontsize=11, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(f'check_signifcance/check_significance_scan_eff_N_{n_cat}_fixed_bdt.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 可选：对照图（动态窗口，可能有轻微"抖动/不连续"）
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(xe, Zd, 'k.', markersize=3, label='Scan (per-category HWHM)')
        
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
        ax.text(0.05, 0.88, f'{n_cat-1} → {n_cat} categories\n(dynamic HWHM)', 
                transform=ax.transAxes, fontsize=11, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(f'check_signifcance/check_significance_scan_eff_N_{n_cat}_dynamic_bdt.png', dpi=300, bbox_inches='tight')
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

    # Build categories iteratively (optimize with per-category HWHM),
    # while also preparing a fixed-window curve for plotting.
    scan_history, global_window, g_mu, g_fwhm = iterative_binning(
        sig_scores, sig_weights, sig_masses,
        bkg_scores, bkg_weights, bkg_masses,
        num_thresholds=num_thresholds,
        max_categories=max_categories
    )

    # Make diagnostic plots (smooth fixed-window + dynamic for reference)
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
        f.write(f"Global window: [{global_window[0]:.3f}, {global_window[1]:.3f}] GeV ")
        f.write(f"(mu={g_mu:.3f}, FWHM={g_fwhm:.3f})\n\n")
        
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
            bestZ = entry["bestZ_dyn"]
            
            f.write(f"  BDT optimal cut     = {None if bdt_opt is None else round(bdt_opt, 6)}\n")
            f.write(f"  1 - epsilon_sig     = {None if eff_opt is None else round(eff_opt, 6)}\n")
            f.write(f"  Best Z (dynamic)    = {bestZ:.3f}\n")
            
            # 收集当前配置的所有边界
            if i > 0:
                current_bounds = entry["prev_bdt"] + [bdt_opt] if bdt_opt is not None else entry["prev_bdt"]
                current_bounds = sorted(current_bounds)
                all_boundaries.append((n_categories, current_bounds, bestZ))
                f.write(f"  All boundaries      = {[round(b, 6) for b in current_bounds]}\n")
        
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
        print(f"  Best Z (dynamic)    = {entry['bestZ_dyn']:.3f}")
    print(f"\nGlobal fixed window used for smooth curve: [{global_window[0]:.3f}, {global_window[1]:.3f}] GeV "
          f"(mu≈{g_mu:.3f}, FWHM≈{g_fwhm:.3f})")
    print(f"\n[INFO] Boundaries saved to: {boundaries_file}")
