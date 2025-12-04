# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Higgs Categorization Optimization (Triple Gaussian Signal + Sideband Bkg)
# Features:
# 1. Caching for speedup (Memoization)
# 2. Analytical Error Propagation (Jacobian method)
# 3. Full Scan Plotting capability with Improved Formatting
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os
import time
import uproot
from scipy.optimize import curve_fit, brentq

# Set CMS style
plt.style.use(hep.style.CMS)

# ------------------------
# ----- CONFIGURATION ----
# ------------------------

base_dir = "/eos/user/j/jiehan/HMuMuShare/outputs_ggH_2223_bdt_1201/ggH"
sig_file   = f"{base_dir}/sig.root"
bkg_file   = f"{base_dir}/bkg.root"
data_file  = f"{base_dir}/data.root"

sig_proc = ["GluGluHToMuMu_M125_ggHUnc","VBFHToMuMu_M125_ggHUnc"]
bkg_proc = ["EWK_LLJJ_M50", "ST_tW_antitop", "TTTo2L2Nu", "WWTo2L2Nu", "WWZ", "WZTo3LNu", "ZZTo2L2Nu", "ZZTo4L", "DY_105To160_ZpT-reweighted", "ST_tW_top", "WWW", "WZTo2L2Q", "WZZ", "ZZTo2L2Q", "ZZZ"]

print(f"[INFO] Merging signal files: hadd -f {sig_file} {' '.join([f'{base_dir}/{proc}.root' for proc in sig_proc])}")
os.system(f"hadd -f {sig_file} {' '.join([f'{base_dir}/{proc}.root' for proc in sig_proc])}")
print(f"[INFO] Merging background files: hadd -f {bkg_file} {' '.join([f'{base_dir}/{proc}.root' for proc in bkg_proc])}")
os.system(f"hadd -f {bkg_file} {' '.join([f'{base_dir}/{proc}.root' for proc in bkg_proc])}")

tree_name  = "data_two_jet_m110To150_ggH_test"
score_var  = "bdt_score"
weight_var = "eventWeight"
mass_var   = "diMufsr_rc_mass"

year = "2023"

num_thresholds = 200
max_categories = 6
random_seed    = 12345

mass_nbins = 80
mass_min   = 110.0
mass_max   = 150.0

SAVE_ALL_SCAN_PLOTS = False

CONFIG = {
    "2022": {"lumi": 62.3, "com": 13.6},
    "2023": {"lumi": 62.3, "com": 13.6},
    "2024": {"lumi": 109.8, "com": 13.6},
}

GLOBAL_WIN_LO = 123.0
GLOBAL_WIN_HI = 127.0
PLOT_DIR = "check_significance_fit/plots"
SCAN_PLOT_DIR = "check_significance_fit/scans_summary"
WINDOW_RATIO = 0.04

BIN_CACHE = {}
np.random.seed(random_seed)

# ------------------------
# ----- MODELS -----------
# ------------------------

def triple_gaussian(x, N, m1, s1, m2, s2, m3, s3, f1, f2):
    f3 = 1.0 - f1 - f2
    def g(val, mu, sig):
        return (1.0 / (np.sqrt(2 * np.pi) * sig)) * np.exp(-0.5 * ((val - mu) / sig)**2)
    return N * (f1 * g(x, m1, s1) + f2 * g(x, m2, s2) + f3 * g(x, m3, s3))

# --- Background Models ---

def exp_1(x, N, t):
    return N * np.exp(-t * (x - 110.0))

def exp_2(x, N, f1, t1, t2):
    return N * (f1 * np.exp(-t1 * (x - 110.0)) + (1-f1) * np.exp(-t2 * (x - 110.0)))

def exp_3(x, N, f1, f2, t1, t2, t3):
    f3 = 1.0 - f1 - f2
    return N * (f1 * np.exp(-t1 * (x - 110.0)) + f2 * np.exp(-t2 * (x - 110.0)) + f3 * np.exp(-t3 * (x - 110.0)))

# ------------------------
# ----- INTEGRATION ------
# ------------------------

def calc_exp_integral_term(t, lo, hi, x0=110.0):
    """Calculates I and dI/dt analytically."""
    A = lo - x0; B = hi - x0
    exp_A = np.exp(-t * A); exp_B = np.exp(-t * B)
    term_val = (1.0 / t) * (exp_A - exp_B)
    term_deriv = (-1.0/t) * term_val + (1.0/t) * (B * exp_B - A * exp_A)
    return term_val, term_deriv

def get_analytical_integral_and_error(model_func, params, cov, lo, hi, x0=110.0):
    """Analytically calculates integral and error (J @ Cov @ J.T)."""
    if hi <= lo: return 0.0, 0.0
    J = np.zeros(len(params))
    Integral = 0.0

    if len(params) == 2: # exp_1
        N, t = params
        I_t, dI_dt = calc_exp_integral_term(t, lo, hi, x0)
        Integral = N * I_t
        J[0] = I_t; J[1] = N * dI_dt

    elif len(params) == 4: # exp_2
        N, f1, t1, t2 = params
        I_1, dI_dt1 = calc_exp_integral_term(t1, lo, hi, x0)
        I_2, dI_dt2 = calc_exp_integral_term(t2, lo, hi, x0)
        Integral = N * (f1 * I_1 + (1 - f1) * I_2)
        J[0] = f1 * I_1 + (1 - f1) * I_2
        J[1] = N * (I_1 - I_2)
        J[2] = N * f1 * dI_dt1
        J[3] = N * (1 - f1) * dI_dt2

    elif len(params) == 6: # exp_3
        N, f1, f2, t1, t2, t3 = params
        I_1, dI_dt1 = calc_exp_integral_term(t1, lo, hi, x0)
        I_2, dI_dt2 = calc_exp_integral_term(t2, lo, hi, x0)
        I_3, dI_dt3 = calc_exp_integral_term(t3, lo, hi, x0)
        f3 = 1.0 - f1 - f2
        Integral = N * (f1 * I_1 + f2 * I_2 + f3 * I_3)
        J[0] = f1 * I_1 + f2 * I_2 + f3 * I_3
        J[1] = N * (I_1 - I_3); J[2] = N * (I_2 - I_3)
        J[3] = N * f1 * dI_dt1; J[4] = N * f2 * dI_dt2; J[5] = N * f3 * dI_dt3

    variance = np.dot(J, np.dot(cov, J))
    return Integral, np.sqrt(max(0.0, variance))

# ------------------------
# ----- FITTING TOOLS ----
# ------------------------

def get_fwhm_from_pdf(func, params, x_range=(110, 150)):
    x_fine = np.linspace(x_range[0], x_range[1], 10000)
    y_fine = func(x_fine, *params)
    peak_x = x_fine[np.argmax(y_fine)]
    peak_y = np.max(y_fine)
    half_max = peak_y * WINDOW_RATIO
    
    def diff(x): return func(x, *params) - half_max
    
    try: low_x = brentq(diff, x_range[0], peak_x)
    except: low_x = max(x_range[0], peak_x - 3.0)
        
    try: high_x = brentq(diff, peak_x, x_range[1])
    except: high_x = min(x_range[1], peak_x + 3.0)
        
    return peak_x, low_x, high_x

def fit_signal_triple_gauss(masses, weights, plot_path=None):
    hist, edges = np.histogram(masses, bins=mass_nbins, range=(mass_min, mass_max), weights=weights)
    centers = 0.5 * (edges[:-1] + edges[1:])
    errs = np.sqrt(np.histogram(masses, bins=mass_nbins, range=(mass_min, mass_max), weights=weights**2)[0])
    errs[errs == 0] = 1.0
    
    p0_3 = [np.sum(hist), 125, 1.5, 125, 3.0, 124.0, 6.0, 0.6, 0.3]
    bounds_3 = ([0, 115, 0.3, 110, 0.5, 110, 1.0, 0, 0], [np.inf, 135, 3.0, 140, 10.0, 150, 20.0, 1, 1])
    
    try:
        popt, _ = curve_fit(triple_gaussian, centers, hist, p0=p0_3, sigma=errs, bounds=bounds_3, maxfev=20000)
        peak, win_lo, win_hi = get_fwhm_from_pdf(triple_gaussian, popt)
    except:
        peak, win_lo, win_hi = 125.0, GLOBAL_WIN_LO, GLOBAL_WIN_HI
        popt = None

    if plot_path and popt is not None:
        fig, ax = plt.subplots(figsize=(10, 8))
        hep.histplot(hist, bins=edges, yerr=errs, histtype='errorbar', color='k', label='Signal MC', ax=ax)
        x_plot = np.linspace(110, 135, 500)
        ax.plot(x_plot, triple_gaussian(x_plot, *popt), color='r', lw=2, label='Fit (3 Gauss)')
        ax.axvspan(win_lo, win_hi, alpha=0.1, color='b', label='Signal Window')
        S_val = np.sum(weights[(masses >= win_lo) & (masses <= win_hi)])
        info_text = (
            f"Range: [{win_lo:.2f}, {win_hi:.2f}] GeV\n"
            f"S (yield): {S_val:.2f}\n"
        )
        ax.text(0.05, 0.75, info_text, transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        ax.legend(loc='upper right')
        hep.cms.label("Preliminary", data=True, lumi=CONFIG[year]['lumi'], com=CONFIG[year]['com'], year=year, ax=ax)
        plt.tight_layout()
        plt.savefig(plot_path); plt.close()

    return peak, win_lo, win_hi

def estimate_background_and_plot(data_mass, win_lo, win_hi, plot_path=None):
    hist, edges = np.histogram(data_mass, bins=mass_nbins, range=(mass_min, mass_max))
    centers = 0.5 * (edges[:-1] + edges[1:])
    
    # Mask for Sideband Fitting (Blind 120-130)
    mask_sb = (centers < 120) | (centers > 130)
    x_fit = centers[mask_sb]; y_fit = hist[mask_sb]; y_err = np.sqrt(y_fit)
    y_err[y_err == 0] = 1.0
    
    if np.sum(y_fit) < 20: return 0.0, 0.0

    best_aic = np.inf
    best_res = (None, None, None, "") # func, popt, pcov, name

    # Define models with their names, functions, initial guesses, bounds, and num params (k)
    models = [
        # ("1 Exp", exp_1, [np.max(y_fit), 0.05], ([0,0], [np.inf, 1]), 2),
        ("2 Exp", exp_2, [np.max(y_fit), 0.8, 0.05, 0.15], ([0,0,0,0], [np.inf, 1, 1, 1]), 4),
        # ("3 Exp", exp_3, [np.max(y_fit), 0.4, 0.4, 0.05, 0.15, 0.3], ([0,0,0,0,0,0], [np.inf, 1, 1, 1, 1, 1]), 6)
    ]

    for name, func, p0, bounds, k in models:
        try:
            popt, pcov = curve_fit(func, x_fit, y_fit, p0=p0, sigma=y_err, absolute_sigma=True, bounds=bounds, maxfev=10000)
            rss = np.sum(((y_fit - func(x_fit, *popt))/y_err)**2)
            aic = rss + 2*k + (2*k*(k+1))/(len(x_fit)-k-1)
            
            if aic < (best_aic - 2.0):
                best_aic = aic; best_res = (func, popt, pcov, name)
        except:
            continue

    func, popt, pcov, name = best_res
    if func is None: return 0.0, 0.0

    # Calculate Integral and Error analytically
    B_est, sigma_B = get_analytical_integral_and_error(func, popt, pcov, win_lo, win_hi)

    if plot_path:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 1. Plot Data
        hist_plot = hist.astype(float)
        mask_blind = (centers > win_lo) & (centers < win_hi)
        hist_plot[mask_blind] = 0
        err_plot = np.sqrt(hist)
        err_plot[mask_blind] = 0
        
        hep.histplot(hist_plot, bins=edges, yerr=err_plot, histtype='errorbar', color='k', label='Data (Sideband)', ax=ax)
        
        # 2. Plot Fit
        x_plot = np.linspace(mass_min, mass_max, 500)
        ax.plot(x_plot, func(x_plot, *popt), color='b', label=f'Bkg Fit ({name})')
        
        # 3. Highlight Blinded Region
        ax.axvspan(win_lo, win_hi, color='red', alpha=0.2, label='Blinded Region')
        
        # 4. Construct Info Text with Parameters
        perr = np.sqrt(np.diag(pcov)) # Get diagonal errors
        param_str = ""
        
        if func == exp_1:
            param_str = (f"N={popt[0]:.1f} $\\pm$ {perr[0]:.1f}\n"
                         f"t={popt[1]:.3f} $\\pm$ {perr[1]:.3f}")
        elif func == exp_2:
            param_str = (f"N={popt[0]:.1f} $\\pm$ {perr[0]:.1f}, f1={popt[1]:.2f} $\\pm$ {perr[1]:.2f}\n"
                         f"t1={popt[2]:.3f} $\\pm$ {perr[2]:.3f}, t2={popt[3]:.3f} $\\pm$ {perr[3]:.3f}")
        elif func == exp_3:
            param_str = (f"N={popt[0]:.1f} $\\pm$ {perr[0]:.1f}, f1={popt[1]:.2f} $\\pm$ {perr[1]:.2f}\n"
                         f"f2={popt[2]:.2f} $\\pm$ {perr[2]:.2f}, t1={popt[3]:.3f} $\\pm$ {perr[3]:.3f}\n"
                         f"t2={popt[4]:.3f} $\\pm$ {perr[4]:.3f}, t3={popt[5]:.3f} $\\pm$ {perr[5]:.3f}")

        info_text = (
            f"Model: {name}\n"
            f"Window: [{win_lo:.2f}, {win_hi:.2f}] GeV\n"
            f"B = {B_est:.2f} $\\pm$ {sigma_B:.2f} (Analytical)\n"
            f"Params:\n{param_str}"
        )
        
        ax.text(0.05, 0.55, info_text, transform=ax.transAxes, fontsize=11,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

        ax.legend(loc='upper right')
        hep.cms.label("Preliminary", data=True, lumi=CONFIG[year]['lumi'], com=CONFIG[year]['com'], year=year, ax=ax)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    return B_est, sigma_B

# ------------------------
# ----- METRICS & OPT ----
# ------------------------

def calc_significance_with_error(S, B, sigma_B):
    if B <= 0 or S <= 0: return 0.0
    if sigma_B <= 1e-9: return np.sqrt(2.0 * ((S + B) * np.log(1.0 + S / B) - S))
    sb = S + B; sig2 = sigma_B**2
    val = 2.0 * (sb * np.log((sb * (B + sig2)) / (B**2 + sb * sig2)) - (B**2 / sig2) * np.log(1.0 + (sig2 * S) / (B * (B + sig2))))
    return np.sqrt(val) if val > 0 else 0.0

def evaluate_boundaries(sig_data, obs_data, boundaries, category_step, save_dir=None, fixed_window=None):
    (s_sc, s_w, s_m) = sig_data; (d_sc, d_w, d_m) = obs_data
    S_list, B_list, err_list, stats = [], [], [], []
    lower = -np.inf; all_bounds = list(boundaries) + [np.inf]
    if save_dir and not os.path.exists(save_dir): os.makedirs(save_dir)

    for bin_idx, boundary in enumerate(all_bounds):
        cache_key = (round(lower, 6) if lower != -np.inf else -999.0, round(boundary, 6) if boundary != np.inf else 999.0)
        if (save_dir is None) and (cache_key in BIN_CACHE):
            res = BIN_CACHE[cache_key]
            S_list.append(res['S']); B_list.append(res['B']); err_list.append(res['sigma_B']); stats.append(res)
            lower = boundary; continue

        mask_s = (s_sc > lower) & (s_sc <= boundary); mask_d = (d_sc > lower) & (d_sc <= boundary)
        p_sig = f"{save_dir}/bin_{bin_idx}_sig.pdf" if save_dir else None
        
        if fixed_window: lo, hi = fixed_window
        else:
            if mask_s.sum() > 5: _, lo, hi = fit_signal_triple_gauss(s_m[mask_s], s_w[mask_s], plot_path=p_sig)
            else: lo, hi = GLOBAL_WIN_LO, GLOBAL_WIN_HI

        S = np.sum(s_w[mask_s & (s_m >= lo) & (s_m <= hi)])
        B, sigma_B = estimate_background_and_plot(d_m[mask_d], lo, hi, plot_path=f"{save_dir}/bin_{bin_idx}_bkg.pdf" if save_dir else None)
        
        res = {'S': S, 'B': B, 'sigma_B': sigma_B, 'win': (lo, hi)}
        S_list.append(S); B_list.append(B); err_list.append(sigma_B); stats.append(res)
        if save_dir is None: BIN_CACHE[cache_key] = res
        lower = boundary

    zs = np.array([calc_significance_with_error(s, b, e) for s, b, e in zip(S_list, B_list, err_list)])
    return np.sqrt(np.sum(zs**2)), stats

def run_optimization(sig_data, obs_data):
    print(f"\n[INFO] Starting optimization...")
    BIN_CACHE.clear()
    _, g_lo, g_hi = fit_signal_triple_gauss(sig_data[2], sig_data[1])
    GLOBAL_FIXED_WIN = (g_lo, g_hi)
    print(f"[INFO] Global Window: {g_lo:.2f} - {g_hi:.2f} GeV")

    s_sc, s_w = sig_data[0], sig_data[1]
    idx = np.argsort(s_sc); cum_w = np.cumsum(s_w[idx]); tot_w = cum_w[-1]
    cuts = sorted(list(set([s_sc[idx][np.searchsorted(cum_w, p * tot_w)] for p in np.linspace(0.0, 1.0, num_thresholds) if np.searchsorted(cum_w, p * tot_w) < len(s_sc)])))

    current_bounds, history = [], []
    z0, stats0 = evaluate_boundaries(sig_data, obs_data, [], 1, save_dir=f"{PLOT_DIR}/step_1_baseline")
    history.append({'n': 1, 'z': z0, 'bounds': [], 'stats': stats0})
    print(f"[INFO] 1 Cat Z = {z0:.4f}")

    for n_cat in range(2, max_categories + 1):
        print(f"\n[INFO] Optimizing {n_cat} categories...")
        best_z, best_cut = -1.0, None
        scan_eff, scan_z_dyn, scan_z_fix = [], [], []
        prev_effs = [s_w[s_sc < b].sum() / tot_w for b in current_bounds]

        for i, cut in enumerate(cuts):
            if cut in current_bounds: continue
            tb = sorted(current_bounds + [cut])
            z_dyn, _ = evaluate_boundaries(sig_data, obs_data, tb, n_cat, save_dir=f"{PLOT_DIR}/step_{n_cat}/cut_{i}" if SAVE_ALL_SCAN_PLOTS else None)
            
            scan_eff.append(s_w[s_sc < cut].sum() / tot_w)
            scan_z_dyn.append(z_dyn)
            if z_dyn > best_z: best_z = z_dyn; best_cut = cut
            if i % 10 == 0: print(f"\r   Scan {i}/{len(cuts)} | Z_dyn: {best_z:.4f}", end="")

        if best_cut is not None:
            current_bounds.append(float(best_cut)); current_bounds.sort()
            _, final_stats = evaluate_boundaries(sig_data, obs_data, current_bounds, n_cat, save_dir=f"{PLOT_DIR}/step_{n_cat}/WINNER_cut_{best_cut:.4f}")
            history.append({
                'n': n_cat, 'z': best_z, 'bounds': list(current_bounds), 'stats': final_stats,
                'scan': {'x_eff': scan_eff, 'Z_dyn': scan_z_dyn, 'prev_eff': prev_effs}
            })
            print(f"\n   Found best cut: {best_cut:.4f} (Z={best_z:.4f})")
        else: break
            
    return history

def plot_iterative_scan(scan_history):
    if not os.path.exists(SCAN_PLOT_DIR): os.makedirs(SCAN_PLOT_DIR)
    
    for entry in scan_history:
        n_cat = entry['n']
        if n_cat < 2: continue
        
        scan = entry['scan']
        eff = np.array(scan["x_eff"])
        Zd = np.array(scan["Z_dyn"])
        prev_eff = scan["prev_eff"]
        
        # Identify winner
        idx_max = np.argmax(Zd)
        best_eff = eff[idx_max]
        
        # Sort for clean line/scatter plot
        idx_sort = np.argsort(eff)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot Scan Points
        ax.plot(eff[idx_sort], Zd[idx_sort], 'b.', markersize=5, label='Scan Points')
        
        # Plot Previous Boundaries (Vertical Lines)
        # Only label the first one to avoid legend clutter
        label_prev = 'Prev. Boundaries' if len(prev_eff) > 0 else None
        for i, b in enumerate(prev_eff):
            lbl = label_prev if i == 0 else None
            ax.axvline(b, color='gray', linestyle='-', linewidth=1.5, alpha=0.6, label=lbl)
            
        # Plot New Boundary (Vertical Line)
        ax.axvline(best_eff, color='red', linestyle='--', linewidth=2.0, label='New Boundary')
        
        # Labels
        ax.set_xlabel(r'$1 - \epsilon_{\mathrm{sig}}$', fontsize=18)
        ax.set_ylabel(r'Significance ($\sigma$)', fontsize=18)
        
        # CMS Style
        hep.cms.label("Preliminary", data=False, lumi=None, year=year, ax=ax, loc=0, fontsize=16)
        
        # Legend
        ax.legend(loc='upper left', fontsize=14, frameon=True)
        
        # Info Text (formerly Title) - Positioned below legend or free space
        info_text = f"{n_cat-1} $\\to$ {n_cat} Categories\nScan for max significance"
        ax.text(0.05, 0.75, info_text, transform=ax.transAxes, fontsize=14, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        plt.tight_layout()
        plt.savefig(f'{SCAN_PLOT_DIR}/scan_N_{n_cat}.pdf')
        plt.close()

def load_data(file_path, is_data=False):
    try:
        with uproot.open(file_path) as f:
            data = f[tree_name].arrays([score_var, mass_var, weight_var] if not is_data else [score_var, mass_var], library="np")
            return data[score_var], (data[weight_var] if not is_data else np.ones_like(data[score_var])), data[mass_var]
    except Exception as e:
        print(f"[ERROR] Loading {file_path}: {e}")
        return np.array([]), np.array([]), np.array([])

def save_results(history):
    out_dir = "check_significance_fit"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    with open(f"{out_dir}/results.txt", "w") as f:
        f.write("RESULTS\n" + "="*60 + "\n")
        for entry in history:
            f.write(f"--- {entry['n']} Categories (Total Z = {entry['z']:.4f}) ---\n")
            f.write(f"Boundaries: {entry['bounds']}\n")
            if 'stats' in entry:
                for i, s in enumerate(entry['stats']):
                    win_str = f"[{s['win'][0]:.2f}, {s['win'][1]:.2f}]"
                    f.write(f"Bin {i}: S={s['S']:.2f}, B={s['B']:.2f}, err={s['sigma_B']:.2f}, Z={calc_significance_with_error(s['S'], s['B'], s['sigma_B']):.4f}, Win={win_str}\n")
            f.write("\n")

if __name__ == "__main__":
    start_t = time.time()
    sig_data = load_data(sig_file); obs_data = load_data(data_file, is_data=True)
    if len(sig_data[0]) > 0:
        history = run_optimization(sig_data, obs_data)
        save_results(history)
        plot_iterative_scan(history)
    print(f"[INFO] Done in {time.time() - start_t:.1f}s")