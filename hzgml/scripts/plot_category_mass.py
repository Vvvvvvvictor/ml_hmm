# -*- coding: utf-8 -*-
# Plot mass distributions for different BDT categories
# Based on optimal boundaries from categorization optimization

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import uproot
import os
from matplotlib.gridspec import GridSpec

# 设置CMS风格
plt.style.use(hep.style.CMS)

# ------------------------
# ----- CONFIGURATION ----
# ------------------------
input_dir = "/eos/user/j/jiehan/root_mumu/outputs_ggH_reweight_bdt_1111/ggH"
output_dir = "./check_category_mass"
os.makedirs(output_dir, exist_ok=True)

# BDT score boundaries (from optimal_boundaries.txt)
# Best configuration: 6 categories
boundaries = [0.096485, 0.16874, 0.216712, 0.289974, 0.44982]

tree_name = "test"
score_var = "bdt_score"
weight_var = "eventWeight"
mass_var = "diMufsr_mass"

# Mass range
mass_min = 110.0
mass_max = 150.0
mass_nbins = 40

# Background samples (order matters for stacking)
bkg_samples = {
    "DY_105To160": {"color": "#7fc97f", "label": "DY"},
    "TTTo2L2Nu": {"color": "#beaed4", "label": r"$t\bar{t}$"},
    "ST_tW_top": {"color": "#fdc086", "label": "Single top"},
    "ST_tW_antitop": {"color": "#fdc086", "label": None},  # Same as ST_tW_top
    "EWK_LLJJ_M50": {"color": "#ffff99", "label": "EWK"},
    "WWTo2L2Nu": {"color": "#386cb0", "label": "VV"},
    "WZTo3LNu": {"color": "#386cb0", "label": None},
    "WZTo2L2Q": {"color": "#386cb0", "label": None},
    "ZZTo2L2Nu": {"color": "#386cb0", "label": None},
    "ZZTo2L2Q": {"color": "#386cb0", "label": None},
    "ZZTo4L": {"color": "#386cb0", "label": None},
    "WWW": {"color": "#f0027f", "label": "VVV"},
    "WWZ": {"color": "#f0027f", "label": None},
    "WZZ": {"color": "#f0027f", "label": None},
    "ZZZ": {"color": "#f0027f", "label": None},
}

# Signal samples
sig_samples = {
    "GluGluHToMuMu_M125": {"color": "red", "label": "ggH", "linestyle": "-"},
    "VBFHToMuMu_M125": {"color": "blue", "label": "VBF", "linestyle": "-"},
}

data_sample = "data"

# ------------------------
# ----- LOAD DATA --------
# ------------------------
def load_sample(file_path, tree=tree_name):
    """Load data from a ROOT file"""
    try:
        with uproot.open(file_path) as f:
            tree_obj = f[tree]
            arrays = tree_obj.arrays([score_var, weight_var, mass_var], library="np")
            
            scores = arrays[score_var].astype(np.float64)
            weights = arrays[weight_var].astype(np.float64)
            masses = arrays[mass_var].astype(np.float64)
            
            return scores, weights, masses
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None, None

def get_category_mask(scores, cat_idx, boundaries):
    """Get mask for events in a specific category"""
    if cat_idx == 0:
        return scores <= boundaries[0]
    elif cat_idx == len(boundaries):
        return scores > boundaries[-1]
    else:
        return (scores > boundaries[cat_idx-1]) & (scores <= boundaries[cat_idx])

# ------------------------
# ----- PLOT FUNCTION ----
# ------------------------
def plot_category_mass(cat_idx, boundaries, bkg_data, sig_data, data_data):
    """Plot mass distribution for one category with data/MC ratio"""
    
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 1, hspace=0.0, height_ratios=[3, 1, 0.05, 0.05])
    ax_main = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)
    
    # Define category label
    if cat_idx == 0:
        cat_label = f"Cat {cat_idx}: BDT ≤ {boundaries[0]:.3f}"
    elif cat_idx == len(boundaries):
        cat_label = f"Cat {cat_idx}: BDT > {boundaries[-1]:.3f}"
    else:
        cat_label = f"Cat {cat_idx}: {boundaries[cat_idx-1]:.3f} < BDT ≤ {boundaries[cat_idx]:.3f}"
    
    # Prepare histograms
    bin_edges = np.linspace(mass_min, mass_max, mass_nbins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Stack backgrounds
    bkg_hists = []
    bkg_colors = []
    bkg_labels = []
    bkg_weights_dict = {}  # Store weights for each process type
    bkg_total = np.zeros(mass_nbins)
    
    for sample_name, info in bkg_samples.items():
        if sample_name in bkg_data:
            masses, weights = bkg_data[sample_name]
            hist, _ = np.histogram(masses, bins=bin_edges, weights=weights)
            bkg_hists.append(hist)
            bkg_colors.append(info["color"])
            
            # Accumulate weights for each process type
            if info["label"] is not None:
                if info["label"] not in bkg_weights_dict:
                    bkg_weights_dict[info["label"]] = 0
                bkg_weights_dict[info["label"]] += np.sum(weights)
            
            # Only add label if not None and not already added
            if info["label"] is not None and info["label"] not in bkg_labels:
                bkg_labels.append(info["label"])
            else:
                bkg_labels.append("")
            bkg_total += hist
    
    # Calculate background scale factor from sidebands (110-115 + 135-150)
    if data_data is not None:
        data_masses, data_weights = data_data
        # Sideband masks
        sideband_mask = ((data_masses >= 110) & (data_masses <= 115)) | \
                        ((data_masses >= 135) & (data_masses <= 150))
        
        data_sideband = np.sum(data_weights[sideband_mask])
        
        bkg_sideband = 0
        for sample_name, info in bkg_samples.items():
            if sample_name in bkg_data:
                masses, weights = bkg_data[sample_name]
                sb_mask = ((masses >= 110) & (masses <= 115)) | \
                          ((masses >= 135) & (masses <= 150))
                bkg_sideband += np.sum(weights[sb_mask])
        
        if bkg_sideband > 0:
            scale_factor = data_sideband / bkg_sideband
        else:
            scale_factor = 1.0
    else:
        scale_factor = 1.0
    
    # Apply scale factor to backgrounds
    bkg_hists_scaled = [h * scale_factor for h in bkg_hists]
    bkg_total_scaled = bkg_total * scale_factor
    
    # Scale the weights dictionary
    bkg_weights_scaled = {k: v * scale_factor for k, v in bkg_weights_dict.items()}
    
    # Calculate signal scale factor (based on 120-130 GeV background)
    mass_mask_120_130 = (bin_centers >= 120) & (bin_centers <= 130)
    bkg_in_signal_region = np.sum(bkg_total_scaled[mass_mask_120_130])
    
    # Find a reasonable signal scale (round to nearest 10)
    if bkg_in_signal_region > 0:
        # Start with signals
        sig_total_in_region = 0
        for sample_name in sig_samples.keys():
            if sample_name in sig_data:
                masses, weights = sig_data[sample_name]
                sig_mask = (masses >= 120) & (masses <= 130)
                sig_total_in_region += np.sum(weights[sig_mask])
        
        if sig_total_in_region > 0:
            raw_scale = bkg_in_signal_region / sig_total_in_region * 0.5  # Make signal visible
            # Round to nearest power of 10
            magnitude = 10 ** np.floor(np.log10(raw_scale))
            sig_scale = np.round(raw_scale / magnitude) * magnitude
            sig_scale = int(max(10, sig_scale))  # At least 10x
        else:
            sig_scale = 100
    else:
        sig_scale = 100
    
    # Plot stacked backgrounds with weights in labels
    bkg_labels_with_weights = []
    for label in bkg_labels:
        if label != "" and label in bkg_weights_scaled:
            bkg_labels_with_weights.append(f"{label} (W={bkg_weights_scaled[label]:.1f})")
        else:
            bkg_labels_with_weights.append(label)
    
    ax_main.hist([bin_centers] * len(bkg_hists_scaled), bins=bin_edges, weights=bkg_hists_scaled,
                 stacked=True, color=bkg_colors, label=bkg_labels_with_weights, histtype='stepfilled',
                 edgecolor='none', alpha=0.8)
    
    # Background uncertainty (gray band)
    bkg_errors = np.sqrt(bkg_total_scaled)  # Poisson uncertainty
    ax_main.fill_between(bin_centers, bkg_total_scaled - bkg_errors, bkg_total_scaled + bkg_errors,
                         step='mid', alpha=0, hatch='///', label='Bkg. unc.', edgecolor='black', linewidth=1)
    
    # Plot data points with error bars
    if data_data is not None:
        data_masses, data_weights = data_data[0][sideband_mask], data_data[1][sideband_mask]
        data_hist, _ = np.histogram(data_masses, bins=bin_edges, weights=data_weights)
        data_errors = np.sqrt(data_hist)  # Poisson errors
        
        ax_main.errorbar(bin_centers, data_hist, yerr=data_errors, fmt='ko', 
                        markersize=4, label=f'Data (W={np.sum(data_weights):.1f})', 
                        capsize=2, linewidth=1)
    
    # Plot signals
    signal_bin_edges = np.linspace(mass_min, mass_max, mass_nbins*4 + 1)
    signal_bin_centers = 0.5 * (signal_bin_edges[:-1] + signal_bin_edges[1:])
    sig_total_hist = np.zeros(len(signal_bin_centers))
    sig_total_weight = 0
    sig_labels_added = []
    for sample_name, info in sig_samples.items():
        if sample_name in sig_data:
            masses, weights = sig_data[sample_name]
            hist, _ = np.histogram(masses, bins=signal_bin_edges, weights=weights * sig_scale)
            sig_total_hist += hist
            sig_total_weight += np.sum(weights)
            label = f"{info['label']} (×{int(sig_scale)}, W={np.sum(weights):.1f})"
            ax_main.plot(signal_bin_centers, hist, color=info['color'], linestyle=info['linestyle'],
                        linewidth=2, label=label)
            sig_labels_added.append(sample_name)
    
    # Plot total signal
    ax_main.plot(signal_bin_centers, sig_total_hist, color='black', 
                    linestyle='-', linewidth=2, 
                    label=f'Total signal (×{int(sig_scale)}, W={sig_total_weight:.1f})')
    
    # Calculate and display sum of weights
    total_bkg_weight = sum(np.sum(bkg_data[s][1]) for s in bkg_samples.keys() if s in bkg_data)
    
    # Legend
    handles, labels = ax_main.get_legend_handles_labels()
    # Reverse to show in correct order
    ax_main.legend(handles[::-1], labels[::-1], loc='upper right', fontsize=14, ncol=2)
    
    # Add scale factor text
    ax_main.text(0.65, 0.7, f'Bkg scale: {scale_factor:.3f}\n(from sidebands)',
                transform=ax_main.transAxes, fontsize=14, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax_main.set_ylabel('Events / {:.1f} GeV'.format(bin_width), fontsize=20)
    ax_main.set_xlim(mass_min, mass_max)
    ax_main.tick_params(axis='x', labelbottom=False)
    
    # Add CMS label and category info
    hep.cms.label("Preliminary", data=True, lumi=62.31, year="2022-2023", com=13.6, ax=ax_main, loc=0, fontsize=14)
    ax_main.text(0.65, 0.6, cat_label, transform=ax_main.transAxes, fontsize=14,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ==================
    # Ratio plot
    # ==================
    if data_data is not None:
        ratio = np.divide(data_hist, bkg_total_scaled, 
                         out=np.ones_like(data_hist), where=bkg_total_scaled>0)
        
        # Error propagation for ratio
        # σ(data/MC) = data/MC * sqrt((σ_data/data)² + (σ_MC/MC)²)
        ratio_errors = ratio * np.sqrt(
            np.divide(data_errors**2, data_hist**2, out=np.zeros_like(data_hist), where=data_hist>0) +
            np.divide(bkg_errors**2, bkg_total_scaled**2, out=np.zeros_like(bkg_total_scaled), where=bkg_total_scaled>0)
        )
        
        ax_ratio.errorbar(bin_centers, ratio, yerr=ratio_errors, fmt='ko', 
                         markersize=4, capsize=2, linewidth=1)
        
        # MC uncertainty band at 1.0
        mc_rel_error = np.divide(bkg_errors, bkg_total_scaled, 
                                out=np.zeros_like(bkg_errors), where=bkg_total_scaled>0)
        ax_ratio.fill_between(bin_centers, 1 - mc_rel_error, 1 + mc_rel_error,
                             step='mid', color='gray', alpha=0.3)
    
    ax_ratio.axhline(1, color='black', linestyle='--', linewidth=1)
    ax_ratio.set_xlabel(r'$m_{\mu\mu}$ [GeV]', fontsize=20)
    ax_ratio.set_ylabel('Data/MC', fontsize=16)
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_xlim(mass_min, mass_max)
    ax_ratio.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_name = f"category_{cat_idx}_mass"
    plt.savefig(os.path.join(output_dir, f"{output_name}.png"), dpi=400, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"{output_name}.pdf"), bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot for category {cat_idx}")

# ------------------------
# ----- MAIN -------------
# ------------------------
if __name__ == "__main__":
    print("="*60)
    print("Plotting mass distributions for BDT categories")
    print("="*60)
    print(f"Boundaries: {boundaries}")
    print(f"Number of categories: {len(boundaries) + 1}")
    
    # Load all samples
    print("\n[INFO] Loading samples...")
    
    all_bkg_data = {}
    all_sig_data = {}
    all_data_data = None
    
    # Load backgrounds
    for sample_name in bkg_samples.keys():
        file_path = os.path.join(input_dir, f"{sample_name}.root")
        if os.path.exists(file_path):
            print(f"Loading background: {sample_name}")
            scores, weights, masses = load_sample(file_path)
            if scores is not None:
                all_bkg_data[sample_name] = (scores, weights, masses)
        else:
            print(f"Warning: {file_path} not found")
    
    # Load signals
    for sample_name in sig_samples.keys():
        file_path = os.path.join(input_dir, f"{sample_name}.root")
        if os.path.exists(file_path):
            print(f"Loading signal: {sample_name}")
            scores, weights, masses = load_sample(file_path)
            if scores is not None:
                all_sig_data[sample_name] = (scores, weights, masses)
        else:
            print(f"Warning: {file_path} not found")
    
    # Load data
    data_file_path = os.path.join(input_dir, f"{data_sample}.root")
    if os.path.exists(data_file_path):
        print(f"Loading data: {data_sample}")
        scores, weights, masses = load_sample(data_file_path)
        if scores is not None:
            all_data_data = (scores, weights, masses)
    else:
        print(f"Warning: {data_file_path} not found")
    
    # Plot each category
    print("\n[INFO] Generating plots for each category...")
    num_categories = len(boundaries) + 1
    
    for cat_idx in range(num_categories):
        print(f"\n[INFO] Processing category {cat_idx}...")
        
        # Filter data for this category
        cat_bkg_data = {}
        for sample_name, (scores, weights, masses) in all_bkg_data.items():
            mask = get_category_mask(scores, cat_idx, boundaries)
            cat_bkg_data[sample_name] = (masses[mask], weights[mask])
            print(f"  {sample_name}: {np.sum(mask)} events, weight={np.sum(weights[mask]):.2f}")
        
        cat_sig_data = {}
        for sample_name, (scores, weights, masses) in all_sig_data.items():
            mask = get_category_mask(scores, cat_idx, boundaries)
            cat_sig_data[sample_name] = (masses[mask], weights[mask])
            print(f"  {sample_name}: {np.sum(mask)} events, weight={np.sum(weights[mask]):.2f}")
        
        cat_data_data = None
        if all_data_data is not None:
            scores, weights, masses = all_data_data
            mask = get_category_mask(scores, cat_idx, boundaries)
            cat_data_data = (masses[mask], weights[mask])
            print(f"  Data: {np.sum(mask)} events, weight={np.sum(weights[mask]):.2f}")
        
        # Plot
        plot_category_mass(cat_idx, boundaries, cat_bkg_data, cat_sig_data, cat_data_data)
    
    print(f"\n[INFO] All plots saved to {output_dir}")
    print("="*60)
