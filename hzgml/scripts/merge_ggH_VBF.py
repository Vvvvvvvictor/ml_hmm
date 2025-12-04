#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge m110To150_ggH and m110To150_VBF trees inside each ROOT file
into a single concatenated tree named m110To150_Dilep using uproot with chunked processing.

Requirements:
    - Python packages: uproot>=5, numpy, pandas

Usage examples:
    python scripts/merge_ggH_VBF.py -i /eos/user/q/qguo/vbfhmm/ml/RunIII/skimmed_ntuples_ggH_v1/ -o /eos/user/j/jiehan/root_mumu/skimmed_ntuples_ggH_v1 --overwrite
    python scripts/merge_ggH_VBF.py -i /eos/user/q/qguo/vbfhmm/ml/RunIII/skimmed_ntuples_ggH_v1/ -o /eos/user/j/jiehan/root_mumu/skimmed_ntuples_ggH_v1 --dry-run
    python scripts/merge_ggH_VBF.py -i /eos/user/j/jiehan/root_mumu/skimmed_ntuples_ggH_v1 -p ZZZ.root
    python scripts/merge_ggH_VBF.py -i /eos/user/j/jiehan/root_mumu/skimmed_ntuples_ggH_v1 --chunk-size 100000

Notes:
    - If output directory is specified, writes merged files there; otherwise overwrites input files.
    - Adds the new tree m110To150_Dilep.
    - Skips files where the source trees are missing or target already exists (unless --overwrite).
    - No extra branches are added.
    - Uses chunked processing to handle large files without memory issues.
"""

import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd
import uproot
from pdb import set_trace as bp

def _tree_exists(file_path: str, tree_name: str) -> bool:
    try:
        with uproot.open(file_path) as f:
            return tree_name in f
    except Exception:
        return False


def merge_file(file_path: str, output_dir: str = None, dry_run: bool = False, overwrite: bool = False, chunk_size: int = 100000):
    if dry_run:
        print(f"[DRY-RUN] Would process: {file_path}")
        return

    target_name = "m110To150_Dilep"
    
    # Determine output file path
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(file_path))
    else:
        output_path = file_path

    # Skip if target exists and not overwriting
    if _tree_exists(output_path, target_name) and not overwrite:
        print(f"[SKIP] {target_name} already exists in {output_path}")
        return

    t1_name = "m110To150_ggH"
    t2_name = "m110To150_VBF"

    try:
        with uproot.open(file_path) as f:
            if t1_name not in f or t2_name not in f:
                print(f"[SKIP] Missing source trees in {file_path} (ggH? {t1_name in f} VBF? {t2_name in f})")
                return
            t1 = f[t1_name]
            t2 = f[t2_name]

            # Get branch names and check consistency
            branches1 = set(t1.keys())
            branches2 = set(t2.keys())
            
            if branches1 != branches2:
                print(f"[ERROR] Branch mismatch in {file_path}")
                print(f"        ggH has: {sorted(branches1)}")
                print(f"        VBF has: {sorted(branches2)}")
                return
            
            branch_names = sorted(branches1)
            n_entries_t1 = t1.num_entries
            n_entries_t2 = t2.num_entries
            total_entries = n_entries_t1 + n_entries_t2
            
            print(f"[INFO] Processing {file_path}")
            print(f"       ggH: {n_entries_t1} entries, VBF: {n_entries_t2} entries")
            print(f"       Total: {total_entries} entries")

    except Exception as e:
        print(f"[ERROR] Failed reading trees from {file_path}: {e}")
        return

    # Create temporary output file
    temp_file = output_path + ".tmp"
    
    try:
        # Open output file for writing
        with uproot.recreate(temp_file) as f_out:
            # First: copy original ggH tree
            print(f"[INFO] Copying original ggH tree ({t1_name})...")
            with uproot.open(file_path) as f_in:
                t1 = f_in[t1_name]
                
                for start in range(0, n_entries_t1, chunk_size):
                    stop = min(start + chunk_size, n_entries_t1)
                    
                    chunk_dict = {}
                    for branch in branch_names:
                        arr = t1[branch].array(library="np", entry_start=start, entry_stop=stop)
                        chunk_dict[branch] = arr
                    
                    if start == 0:
                        f_out[t1_name] = chunk_dict
                    else:
                        f_out[t1_name].extend(chunk_dict)
                    
                    if (stop - start) > 0:
                        print(f"       Copied ggH entries {start}-{stop}")
            
            # Second: copy original VBF tree
            print(f"[INFO] Copying original VBF tree ({t2_name})...")
            with uproot.open(file_path) as f_in:
                t2 = f_in[t2_name]
                
                for start in range(0, n_entries_t2, chunk_size):
                    stop = min(start + chunk_size, n_entries_t2)
                    
                    chunk_dict = {}
                    for branch in branch_names:
                        arr = t2[branch].array(library="np", entry_start=start, entry_stop=stop)
                        chunk_dict[branch] = arr
                    
                    if start == 0:
                        f_out[t2_name] = chunk_dict
                    else:
                        f_out[t2_name].extend(chunk_dict)
                    
                    if (stop - start) > 0:
                        print(f"       Copied VBF entries {start}-{stop}")
            
            # Third: create merged Dilep tree with ggH data
            print(f"[INFO] Creating merged tree ({target_name}) with ggH data...")
            with uproot.open(file_path) as f_in:
                t1 = f_in[t1_name]
                
                for start in range(0, n_entries_t1, chunk_size):
                    stop = min(start + chunk_size, n_entries_t1)
                    
                    chunk_dict = {}
                    for branch in branch_names:
                        arr = t1[branch].array(library="np", entry_start=start, entry_stop=stop)
                        chunk_dict[branch] = arr
                    
                    if start == 0:
                        f_out[target_name] = chunk_dict
                    else:
                        f_out[target_name].extend(chunk_dict)
                    
                    if (stop - start) > 0:
                        print(f"       Wrote ggH entries {start}-{stop}")
            
            # Fourth: append VBF data to merged tree
            print(f"[INFO] Appending VBF data to merged tree...")
            with uproot.open(file_path) as f_in:
                t2 = f_in[t2_name]
                
                for start in range(0, n_entries_t2, chunk_size):
                    stop = min(start + chunk_size, n_entries_t2)
                    
                    chunk_dict = {}
                    for branch in branch_names:
                        arr = t2[branch].array(library="np", entry_start=start, entry_stop=stop)
                        chunk_dict[branch] = arr
                    
                    f_out[target_name].extend(chunk_dict)
                    
                    if (stop - start) > 0:
                        print(f"       Wrote VBF entries {start}-{stop}")
        
        # Calculate yields before replacing file
        ggH_yields = None
        VBF_yields = None
        total_yields = None
        
        try:
            with uproot.open(file_path) as f:
                if "eventWeight" in f[t1_name].keys():
                    ggH_yields = np.sum(f[t1_name]["eventWeight"].array(library="np"))
                if "eventWeight" in f[t2_name].keys():
                    VBF_yields = np.sum(f[t2_name]["eventWeight"].array(library="np"))
            
            with uproot.open(temp_file) as f:
                if "eventWeight" in f[target_name].keys():
                    total_yields = np.sum(f[target_name]["eventWeight"].array(library="np"))
        except Exception as e:
            print(f"[WARN] Could not calculate yields: {e}")
        
        # Replace/move temp file to final output path
        os.replace(temp_file, output_path)
        
        print(f"[OK] {output_path}: wrote {target_name} with {total_entries} entries (ggH+VBF)")
        if ggH_yields is not None and VBF_yields is not None and total_yields is not None:
            print(f"     ggH yields: {ggH_yields:.2f}, VBF yields: {VBF_yields:.2f}, Total: {total_yields:.2f}")
            
    except Exception as e:
        print(f"[ERROR] Writing to {output_path} failed: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
        return


def main():
    parser = argparse.ArgumentParser(description="Merge ggH and VBF trees into Dilep (uproot with chunked processing)")
    parser.add_argument('-i', '--input-dir', default='/eos/home-j/jiehan/root_mumu/skimmed_ntuples_ggH_v1', help='Directory containing ROOT files')
    parser.add_argument('-o', '--output-dir', default=None, help='Output directory for merged files (if not specified, overwrites input files)')
    parser.add_argument('-p', '--pattern', default='*.root', help='Glob pattern for files (default: *.root)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing m110To150_Dilep tree if present')
    parser.add_argument('--dry-run', action='store_true', help='List target files without modifying')
    parser.add_argument('--chunk-size', type=int, default=1000000, help='Number of entries to process per chunk (default: 1000000)')
    args = parser.parse_args()

    directory = args.input_dir
    if not os.path.isdir(directory):
        print(f"[ERROR] Input directory not found: {directory}")
        sys.exit(1)

    paths = sorted(glob.glob(os.path.join(directory, args.pattern)))
    if not paths:
        print(f"[WARN] No files matched pattern {args.pattern} in {directory}")
        return

    if args.output_dir:
        print(f"[INFO] Output directory: {args.output_dir}")
    else:
        print(f"[INFO] No output directory specified, will overwrite input files")
    
    print(f"[INFO] Found {len(paths)} files. Starting merge with chunk size {args.chunk_size}...")
    for fp in paths:
        merge_file(fp, output_dir=args.output_dir, dry_run=args.dry_run, overwrite=args.overwrite, chunk_size=args.chunk_size)

if __name__ == '__main__':
    main()
