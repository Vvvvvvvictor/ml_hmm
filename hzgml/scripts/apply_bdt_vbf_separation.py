#!/usr/bin/env python
import copy
import os
from argparse import ArgumentParser
import json
import numpy as np
import pandas as pd
import uproot
import pickle
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import xgboost as xgb
from tqdm import tqdm
import logging
import gc
from pdb import set_trace
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
import ROOT
ROOT.gErrorIgnoreLevel = ROOT.kError + 1
pd.options.mode.chained_assignment = None

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("WARNING: psutil not available, memory monitoring disabled")

from pdb import set_trace as bp

import os
from datetime import datetime

# Get the current date in the format MMDD
current_date = datetime.now().strftime("%m%d")

def get_memory_usage():
    """Get current memory usage in GB."""
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024 / 1024
    else:
        return 0.0

def getArgs():
    """Get arguments from command line."""
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', action='store', nargs=2, default=['data/training_config_BDT_VBFsep.json', 'data/apply_config_BDT.json'], help='Training and apply config files')
    parser.add_argument('-i', '--inputFolder', action='store', default='/eos/user/q/qguo/vbfhmm/ml/2018/skimmed_ntuples_v4/', help='directory of training inputs')
    parser.add_argument('-m', '--modelFolder', action='store', default='models', help='directory of BDT models')
    parser.add_argument('-o', '--outputFolder', action='store', default='', help='directory for outputs (if empty, will update files in place)')
    parser.add_argument('-r', '--region', action='store', choices=['two_jet', 'one_jet', 'zero_jet', 'zero_to_one_jet', 'VH_ttH', 'all_jet', "ggH", "VBFsep"], default='VBFsep', help='Region to process')
    parser.add_argument('-cat', '--category', action='store', nargs='+', help='apply only for specific categories')
    parser.add_argument('--vbf-sample', action='store', default='VBF', help='Name of VBF sample for threshold calculation')
    parser.add_argument('--ggf-sample', action='store', default='ggH', help='Name of ggF sample for threshold calculation')
    parser.add_argument('--vbf-purity', action='store', type=float, default=0.5, help='Target VBF purity for threshold (default: 0.5)')
    parser.add_argument('--selection', action='store', default='njets >= 2', help='Selection to apply before calculating scores for threshold determination')
    parser.add_argument('-s', '--shield', action='store', type=int, default=-1, help='Which variables needs to be shielded')
    parser.add_argument('-a', '--add', action='store', type=int, default=-1, help='Which variables needs to be added')
    parser.add_argument('-F', '--FixSBH125', action='store_true', default=False, help='Fix the H mass to be 125GeV to get the scores')
    parser.add_argument('-y', '--year', action='store', default='', help='directory name')

    return parser.parse_args()

class ApplyXGBVBFSeparation(object):
    """Class for applying XGBoost with VBF/ggF separation"""

    def __init__(self, configPath, region=''):

        print('===============================================')
        print('  ApplyXGBVBFSeparation initialized')
        print('===============================================')

        args = getArgs()
        self._shield = args.shield
        self._add = args.add
        self._FixSBH125 = args.FixSBH125
        self._year = args.year
        self.vbf_sample = args.vbf_sample
        self.ggf_sample = args.ggf_sample
        self.vbf_purity = args.vbf_purity
        self.selection = args.selection

        self._region = region
        self._inputFolder = ''
        self._inputTree = "m110To150_Dilep"
        print("inputTree: ", self._inputTree)
        self._modelFolder = ''
        self._outputFolder = ''
        self._chunksize = 500000
        self._batch_write_size = 500000  # Write to file every N events for large files
        self._category = []
        self._branches = []
        self._outbranches = []

        self.m_models = {}
        self.m_tsfs = {}

        self.train_variables = {}
        self.randomIndex = 'event'

        self.models = {}
        self.observables = []
        self.preselections = []

        self.readApplyConfig(configPath[1])
        self.readTrainConfig(configPath[0])
        self.arrangeBranches()
        self.arrangePreselections()
        
        # Store thresholds for VBF/ggF separation at different purities
        self.thresholds = {}  # {purity: threshold}
        self.purity_levels = [0.4, 0.5, 0.55, 0.7, 0.85]

    def readApplyConfig(self, configPath):
        """Read configuration file formated in json to extract information to fill TemplateMaker variables."""
        try:
            member_variables = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("_") and not attr.startswith('m_')]

            stream = open(configPath, 'r')
            configs = json.loads(stream.read())

            # read from the common settings
            config = configs["common"]
            for member in config.keys():
                if member in member_variables:
                    setattr(self, member, config[member])

            # read from the region specific settings
            if self._region:
                config = configs[self._region]
                for member in config.keys():
                    if member in member_variables:
                        setattr(self, member, config[member])
                if '+preselections' in config.keys():
                    self.preselections += config['+preselections']
                if '+observables' in config.keys():
                    self.observables += config['+observables']

        except Exception as e:
            logging.error("Error reading apply configuration '{config}'".format(config=configPath))
            logging.error(e)

    def readTrainConfig(self, configPath):
        
        try:
            stream = open(configPath, 'r')
            configs = json.loads(stream.read())
   
            config = configs["common"]

            if 'randomIndex' in config.keys(): self.randomIndex = config['randomIndex']
 
            if self.models:
                for model in self.models:
    
                    # read from the common settings
                    config = configs["common"]
                    if 'train_variables' in config.keys(): self.train_variables[model] = config['train_variables'][:]
    
                    # read from the region specific settings
                    if model in configs.keys():
                        config = configs[model]
                        if self._add >= 0:
                            config["+train_variables"].append(config["test_variables"][self._add])
                        if 'train_variables' in config.keys(): self.train_variables[model] = config['train_variables'][:]
                        if '+train_variables' in config.keys(): self.train_variables[model] += config['+train_variables']
                        if self._shield >= 0:
                            self.train_variables[model].pop(self._shield)

                        print("\n\n")
                        print(self.train_variables[model])
                        print(len(self.train_variables[model]))
                        print("\n\n")

        except Exception as e:
            logging.error("Error reading training configuration '{config}'".format(config=configPath))
            logging.error(e)

    def arrangeBranches(self):

        self._branches = set()
        for model in self.models:
            self._branches = self._branches | set(self.train_variables[model])

        self._branches = self._branches | set([self.randomIndex]) | set([p.split()[0] for p in self.preselections]) | set(self.observables)
        self._branches = list(self._branches)

        for model in self.models:
            self.train_variables[model] = [x.replace('noexpand:', '') for x in self.train_variables[model]]
        self.preselections = [x.replace('noexpand:', '') for x in self.preselections]
        self.randomIndex = self.randomIndex.replace('noexpand:', '')

        self._outbranches = [branch for branch in self._branches if 'noexpand' not in branch]

    def arrangePreselections(self):

        if self.preselections:
            print("XGB INFO: Preselections applied:")
            for p in self.preselections:
                print(f"  {p}")
            self.preselections = ['data.' + p for p in self.preselections]

    def setInputFolder(self, inputFolder):
        self._inputFolder = inputFolder

    def setModelFolder(self, modelFolder):
        self._modelFolder = modelFolder

    def setOutputFolder(self, outputFolder):
        if outputFolder:
            self._outputFolder = outputFolder + f'_vbfsep_{current_date}'
            if self._year:  self._outputFolder = self._outputFolder + '_' + self._year
            if self._FixSBH125:  self._outputFolder += '_SB_HM125'
        else:
            # Empty means in-place update
            self._outputFolder = ''

    def preselect(self, data):

        for p in self.preselections:
            data = data[eval(p)]

        return data

    def loadModels(self):

        if self.models:
            for model in self.models:
                print('XGB INFO: Loading BDT model: ', model)
                self.m_models[model] = []
                for i in range(4):
                    bst = xgb.Booster()
                    model_path_base = '%s/BDT_%s_%d' % (self._modelFolder, model, i)
                    if os.path.exists(model_path_base + '.json'):
                        bst.load_model(model_path_base + '.json')
                    elif os.path.exists(model_path_base + '.ubj'):
                        bst.load_model(model_path_base + '.ubj')
                    elif os.path.exists(model_path_base + '.h5'):
                        bst.load_model(model_path_base + '.h5')
                    else:
                        raise FileNotFoundError(f"Model file not found: {model_path_base}.[json|ubj|h5]")
                    self.m_models[model].append(bst)
                    del bst

    def loadTransformer(self):
        
        if self.models:
            for model in self.models:
                print('XGB INFO: Loading score transformer for model: ', model)
                self.m_tsfs[model] = []
                for i in range(4):
                    tsf = pickle.load(open('%s/BDT_tsf_%s_%d.pkl'%(self._modelFolder, model, i), "rb" ), encoding = 'latin1' )
                    self.m_tsfs[model].append(tsf)

    def calculate_thresholds(self):
        """Calculate thresholds for multiple VBF purities: 0.55, 0.7, 0.85."""
        print(f'\n{"="*60}')
        print(f'Calculating VBF/ggF separation thresholds')
        print(f'Target VBF purities: {self.purity_levels}')
        print(f'{"="*60}\n')
        
        # Get VBF and ggF scores
        vbf_scores = []
        ggf_scores = []
        
        for sample_name in [self.vbf_sample, self.ggf_sample]:
            sample_scores = []
            cat_folder = self._inputFolder + '/'
            f_list = []
            for f in os.listdir(cat_folder):
                if f.endswith('{}.root'.format(sample_name)):
                    f_list.append(cat_folder + '/' + f)
                    
            if not f_list:
                print(f'WARNING: No files found for sample {sample_name}')
                continue
            
            print(f'Processing {sample_name} for threshold calculation...')
            
            for filename in tqdm(sorted(f_list), desc=f'Loading {sample_name}'):
                try:
                    file = uproot.open(filename)
                except Exception as e:
                    print(f'ERROR: Failed to open file: {filename}')
                    continue
                
                for data in file[self._inputTree].iterate(library='pd', step_size=self._chunksize):
                    data = self.preselect(data)
                    
                    if self._FixSBH125:
                        mask = ((data['diMufsr_rc_mass'] > 110) & (data['diMufsr_rc_mass'] < 115)) | ((data['diMufsr_rc_mass'] > 135) & (data['diMufsr_rc_mass'] < 150))
                        data.loc[mask, 'diMufsr_rc_mass'] = 125
                    
                    for i in range(4):
                        data_s = data[(data[self.randomIndex])%314159%4 == i]
                        data_s = data_s.query(self.selection)
                        if data_s.shape[0] == 0: continue
                        
                        for model in self.train_variables.keys():
                            x_Events = data_s[self.train_variables[model]]
                            dEvents = xgb.DMatrix(x_Events)
                            scores = self.m_models[model][i].predict(dEvents)
                            data_s['vbf_score'] = scores
                            sample_scores.append(data_s[['vbf_score', 'eventWeight']])
                        
                        del data_s
                    
                    del data
                    gc.collect()
                
                file.close()
                gc.collect()
            
            # Convert to numpy array
            if sample_scores:
                sample_scores_array = pd.concat(sample_scores)
                if sample_name == self.vbf_sample:
                    vbf_scores = sample_scores_array
                else:
                    ggf_scores = sample_scores_array
    
        if len(vbf_scores) == 0 or len(ggf_scores) == 0:
            raise ValueError("Failed to collect scores from VBF or ggF samples")
        
        print(f'\nCollected {len(vbf_scores)} VBF scores and {len(ggf_scores)} ggF scores')
        
        # Calculate histograms for VBF and all scores
        bins = np.arange(0, 1.0001, 0.0001)
        vbf_hist, _ = np.histogram(vbf_scores['vbf_score'], bins=bins, weights=vbf_scores['eventWeight'])
        all_scores = pd.concat([vbf_scores, ggf_scores])
        all_hist, _ = np.histogram(all_scores['vbf_score'], bins=bins, weights=all_scores['eventWeight'])
        
        # Calculate cumulative sum from right to left (events with score >= threshold)
        vbf_cumsum_ge = np.cumsum(vbf_hist[::-1])[::-1]
        all_cumsum_ge = np.cumsum(all_hist[::-1])[::-1]
        
        # Calculate purity at each bin
        with np.errstate(divide='ignore', invalid='ignore'):
            purity_arr = np.where(all_cumsum_ge > 0, 
                                  vbf_cumsum_ge / all_cumsum_ge, 
                                  1.0)
        
        # Calculate threshold for each target purity
        for target_purity in self.purity_levels:
            valid_mask = purity_arr >= target_purity
            
            if np.any(valid_mask):
                valid_indices = np.where(valid_mask)[0]
                best_idx = valid_indices[1]
                best_threshold = bins[best_idx]
                best_purity = purity_arr[best_idx]
            else:
                print(f'WARNING: Could not find threshold with purity >= {target_purity}')
                print('Using threshold with highest achievable purity')
                best_idx = np.argmax(purity_arr)
                best_threshold = bins[best_idx]
                best_purity = purity_arr[best_idx]
            
            self.thresholds[target_purity] = best_threshold
            
            # Calculate statistics for this purity
            threshold_val = best_threshold
            n_vbf_high = np.sum(vbf_scores.query('vbf_score >= @threshold_val')['eventWeight'])
            n_vbf_low = np.sum(vbf_scores.query('vbf_score < @threshold_val')['eventWeight'])
            n_ggf_high = np.sum(ggf_scores.query('vbf_score >= @threshold_val')['eventWeight'])
            n_ggf_low = np.sum(ggf_scores.query('vbf_score < @threshold_val')['eventWeight'])

            print(f'\n{"="*60}')
            print(f'Purity {target_purity:.2f} - Threshold: {best_threshold:.6f}')
            print(f'Actual VBF purity above threshold: {best_purity:.4f}')
            print(f'\nVBF sample:')
            print(f'  - High score (VBFlike): {n_vbf_high:.2f} events ({100*n_vbf_high/(n_vbf_high+n_vbf_low):.2f}%)')
            print(f'  - Low score (ggHlike): {n_vbf_low:.2f} events ({100*n_vbf_low/(n_vbf_high+n_vbf_low):.2f}%)')
            print(f'\nggF sample:')
            print(f'  - High score (VBFlike): {n_ggf_high:.2f} events ({100*n_ggf_high/(n_ggf_high+n_ggf_low):.2f}%)')
            print(f'  - Low score (ggHlike): {n_ggf_low:.2f} events ({100*n_ggf_low/(n_ggf_high+n_ggf_low):.2f}%)')
            print(f'{"="*60}\n')

    def applyBDT(self, category, scale=1, shift=0):
        """Apply BDT and add VBFlike/ggHlike trees to the original ROOT file."""
        
        if not self.thresholds:
            raise ValueError("Thresholds not calculated. Call calculate_thresholds() first.")
        
        outputbraches = copy.deepcopy(self._outbranches)
        branches = copy.deepcopy(self._branches)
        outputbraches += ["eventWeight", "trg_single_mu24", "nmuons"]
        
        cat_folder = self._inputFolder + '/'
        f_list = []
        for f in os.listdir(cat_folder):
            if f.endswith('{}.root'.format(category)):
                f_list.append(cat_folder + '/' + f)

        print('-------------------------------------------------')
        for f in f_list: print('XGB INFO: Including sample: ', f)

        # Process each file separately
        for filename in f_list:
            self._process_single_file_with_separation(filename, branches, outputbraches, scale, shift, category)

    def _process_single_file_with_separation(self, filename, branches, outputbraches, scale, shift, category):
        """Process a single file and add VBFlike/ggHlike trees for multiple purities."""
        try:
            file = uproot.open(filename)
        except Exception as e:
            print(f'XGB ERROR: Failed to open file: {filename}')
            return

        chunksize = self._chunksize

        # Use temporary ROOT files to accumulate data for large files
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix='xgb_vbfsep_')
        temp_files = {
            'all': [],
        }
        for purity in self.purity_levels:
            temp_files[f'vbf_{purity}'] = []
            temp_files[f'ggh_{purity}'] = []
        
        event_counts = {p: {'vbf': 0, 'ggh': 0} for p in self.purity_levels}
        
        print(f'XGB INFO: Processing {filename}...')
        print(f'XGB INFO: Using temporary directory: {temp_dir}')
        chunk_count = 0
        batch_size = 5  # Write to temp file every 5 chunks

        vbflike_batch = {p: [] for p in self.purity_levels}
        gghlike_batch = {p: [] for p in self.purity_levels}
        all_batch = []

        for data in file[self._inputTree].iterate(library='pd', step_size=chunksize):
            chunk_count += 1
            if chunk_count % 5 == 0:
                print(f'  - Processed {chunk_count} chunks, memory: {get_memory_usage():.2f} GB')
            
            data = self.preselect(data)

            if self._FixSBH125:
                mask = ((data['diMufsr_rc_mass'] > 110) & (data['diMufsr_rc_mass'] < 115)) | ((data['diMufsr_rc_mass'] > 135) & (data['diMufsr_rc_mass'] < 150))
                data.loc[mask, 'diMufsr_rc_mass'] = 125

            for i in range(4):
                data_s = data[(data[self.randomIndex]-shift)%314159%4 == i]
                if data_s.shape[0] == 0: continue

                data_o = data_s.copy()

                # 只用第一个model作为vbf_score
                model = list(self.train_variables.keys())[0]
                x_Events = data_s[self.train_variables[model]]
                dEvents = xgb.DMatrix(x_Events)
                scores = self.m_models[model][i].predict(dEvents)
                if len(scores) > 0:
                    scores_t = self.m_tsfs[model][i].transform(scores.reshape(-1,1)).reshape(-1)
                else:
                    scores_t = scores

                # 保存分数
                data_o['vbf_score'] = scores
                data_o['vbf_score_t'] = scores_t

                # 兼容原有接口
                xgb_basename = self.models[model]
                data_o[xgb_basename] = scores
                data_o[xgb_basename+'_t'] = scores_t

                # Separate by threshold for each purity level (only for 2jet events)
                is_2jet = data_o['njets'] >= 2
                for purity in self.purity_levels:
                    threshold = self.thresholds[purity]
                    # VBFlike: only for 2jet events with high score
                    mask_vbflike = is_2jet & (scores >= threshold)
                    # ggHlike: low score regardless of njets
                    mask_gghlike = (scores < threshold) | (~is_2jet)

                    if np.any(mask_vbflike):
                        vbflike_batch[purity].append(data_o[mask_vbflike].copy())
                        event_counts[purity]['vbf'] += np.sum(data_o[mask_vbflike]['eventWeight'])
                    if np.any(mask_gghlike):
                        gghlike_batch[purity].append(data_o[mask_gghlike].copy())
                        event_counts[purity]['ggh'] += np.sum(data_o[mask_gghlike]['eventWeight'])

                # 所有事件都加分数
                all_batch.append(data_o.copy())

                del data_o, data_s, dEvents
                gc.collect()
            
            del data
            gc.collect()

            # Write batch to temporary files every batch_size chunks
            if chunk_count % batch_size == 0:
                self._write_temp_batch(temp_dir, temp_files, vbflike_batch, gghlike_batch, all_batch, chunk_count)
                # Clear batches
                vbflike_batch = {p: [] for p in self.purity_levels}
                gghlike_batch = {p: [] for p in self.purity_levels}
                all_batch = []
                gc.collect()

        file.close()

        # Write remaining batch
        if all_batch:
            self._write_temp_batch(temp_dir, temp_files, vbflike_batch, gghlike_batch, all_batch, chunk_count)
            vbflike_batch = {p: [] for p in self.purity_levels}
            gghlike_batch = {p: [] for p in self.purity_levels}
            all_batch = []
            gc.collect()

        print(f'XGB INFO: Finished reading file: {filename}')
        print(f'XGB INFO: Purity 0.55: vbflike events = {event_counts[0.55]["vbf"]:.2f}, ggHlike events = {event_counts[0.55]["ggh"]:.2f}')
        print(f'XGB INFO: Purity 0.70: vbflike events = {event_counts[0.7]["vbf"]:.2f}, ggHlike events = {event_counts[0.7]["ggh"]:.2f}')
        print(f'XGB INFO: Purity 0.85: vbflike events = {event_counts[0.85]["vbf"]:.2f}, ggHlike events = {event_counts[0.85]["ggh"]:.2f}')
        
        # Merge temporary ROOT files using hadd
        print(f'XGB INFO: Merging temporary files using hadd...')
        merged_files = {}
        
        for purity in self.purity_levels:
            purity_str = str(int(purity * 100)).zfill(3)
            
            # Merge VBFlike files
            vbf_key = f'vbf_{purity}'
            if temp_files[vbf_key]:
                merged_vbf = os.path.join(temp_dir, f'merged_VBFlike_{purity_str}.root')
                self._hadd_files(merged_vbf, temp_files[vbf_key], f'VBFlike_{purity_str}')
                merged_files[f'VBFlike_{purity_str}'] = merged_vbf
                print(f'  - Merged VBFlike_{purity_str}')
            else:
                merged_files[f'VBFlike_{purity_str}'] = None

            # Merge ggHlike files
            ggh_key = f'ggh_{purity}'
            if temp_files[ggh_key]:
                merged_ggh = os.path.join(temp_dir, f'merged_ggHlike_{purity_str}.root')
                self._hadd_files(merged_ggh, temp_files[ggh_key], f'ggHlike_{purity_str}')
                merged_files[f'ggHlike_{purity_str}'] = merged_ggh
                print(f'  - Merged ggHlike_{purity_str}')
            else:
                merged_files[f'ggHlike_{purity_str}'] = None
        
        # Merge all events
        if temp_files['all']:
            merged_all = os.path.join(temp_dir, 'merged_all.root')
            self._hadd_files(merged_all, temp_files['all'], f'{self._inputTree}')
            merged_files['all'] = merged_all
            print(f'  - Merged {self._inputTree}')
        else:
            merged_files['all'] = None

        gc.collect()

        # Add merged trees to final ROOT file
        print(f'XGB INFO: Adding trees to final file...')
        self._add_merged_trees_to_file(filename, merged_files)
        
        print(f'XGB INFO: Successfully updated {filename}')
        for purity in self.purity_levels:
            purity_str = str(int(purity * 100)).zfill(3)
            print(f'  - Added VBFlike_{purity_str} tree')
            print(f'  - Added ggHlike_{purity_str} tree')
        print(f'  - Updated original tree with vbf_score')

        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir)
        print(f'XGB INFO: Cleaned up temporary directory')

        del merged_files, temp_files
        gc.collect()

    def _write_temp_batch(self, temp_dir, temp_files, vbflike_batch, gghlike_batch, all_batch, chunk_count):
        """Write current batch to temporary ROOT files."""
        import os
        
        # Write VBFlike batches
        for purity in self.purity_levels:
            purity_str = str(int(purity * 100)).zfill(3)
            
            if vbflike_batch[purity]:
                df = pd.concat(vbflike_batch[purity], ignore_index=True)
                if "index" in df.columns:
                    df = df.drop('index', axis=1)
                temp_file = os.path.join(temp_dir, f'vbf_{purity_str}_{chunk_count}.root')
                tree_name = f'VBFlike_{purity_str}'
                with uproot.recreate(temp_file) as f:
                    f[tree_name] = df
                temp_files[f'vbf_{purity}'].append(temp_file)
                del df

            if gghlike_batch[purity]:
                df = pd.concat(gghlike_batch[purity], ignore_index=True)
                if "index" in df.columns:
                    df = df.drop('index', axis=1)
                temp_file = os.path.join(temp_dir, f'ggh_{purity_str}_{chunk_count}.root')
                tree_name = f'ggHlike_{purity_str}'
                with uproot.recreate(temp_file) as f:
                    f[tree_name] = df
                temp_files[f'ggh_{purity}'].append(temp_file)
                del df

        # Write all events batch
        if all_batch:
            df = pd.concat(all_batch, ignore_index=True)
            if "index" in df.columns:
                df = df.drop('index', axis=1)
            temp_file = os.path.join(temp_dir, f'all_{chunk_count}.root')
            tree_name = f'{self._inputTree}_with_vbfscore'
            with uproot.recreate(temp_file) as f:
                f[tree_name] = df
            temp_files['all'].append(temp_file)
            del df

        gc.collect()

    def _hadd_files(self, output_file, input_files, tree_name):
        """Use ROOT's hadd to merge multiple ROOT files."""
        if not input_files:
            return
        
        # Build hadd command
        cmd = ['hadd', '-f', output_file] + input_files
        
        import subprocess
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            print(f'WARNING: hadd failed for {tree_name}')
            print(f'STDERR: {result.stderr}')
        
        # Clean up individual temp files after merging
        for f in input_files:
            try:
                os.remove(f)
            except:
                pass

    def _add_merged_trees_to_file(self, filename, merged_files):
        """Add merged trees from temporary ROOT files to the final ROOT file."""
        # First, copy all original trees to a temporary file
        import shutil
        temp_original = filename + '.original.tmp'
        shutil.copy2(filename, temp_original)
        
        root_file = ROOT.TFile.Open(filename, "recreate")
        if not root_file or root_file.IsZombie():
            print(f'ERROR: Cannot open file {filename} for updating')
            return

        # Collect tree names that will be updated/added
        trees_to_update = set()
        if merged_files.get('all'):
            trees_to_update.add(self._inputTree)
        for purity in self.purity_levels:
            purity_str = str(int(purity * 100)).zfill(3)
            if merged_files.get(f'VBFlike_{purity_str}'):
                trees_to_update.add(f'VBFlike_{purity_str}')
            if merged_files.get(f'ggHlike_{purity_str}'):
                trees_to_update.add(f'ggHlike_{purity_str}')

        # Copy all original trees from the backup (except those we will update)
        print(f'  - Copying original trees from input file...')
        original_file = ROOT.TFile.Open(temp_original, "READ")
        if original_file and not original_file.IsZombie():
            for key in original_file.GetListOfKeys():
                obj_name = key.GetName()
                # Skip trees that will be updated
                if obj_name in trees_to_update:
                    print(f'    - Skipping {obj_name} (will be updated)')
                    continue
                    
                obj = key.ReadObj()
                if obj.InheritsFrom("TTree"):
                    root_file.cd()
                    tree_copy = obj.CloneTree()
                    tree_copy.Write(obj_name, ROOT.TObject.kOverwrite)
                    print(f'    - Copied original tree: {obj_name}')
                    del tree_copy
            original_file.Close()
        
        # Add updated all events tree (original tree with vbf_score added)
        if merged_files.get('all'):
            tree_name = f"{self._inputTree}"
            print(f'  - Adding/Updating {tree_name} tree with vbf_score')
            
            temp_root = ROOT.TFile.Open(merged_files['all'], "READ")
            temp_tree_name = f"{self._inputTree}_with_vbfscore"
            tree_all = temp_root.Get(temp_tree_name)
            if tree_all:
                root_file.cd()
                tree_all_copy = tree_all.CloneTree()
                tree_all_copy.Write(tree_name, ROOT.TObject.kOverwrite)
                del tree_all_copy
            temp_root.Close()
            gc.collect()

        # Add VBFlike and ggHlike trees for each purity level
        for purity in self.purity_levels:
            purity_str = str(int(purity * 100)).zfill(3)
            
            # Add VBFlike tree
            vbf_key = f'VBFlike_{purity_str}'
            if merged_files.get(vbf_key):
                tree_name = vbf_key
                print(f'  - Adding/Updating {tree_name} tree')
                
                temp_root = ROOT.TFile.Open(merged_files[vbf_key], "READ")
                tree_vbf = temp_root.Get(tree_name)
                if tree_vbf:
                    root_file.cd()
                    tree_vbf_copy = tree_vbf.CloneTree()
                    tree_vbf_copy.Write(tree_name, ROOT.TObject.kOverwrite)
                    del tree_vbf_copy
                temp_root.Close()
                gc.collect()

            # Add ggHlike tree
            ggh_key = f'ggHlike_{purity_str}'
            if merged_files.get(ggh_key):
                tree_name = ggh_key
                print(f'  - Adding/Updating {tree_name} tree')
                
                temp_root = ROOT.TFile.Open(merged_files[ggh_key], "READ")
                tree_ggh = temp_root.Get(tree_name)
                if tree_ggh:
                    root_file.cd()
                    tree_ggh_copy = tree_ggh.CloneTree()
                    tree_ggh_copy.Write(tree_name, ROOT.TObject.kOverwrite)
                    del tree_ggh_copy
                temp_root.Close()
                gc.collect()

        root_file.Close()
        
        # Clean up temporary backup file
        try:
            os.remove(temp_original)
        except:
            pass
        
        print(f'  - File {filename} successfully updated')


def main():

    args = getArgs()
    
    configPath = args.config
    xgb = ApplyXGBVBFSeparation(configPath, args.region)

    xgb.setInputFolder(args.inputFolder)
    xgb.setModelFolder(args.modelFolder)
    xgb.setOutputFolder(args.outputFolder)

    xgb.loadModels()
    xgb.loadTransformer()
    
    # Calculate thresholds for multiple purities using VBF and ggF samples
    xgb.calculate_thresholds()

    # Load sample list
    with open('data/inputs_config_ggH.json') as f:
        config = json.load(f)
    sample_list = config['sample_list']

    # Apply to all samples
    for category in sample_list:
        if args.category and category not in args.category: continue
        xgb.applyBDT(category)

    print('\n' + '='*60)
    print('Processing completed!')
    print('='*60)

    return

if __name__ == '__main__':
    main()
