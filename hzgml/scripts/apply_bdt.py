#!/usr/bin/env python
import copy
import os
from argparse import ArgumentParser
import json
import numpy as np
import pandas as pd
import uproot
# from root_pandas import *
import pickle
from sklearn.preprocessing import StandardScaler
from weighted_quantile_transformer import WeightedQuantileTransformer
import xgboost as xgb
from tqdm import tqdm
import logging
import gc  # Add garbage collection
from pdb import set_trace
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
import ROOT
ROOT.gErrorIgnoreLevel = ROOT.kError + 1
pd.options.mode.chained_assignment = None

try:
    import psutil  # For memory monitoring
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("WARNING: psutil not available, memory monitoring disabled")

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    print("WARNING: pyarrow not available, using pickle for temporary files")

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

def check_hadd_available():
    """Check if ROOT's hadd command is available."""
    import subprocess
    try:
        result = subprocess.run(['which', 'hadd'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

HADD_AVAILABLE = check_hadd_available()

def getArgs():
    """Get arguments from command line."""
    parser = ArgumentParser()
    #parser.add_argument('-c', '--config', action='store', nargs=2, default=['data/training_config_BDT.json', 'data/apply_config_BDT.json'], help='Region to process')
    parser.add_argument('-c', '--config', action='store', nargs=2, default=['data/training_config_BDT_Hmm_RunIII.json', 'data/apply_config_BDT.json'], help='Region to process')
    parser.add_argument('-i', '--inputFolder', action='store', default='/eos/user/q/qguo/vbfhmm/ml/2018/skimmed_ntuples_v4/', help='directory of training inputs')
    parser.add_argument('-t', '--inputTree', action='store', default='two_jet', help='input tree name')
    parser.add_argument('-m', '--modelFolder', action='store', default='models', help='directory of BDT models')
    parser.add_argument('-o', '--outputFolder', action='store', default='outputs', help='directory for outputs')
    parser.add_argument('-r', '--region', action='store', choices=['two_jet', 'one_jet', 'zero_jet', 'zero_to_one_jet', 'VH_ttH', 'all_jet', "ggH", "ggH_sep"], default='two_jet', help='Region to process')
    parser.add_argument('-cat', '--category', action='store', nargs='+', help='apply only for specific categories')

    parser.add_argument('-s', '--shield', action='store', type=int, default=-1, help='Which variables needs to be shielded')
    parser.add_argument('-a', '--add', action='store', type=int, default=-1, help='Which variables needs to be added')
    parser.add_argument('-F', '--FixSBH125', action='store_true', default=False, help='Fix the H mass to be 125GeV to get the scores')
    parser.add_argument('-y', '--year', action='store', default='', help='directory name')
    
    # [GPU Update] Added GPU argument
    parser.add_argument('-g', '--gpu', action='store_true', default=True, help='Use GPU for XGBoost inference')

    return parser.parse_args()

class ApplyXGBHandler(object):
    "Class for applying XGBoost"

    def __init__(self, configPath, region=''):

        print('===============================')
        print('  ApplyXGBHandler initialized')
        print('===============================')

        args=getArgs()
        self._shield = args.shield
        self._add = args.add
        self._FixSBH125 = args.FixSBH125
        self._year = args.year
        self._use_gpu = args.gpu  # [GPU Update] Store GPU flag

        self._region = region
        self._inputFolder = ''
        # self._inputTree = region if region else 'inclusive'
        self._inputTree = args.inputTree
        print("inputTree: ", self._inputTree)
        if self._use_gpu:
            print("XGB INFO: GPU Acceleration Enabled for Inference")
            
        #self._inputTree = 'two_jet'
        self._modelFolder = ''
        self._outputFolder = ''
        #self._outputFolder = f'models_{current_date}'
        self._chunksize = 500000
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
            # if (self._add>=0):
            #     configs["common"]["train_variables"].append(configs["common"]["+train_variables"][self._add])
   
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
            self.preselections = ['data.' + p for p in self.preselections]

    def setInputFolder(self, inputFolder):
        self._inputFolder = inputFolder

    def setModelFolder(self, modelFolder):
        self._modelFolder = modelFolder

    def setOutputFolder(self, outputFolder):
        self._outputFolder = outputFolder + f'_bdt_{current_date}'
        if self._year:  self._outputFolder = self._outputFolder + '_' + self._year
        if self._FixSBH125:  self._outputFolder += '_SB_HM125'

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
                    # Try to load with different formats in order of preference
                    model_path_base = '%s/BDT_%s_%d' % (self._modelFolder, model, i)
                    if os.path.exists(model_path_base + '.json'):
                        bst.load_model(model_path_base + '.json')
                    elif os.path.exists(model_path_base + '.ubj'):
                        bst.load_model(model_path_base + '.ubj')
                    elif os.path.exists(model_path_base + '.h5'):
                        bst.load_model(model_path_base + '.h5')
                    else:
                        raise FileNotFoundError(f"Model file not found: {model_path_base}.[json|ubj|h5]")
                    
                    # [GPU Update] Force model to use GPU predictor if enabled
                    if self._use_gpu:
                        try:
                            # Try modern XGBoost (2.0+) syntax
                            bst.set_param({"device": "cuda", "predictor": "gpu_predictor"})
                        except:
                            # Fallback for older XGBoost versions
                            try:
                                bst.set_param({"gpu_id": 0, "tree_method": "gpu_hist", "predictor": "gpu_predictor"})
                            except Exception as e:
                                print(f"XGB WARNING: Failed to set GPU parameters for model {model}_{i}: {e}")

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

    def applyBDT(self, category, scale=1, shift=0):
        outputbraches = copy.deepcopy(self._outbranches)
        branches = copy.deepcopy(self._branches)
        # branches += ["Z_sublead_lepton_pt", "gamma_mvaID_WP80", "gamma_mvaID_WPL"]
        # branches += ["eventWeight", "trg_single_mu24", "nmuons"]
        outputbraches += ["eventWeight", "trg_single_mu24", "nmuons"]
        # if category == "DYJetsToLL":
        #     branches.append('n_iso_photons')
        # if category != "data_fake" and category != "mc_true" and category != "mc_med":
        #     branches.append('gamma_mvaID_WP80')
        # if category == "data_fake" or category == "mc_true" or category == "mc_med":
        #     branches += ['weight_err']
        #     outputbraches += ['weight_err']
        # if category == "mc_true" or category == "mc_med":
        #     branches += ['tagger']
        #     outputbraches += ['tagger']

        # print(branches)
        # print(outputbraches)
        
        outputContainer = self._outputFolder + '/' + self._region
        print("outputContainer: ",outputContainer)
        output_path = outputContainer + '/%s.root' % category
        if not os.path.isdir(outputContainer): os.makedirs(outputContainer)
        if os.path.isfile(output_path): os.remove(output_path)

        f_list = []
        cat_folder = self._inputFolder + '/' 
        for f in os.listdir(cat_folder):
            #if f.endswith('{}_ml.root'.format(category)): f_list.append(cat_folder + '/' + f)
            if f.endswith('{}.root'.format(category)): f_list.append(cat_folder + '/' + f)

        print('-------------------------------------------------')
        for f in f_list: print('XGB INFO: Including sample: ', f)

        # Use ultra-low memory mode by default
        print("XGB INFO: Using ultra-low memory mode")
        self._process_files(f_list, output_path, branches, outputbraches, scale, shift, category)

    def _process_files(self, f_list, output_path, branches, outputbraches, scale, shift, category):
        """Process files with ultra-low memory usage."""
        import tempfile
        temp_files = []
        
        initial_memory = get_memory_usage()
        print(f"XGB INFO: Initial memory usage: {initial_memory:.2f} GB")
        
        # Process each file individually and save to temp files immediately
        for file_idx, filename in enumerate(tqdm(sorted(f_list), desc='XGB INFO: Applying BDTs to %s samples' % category, bar_format='{desc}: {percentage:3.0f}%|{bar:20}{r_bar}')):
            try:
                file = uproot.open(filename)
            except Exception as e:
                print('XGB ERROR: Failed to open file: ', filename)
                continue
            
            # Process each chunk immediately without batching
            chunk_count = 0
            tree_with_name = [name.split(';')[0] for name in file.keys() if self._inputTree in name]
            for name in tree_with_name:
                if name != self._inputTree:
                    print(f"XGB INFO: Skipping tree '{name}' (not matching '{self._inputTree}')")
                    continue
                # if self._inputTree not in name:
                #     print(f"XGB INFO: Skipping tree '{name}' ('{self._inputTree}' not in)")
                #     continue
                for data in file[name].iterate(library='pd', step_size=self._chunksize):
                    data = self.preselect(data)
                    
                    # Apply FixSBH125 if requested
                    if self._FixSBH125:
                        mask = ((data['diMufsr_rc_mass'] > 110) & (data['diMufsr_rc_mass'] < 115)) | ((data['diMufsr_rc_mass'] > 135) & (data['diMufsr_rc_mass'] < 150))
                        data.loc[mask, 'diMufsr_rc_mass'] = 125
                    
                    for i in range(4):
                        data_s = data[(data[self.randomIndex]-shift)%314159%4 == i]
                        if data_s.shape[0] == 0: continue
                        
                        data_o = data_s.copy()

                        for model in self.train_variables.keys():
                            x_Events = data_s[self.train_variables[model]]
                            
                            # [GPU Update] Pass nthread=-1 to let XGBoost handle concurrency, GPU context handles the rest
                            # If using GPU, the Booster param 'predictor':'gpu_predictor' set in loadModels handles the switch
                            dEvents = xgb.DMatrix(x_Events)
                            
                            scores = self.m_models[model][i].predict(dEvents)
                            if len(scores) > 0:
                                # Transformer usually runs on CPU (sklearn), so we don't change this
                                scores_t = self.m_tsfs[model][i].transform(scores.reshape(-1,1)).reshape(-1)
                            else:
                                scores_t = scores
                        
                            xgb_basename = self.models[model]
                            data_o[xgb_basename] = scores
                            data_o[xgb_basename+'_t'] = scores_t
                        
                        # Save immediately to temp ROOT file
                        if len(data_o) > 0:
                            # Remove index column if it exists
                            if "index" in data_o.columns:
                                data_o = data_o.drop('index', axis=1)
                            
                            temp_file = tempfile.NamedTemporaryFile(suffix='.root', delete=False)
                            temp_file.close()

                            with uproot.recreate(temp_file.name) as root_file:
                                # root_file[f"{name}_test"] = data_o
                                root_file[f"test"] = data_o
                            
                            temp_files.append(temp_file.name)
                        
                        # Clear immediately
                        del data_o
                        chunk_count += 1
                        
                        # Force garbage collection more frequently
                        if chunk_count % 4 == 0:
                            gc.collect()
                            if chunk_count % 20 == 0:
                                current_memory = get_memory_usage()
                                print(f"XGB INFO: Processed {chunk_count} chunks from file {file_idx+1}/{len(f_list)}, Memory: {current_memory:.2f} GB", end='\r', flush=True)
                    
                    # Clear chunk data
                    del data, data_s
                    gc.collect()
                
            file.close()
            gc.collect()
                
            current_memory = get_memory_usage()
            print(f"XGB INFO: Processed file {file_idx+1}/{len(f_list)}, Memory: {current_memory:.2f} GB, Temp files: {len(temp_files)}")
        
        # Combine temp files with minimal memory usage
        print(f"XGB INFO: Combining {len(temp_files)} temporary files...")
        self._combine_temp_files(temp_files, output_path)
        
        # Clean up
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
        
        final_memory = get_memory_usage()
        print(f"XGB INFO: Final memory usage: {final_memory:.2f} GB")
        
    def _combine_temp_files(self, temp_files, output_path):
        """Combine temporary ROOT files using ROOT hadd for optimal performance."""
        if not temp_files:
            return
        
        print(f"XGB INFO: Combining {len(temp_files)} temporary ROOT files...")
        
        # All files are already in ROOT format, just use hadd directly
        if HADD_AVAILABLE:
            print("XGB INFO: Using ROOT hadd to combine files...")
            hadd_command = f"hadd -f {output_path} " + " ".join(temp_files)
            
            import subprocess
            try:
                result = subprocess.run(hadd_command, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"XGB INFO: Successfully combined {len(temp_files)} files using hadd")
                else:
                    print(f"XGB ERROR: hadd failed: {result.stderr}")
                    # Fallback to manual combination
                    self._fallback_combine_files(temp_files, output_path)
            except Exception as e:
                print(f"XGB ERROR: Failed to run hadd: {e}")
                # Fallback to manual combination
                self._fallback_combine_files(temp_files, output_path)
        else:
            print("XGB WARNING: hadd not available, using manual combination...")
            self._fallback_combine_files(temp_files, output_path)
    
    def _fallback_combine_files(self, root_files, output_path):
        """Fallback method to combine ROOT files manually if hadd fails."""
        print("XGB INFO: Using fallback method to combine files...")
        
        if not root_files:
            return
        
        # Read first file to initialize
        with uproot.open(root_files[0]) as first_file:
            combined_data = first_file["test"].arrays(library='pd')
        
        # Append remaining files one by one
        for root_file in root_files[1:]:
            try:
                with uproot.open(root_file) as file:
                    data = file["test"].arrays(library='pd')
                    combined_data = pd.concat([combined_data, data], ignore_index=True, sort=False)
                    del data
                    gc.collect()
            except Exception as e:
                print(f"XGB WARNING: Failed to read {root_file}: {e}")
                continue
        
        # Write final result
        with uproot.recreate(output_path) as output_file:
            output_file["test"] = combined_data
        
        del combined_data
        gc.collect()


def main():

    args=getArgs()
    
    configPath = args.config
    xgb = ApplyXGBHandler(configPath, args.region)

    xgb.setInputFolder(args.inputFolder)
    xgb.setModelFolder(args.modelFolder)
    xgb.setOutputFolder(args.outputFolder)

    xgb.loadModels()
    xgb.loadTransformer()

    #with open('data/inputs_config_22EE_herwig.json') as f:
    #with open('data/inputs_config_22EE_v2.json') as f:
    with open('data/inputs_config_ggH.json') as f:
        config = json.load(f)
    sample_list = config['sample_list']

    for category in sample_list:
        if args.category and category not in args.category: continue
        xgb.applyBDT(category)

    return

if __name__ == '__main__':
    main()