#!/usr/bin/env python
"""Apply NN VBF 3-class models to ROOT files"""
from argparse import ArgumentParser
import os
import sys
import json
import copy
import pickle
import uproot
import numpy as np
from sklearn.preprocessing import StandardScaler
from weighted_quantile_transformer import WeightedQuantileTransformer
import logging
from tqdm import tqdm
import gc
from pdb import set_trace
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
import ROOT
import pandas as pd
ROOT.gErrorIgnoreLevel = ROOT.kError + 1
pd.options.mode.chained_assignment = None

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("WARNING: psutil not available, memory monitoring disabled")

try:
    import pyarrow
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    print("WARNING: pyarrow not available, using pickle for temporary files")

from pdb import set_trace as bp
import tempfile
import subprocess
from datetime import datetime

# TensorFlow and Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import load_model

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
    try:
        result = subprocess.run(['which', 'hadd'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

HADD_AVAILABLE = check_hadd_available()

def getArgs():
    """Get arguments from command line."""
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', action='store', nargs=2, 
                        default=['data/training_config_NN_VBF_3class.json', 'data/apply_config_NN.json'], 
                        help='Config files: training and apply')
    parser.add_argument('-i', '--inputFolder', action='store', 
                        default='/eos/user/q/qguo/vbfhmm/ml/2018/skimmed_ntuples_v4/', 
                        help='directory of training inputs')
    parser.add_argument('-m', '--modelFolder', action='store', default='models_VBF_3class', 
                        help='directory of NN models')
    parser.add_argument('-o', '--outputFolder', action='store', default='outputs', 
                        help='directory for outputs')
    parser.add_argument('-r', '--region', action='store', 
                        choices=['two_jet', 'one_jet', 'zero_jet', 'zero_to_one_jet', 'VH_ttH', 'all_jet', "ggH", "VBF", "VBFsep"], 
                        default='two_jet', help='Region to process')
    parser.add_argument('-cat', '--category', action='store', nargs='+', 
                        help='apply only for specific categories')
    parser.add_argument('--tree-name', action='store', default='two_jet_m110To150', 
                        help='Name of the tree in ROOT files')
    parser.add_argument('-s', '--shield', action='store', type=int, default=-1, 
                        help='Which variables needs to be shielded')
    parser.add_argument('-a', '--add', action='store', type=int, default=-1, 
                        help='Which variables needs to be added')
    parser.add_argument('-F', '--FixSBH125', action='store_true', default=False, 
                        help='Fix the H mass to be 125GeV to get the scores')
    parser.add_argument('-y', '--year', action='store', default='', help='directory name')

    return parser.parse_args()

class Apply3ClassNNHandler(object):
    "Class for applying 3-class NN models"

    def __init__(self, configPath, region=''):

        print('===============================')
        print('  Apply3ClassNNHandler initialized')
        print('===============================')

        args=getArgs()
        self._shield = args.shield
        self._add = args.add
        self._FixSBH125 = args.FixSBH125
        self._year = args.year

        self._region = region
        self._inputFolder = ''
        self._inputTree = args.tree_name if args.tree_name else 'm110To150'
        print("inputTree: ", self._inputTree)
        self._modelFolder = ''
        self._outputFolder = ''
        self._chunksize = 500000
        self._category = []
        self._branches = []
        self._outbranches = []

        self.m_models = {}
        self.m_scalers = {}
        self.m_tsfs = {}

        self.train_variables = {}
        self.randomIndex = 'event'

        self.models = {}
        self.observables = []
        self.preselections = []
        
        # For hierarchical model: first 3 variables are mass, rest are topology
        self.n_mass_features = 3

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
                if member in member_variables: setattr(self, member, config[member])

            # read from the region specific settings
            if self._region:
                config = configs[self._region]
                for member in config.keys():
                    if member.startswith('#'): continue
                    if member.startswith('+'):
                        member_name = member[1:]
                        if member_name in member_variables:
                            getattr(self, member_name).extend(config[member])
                    else:
                        if member in member_variables: setattr(self, member, config[member])

        except Exception as e:
            logging.error("Error reading apply configuration '{config}'".format(config=configPath))
            logging.error(e)

    def readTrainConfig(self, configPath):
        """Read training configuration to get training variables."""
        try:
            stream = open(configPath, 'r')
            configs = json.loads(stream.read())
   
            config = configs["common"]

            if 'randomIndex' in config.keys(): self.randomIndex = config['randomIndex']
 
            if self.models:
                for model in self.models:
                    self.train_variables[model] = config['train_variables']
                    if self._region in configs.keys():
                        if 'train_variables' in configs[self._region].keys():
                            self.train_variables[model] = configs[self._region]['train_variables']

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
        self._outputFolder = outputFolder + f'_3class_{current_date}'
        if self._year:  self._outputFolder = self._outputFolder + '_' + self._year
        if self._FixSBH125:  self._outputFolder += '_SB_HM125'

    def preselect(self, data):
        for p in self.preselections:
            print("Applying preselection: ", p)
            data = data[eval(p)]
        return data

    def loadModels(self):
        """Load Keras NN 3-class models."""
        if self.models:
            for model in self.models:
                print('NN INFO: Loading NN 3-class model: ', model)
                self.m_models[model] = []
                for i in range(4):
                    model_path = '%s/model_fold_%d.keras' % (self._modelFolder, i)
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f"Model file not found: {model_path}")
                    keras_model = load_model(model_path)
                    self.m_models[model].append(keras_model)
                    print(f'  Loaded fold {i}: {model_path}')

    def loadScaler(self):
        """Load StandardScaler for each fold."""
        if self.models:
            for model in self.models:
                print('NN INFO: Loading scaler for model: ', model)
                self.m_scalers[model] = []
                for i in range(4):
                    scaler_path = '%s/scaler_fold_%d.pkl' % (self._modelFolder, i)
                    if not os.path.exists(scaler_path):
                        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
                    scaler = pickle.load(open(scaler_path, "rb"))
                    self.m_scalers[model].append(scaler)
                    print(f'  Loaded fold {i}: {scaler_path}')
                
                # Try to load the two derived score transformers if present
                self.m_tsfs[model] = []
                for i in range(4):
                    fold_tsfs = {}
                    
                    # Load vbf_score transformer (VBF - ggF)
                    vbf_tsf_path = '%s/vbf_score_tsf_fold_%d.pkl' % (self._modelFolder, i)
                    if os.path.exists(vbf_tsf_path):
                        try:
                            tsf = pickle.load(open(vbf_tsf_path, 'rb'))
                            print(f'  Loaded vbf_score transformer for fold {i}: {vbf_tsf_path}')
                            fold_tsfs['vbf_score_tsf'] = tsf
                        except Exception as e:
                            print(f'NN WARNING: Failed to load vbf_score transformer {vbf_tsf_path}: {e}')
                            fold_tsfs['vbf_score_tsf'] = None
                    else:
                        fold_tsfs['vbf_score_tsf'] = None
                    
                    # Load bdt_score transformer ((VBF + ggF) - bkg)
                    bdt_tsf_path = '%s/bdt_score_tsf_fold_%d.pkl' % (self._modelFolder, i)
                    if os.path.exists(bdt_tsf_path):
                        try:
                            tsf = pickle.load(open(bdt_tsf_path, 'rb'))
                            print(f'  Loaded bdt_score transformer for fold {i}: {bdt_tsf_path}')
                            fold_tsfs['bdt_score_tsf'] = tsf
                        except Exception as e:
                            print(f'NN WARNING: Failed to load bdt_score transformer {bdt_tsf_path}: {e}')
                            fold_tsfs['bdt_score_tsf'] = None
                    else:
                        fold_tsfs['bdt_score_tsf'] = None
                    
                    self.m_tsfs[model].append(fold_tsfs)

    def applyNN(self, category, scale=1, shift=0):
        """Apply 3-class NN models to category samples."""
        outputbraches = copy.deepcopy(self._outbranches)
        branches = copy.deepcopy(self._branches)
        outputbraches += ["eventWeight", "trg_single_mu24", "nmuons"]
        
        outputContainer = self._outputFolder + '/' + self._region
        print("outputContainer: ",outputContainer)
        output_path = outputContainer + '/%s.root' % category
        if not os.path.isdir(outputContainer): os.makedirs(outputContainer)
        if os.path.isfile(output_path): os.remove(output_path)

        f_list = []
        cat_folder = self._inputFolder + '/' 
        for f in os.listdir(cat_folder):
            if f.endswith('{}.root'.format(category)): 
                f_list.append(cat_folder + '/' + f)

        print('-------------------------------------------------')
        for f in f_list: print('NN INFO: Including sample: ', f)

        # Use ultra-low memory mode by default
        print("NN INFO: Using ultra-low memory mode")
        self._process_files(f_list, output_path, branches, outputbraches, scale, shift, category)

    def _process_files(self, f_list, output_path, branches, outputbraches, scale, shift, category):
        """Process files with ultra-low memory usage."""
        # Temp files to store results
        temp_files = []
        
        initial_memory = get_memory_usage()
        print(f"NN INFO: Initial memory usage: {initial_memory:.2f} GB")
        
        # Process each file individually and save to temp files immediately
        for file_idx, filename in enumerate(tqdm(sorted(f_list), desc='NN INFO: Applying 3-class NNs to %s samples' % category, bar_format='{desc}: {percentage:3.0f}%|{bar:20}{r_bar}')):
            try:
                file = uproot.open(filename)
            except Exception as e:
                print('NN ERROR: Failed to open file: ', filename)
                continue
            
            # Process each chunk immediately without batching
            chunk_count = 0
            for data in file[self._inputTree].iterate(library='pd', step_size=self._chunksize):
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
                        
                        # Scale the input using the scaler
                        x_Events_scaled = self.m_scalers[model][i].transform(x_Events)
                        
                        # Split into mass and topology features for hierarchical model
                        x_mass = x_Events_scaled[:, :self.n_mass_features]
                        x_topo = x_Events_scaled[:, self.n_mass_features:]
                        
                        # Predict using the 3-class NN model with two inputs
                        # Returns shape (n_samples, 3) where columns are [VBF, ggF, bkg]
                        scores_3class = self.m_models[model][i].predict([x_mass, x_topo], batch_size=81920, verbose=0)
                        
                        # Extract individual class scores
                        score_vbf = scores_3class[:, 0]  # VBF output
                        score_ggf = scores_3class[:, 1]  # ggF output
                        score_bkg = scores_3class[:, 2]  # background output
                        
                        # Calculate derived scores as requested:
                        # vbf_score = VBF - ggF
                        # bdt_score = (VBF + ggF) - bkg
                        vbf_score = score_vbf - score_ggf
                        bdt_score = (score_vbf + score_ggf) - score_bkg
                        
                        nn_basename = self.models[model]
                        
                        # Store raw outputs
                        data_o[nn_basename + '_vbf'] = score_vbf
                        data_o[nn_basename + '_ggf'] = score_ggf
                        data_o[nn_basename + '_bkg'] = score_bkg
                        
                        # Store derived scores
                        data_o[nn_basename + '_vbf_score'] = vbf_score
                        data_o[nn_basename + '_bdt_score'] = bdt_score
                        
                        # Try to apply saved transformers to get _t versions
                        fold_tsfs = None
                        if model in self.m_tsfs:
                            try:
                                fold_tsfs = self.m_tsfs[model][i]
                            except Exception:
                                fold_tsfs = None
                        
                        if fold_tsfs is not None:
                            # Transform vbf_score (VBF - ggF)
                            if 'vbf_score_tsf' in fold_tsfs and fold_tsfs['vbf_score_tsf'] is not None:
                                try:
                                    vbf_score_t = fold_tsfs['vbf_score_tsf'].transform(
                                        vbf_score.reshape(-1, 1)).reshape(-1)
                                    data_o[nn_basename + '_vbf_score_t'] = vbf_score_t
                                except Exception as e:
                                    print(f'NN WARNING: vbf_score transformer failed, fold {i}: {e}')
                                    data_o[nn_basename + '_vbf_score_t'] = vbf_score
                            else:
                                data_o[nn_basename + '_vbf_score_t'] = vbf_score
                            
                            # Transform bdt_score ((VBF + ggF) - bkg)
                            if 'bdt_score_tsf' in fold_tsfs and fold_tsfs['bdt_score_tsf'] is not None:
                                try:
                                    bdt_score_t = fold_tsfs['bdt_score_tsf'].transform(
                                        bdt_score.reshape(-1, 1)).reshape(-1)
                                    data_o[nn_basename + '_bdt_score_t'] = bdt_score_t
                                except Exception as e:
                                    print(f'NN WARNING: bdt_score transformer failed, fold {i}: {e}')
                                    data_o[nn_basename + '_bdt_score_t'] = bdt_score
                            else:
                                data_o[nn_basename + '_bdt_score_t'] = bdt_score
                        else:
                            # No transformers available, use untransformed scores
                            data_o[nn_basename + '_vbf_score_t'] = vbf_score
                            data_o[nn_basename + '_bdt_score_t'] = bdt_score
                    
                    # Save immediately to temp ROOT file
                    if len(data_o) > 0:
                        # Remove index column if it exists
                        if "index" in data_o.columns:
                            data_o = data_o.drop('index', axis=1)
                        
                        temp_file = tempfile.NamedTemporaryFile(suffix='.root', delete=False)
                        temp_file.close()

                        with uproot.recreate(temp_file.name) as root_file:
                            root_file["test"] = data_o
                        
                        temp_files.append(temp_file.name)
                    
                    # Clear immediately
                    del data_o
                    chunk_count += 1
                    
                    # Force garbage collection more frequently
                    if chunk_count % 4 == 0:
                        gc.collect()
                
                # Clear chunk data
                del data, data_s
                gc.collect()
            
            file.close()
            gc.collect()
            
            current_memory = get_memory_usage()
            print(f"NN INFO: Processed file {file_idx+1}/{len(f_list)}, Memory: {current_memory:.2f} GB, Temp files: {len(temp_files)}")
        
        # Combine temp files with minimal memory usage
        print(f"NN INFO: Combining {len(temp_files)} temporary files...")
        self._combine_temp_files(temp_files, output_path)
        
        # Clean up
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
        
        final_memory = get_memory_usage()
        print(f"NN INFO: Final memory usage: {final_memory:.2f} GB")
        
    def _combine_temp_files(self, temp_files, output_path):
        """Combine temporary ROOT files using ROOT hadd for optimal performance."""
        if not temp_files:
            print("NN WARNING: No temporary files to combine!")
            return
        
        print(f"NN INFO: Combining {len(temp_files)} temporary ROOT files...")
        
        # All files are already in ROOT format, just use hadd directly
        if HADD_AVAILABLE:
            try:
                # Use hadd command - much faster and more memory efficient
                hadd_cmd = ['hadd', '-f', output_path] + temp_files
                print(f"NN INFO: Running hadd command...")
                result = subprocess.run(hadd_cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"NN INFO: Successfully combined files using hadd")
                    return
                else:
                    print(f"NN WARNING: hadd failed with return code {result.returncode}")
                    print(f"stderr: {result.stderr}")
                    print("NN INFO: Falling back to manual combination")
            except Exception as e:
                print(f"NN WARNING: Error running hadd: {e}")
                print("NN INFO: Falling back to manual combination")
        
        # Fallback: manual combination
        self._fallback_combine_files(temp_files, output_path)
    
    def _fallback_combine_files(self, root_files, output_path):
        """Fallback method to combine ROOT files manually if hadd fails."""
        print("NN INFO: Using fallback method to combine files...")
        
        if not root_files:
            print("NN WARNING: No files to combine!")
            return
        
        # Read first file to initialize
        with uproot.open(root_files[0]) as first_file:
            combined_data = first_file["test"].arrays(library='pd')
        
        # Append remaining files one by one
        for root_file in root_files[1:]:
            try:
                with uproot.open(root_file) as f:
                    data = f["test"].arrays(library='pd')
                    combined_data = pd.concat([combined_data, data], ignore_index=True)
                    del data
                    gc.collect()
            except Exception as e:
                print(f"NN WARNING: Failed to read {root_file}: {e}")
        
        # Write final result
        with uproot.recreate(output_path) as output_file:
            output_file["test"] = combined_data
        
        del combined_data
        gc.collect()


def main():

    args=getArgs()
    
    configPath = args.config
    nn = Apply3ClassNNHandler(configPath, args.region)

    nn.setInputFolder(args.inputFolder)
    nn.setModelFolder(args.modelFolder)
    nn.setOutputFolder(args.outputFolder)

    nn.loadModels()
    nn.loadScaler()

    with open('data/inputs_config_ggH.json') as f:
        config = json.load(f)
    sample_list = config['sample_list']

    for category in sample_list:
        if args.category and category not in args.category: continue
        nn.applyNN(category)

    return

if __name__ == '__main__':
    main()
