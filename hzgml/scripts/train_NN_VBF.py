#!/usr/bin/env python
"""
VBF DNN Training Script
Based on VBF_dnn.ipynb - Implements hierarchical DNN model for VBF H→μμ analysis
"""
import os
import sys
import json
import pickle
import logging
from argparse import ArgumentParser
from datetime import datetime
from tqdm import tqdm
from tabulate import tabulate
from pdb import set_trace
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers as tf_optimizers

# Scikit-learn
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from weighted_quantile_transformer import WeightedQuantileTransformer
from sklearn.metrics import roc_curve, auc, confusion_matrix

# ROOT and uproot
import uproot
import ROOT
ROOT.gErrorIgnoreLevel = ROOT.kError + 1

# Optuna for hyperparameter tuning
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances

# Configure logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# Logger setup
logger = logging.getLogger('log')
logger.setLevel(logging.INFO)

# Configure GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"GPU available: {physical_devices}")
    print("GPU memory growth enabled")
else:
    print("No GPU found, using CPU")

def getArgs():
    """Get arguments from command line."""
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', action='store', default='data/training_config_NN_VBF.json', 
                        help='Configuration file path')
    parser.add_argument('-i', '--inputFolder', action='store', 
                        help='Directory of training inputs')
    parser.add_argument('-o', '--outputFolder', action='store', default='models_VBF_dnn',
                        help='Directory for outputs')
    parser.add_argument('-r', '--region', action='store', default='VBF',
                        help='Region to process')
    parser.add_argument('-f', '--fold', action='store', type=int, nargs='+', 
                        choices=[0, 1, 2, 3], default=[0, 1, 2, 3], 
                        help='Specify the fold for training')
    parser.add_argument('--tree-name', action='store', default='two_jet_m110To150',
                        help='Name of the tree in ROOT files')
    parser.add_argument('--save', action='store_true', default=True,
                        help='Save model weights')
    parser.add_argument('--corr', action='store_true', default=True,
                        help='Plot correlation between training variables')
    parser.add_argument('--importance', action='store_true', default=True,
                        help='Plot feature importance')
    parser.add_argument('--roc', action='store_true', default=True,
                        help='Plot ROC curve')
    parser.add_argument('--optuna', action='store_true', default=False,
                        help='Run hyperparameter tuning using Optuna')
    parser.add_argument('--n-trials', action='store', type=int, default=100,
                        help='Number of Optuna trials')
    parser.add_argument('--optuna-suffix', action='store', default='',
                        help='Suffix for optuna experiment directory')
    parser.add_argument('--continue-optuna', action='store', type=int, default=0,
                        help='If 1, delete old DB and restart; if 0, continue')
    parser.add_argument('--hyperparams_path', action='store', default=None,
                        help='Path to best-params JSON file (used for final training)')
    parser.add_argument('--epochs', action='store', type=int, default=10000,
                        help='Maximum number of epochs')
    parser.add_argument('--batch-size', action='store', type=int, default=None,
                        help='Batch size for training')
    
    return parser.parse_args()


class VBFDNNHandler:
    """Handler class for VBF DNN training based on hierarchical model structure"""
    
    def __init__(self, configPath, region='VBF'):
        
        print("""
========================================================================
||                     VBF DNN Training Script                        ||
||          Hierarchical Deep Neural Network for VBF H→μμ             ||
========================================================================
              """)
        
        args = getArgs()
        
        self._folds = args.fold
        self._region = region
        self._inputFolder = args.inputFolder if args.inputFolder else ''
        self._outputFolder = args.outputFolder
        self._plotFolder = f'plots_{args.outputFolder}' if args.outputFolder else 'plots_VBF_dnn'
        self._tree_name = args.tree_name
        self._epochs = args.epochs
        self._batch_size = args.batch_size
        self._current_fold = 0
        self.optuna_suffix = args.optuna_suffix
        self.continue_optuna = int(getattr(args, 'continue_optuna', 0))
        self.hyperparams_path = args.hyperparams_path
        
        # Data storage
        self.data_sig = pd.DataFrame()
        self.data_bkg = pd.DataFrame()
        self.data_bkg_vbfz = pd.DataFrame()  # Background for VBF Z model
        self.data_bkg_dy = pd.DataFrame()    # Background for DY model
        
        # Training data dictionaries
        self.X_train = {}
        self.X_val = {}
        self.X_test = {}
        self.y_train = {}
        self.y_val = {}
        self.y_test = {}
        self.train_wt = {}  # Reweighted weights for training
        self.val_wt = {}    # Reweighted weights for validation
        self.test_wt = {}   # Reweighted weights for testing
        self.train_wt_raw = {}  # Raw weights for transformScore
        self.val_wt_raw = {}
        self.test_wt_raw = {}
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.histories = {}
        
        # Predictions storage
        self.y_train_pred = {}
        self.y_val_pred = {}
        self.y_test_pred = {}
        # Score transformers (for mapping scores to uniform via weighted quantiles)
        self.m_tsf = {}
        
        # Configuration parameters
        self.train_signal = []
        self.train_mc_background = []
        self.vbfz_mc_background = []  # Background for VBF Z sub-model
        self.dy_mc_background = []    # Background for DY sub-model
        self.train_variables = []
        self.mass_variables = []  # First 3 variables: mass-related
        self.topo_variables = []  # Remaining variables: topology
        self.preselections = []
        self.signal_preselections = []
        self.background_preselections = []
        self.randomIndex = 'event'
        self.weight = 'eventWeight'
        
        # Model hyperparameters (will be set from config or optuna)
        self.params = {
            'dense_units_1': 64,
            'dense_units_2': 64,
            'dense_units_3': 16,
            'dropout_rate': 0.1,
            'mass_dense_units_1': 16,
            'mass_dense_units_2': 8,
            'mass_dense_units_3': 2,
            'mass_dropout_rate': 0.1,
            'learning_rate': 0.0005,
            'first_decay_steps': 2000,
            'decay_rate': 0.89,
            'decay_steps': 1600,
            'pre_epoch': 50,
            'merged_dense_1': 128,
            'merged_dense_2': 16,
            'merged_dense_3': 128,
            'activation': 'elu',
            'optimizer': 'adam',
            'lr_schedule': 'exp',  # one of: 'plateau', 'exp', 'cosine', 'none'
        }
        
        # Read configuration
        self.readConfig(configPath)
        self.checkConfig()
    
    def readConfig(self, configPath):
        """Read configuration file formatted in json."""
        try:
            with open(configPath, 'r') as stream:
                configs = json.load(stream)
            
            # Read common settings
            config = configs.get("common", {})
            self.train_signal = config.get("train_signal", [])
            self.train_mc_background = config.get("train_mc_background", [])
            self.train_variables = config.get("train_variables", [])
            self.preselections = config.get("preselections", [])
            self.signal_preselections = config.get("signal_preselections", [])
            self.background_preselections = config.get("background_preselections", [])
            self.randomIndex = config.get("randomIndex", "event")
            self.weight = config.get("weight", "eventWeight")
            
            # Read region-specific settings
            if self._region and self._region in configs:
                region_config = configs[self._region]
                for key in region_config:
                    if hasattr(self, key):
                        setattr(self, key, region_config[key])
                # load params overrides if provided
                if isinstance(region_config.get('params', None), dict):
                    self.params.update(region_config['params'])
            
            # Read sub-model specific backgrounds if not already set by region config
            if not self.vbfz_mc_background and self._region and self._region in configs:
                self.vbfz_mc_background = configs[self._region].get('vbfz_mc_background', [])
            if not self.dy_mc_background and self._region and self._region in configs:
                self.dy_mc_background = configs[self._region].get('dy_mc_background', [])
            
            # Split variables into mass and topology
            if len(self.train_variables) >= 3:
                self.mass_variables = self.train_variables[:3]
                self.topo_variables = self.train_variables[3:]
            else:
                raise ValueError("Need at least 3 variables (mass variables)")
            
            print("\n" + "="*60)
            print(f"Training variables ({len(self.train_variables)}):")
            print(f"  Mass variables ({len(self.mass_variables)}): {self.mass_variables}")
            print(f"  Topology variables ({len(self.topo_variables)}): {self.topo_variables}")
            print(f"Tree name: {self._tree_name}")
            print("="*60 + "\n")
            
        except Exception as e:
            logging.error(f"Error reading configuration '{configPath}'")
            logging.error(e)
            raise
    
    def checkConfig(self):
        """Check if configuration is valid."""
        if not self.train_signal:
            raise ValueError('ERROR: no training signal!!')
        if not self.train_mc_background:
            raise ValueError('ERROR: no training background!!')
        if not self.train_variables:
            raise ValueError('ERROR: no training variables!!')
    
    def load_data_from_root(self, files, tree_name, branches):
        """Load data from ROOT files using uproot."""
        data_frames = []
        for file in tqdm(files, desc='Loading ROOT files'):
            try:
                with uproot.open(file) as f:
                    tree = f[tree_name]
                    df = tree.arrays(branches, library="pd")
                    data_frames.append(df)
            except Exception as e:
                logging.warning(f"Failed to load {file}: {e}")
                continue
        
        if not data_frames:
            raise ValueError("No data loaded from files")
        
        return pd.concat(data_frames, ignore_index=True)
    
    def preselect(self, data, sample=''):
        """Apply preselections to data."""
        if sample == 'signal':
            for p in self.signal_preselections:
                data.query(p, inplace=True)
        elif sample == 'background':
            for p in self.background_preselections:
                data.query(p, inplace=True)
        
        for p in self.preselections:
            data.query(p, inplace=True)
        
        return data
    
    def readData(self):
        """Read signal and background data from ROOT files."""
        
        # Get all required branches
        all_branches = list(set(self.train_variables + [self.randomIndex, self.weight]))
        
        # Collect signal files
        sig_list = []
        for sig_cat in self.train_signal:
            sig_file = os.path.join(self._inputFolder, f"{sig_cat}.root")
            if os.path.exists(sig_file):
                sig_list.append(sig_file)
        
        # Collect background files
        bkg_list = []
        for bkg_cat in self.train_mc_background:
            bkg_file = os.path.join(self._inputFolder, f"{bkg_cat}.root")
            if os.path.exists(bkg_file):
                bkg_list.append(bkg_file)
        
        print('-' * 60)
        print('Loading signal samples:')
        for sig in sig_list:
            print(f'  {sig}')
        
        # Load signal data
        if sig_list:
            self.data_sig = self.load_data_from_root(sig_list, self._tree_name, all_branches)
            self.data_sig = self.preselect(self.data_sig, 'signal')
            print(f'Signal events loaded: {len(self.data_sig)}')
        
        print('-' * 60)
        print('Loading background samples:')
        for bkg in bkg_list:
            print(f'  {bkg}')
        
        # Load background data
        if bkg_list:
            self.data_bkg = self.load_data_from_root(bkg_list, self._tree_name, all_branches)
            self.data_bkg = self.preselect(self.data_bkg, 'background')
            print(f'Background events loaded: {len(self.data_bkg)}')
        
        # Load VBF Z model specific background
        if self.vbfz_mc_background:
            vbfz_bkg_list = []
            for bkg_cat in self.vbfz_mc_background:
                pattern = f'{self._inputFolder}/{bkg_cat}*.root'
                vbfz_bkg_list.extend(glob.glob(pattern))
            
            if vbfz_bkg_list:
                print('-' * 60)
                print('Loading VBF Z model background samples:')
                for bkg in vbfz_bkg_list:
                    print(f'  {bkg}')
                self.data_bkg_vbfz = self.load_data_from_root(vbfz_bkg_list, self._tree_name, all_branches)
                self.data_bkg_vbfz = self.preselect(self.data_bkg_vbfz, 'background')
                print(f'VBF Z background events loaded: {len(self.data_bkg_vbfz)}')
        
        # Load DY model specific background
        if self.dy_mc_background:
            dy_bkg_list = []
            for bkg_cat in self.dy_mc_background:
                pattern = f'{self._inputFolder}/{bkg_cat}*.root'
                dy_bkg_list.extend(glob.glob(pattern))
            
            if dy_bkg_list:
                print('-' * 60)
                print('Loading DY model background samples:')
                for bkg in dy_bkg_list:
                    print(f'  {bkg}')
                self.data_bkg_dy = self.load_data_from_root(dy_bkg_list, self._tree_name, all_branches)
                self.data_bkg_dy = self.preselect(self.data_bkg_dy, 'background')
                print(f'DY background events loaded: {len(self.data_bkg_dy)}')
        
        print('-' * 60)
    
    def check_nan_inf(self, data, name="Data"):
        """Check for NaN and inf values in DataFrame."""
        nan_info = data.isnull().sum()
        inf_info = np.isinf(data.select_dtypes(include=[np.number])).sum()
        
        has_issues = False
        for col in data.columns:
            if nan_info.get(col, 0) > 0:
                logging.warning(f"{name} - Column '{col}' contains {nan_info[col]} NaN values")
                has_issues = True
            if inf_info.get(col, 0) > 0:
                logging.warning(f"{name} - Column '{col}' contains {inf_info[col]} inf values")
                has_issues = True
        
        if not has_issues:
            logging.info(f"{name} - No NaN or inf values found")
    
    def reweight_binary(self, labels, wt_raw):
        """Reweight binary classification data independently.
        
        Args:
            labels: Binary labels (0 or 1)
            wt_raw: Raw weights
            
        Returns:
            tuple: (reweighted_weights, N_sig, N_bkg, mean_sig, mean_bkg)
        """
        sig_mask, bkg_mask = labels == 1, labels == 0
        N_sig, N_bkg = np.sum(sig_mask), np.sum(bkg_mask)
        mean_sig = np.mean(wt_raw[sig_mask]) if N_sig > 0 else 1.0
        mean_bkg = np.mean(wt_raw[bkg_mask]) if N_bkg > 0 else 1.0
        wt = wt_raw.copy()
        if N_sig > 0: wt[sig_mask] = wt_raw[sig_mask] / mean_sig
        if N_bkg > 0: wt[bkg_mask] = wt_raw[bkg_mask] / mean_bkg * N_sig / N_bkg
        print(f"  Reweighting: sig sum = {np.sum(wt[sig_mask]):.2f}, bkg sum = {np.sum(wt[bkg_mask]):.2f}")
        return wt, N_sig, N_bkg, mean_sig, mean_bkg
    
    def plot_correlation(self, data, data_type, fold):
        """Plot correlation matrix for the training variables."""
        if not os.path.isdir(f"{self._plotFolder}/corr/"):
            os.makedirs(f"{self._plotFolder}/corr/")
        
        # Select only training variables
        data_corr = data[self.train_variables].copy()
        
        # Replace invalid values for correlation calculation
        for col in data_corr.columns:
            data_corr[col] = data_corr[col].replace([np.inf, -np.inf], np.nan)
        
        # Compute correlation matrix
        corr_matrix = data_corr.corr() * 100
        
        # Plot
        plt.figure(figsize=(14, 12), dpi=300)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        # Plot heatmap
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                    vmin=-100, vmax=100, annot=False, fmt='.0f')

        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                if i < j:
                    continue
                value = corr_matrix.iloc[i, j]
                plt.text(j + 0.5, i + 0.5, f"{value:.0f}%", ha='center', va='center', fontsize=12, color='black')
        
        plt.title(f'Correlation Matrix - {data_type} (fold {fold})', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{self._plotFolder}/corr/corr_{self._region}_{data_type}_fold{fold}.pdf")
        plt.savefig(f"{self._plotFolder}/corr/corr_{self._region}_{data_type}_fold{fold}.png")
        plt.close()
        
        # Print strong correlations
        print(f"\n{data_type} - Strong correlations (|corr| > 40%):")
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 40:
                    print(f"  {corr_matrix.columns[i]:30s} <-> {corr_matrix.columns[j]:30s}: {corr_matrix.iloc[i, j]:6.1f}%")
    
    def prepareData(self, fold=0):
        """Prepare data for training with k-fold cross-validation."""
        
        self._current_fold = fold
        
        print('=' * 60)
        print(f'Preparing data for fold {fold}')
        print('=' * 60)
        
        # Add labels
        data_sig_labeled = self.data_sig.copy()
        data_bkg_labeled = self.data_bkg.copy()
        data_sig_labeled['label'] = 1
        data_bkg_labeled['label'] = 0
        
        # Combine data
        data = pd.concat([data_sig_labeled, data_bkg_labeled], ignore_index=True)
        
        # Split based on randomIndex
        test_data = data[data[self.randomIndex] % 4 == fold]
        val_data = data[(data[self.randomIndex] - 1) % 4 == fold]
        train_data = data[((data[self.randomIndex] - 2) % 4 == fold) | 
                         ((data[self.randomIndex] - 3) % 4 == fold)]
        
        # Extract features and labels
        X_train = train_data[self.train_variables].values
        X_val = val_data[self.train_variables].values
        X_test = test_data[self.train_variables].values
        
        self.y_train[fold] = train_data['label'].values
        self.y_val[fold] = val_data['label'].values
        self.y_test[fold] = test_data['label'].values
        
        # Extract raw weights (for transformScore)
        self.train_wt_raw[fold] = train_data[self.weight].values
        self.val_wt_raw[fold] = val_data[self.weight].values
        self.test_wt_raw[fold] = test_data[self.weight].values
        
        # Reweight each dataset independently
        self.train_wt[fold], Ns, Nb, ms, mb = self.reweight_binary(train_data['label'].values, self.train_wt_raw[fold])
        self.val_wt[fold], _, _, _, _ = self.reweight_binary(val_data['label'].values, self.val_wt_raw[fold])
        self.test_wt[fold], _, _, _, _ = self.reweight_binary(test_data['label'].values, self.test_wt_raw[fold])
        
        print(f"\nReweight(fold{fold}): sig={Ns}(m={ms:.6f}), bkg={Nb}(m={mb:.6f})")
        
        # Check for NaN and inf
        print("\nChecking for invalid values...")
        X_combined = np.vstack([X_train, X_val, X_test])
        X_df = pd.DataFrame(X_combined, columns=self.train_variables)
        self.check_nan_inf(X_df, f"Fold {fold}")
        
        # Replace inf with NaN, then fill NaN with -1
        X_train = np.where(np.isinf(X_train), np.nan, X_train)
        X_val = np.where(np.isinf(X_val), np.nan, X_val)
        X_test = np.where(np.isinf(X_test), np.nan, X_test)
        
        X_train = np.nan_to_num(X_train, nan=-1.0)
        X_val = np.nan_to_num(X_val, nan=-1.0)
        X_test = np.nan_to_num(X_test, nan=-1.0)
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler
        self.scalers[fold] = scaler
        
        # Store scaled data
        self.X_train[fold] = X_train_scaled
        self.X_val[fold] = X_val_scaled
        self.X_test[fold] = X_test_scaled
        
        # Prepare data for VBF Z and DY sub-models
        self._prepare_submodel_data(fold)
        
        # Print dataset statistics
        headers = ['Set', 'Signal', 'Background', 'Total']
        train_sig = np.sum(self.y_train[fold] == 1)
        train_bkg = np.sum(self.y_train[fold] == 0)
        val_sig = np.sum(self.y_val[fold] == 1)
        val_bkg = np.sum(self.y_val[fold] == 0)
        test_sig = np.sum(self.y_test[fold] == 1)
        test_bkg = np.sum(self.y_test[fold] == 0)
        
        table = [
            ['Train', train_sig, train_bkg, train_sig + train_bkg],
            ['Val', val_sig, val_bkg, val_sig + val_bkg],
            ['Test', test_sig, test_bkg, test_sig + test_bkg]
        ]
        print("\n" + tabulate(table, headers=headers, tablefmt='grid'))
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def _prepare_submodel_data(self, fold):
        """Prepare training data for VBF Z and DY sub-models."""
        
        # Prepare VBF Z model data (signal vs VBF Z background)
        if hasattr(self, 'data_bkg_vbfz') and not self.data_bkg_vbfz.empty:
            data_sig_vbfz = self.data_sig.copy()
            data_bkg_vbfz = self.data_bkg_vbfz.copy()
            data_sig_vbfz['label'] = 1
            data_bkg_vbfz['label'] = 0
            
            data_vbfz = pd.concat([data_sig_vbfz, data_bkg_vbfz], ignore_index=True)
            
            # Split based on randomIndex
            test_vbfz = data_vbfz[data_vbfz[self.randomIndex] % 4 == fold]
            val_vbfz = data_vbfz[(data_vbfz[self.randomIndex] - 1) % 4 == fold]
            train_vbfz = data_vbfz[((data_vbfz[self.randomIndex] - 2) % 4 == fold) | 
                                   ((data_vbfz[self.randomIndex] - 3) % 4 == fold)]
            
            X_train_vbfz = train_vbfz[self.train_variables].values
            X_val_vbfz = val_vbfz[self.train_variables].values
            
            # Handle NaN and inf
            X_train_vbfz = np.where(np.isinf(X_train_vbfz), np.nan, X_train_vbfz)
            X_val_vbfz = np.where(np.isinf(X_val_vbfz), np.nan, X_val_vbfz)
            X_train_vbfz = np.nan_to_num(X_train_vbfz, nan=-1.0)
            X_val_vbfz = np.nan_to_num(X_val_vbfz, nan=-1.0)
            
            # Standardize using the same scaler as main data
            X_train_vbfz = self.scalers[fold].transform(X_train_vbfz)
            X_val_vbfz = self.scalers[fold].transform(X_val_vbfz)
            
            # Store VBF Z data
            if not hasattr(self, 'X_train_vbfz'):
                self.X_train_vbfz = {}
                self.X_val_vbfz = {}
                self.y_train_vbfz = {}
                self.y_val_vbfz = {}
                self.wt_train_vbfz = {}
                self.wt_val_vbfz = {}
            
            self.X_train_vbfz[fold] = X_train_vbfz
            self.X_val_vbfz[fold] = X_val_vbfz
            self.y_train_vbfz[fold] = train_vbfz['label'].values
            self.y_val_vbfz[fold] = val_vbfz['label'].values
            
            # Get raw weights
            wt_train_vbfz_raw = train_vbfz[self.weight].values
            wt_val_vbfz_raw = val_vbfz[self.weight].values
            
            # Reweight independently
            self.wt_train_vbfz[fold] = self.reweight_binary(self.y_train_vbfz[fold], wt_train_vbfz_raw)[0]
            self.wt_val_vbfz[fold] = self.reweight_binary(self.y_val_vbfz[fold], wt_val_vbfz_raw)[0]
            
            print(f"\nVBF Z model data - Train: {len(train_vbfz)} (sig: {np.sum(self.y_train_vbfz[fold])}, bkg: {np.sum(1-self.y_train_vbfz[fold])}), "
                  f"Val: {len(val_vbfz)} (sig: {np.sum(self.y_val_vbfz[fold])}, bkg: {np.sum(1-self.y_val_vbfz[fold])})")
        
        # Prepare DY model data (signal vs DY background)
        if hasattr(self, 'data_bkg_dy') and not self.data_bkg_dy.empty:
            data_sig_dy = self.data_sig.copy()
            data_bkg_dy = self.data_bkg_dy.copy()
            data_sig_dy['label'] = 1
            data_bkg_dy['label'] = 0
            
            data_dy = pd.concat([data_sig_dy, data_bkg_dy], ignore_index=True)
            
            # Split based on randomIndex
            test_dy = data_dy[data_dy[self.randomIndex] % 4 == fold]
            val_dy = data_dy[(data_dy[self.randomIndex] - 1) % 4 == fold]
            train_dy = data_dy[((data_dy[self.randomIndex] - 2) % 4 == fold) | 
                              ((data_dy[self.randomIndex] - 3) % 4 == fold)]
            
            X_train_dy = train_dy[self.train_variables].values
            X_val_dy = val_dy[self.train_variables].values
            
            # Handle NaN and inf
            X_train_dy = np.where(np.isinf(X_train_dy), np.nan, X_train_dy)
            X_val_dy = np.where(np.isinf(X_val_dy), np.nan, X_val_dy)
            X_train_dy = np.nan_to_num(X_train_dy, nan=-1.0)
            X_val_dy = np.nan_to_num(X_val_dy, nan=-1.0)
            
            # Standardize using the same scaler as main data
            X_train_dy = self.scalers[fold].transform(X_train_dy)
            X_val_dy = self.scalers[fold].transform(X_val_dy)
            
            # Store DY data
            if not hasattr(self, 'X_train_dy'):
                self.X_train_dy = {}
                self.X_val_dy = {}
                self.y_train_dy = {}
                self.y_val_dy = {}
                self.wt_train_dy = {}
                self.wt_val_dy = {}
            
            self.X_train_dy[fold] = X_train_dy
            self.X_val_dy[fold] = X_val_dy
            self.y_train_dy[fold] = train_dy['label'].values
            self.y_val_dy[fold] = val_dy['label'].values
            
            # Get raw weights
            wt_train_dy_raw = train_dy[self.weight].values
            wt_val_dy_raw = val_dy[self.weight].values
            
            # Reweight independently
            self.wt_train_dy[fold] = self.reweight_binary(self.y_train_dy[fold], wt_train_dy_raw)[0]
            self.wt_val_dy[fold] = self.reweight_binary(self.y_val_dy[fold], wt_val_dy_raw)[0]
            
            print(f"DY model data - Train: {len(train_dy)} (sig: {np.sum(self.y_train_dy[fold])}, bkg: {np.sum(1-self.y_train_dy[fold])}), "
                  f"Val: {len(val_dy)} (sig: {np.sum(self.y_val_dy[fold])}, bkg: {np.sum(1-self.y_val_dy[fold])})")
    
    def create_vbf_model(self, n_mass_features=3, n_topo_features=16, params=None):
        """
        Create the VBF DNN model with hierarchical structure.
        
        This model implements the architecture from VBF_dnn.ipynb:
        1. Four specialized sub-models train on different feature combinations
        2. Intermediate features are extracted from these sub-models
        3. Features are merged and fed to final classifier
        """
        
        if params is None:
            params = self.params
        
        # Input layers
        input_mass_res = Input(shape=(n_mass_features,), name='mass_res')
        input_vbf_topo = Input(shape=(n_topo_features,), name='vbf_topo')
        
        dense_units_1 = params.get('dense_units_1', 64)
        dense_units_2 = params.get('dense_units_2', 32)
        dense_units_3 = params.get('dense_units_3', 16)
        dropout_rate = params.get('dropout_rate', 0.2)
        activation = params.get('activation', 'relu')

        mass_dense_units_1 = params.get('mass_dense_units_1', 64)
        mass_dense_units_2 = params.get('mass_dense_units_2', 32)
        mass_dense_units_3 = params.get('mass_dense_units_3', 16)
        mass_dropout_rate = params.get('mass_dropout_rate', 0.2)
        
        # Sub-model 1: Signal vs VBF Z (uses both mass and topology)
        x1 = concatenate([input_mass_res, input_vbf_topo])
        x1 = Dense(dense_units_1, activation=activation)(x1)
        x1 = Dropout(dropout_rate)(x1)
        x1 = Dense(dense_units_2, activation=activation)(x1)
        x1 = Dropout(dropout_rate)(x1)
        x1 = Dense(dense_units_3, activation=activation)(x1)
        x1 = Dropout(dropout_rate)(x1)
        x1_out = Dense(1, activation='sigmoid', name='VBF_Z_Output')(x1)
        
        # Sub-model 2: Signal vs DY (uses both mass and topology)
        x2 = concatenate([input_mass_res, input_vbf_topo])
        x2 = Dense(dense_units_1, activation=activation)(x2)
        x2 = Dropout(dropout_rate)(x2)
        x2 = Dense(dense_units_2, activation=activation)(x2)
        x2 = Dropout(dropout_rate)(x2)
        x2 = Dense(dense_units_3, activation=activation)(x2)
        x2 = Dropout(dropout_rate)(x2)
        x2_out = Dense(1, activation='sigmoid', name='DY_Output')(x2)
        
        # Sub-model 3: Signal vs Background (topology only, no mass)
        x3 = Dense(dense_units_1, activation=activation)(input_vbf_topo)
        x3 = Dropout(dropout_rate)(x3)
        x3 = Dense(dense_units_2, activation=activation)(x3)
        x3 = Dropout(dropout_rate)(x3)
        x3 = Dense(dense_units_3, activation=activation)(x3)
        x3 = Dropout(dropout_rate)(x3)
        x3_out = Dense(1, activation='sigmoid', name='No_Mass_Output')(x3)
        
        # Sub-model 4: Signal vs Background (mass only)
        x4 = Dense(mass_dense_units_1, activation=activation)(input_mass_res)
        x4 = Dropout(mass_dropout_rate)(x4)
        x4 = Dense(mass_dense_units_2, activation=activation)(x4)
        x4 = Dropout(mass_dropout_rate)(x4)
        x4 = Dense(mass_dense_units_3, activation=activation)(x4)
        x4 = Dropout(mass_dropout_rate)(x4)
        x4_out = Dense(1, activation='sigmoid', name='Mass_Only_Output')(x4)
        
        # Create individual models (for pre-training if needed)
        model_vbf_z = Model(inputs=[input_mass_res, input_vbf_topo], outputs=x1_out)
        model_dy = Model(inputs=[input_mass_res, input_vbf_topo], outputs=x2_out)
        model_no_mass = Model(inputs=input_vbf_topo, outputs=x3_out)
        model_mass_only = Model(inputs=input_mass_res, outputs=x4_out)
        
        # Compile sub-models
        for model in [model_vbf_z, model_dy, model_no_mass, model_mass_only]:
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Extract intermediate layers (second to last layer before output)
        intermediate_vbf_z = Model(inputs=model_vbf_z.input, 
                                   outputs=model_vbf_z.layers[-2].output)
        intermediate_dy = Model(inputs=model_dy.input, 
                               outputs=model_dy.layers[-2].output)
        intermediate_no_mass = Model(inputs=model_no_mass.input, 
                                     outputs=model_no_mass.layers[-2].output)
        intermediate_mass_only = Model(inputs=model_mass_only.input, 
                                       outputs=model_mass_only.layers[-2].output)
        
        # Get intermediate features
        input_vbf_z_features = intermediate_vbf_z([input_mass_res, input_vbf_topo])
        input_dy_features = intermediate_dy([input_mass_res, input_vbf_topo])
        input_no_mass_features = intermediate_no_mass(input_vbf_topo)
        input_mass_only_features = intermediate_mass_only(input_mass_res)
        
        # Merge all intermediate features
        merged_features = concatenate([
            input_vbf_z_features, 
            input_dy_features, 
            input_no_mass_features, 
            input_mass_only_features
        ])
        
        # Final merged model
        merged_dense_1 = params.get('merged_dense_1', 64)
        merged_dense_2 = params.get('merged_dense_2', 32)
        merged_dense_3 = params.get('merged_dense_3', 16)
        
        x = Dense(merged_dense_1, activation=activation)(merged_features)
        x = Dropout(dropout_rate)(x)
        x = Dense(merged_dense_2, activation=activation)(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(merged_dense_3, activation=activation)(x)
        x = Dropout(dropout_rate)(x)
        final_output = Dense(1, activation='sigmoid', name='Final_Output')(x)
        
        # Create final merged model
        merged_model = Model(inputs=[input_mass_res, input_vbf_topo], outputs=final_output)

        return merged_model, (model_vbf_z, model_dy, model_no_mass, model_mass_only)

    def _build_optimizer_and_callbacks(self, params):
        """Create optimizer and LR schedule callbacks from params."""
        opt_name = str(params.get('optimizer', 'adam')).lower()
        base_lr = float(params.get('learning_rate', 1e-3))
        lr_sched = str(params.get('lr_schedule', 'plateau')).lower()

        # Learning rate setup
        lr = base_lr
        lr_callbacks = []
        if lr_sched == 'exp':
            decay_rate = float(params.get('decay_rate', 0.96))
            decay_steps = int(params.get('decay_steps', 1000))
            lr = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=base_lr,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=True
            )
        elif lr_sched == 'cosine':
            first_decay_steps = int(params.get('first_decay_steps', 1000))
            t_mul = float(params.get('t_mul', 2.0))
            m_mul = float(params.get('m_mul', 1.0))
            alpha = float(params.get('alpha', 0.0))
            lr = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=base_lr,
                first_decay_steps=first_decay_steps,
                t_mul=t_mul,
                m_mul=m_mul,
                alpha=alpha
            )
        elif lr_sched == 'plateau':
            lr_callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-5, verbose=1))

        # Optimizer
        if opt_name == 'adam':
            opt = tf_optimizers.Adam(learning_rate=lr)
        elif opt_name == 'adamw':
            try:
                opt = tf_optimizers.AdamW(learning_rate=lr)
            except Exception:
                opt = tf_optimizers.Adam(learning_rate=lr)
        elif opt_name == 'sgd':
            momentum = float(params.get('momentum', 0.9))
            nesterov = bool(params.get('nesterov', True))
            opt = tf_optimizers.SGD(learning_rate=lr, momentum=momentum, nesterov=nesterov)
        elif opt_name == 'rmsprop':
            opt = tf_optimizers.RMSprop(learning_rate=lr)
        else:
            opt = tf_optimizers.Adam(learning_rate=lr)

        return opt, lr_callbacks
    
    def trainModel(self, fold, params=None):
        """Train the VBF DNN model for a specific fold."""
        
        if params is None:
            params = self.params
        
        print(f"\n{'='*60}")
        print(f"Training fold {fold}")
        print(f"{'='*60}")
        print(f"Hyperparameters: {params}")
        
        # Create model
        n_mass = len(self.mass_variables)
        n_topo = len(self.topo_variables)
        
        merged_model, sub_models = self.create_vbf_model(n_mass, n_topo, params)
        model_vbf_z, model_dy, model_no_mass, model_mass_only = sub_models
        
        # Get training data
        X_train = self.X_train[fold]
        X_val = self.X_val[fold]
        y_train = self.y_train[fold]
        y_val = self.y_val[fold]
        
        # Split features into mass and topology
        X_train_mass = X_train[:, :n_mass]
        X_train_topo = X_train[:, n_mass:]
        X_val_mass = X_val[:, :n_mass]
        X_val_topo = X_val[:, n_mass:]
        
        # optimizer and lr schedule
        optimizer, lr_cbs = self._build_optimizer_and_callbacks(params)
        batch_size = int(self._batch_size) if self._batch_size else int(params.get('batch_size', 2048))
        
        # Pre-train sub-models
        print("\nPre-training sub-models...")
        
        # Setup callbacks for sub-model training (use simple early stopping)
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )

        pre_epoch = int(params.get('pre_epoch', 10))

        # Train VBF Z model with its specific background
        print("  Training VBF Z model...")
        X_train_vbfz_mass = self.X_train_vbfz[fold][:, :n_mass]
        X_train_vbfz_topo = self.X_train_vbfz[fold][:, n_mass:]
        X_val_vbfz_mass = self.X_val_vbfz[fold][:, :n_mass]
        X_val_vbfz_topo = self.X_val_vbfz[fold][:, n_mass:]
        model_vbf_z.fit([X_train_vbfz_mass, X_train_vbfz_topo], self.y_train_vbfz[fold], 
                        sample_weight=self.wt_train_vbfz[fold],
                        epochs=pre_epoch, batch_size=batch_size,
                        validation_data=([X_val_vbfz_mass, X_val_vbfz_topo], self.y_val_vbfz[fold], self.wt_val_vbfz[fold]),
                        callbacks=[early_stop], verbose=2)
        
        # Train DY model with its specific background
        print("  Training DY model...")
        X_train_dy_mass = self.X_train_dy[fold][:, :n_mass]
        X_train_dy_topo = self.X_train_dy[fold][:, n_mass:]
        X_val_dy_mass = self.X_val_dy[fold][:, :n_mass]
        X_val_dy_topo = self.X_val_dy[fold][:, n_mass:]
        model_dy.fit([X_train_dy_mass, X_train_dy_topo], self.y_train_dy[fold],
                    sample_weight=self.wt_train_dy[fold],
                    epochs=pre_epoch, batch_size=batch_size,
                    validation_data=([X_val_dy_mass, X_val_dy_topo], self.y_val_dy[fold], self.wt_val_dy[fold]),
                    callbacks=[early_stop], verbose=2)
        
        print("  Training no-mass model...")
        model_no_mass.fit(X_train_topo, y_train,
                         sample_weight=self.train_wt[fold],
                         epochs=pre_epoch, batch_size=batch_size,
                         validation_data=(X_val_topo, y_val, self.val_wt[fold]),
                         callbacks=[early_stop], verbose=2)
        
        print("  Training mass-only model...")
        model_mass_only.fit(X_train_mass, y_train,
                           sample_weight=self.train_wt[fold],
                           epochs=pre_epoch , batch_size=batch_size,
                           validation_data=(X_val_mass, y_val, self.val_wt[fold]),
                           callbacks=[early_stop], verbose=2)
        
        # # Make upstream layers trainable
        # for layer in merged_model.layers[:-3]:
        #     layer.trainable = True

        # Compile merged model
        merged_model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        print("\nMerged model summary:")
        merged_model.summary()
        
        # LR schedule callbacks (if any)
        reduce_lr_merged = lr_cbs

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        # Train merged model
        print("\nTraining merged model...")
        history = merged_model.fit(
            [X_train_mass, X_train_topo],
            y_train,
            sample_weight=self.train_wt[fold],
            epochs=self._epochs,
            batch_size=batch_size,
            validation_data=([X_val_mass, X_val_topo], y_val, self.val_wt[fold]),
            callbacks=[early_stopping] + reduce_lr_merged,
            verbose=2
        )
        
        # Store model and history
        self.models[fold] = merged_model
        self.histories[fold] = history
        
        return merged_model, history
    
    def evaluate_on_val(self, fold):
        """Generate predictions and evaluate on validation set only (no test usage)."""
        model = self.models[fold]
        n_mass = len(self.mass_variables)

        # Get data
        X_train = self.X_train[fold]
        X_val = self.X_val[fold]
        X_test = self.X_test[fold]

        # Split into mass and topology
        X_train_mass, X_train_topo = X_train[:, :n_mass], X_train[:, n_mass:]
        X_val_mass, X_val_topo = X_val[:, :n_mass], X_val[:, n_mass:]
        X_test_mass, X_test_topo = X_test[:, :n_mass], X_test[:, n_mass:]

        # Generate predictions
        print(f"\nGenerating predictions (train/val) for fold {fold}...")
        self.y_train_pred[fold] = model.predict([X_train_mass, X_train_topo], batch_size=81920, verbose=0)
        self.y_val_pred[fold] = model.predict([X_val_mass, X_val_topo], batch_size=81920, verbose=0)
        self.y_test_pred[fold] = model.predict([X_test_mass, X_test_topo], batch_size=81920, verbose=0)

        # Evaluate on validation set
        t_loss, t_accuracy, t_auc = model.evaluate([X_train_mass, X_train_topo], self.y_train[fold], batch_size=81920, verbose=0)
        loss, accuracy, auc_score = model.evaluate([X_val_mass, X_val_topo], self.y_val[fold], batch_size=81920, verbose=0)
        print(f"Fold {fold} - Train Loss: {t_loss:.4f}, Accuracy: {t_accuracy:.4f}, AUC: {t_auc:.4f}")
        print(f"Fold {fold} - Val Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")

        return t_auc, auc_score

    def transformScore(self, fold=0, sample='sig'):

        print(f'NN INFO: transforming scores based on {sample} (fold {fold})')
        # create transformer
        self.m_tsf[fold] = WeightedQuantileTransformer(n_quantiles=1000, output_distribution='uniform', subsample=1000000000, random_state=0)

        # choose which scores and weights to use
        scores = self.y_test_pred.get(fold)
        if scores is None:
            print(f'NN WARNING: No test predictions for fold {fold} to fit transformer')
            return

        scores = np.asarray(scores).reshape(-1)

        # use sample selection if requested
        labels = self.y_test.get(fold)
        weights_raw = self.test_wt_raw.get(fold)  # Use RAW weights for transformScore

        if labels is not None:
            if sample == 'sig':
                mask = labels == 1
            elif sample == 'bkg':
                mask = labels == 0
            else:
                mask = np.ones_like(labels, dtype=bool)
            scores_to_fit = scores[mask]
            sample_weight = weights_raw[mask] if weights_raw is not None else None
        else:
            scores_to_fit = scores
            sample_weight = weights_raw

        # Fit transformer (it handles edge cases internally)
        try:
            if sample_weight is not None:
                self.m_tsf[fold].fit(scores_to_fit.reshape(-1, 1), sample_weight=sample_weight)
                print(f'NN INFO: Score transformation fitted using raw weights')
            else:
                self.m_tsf[fold].fit(scores_to_fit.reshape(-1, 1))
                print(f'NN INFO: Score transformation fitted without weights')
        except Exception as e:
            print(f'NN WARNING: transformer fit failed: {e}')
            # fallback: keep transformer but don't fit
            self.m_tsf.pop(fold, None)
            return

        return
    
    def plot_feature_importance(self, fold):
        """
        Plot feature importance using permutation importance.
        For neural networks, we use a simple approach: measure prediction change when shuffling each feature.
        """
        if not os.path.isdir(f"{self._plotFolder}/feature_importance/"):
            os.makedirs(f"{self._plotFolder}/feature_importance/")
        
        model = self.models[fold]
        X_val = self.X_val[fold]
        y_val = self.y_val[fold]
        n_mass = len(self.mass_variables)

        X_val_mass, X_val_topo = X_val[:, :n_mass], X_val[:, n_mass:]

        # Baseline AUC on validation set
        y_pred_base = model.predict([X_val_mass, X_val_topo], verbose=0).ravel()
        fpr, tpr, _ = roc_curve(y_val, y_pred_base)
        baseline_auc = auc(fpr, tpr)
        
        # Compute importance by shuffling each feature
        importances = []
        for i in range(X_val.shape[1]):
            X_val_perm = X_val.copy()
            np.random.shuffle(X_val_perm[:, i])

            X_val_mass_perm = X_val_perm[:, :n_mass]
            X_val_topo_perm = X_val_perm[:, n_mass:]

            y_pred_perm = model.predict([X_val_mass_perm, X_val_topo_perm], verbose=0).ravel()
            fpr, tpr, _ = roc_curve(y_val, y_pred_perm)
            perm_auc = auc(fpr, tpr)
            
            importance = baseline_auc - perm_auc
            importances.append(importance)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(self.train_variables) * 0.3)))
        
        indices = np.argsort(importances)
        variables = [self.train_variables[i] for i in indices]
        values = [importances[i] for i in indices]
        
        colors = ['blue' if i < n_mass else 'green' for i in indices]

        for i in range(len(variables)):
            plt.text(values[i] + 0.0005, i, f"{values[i]:.4f}", va='center', fontsize=10)
        
        ax.barh(range(len(variables)), values, color=colors)
        ax.set_yticks(range(len(variables)))
        ax.set_yticklabels(variables)
        ax.set_xlabel('Importance (AUC decrease when shuffled)')
        ax.set_title(f'Feature Importance - Fold {fold}')
        ax.grid(axis='x', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='Mass variables'),
            Patch(facecolor='green', label='Topology variables')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(f"{self._plotFolder}/feature_importance/importance_{self._region}_fold{fold}.pdf")
        # plt.savefig(f"{self._plotFolder}/feature_importance/importance_{self._region}_fold{fold}.png")
        plt.close()
        
        print(f"\nFeature importance (fold {fold}):")
        for var, imp in sorted(zip(self.train_variables, importances), key=lambda x: x[1], reverse=True):
            print(f"  {var:30s}: {imp:.6f}")
    
    def plotROC(self, fold, save=True, show=False):
        """Plot ROC curves for train, validation, and test sets."""
        
        if not os.path.isdir(f"{self._plotFolder}/roc_curve/"):
            os.makedirs(f"{self._plotFolder}/roc_curve/")
        
        # Calculate ROC curves
        fpr_train, tpr_train, _ = roc_curve(self.y_train[fold], self.y_train_pred[fold])
        fpr_val, tpr_val, _ = roc_curve(self.y_val[fold], self.y_val_pred[fold])
        fpr_test, tpr_test, _ = roc_curve(self.y_test[fold], self.y_test_pred[fold])
        
        auc_train = auc(fpr_train, tpr_train)
        auc_val = auc(fpr_val, tpr_val)
        auc_test = auc(fpr_test, tpr_test)
        
        # Plot
        plt.figure(figsize=(8, 8))
        plt.plot(fpr_train, tpr_train, linestyle='--', 
                label=f'Train set, AUC = {auc_train:.4f}', linewidth=2)
        plt.plot(fpr_val, tpr_val, linestyle='--', 
                label=f'Validation set, AUC = {auc_val:.4f}', linewidth=2)
        plt.plot(fpr_test, tpr_test, linestyle='-', 
                label=f'Test set, AUC = {auc_test:.4f}', linewidth=2)
        plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='Random', linewidth=1)
        
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title(f'ROC Curve - Fold {fold}', fontsize=16)
        plt.legend(loc='lower right', fontsize=12)
        plt.grid(alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        if save:
            plt.savefig(f"{self._plotFolder}/roc_curve/roc_{self._region}_fold{fold}.pdf")
            # plt.savefig(f"{self._plotFolder}/roc_curve/roc_{self._region}_fold{fold}.png")
        
        if show:
            plt.show()
        
        plt.close()
        
        return auc_train, auc_val, auc_test
    
    def save(self, fold):
        """Save model, scaler, and predictions."""
        
        os.makedirs(self._outputFolder, exist_ok=True)
        
        # Save model
        model_path = f"{self._outputFolder}/model_fold_{fold}.keras"
        self.models[fold].save(model_path)
        print(f"Saved model to {model_path}")
        
        # Save scaler
        scaler_path = f"{self._outputFolder}/scaler_fold_{fold}.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scalers[fold], f)
        print(f"Saved scaler to {scaler_path}")
        
        # # Save predictions (train/val only)
        # np.save(f"{self._outputFolder}/y_train_pred_fold_{fold}.npy", self.y_train_pred[fold])
        # np.save(f"{self._outputFolder}/y_val_pred_fold_{fold}.npy", self.y_val_pred[fold])
        
        # Save training history
        history_path = f"{self._outputFolder}/history_fold_{fold}.pkl"
        with open(history_path, 'wb') as f:
            pickle.dump(self.histories[fold].history, f)
        print(f"Saved training history to {history_path}")

        # Save score transformer if available
        if fold in self.m_tsf.keys():
            tsf_path = f"{self._outputFolder}/DNN_tsf_{fold}.pkl"
            try:
                with open(tsf_path, 'wb') as f:
                    pickle.dump(self.m_tsf[fold], f, -1)
                print(f"Saved DNN transformer to {tsf_path}")
            except Exception as e:
                print(f"NN WARNING: Failed to save transformer: {e}")
    
    def optunaHP(self, n_trials=50):
        """
        Hyperparameter optimization using Optuna.
        Saves DB, logs, and JSON to models/optuna_{region}_{suffix}.
        If continue_optuna == 1, delete old DB and restart.
        """
        
        print(f"\n{'='*60}")
        print(f"Starting Optuna hyperparameter optimization for all folds")
        print(f"Number of trials: {n_trials}")
        print(f"{'='*60}\n")
        
        # Build experiment directory
        suffix = self.optuna_suffix if self.optuna_suffix else datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_dir = f"models/optuna_{self._region}_{suffix}/"
        os.makedirs(exp_dir, exist_ok=True)
        
        # Storage path
        db_path = os.path.join(exp_dir, 'optuna_study.db')
        storage_name = f"sqlite:///{db_path}"
        
        # Restart if requested (per requirement: 1 means delete old and restart, 0 means continue)
        # If DB exists and continue_optuna != 0, delete and restart to avoid value space conflicts
        if os.path.exists(db_path):
            if self.continue_optuna == 0:
                logger.info(f"continue_optuna=0: Continuing with existing database: {db_path}")
                os.remove(db_path)
            elif self.continue_optuna == 1:
                logger.info(f"continue_optuna=1: Removing existing database: {db_path}")
            else:
                # Default behavior: delete to avoid CategoricalDistribution errors
                logger.warning(f"Existing DB found. Removing to avoid value space conflicts: {db_path}")
                os.remove(db_path)
        
        logger.info(f"Experiment dir: {exp_dir}")
        logger.info(f"Storage: {storage_name}")
        
        def objective(trial):
            """Objective function for Optuna optimization."""
            
            # Suggest hyperparameters
            params = {
                'dense_units_1': trial.suggest_categorical('dense_units_1', [32, 64, 128]),
                'dense_units_2': trial.suggest_categorical('dense_units_2', [16, 32, 64]),
                'dense_units_3': trial.suggest_categorical('dense_units_3', [8, 16, 32]),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.05),
                'mass_dense_units_1': trial.suggest_categorical('mass_dense_units_1', [4, 8, 16]),
                'mass_dense_units_2': trial.suggest_categorical('mass_dense_units_2', [2, 4, 8]),
                'mass_dense_units_3': trial.suggest_categorical('mass_dense_units_3', [2, 4, 8]),
                'mass_dropout_rate': trial.suggest_float('mass_dropout_rate', 0.1, 0.4, step=0.05),
                'learning_rate': trial.suggest_float('learning_rate', 5e-5, 1e-3, log=True),
                'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd', 'rmsprop']),
                'activation': trial.suggest_categorical('activation', ['relu', 'elu', 'selu', 'tanh']),
                # 'lr_schedule': trial.suggest_categorical('lr_schedule', ['plateau', 'exp', 'cosine']),
                'batch_size': trial.suggest_categorical('batch_size', [1024, 2048, 4096, 6144, 8192]),
                'pre_epoch': trial.suggest_int('pre_epoch', 20, 60, step=5),
                'decay_rate': trial.suggest_float('decay_rate', 0.75, 0.9),
                'decay_steps': trial.suggest_int('decay_steps', 1000, 4000, step=200),
                'first_decay_steps': trial.suggest_int('first_decay_steps', 500, 4000, step=500),
                'merged_dense_1': trial.suggest_categorical('merged_dense_1', [32, 64, 128, 256]),
                'merged_dense_2': trial.suggest_categorical('merged_dense_2', [16, 32, 64, 128]),
                'merged_dense_3': trial.suggest_categorical('merged_dense_3', [16, 32, 64, 128]),
            }
            logger.info(f"Trial {trial.number}: {params}")
            
            # Train model with these parameters
            try:
                auc_scores, t_aucs = [], []
                for fold in self._folds:
                    self.trainModel(fold, params)
                    t_auc, auc_score = self.evaluate_on_val(fold)
                    auc_scores.append(auc_score)
                    t_aucs.append(t_auc)
                
                return float(2*np.mean(auc_scores)-np.mean(t_aucs))  # maximize val AUC - train AUC
                
            except Exception as e:
                logger.error(f"Trial failed: {e}")
                return 0.0
        
        # Create Optuna study
        study_name = f"NN_{self._region}_allfolds"
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(n_startup_trials=10, multivariate=True, group=True),
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            direction='maximize'
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Print results
        logger.info("Optimization completed!")
        logger.info(f"Best value: {study.best_value:.6f}")
        logger.info(f"Best params: {study.best_params}")
        
        # Save best parameters JSON to exp_dir
        best_params_path = os.path.join(exp_dir, f'best_params_allfolds.json')
        with open(best_params_path, 'w') as f:
            json.dump({'best_value': study.best_value, 'best_params': study.best_params}, f, indent=2)
        logger.info(f"Saved best params to: {best_params_path}")
        
        # Plot optimization history to exp_dir
        try:
            fig = plot_optimization_history(study)
            fig.write_image(os.path.join(exp_dir, f"optuna_history_allfolds.png"))
            fig = plot_param_importances(study)
            fig.write_image(os.path.join(exp_dir, f"optuna_importance_allfolds.png"))
        except Exception as e:
            logger.warning(f"Could not save Optuna plots: {e}")
        
        return study.best_params


def main():
    
    args = getArgs()
    
    # Create handler
    handler = VBFDNNHandler(args.config, args.region)
    
    # Set input folder if provided
    if args.inputFolder:
        handler._inputFolder = args.inputFolder
    
    # Read data
    handler.readData()
    
    # Plot correlations if requested
    if args.corr:
        print("\nPlotting correlations...")
        handler.plot_correlation(handler.data_sig, 'signal', fold=0)
        handler.plot_correlation(handler.data_bkg, 'background', fold=0)
    
    # Load external best hyperparameters json if provided
    if args.hyperparams_path:
        load_path = args.hyperparams_path
        if os.path.isdir(load_path):
            # prefer files with 'best_params' in name, else first json
            candidates = [os.path.join(load_path, f) for f in os.listdir(load_path) if f.endswith('.json')]
            if any('best_params' in os.path.basename(p) for p in candidates):
                candidates = [p for p in candidates if 'best_params' in os.path.basename(p)]
            if not candidates:
                raise FileNotFoundError(f"No JSON files found under {load_path}")
            load_path = sorted(candidates)[0]
        with open(load_path, 'r') as fp:
            j = json.load(fp)
        best_params = j.get('best_params', j)
        handler.params.update(best_params)
        print(f"Loaded hyperparameters from {load_path}:")
        print(best_params)

    # Run Optuna optimization if requested
    if args.optuna:
        for fold in args.fold:
            handler.prepareData(fold)
        best_params = handler.optunaHP(n_trials=args.n_trials)
        for fold in args.fold:
            # Train final model with best parameters
            handler.trainModel(fold, best_params)
            handler.evaluate_on_val(fold)
            if args.importance:
                handler.plot_feature_importance(fold)
            if args.roc:
                handler.plotROC(fold)
            if args.save:
                handler.save(fold)
    else:
        logger.info(f"Starting training....")
        # Regular training for each fold
        for fold in args.fold:
            print(f"\n{'='*70}")
            print(f"Processing fold {fold}")
            print(f"{'='*70}")
            
            # Prepare data
            handler.prepareData(fold)
            
            # Train model with current/default/best params
            handler.trainModel(fold, handler.params)
            
            # Evaluate on validation set only
            handler.evaluate_on_val(fold)
            
            # Plot feature importance
            if args.importance:
                handler.plot_feature_importance(fold)
            
            # Plot ROC
            if args.roc:
                auc_train, auc_val, auc_test = handler.plotROC(fold)
                print(f"\nFold {fold} AUC scores:")
                print(f"  Train: {auc_train:.6f}")
                print(f"  Val:   {auc_val:.6f}")
                print(f"  Test:  {auc_test:.6f}")
        
            handler.transformScore(fold)
            
            # Save model
            if args.save:
                handler.save(fold)

        logger.info("Training completed for all folds.")
    
    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)


if __name__ == '__main__':
    main()
