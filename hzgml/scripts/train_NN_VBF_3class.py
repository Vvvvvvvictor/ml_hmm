#!/usr/bin/env python
"""
VBF 3-Class DNN Training Script
Three-class classification: VBF signal, ggF signal, and background
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from datetime import datetime
from tqdm import tqdm
from tabulate import tabulate
from pdb import set_trace
import ROOT
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import logging

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
    parser.add_argument('-c', '--config', action='store', default='data/training_config_NN_VBF_3class.json', 
                        help='Configuration file path')
    parser.add_argument('-i', '--inputFolder', action='store', 
                        help='Directory of training inputs')
    parser.add_argument('-o', '--outputFolder', action='store', default='models_VBF_3class',
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


class VBF3ClassDNNHandler:
    """Handler class for VBF 3-class DNN training"""
    
    def __init__(self, configPath, region='VBF'):
        
        print("""
========================================================================
||                 VBF 3-Class DNN Training Script                    ||
||      Three-class DNN: VBF signal, ggF signal, background          ||
========================================================================
              """)
        
        args = getArgs()
        
        self._folds = args.fold
        self._region = region
        self._inputFolder = args.inputFolder if args.inputFolder else ''
        self._outputFolder = args.outputFolder
        self._plotFolder = f'plots_{args.outputFolder}' if args.outputFolder else 'plots_VBF_3class'
        self._tree_name = args.tree_name
        self._epochs = args.epochs
        self._batch_size = args.batch_size
        self._current_fold = 0
        self.optuna_suffix = args.optuna_suffix
        self.continue_optuna = int(getattr(args, 'continue_optuna', 0))
        self.hyperparams_path = args.hyperparams_path
        
        # Data storage - now with three classes
        self.data_vbf_sig = pd.DataFrame()  # VBF signal
        self.data_ggf_sig = pd.DataFrame()  # ggF signal
        self.data_bkg = pd.DataFrame()      # Background
        
        # Training data dictionaries
        self.X_train = {}
        self.X_val = {}
        self.X_test = {}
        self.y_train = {}  # Now contains 3 columns: [VBF, ggF, bkg]
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
        self.train_vbf_signal = []
        self.train_ggf_signal = []
        self.train_mc_background = []
        self.vbfz_mc_background = []
        self.dy_mc_background = []
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
            # New: ggF vs VBF submodel parameters
            'ggf_vbf_dense_units_1': 64,
            'ggf_vbf_dense_units_2': 64,
            'ggf_vbf_dense_units_3': 16,
            'ggf_vbf_dropout_rate': 0.15,
            'learning_rate': 0.001,
            'first_decay_steps': 2000,
            'decay_rate': 0.89,
            'decay_steps': 1600,
            'pre_epoch': 50,
            'merged_dense_1': 128,
            'merged_dense_2': 16,
            'merged_dense_3': 128,
            'activation': 'elu',
            'optimizer': 'adam',
            'lr_schedule': 'exp',
            'batch_size': 8192,
            'pre_batch_size': 2048
        }
        
        # Read configuration
        self.readConfig(configPath)
        self.checkConfig()
    
    def readConfig(self, configPath):
        """Read configuration file formatted in json."""
        try:
            member_variables = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__") and not attr.startswith('m_')]
            
            stream = open(configPath, 'r')
            configs = json.loads(stream.read())
            
            # Read from the common settings
            config = configs["common"]
            for member in config.keys():
                if member in member_variables: 
                    setattr(self, member, config[member])
            
            # Read from the region specific settings
            if self._region:
                if self._region in configs.keys():
                    config = configs[self._region]
                    for member in config.keys():
                        if member in member_variables:
                            setattr(self, member, config[member])
            
            # Split training variables into mass and topology
            if self.train_variables:
                self.mass_variables = self.train_variables[:3]
                self.topo_variables = self.train_variables[3:]
            
            print("="*60)
            print("Configuration loaded successfully")
            print(f"VBF Signal samples: {self.train_vbf_signal}")
            print(f"ggF Signal samples: {self.train_ggf_signal}")
            print(f"Background samples: {self.train_mc_background}")
            print(f"Training variables: {len(self.train_variables)}")
            print(f"  - Mass variables (first 3): {self.mass_variables}")
            print(f"  - Topology variables: {len(self.topo_variables)}")
            print("="*60)
            
        except Exception as e:
            logging.error(f"Error reading configuration file '{configPath}'")
            logging.error(e)
            raise
    
    def checkConfig(self):
        """Check if configuration is valid."""
        if not self.train_vbf_signal:
            raise ValueError("No VBF signal samples specified in configuration")
        if not self.train_ggf_signal:
            raise ValueError("No ggF signal samples specified in configuration")
        if not self.train_mc_background:
            raise ValueError("No background samples specified in configuration")
        if not self.train_variables:
            raise ValueError("No training variables specified in configuration")
    
    def load_data_from_root(self, files, tree_name, branches):
        """Load data from ROOT files using uproot."""
        data_frames = []
        for file in tqdm(files, desc='Loading ROOT files'):
            try:
                with uproot.open(file) as f:
                    tree = f[tree_name]
                    df = tree.arrays(branches, library='pd')
                    data_frames.append(df)
            except Exception as e:
                print(f"Warning: Could not load file {file}: {e}")
        
        if not data_frames:
            raise ValueError("No data loaded from ROOT files")
        
        return pd.concat(data_frames, ignore_index=True)
    
    def preselect(self, data, sample=''):
        """Apply preselections to data."""
        if sample == 'signal':
            for p in self.signal_preselections:
                data = data.query(p)
        elif sample == 'background':
            for p in self.background_preselections:
                data = data.query(p)
        
        for p in self.preselections:
            data = data.query(p)
        
        return data
    
    def readData(self):
        """Read VBF signal, ggF signal, and background data from ROOT files."""
        
        # Get all required branches
        all_branches = list(set(self.train_variables + [self.randomIndex, self.weight]))
        
        # Collect VBF signal files
        vbf_sig_list = []
        for sig_cat in self.train_vbf_signal:
            for root, dirs, files in os.walk(self._inputFolder):
                for file in files:
                    if file.endswith(f'{sig_cat}.root'):
                        vbf_sig_list.append(os.path.join(root, file))
        
        # Collect ggF signal files
        ggf_sig_list = []
        for sig_cat in self.train_ggf_signal:
            for root, dirs, files in os.walk(self._inputFolder):
                for file in files:
                    if file.endswith(f'{sig_cat}.root'):
                        ggf_sig_list.append(os.path.join(root, file))
        
        # Collect background files
        bkg_list = []
        for bkg_cat in self.train_mc_background:
            for root, dirs, files in os.walk(self._inputFolder):
                for file in files:
                    if file.endswith(f'{bkg_cat}.root'):
                        bkg_list.append(os.path.join(root, file))
        
        print('-' * 60)
        print('Loading VBF signal samples:')
        for sig in vbf_sig_list:
            print(f'  {sig}')
        
        # Load VBF signal data
        if vbf_sig_list:
            self.data_vbf_sig = self.load_data_from_root(vbf_sig_list, self._tree_name, all_branches)
            self.data_vbf_sig = self.preselect(self.data_vbf_sig, 'signal')
            print(f"VBF Signal events loaded: {len(self.data_vbf_sig)}")
        
        print('-' * 60)
        print('Loading ggF signal samples:')
        for sig in ggf_sig_list:
            print(f'  {sig}')
        
        # Load ggF signal data
        if ggf_sig_list:
            self.data_ggf_sig = self.load_data_from_root(ggf_sig_list, self._tree_name, all_branches)
            self.data_ggf_sig = self.preselect(self.data_ggf_sig, 'signal')
            print(f"ggF Signal events loaded: {len(self.data_ggf_sig)}")
        
        print('-' * 60)
        print('Loading background samples:')
        for bkg in bkg_list:
            print(f'  {bkg}')
        
        # Load background data
        if bkg_list:
            self.data_bkg = self.load_data_from_root(bkg_list, self._tree_name, all_branches)
            self.data_bkg = self.preselect(self.data_bkg, 'background')
            print(f"Background events loaded: {len(self.data_bkg)}")
        
        # Load VBF Z model specific background
        if self.vbfz_mc_background:
            vbfz_list = []
            for bkg_cat in self.vbfz_mc_background:
                for root, dirs, files in os.walk(self._inputFolder):
                    for file in files:
                        if file.endswith(f'{bkg_cat}.root'):
                            vbfz_list.append(os.path.join(root, file))
            if vbfz_list:
                self.data_bkg_vbfz = self.load_data_from_root(vbfz_list, self._tree_name, all_branches)
                self.data_bkg_vbfz = self.preselect(self.data_bkg_vbfz, 'background')
                print(f"VBF Z background events loaded: {len(self.data_bkg_vbfz)}")
        
        # Load DY model specific background
        if self.dy_mc_background:
            dy_list = []
            for bkg_cat in self.dy_mc_background:
                for root, dirs, files in os.walk(self._inputFolder):
                    for file in files:
                        if file.endswith(f'{bkg_cat}.root'):
                            dy_list.append(os.path.join(root, file))
            if dy_list:
                self.data_bkg_dy = self.load_data_from_root(dy_list, self._tree_name, all_branches)
                self.data_bkg_dy = self.preselect(self.data_bkg_dy, 'background')
                print(f"DY background events loaded: {len(self.data_bkg_dy)}")
        
        print('-' * 60)
    
    def check_nan_inf(self, data, name="Data"):
        """Check for NaN and inf values in DataFrame."""
        nan_info = data.isnull().sum()
        inf_info = np.isinf(data.select_dtypes(include=[np.number])).sum()
        
        has_issues = False
        for col in data.columns:
            if nan_info[col] > 0:
                print(f"  {name} - Column '{col}': {nan_info[col]} NaN values")
                has_issues = True
            if col in inf_info.index and inf_info[col] > 0:
                print(f"  {name} - Column '{col}': {inf_info[col]} inf values")
                has_issues = True
        
        if not has_issues:
            print(f"  {name} - No NaN or inf values detected")
    
    def reweight_binary(self, labels, wt_raw):
        """Reweight binary classification data independently.
        
        Args:
            labels: Binary labels (0 or 1)
            wt_raw: Raw weights
            
        Returns:
            reweighted_weights
        """
        sig_mask, bkg_mask = labels == 1, labels == 0
        N_sig, N_bkg = np.sum(sig_mask), np.sum(bkg_mask)
        mean_sig = np.mean(wt_raw[sig_mask]) if N_sig > 0 else 1.0
        mean_bkg = np.mean(wt_raw[bkg_mask]) if N_bkg > 0 else 1.0
        wt = wt_raw.copy()
        if N_sig > 0: wt[sig_mask] = wt_raw[sig_mask] / mean_sig
        if N_bkg > 0: wt[bkg_mask] = wt_raw[bkg_mask] / mean_bkg * N_sig / N_bkg
        print(f"  Reweighting: sig sum = {np.sum(wt[sig_mask]):.2f}, bkg sum = {np.sum(wt[bkg_mask]):.2f}")
        return wt
    
    def reweight_3class(self, data, wt_raw):
        """Reweight 3-class classification data independently.
        
        Args:
            data: DataFrame with label_vbf, label_ggf, label_bkg columns
            wt_raw: Raw weights
            
        Returns:
            tuple: (reweighted_weights, N_list, means_list, N_sig_total)
        """
        masks = [data['label_vbf'].values == 1, data['label_ggf'].values == 1, data['label_bkg'].values == 1]
        N = [np.sum(m) for m in masks]
        N_sig = N[0] + N[1]
        means = [np.mean(wt_raw[m]) if N[i] > 0 else 1.0 for i, m in enumerate(masks)]
        wt = wt_raw.copy()
        for i in range(2):  # VBF and ggF signals
            if N[i] > 0: wt[masks[i]] = wt_raw[masks[i]] / means[i]
        if N[2] > 0: wt[masks[2]] = wt_raw[masks[2]] / means[2] * N_sig / N[2]
        print(f"  Reweighting: VBF sum = {np.sum(wt[masks[0]]):.2f}, ggF sum = {np.sum(wt[masks[1]]):.2f}, bkg sum = {np.sum(wt[masks[2]]):.2f}")
        return wt, N, means, N_sig
    
    def plot_correlation(self, data, data_type, fold):
        """Plot correlation matrix for the training variables."""
        if not os.path.isdir(f"{self._plotFolder}/corr/"):
            os.makedirs(f"{self._plotFolder}/corr/")
        
        # Select only training variables
        data_corr = data[self.train_variables].copy()
        
        # Replace invalid values for correlation calculation
        for col in data_corr.columns:
            data_corr[col] = data_corr[col].replace([np.inf, -np.inf], np.nan).fillna(-1)
        
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
            for j in range(i):
                value = corr_matrix.iloc[i, j]
                if abs(value) > 40:
                    plt.text(j + 0.5, i + 0.5, f'{value:.0f}',
                            ha='center', va='center', fontsize=6, color='black')
        
        plt.title(f'Correlation Matrix - {data_type} (fold {fold})', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{self._plotFolder}/corr/corr_{self._region}_{data_type}_fold{fold}.pdf")
        plt.savefig(f"{self._plotFolder}/corr/corr_{self._region}_{data_type}_fold{fold}.png")
        plt.close()
        
        # Print strong correlations
        print(f"\n{data_type} - Strong correlations (|corr| > 40%):")
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 40:
                    print(f"  {corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.1f}%")
    
    def prepareData(self, fold=0):
        """Prepare data for training with k-fold cross-validation."""
        
        self._current_fold = fold
        
        print('=' * 60)
        print(f'Preparing data for fold {fold}')
        print('=' * 60)
        
        # Add labels: [VBF, ggF, bkg]
        data_vbf_labeled = self.data_vbf_sig.copy()
        data_ggf_labeled = self.data_ggf_sig.copy()
        data_bkg_labeled = self.data_bkg.copy()
        
        # Three-class one-hot encoding
        data_vbf_labeled['label_vbf'] = 1
        data_vbf_labeled['label_ggf'] = 0
        data_vbf_labeled['label_bkg'] = 0
        
        data_ggf_labeled['label_vbf'] = 0
        data_ggf_labeled['label_ggf'] = 1
        data_ggf_labeled['label_bkg'] = 0
        
        data_bkg_labeled['label_vbf'] = 0
        data_bkg_labeled['label_ggf'] = 0
        data_bkg_labeled['label_bkg'] = 1
        
        # Combine data
        data = pd.concat([data_vbf_labeled, data_ggf_labeled, data_bkg_labeled], ignore_index=True)
        
        # Split based on randomIndex
        test_data = data[data[self.randomIndex] % 4 == fold]
        val_data = data[(data[self.randomIndex] - 1) % 4 == fold]
        train_data = data[((data[self.randomIndex] - 2) % 4 == fold) | 
                         ((data[self.randomIndex] - 3) % 4 == fold)]
        
        # Extract features and labels
        X_train = train_data[self.train_variables].values
        X_val = val_data[self.train_variables].values
        X_test = test_data[self.train_variables].values
        
        # Extract three-class labels
        self.y_train[fold] = train_data[['label_vbf', 'label_ggf', 'label_bkg']].values
        self.y_val[fold] = val_data[['label_vbf', 'label_ggf', 'label_bkg']].values
        self.y_test[fold] = test_data[['label_vbf', 'label_ggf', 'label_bkg']].values
        
        # Extract raw weights (for transformScore)
        self.train_wt_raw[fold] = train_data[self.weight].values
        self.val_wt_raw[fold] = val_data[self.weight].values
        self.test_wt_raw[fold] = test_data[self.weight].values
        
        # Reweight each dataset independently
        def reweight_3class(data, wt_raw):
            masks = [data['label_vbf'].values == 1, data['label_ggf'].values == 1, data['label_bkg'].values == 1]
            N = [np.sum(m) for m in masks]
            N_sig = N[0] + N[1]
            means = [np.mean(wt_raw[m]) if N[i] > 0 else 1.0 for i, m in enumerate(masks)]
            wt = wt_raw.copy()
            for i in range(2):  # VBF and ggF signals
                if N[i] > 0: wt[masks[i]] = wt_raw[masks[i]] / means[i]
            if N[2] > 0: wt[masks[2]] = wt_raw[masks[2]] / means[2] * N_sig / N[2]
            return wt, N, means, N_sig
        
        self.train_wt[fold], N_tr, mean_tr, Nsig_tr = reweight_3class(train_data, self.train_wt_raw[fold])
        self.val_wt[fold], _, _, _ = reweight_3class(val_data, self.val_wt_raw[fold])
        self.test_wt[fold], _, _, _ = reweight_3class(test_data, self.test_wt_raw[fold])
        
        print(f"\nReweight(fold{fold}): VBF={N_tr[0]}(m={mean_tr[0]:.6f}), ggF={N_tr[1]}(m={mean_tr[1]:.6f}), "
              f"bkg={N_tr[2]}(m={mean_tr[2]:.6f}), Nsig={Nsig_tr}")
        
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
        headers = ['Set', 'VBF Signal', 'ggF Signal', 'Background', 'Total']
        train_vbf = np.sum(self.y_train[fold][:, 0] == 1)
        train_ggf = np.sum(self.y_train[fold][:, 1] == 1)
        train_bkg = np.sum(self.y_train[fold][:, 2] == 1)
        val_vbf = np.sum(self.y_val[fold][:, 0] == 1)
        val_ggf = np.sum(self.y_val[fold][:, 1] == 1)
        val_bkg = np.sum(self.y_val[fold][:, 2] == 1)
        test_vbf = np.sum(self.y_test[fold][:, 0] == 1)
        test_ggf = np.sum(self.y_test[fold][:, 1] == 1)
        test_bkg = np.sum(self.y_test[fold][:, 2] == 1)
        
        table = [
            ['Train', train_vbf, train_ggf, train_bkg, train_vbf + train_ggf + train_bkg],
            ['Val', val_vbf, val_ggf, val_bkg, val_vbf + val_ggf + val_bkg],
            ['Test', test_vbf, test_ggf, test_bkg, test_vbf + test_ggf + test_bkg]
        ]
        print("\n" + tabulate(table, headers=headers, tablefmt='grid'))
        
        return
    
    def _prepare_submodel_data(self, fold):
        """Prepare training data for VBF Z and DY sub-models."""
        
        # Prepare VBF Z model data (VBF signal vs VBF Z background)
        if hasattr(self, 'data_bkg_vbfz') and not self.data_bkg_vbfz.empty:
            data_vbfz_sig = self.data_vbf_sig.copy()
            data_vbfz_bkg = self.data_bkg_vbfz.copy()
            
            data_vbfz_sig['label'] = 1
            data_vbfz_bkg['label'] = 0
            
            data_vbfz = pd.concat([data_vbfz_sig, data_vbfz_bkg], ignore_index=True)
            
            test_vbfz = data_vbfz[data_vbfz[self.randomIndex] % 4 == fold]
            val_vbfz = data_vbfz[(data_vbfz[self.randomIndex] - 1) % 4 == fold]
            train_vbfz = data_vbfz[((data_vbfz[self.randomIndex] - 2) % 4 == fold) | 
                                   ((data_vbfz[self.randomIndex] - 3) % 4 == fold)]
            
            X_train_vbfz = train_vbfz[self.train_variables].values
            X_val_vbfz = val_vbfz[self.train_variables].values
            y_train_vbfz = train_vbfz['label'].values
            y_val_vbfz = val_vbfz['label'].values
            wt_train_vbfz_raw = train_vbfz[self.weight].values
            wt_val_vbfz_raw = val_vbfz[self.weight].values
            
            # Reweight train and val independently
            wt_train_vbfz = self.reweight_binary(y_train_vbfz, wt_train_vbfz_raw)
            wt_val_vbfz = self.reweight_binary(y_val_vbfz, wt_val_vbfz_raw)
            
            # Standardize using same scaler
            X_train_vbfz_scaled = self.scalers[fold].transform(X_train_vbfz)
            X_val_vbfz_scaled = self.scalers[fold].transform(X_val_vbfz)
            
            # Store
            if not hasattr(self, 'X_train_vbfz'):
                self.X_train_vbfz = {}
                self.X_val_vbfz = {}
                self.y_train_vbfz = {}
                self.y_val_vbfz = {}
                self.wt_train_vbfz = {}
                self.wt_val_vbfz = {}
            
            self.X_train_vbfz[fold] = X_train_vbfz_scaled
            self.X_val_vbfz[fold] = X_val_vbfz_scaled
            self.y_train_vbfz[fold] = y_train_vbfz
            self.y_val_vbfz[fold] = y_val_vbfz
            self.wt_train_vbfz[fold] = wt_train_vbfz
            self.wt_val_vbfz[fold] = wt_val_vbfz
        
        # Prepare DY model data (VBF signal vs DY background)
        if hasattr(self, 'data_bkg_dy') and not self.data_bkg_dy.empty:
            data_dy_sig = self.data_vbf_sig.copy()
            data_dy_bkg = self.data_bkg_dy.copy()
            
            data_dy_sig['label'] = 1
            data_dy_bkg['label'] = 0
            
            data_dy = pd.concat([data_dy_sig, data_dy_bkg], ignore_index=True)
            
            test_dy = data_dy[data_dy[self.randomIndex] % 4 == fold]
            val_dy = data_dy[(data_dy[self.randomIndex] - 1) % 4 == fold]
            train_dy = data_dy[((data_dy[self.randomIndex] - 2) % 4 == fold) | 
                               ((data_dy[self.randomIndex] - 3) % 4 == fold)]
            
            X_train_dy = train_dy[self.train_variables].values
            X_val_dy = val_dy[self.train_variables].values
            y_train_dy = train_dy['label'].values
            y_val_dy = val_dy['label'].values
            wt_train_dy_raw = train_dy[self.weight].values
            wt_val_dy_raw = val_dy[self.weight].values
            
            # Reweight independently
            wt_train_dy = self.reweight_binary(y_train_dy, wt_train_dy_raw)
            wt_val_dy = self.reweight_binary(y_val_dy, wt_val_dy_raw)
            
            # Standardize using same scaler
            X_train_dy_scaled = self.scalers[fold].transform(X_train_dy)
            X_val_dy_scaled = self.scalers[fold].transform(X_val_dy)
            
            # Store
            if not hasattr(self, 'X_train_dy'):
                self.X_train_dy = {}
                self.X_val_dy = {}
                self.y_train_dy = {}
                self.y_val_dy = {}
                self.wt_train_dy = {}
                self.wt_val_dy = {}
            
            self.X_train_dy[fold] = X_train_dy_scaled
            self.X_val_dy[fold] = X_val_dy_scaled
            self.y_train_dy[fold] = y_train_dy
            self.y_val_dy[fold] = y_val_dy
            self.wt_train_dy[fold] = wt_train_dy
            self.wt_val_dy[fold] = wt_val_dy
    
    def create_3class_vbf_model(self, n_mass_features=3, n_topo_features=16, params=None):
        """
        Create the VBF 3-class DNN model with hierarchical structure.
        
        This model includes:
        1. VBF Z sub-model (VBF signal vs VBF Z background)
        2. DY sub-model (VBF signal vs DY background)
        3. No-mass sub-model (signal vs background, topology only)
        4. Mass-only sub-model (signal vs background, mass only)
        5. NEW: ggF vs VBF sub-model (using all features)
        6. Final 3-class classifier with outputs for VBF, ggF, and background
        """
        
        if params is None:
            params = self.params
        
        # Input layers
        input_mass_res = Input(shape=(n_mass_features,), name='mass_res')
        input_vbf_topo = Input(shape=(n_topo_features,), name='vbf_topo')
        
        # Extract hyperparameters
        dense_units_1 = params.get('dense_units_1', 64)
        dense_units_2 = params.get('dense_units_2', 32)
        dense_units_3 = params.get('dense_units_3', 16)
        dropout_rate = params.get('dropout_rate', 0.2)
        activation = params.get('activation', 'relu')

        mass_dense_units_1 = params.get('mass_dense_units_1', 64)
        mass_dense_units_2 = params.get('mass_dense_units_2', 32)
        mass_dense_units_3 = params.get('mass_dense_units_3', 16)
        mass_dropout_rate = params.get('mass_dropout_rate', 0.2)
        
        # NEW: ggF vs VBF sub-model parameters
        ggf_vbf_dense_units_1 = params.get('ggf_vbf_dense_units_1', 32)
        ggf_vbf_dense_units_2 = params.get('ggf_vbf_dense_units_2', 16)
        ggf_vbf_dense_units_3 = params.get('ggf_vbf_dense_units_3', 8)
        ggf_vbf_dropout_rate = params.get('ggf_vbf_dropout_rate', 0.15)
        
        # Sub-model 1: VBF signal vs VBF Z (uses both mass and topology)
        x1 = concatenate([input_mass_res, input_vbf_topo])
        x1 = Dense(dense_units_1, activation=activation)(x1)
        x1 = Dropout(dropout_rate)(x1)
        x1 = Dense(dense_units_2, activation=activation)(x1)
        x1 = Dropout(dropout_rate)(x1)
        x1 = Dense(dense_units_3, activation=activation)(x1)
        x1 = Dropout(dropout_rate)(x1)
        x1_out = Dense(1, activation='sigmoid', name='VBF_Z_Output')(x1)
        
        # Sub-model 2: VBF signal vs DY (uses both mass and topology)
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
        
        # NEW Sub-model 5: ggF vs VBF (uses ALL features - mass + topology)
        x5 = concatenate([input_mass_res, input_vbf_topo])
        x5 = Dense(ggf_vbf_dense_units_1, activation=activation)(x5)
        x5 = Dropout(ggf_vbf_dropout_rate)(x5)
        x5 = Dense(ggf_vbf_dense_units_2, activation=activation)(x5)
        x5 = Dropout(ggf_vbf_dropout_rate)(x5)
        x5 = Dense(ggf_vbf_dense_units_3, activation=activation)(x5)
        x5 = Dropout(ggf_vbf_dropout_rate)(x5)
        x5_out = Dense(1, activation='sigmoid', name='ggF_VBF_Output')(x5)
        
        # Create individual models (for pre-training if needed)
        model_vbf_z = Model(inputs=[input_mass_res, input_vbf_topo], outputs=x1_out)
        model_dy = Model(inputs=[input_mass_res, input_vbf_topo], outputs=x2_out)
        model_no_mass = Model(inputs=input_vbf_topo, outputs=x3_out)
        model_mass_only = Model(inputs=input_mass_res, outputs=x4_out)
        model_ggf_vbf = Model(inputs=[input_mass_res, input_vbf_topo], outputs=x5_out)
        
        # Compile sub-models
        for model in [model_vbf_z, model_dy, model_no_mass, model_mass_only, model_ggf_vbf]:
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
        intermediate_ggf_vbf = Model(inputs=model_ggf_vbf.input,
                                     outputs=model_ggf_vbf.layers[-2].output)
        
        # Get intermediate features
        input_vbf_z_features = intermediate_vbf_z([input_mass_res, input_vbf_topo])
        input_dy_features = intermediate_dy([input_mass_res, input_vbf_topo])
        input_no_mass_features = intermediate_no_mass(input_vbf_topo)
        input_mass_only_features = intermediate_mass_only(input_mass_res)
        input_ggf_vbf_features = intermediate_ggf_vbf([input_mass_res, input_vbf_topo])
        
        # Merge all intermediate features
        merged = concatenate([
            input_vbf_z_features,
            input_dy_features,
            input_no_mass_features,
            input_mass_only_features,
            input_ggf_vbf_features  # NEW
        ])
        
        # Final merged layers
        merged_dense_1 = params.get('merged_dense_1', 128)
        merged_dense_2 = params.get('merged_dense_2', 64)
        merged_dense_3 = params.get('merged_dense_3', 32)
        
        x_merged = Dense(merged_dense_1, activation=activation)(merged)
        x_merged = Dropout(dropout_rate)(x_merged)
        x_merged = Dense(merged_dense_2, activation=activation)(x_merged)
        x_merged = Dropout(dropout_rate)(x_merged)
        x_merged = Dense(merged_dense_3, activation=activation)(x_merged)
        x_merged = Dropout(dropout_rate)(x_merged)
        
        # Three-class output (VBF, ggF, background)
        output = Dense(3, activation='softmax', name='3class_output')(x_merged)
        
        # Create final merged model
        model = Model(inputs=[input_mass_res, input_vbf_topo], outputs=output)
        
        print("\n" + "="*60)
        print("3-Class VBF Model Architecture:")
        print("="*60)
        model.summary()
        print("="*60)
        
        return model, {
            'vbf_z': model_vbf_z,
            'dy': model_dy,
            'no_mass': model_no_mass,
            'mass_only': model_mass_only,
            'ggf_vbf': model_ggf_vbf  # NEW
        }

    def _build_optimizer_and_callbacks(self, params):
        """Build optimizer with learning rate schedule and callbacks."""
        
        # Learning rate schedule
        lr_schedule_type = params.get('lr_schedule', 'exp')
        initial_learning_rate = params.get('learning_rate', 0.001)
        
        use_schedule = False
        if lr_schedule_type == 'exp':
            decay_steps = params.get('decay_steps', 1600)
            decay_rate = params.get('decay_rate', 0.89)
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=False
            )
            use_schedule = True
        elif lr_schedule_type == 'cosine':
            first_decay_steps = params.get('first_decay_steps', 2000)
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate,
                decay_steps=first_decay_steps
            )
            use_schedule = True
        else:
            lr_schedule = initial_learning_rate
            use_schedule = False
        
        # Optimizer
        optimizer_type = params.get('optimizer', 'adam')
        if optimizer_type == 'adam':
            optimizer = tf_optimizers.Adam(learning_rate=lr_schedule)
        elif optimizer_type == 'nadam':
            optimizer = tf_optimizers.Nadam(learning_rate=lr_schedule)
        else:
            optimizer = tf_optimizers.Adam(learning_rate=lr_schedule)
        
        # Callbacks - only use ReduceLROnPlateau if NOT using a learning rate schedule
        callbacks = []
        
        if not use_schedule:
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=2
            )
            callbacks.append(reduce_lr)
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=2
        )
        callbacks.append(early_stopping)
        
        return optimizer, callbacks
    
    def trainModel(self, fold, params=None):
        """Train the 3-class VBF DNN model."""
        
        if params is None:
            params = self.params
        
        print("\n" + "="*60)
        print(f"Training 3-Class VBF DNN for fold {fold}")
        print("="*60)
        
        # Create model
        n_mass = len(self.mass_variables)
        n_topo = len(self.topo_variables)
        model, submodels = self.create_3class_vbf_model(n_mass, n_topo, params)
        
        # Build optimizer and callbacks
        optimizer, callbacks = self._build_optimizer_and_callbacks(params)
        
        # Compile model with categorical crossentropy for 3-class
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Pre-train sub-models if data is available
        pre_epoch = params.get('pre_epoch', 10)
        
        # Setup callbacks for sub-model training (use simple early stopping)
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=2
        )
        
        # Pre-train VBF Z model
        if hasattr(self, 'X_train_vbfz') and fold in self.X_train_vbfz:
            print(f"\nPre-training VBF Z sub-model for {pre_epoch} epochs...")
            X_mass_vbfz = self.X_train_vbfz[fold][:, :n_mass]
            X_topo_vbfz = self.X_train_vbfz[fold][:, n_mass:]
            submodels['vbf_z'].fit(
                [X_mass_vbfz, X_topo_vbfz],
                self.y_train_vbfz[fold],
                sample_weight=self.wt_train_vbfz[fold],
                validation_data=(
                    [self.X_val_vbfz[fold][:, :n_mass], self.X_val_vbfz[fold][:, n_mass:]],
                    self.y_val_vbfz[fold],
                    self.wt_val_vbfz[fold]
                ),
                epochs=pre_epoch,
                batch_size=params.get('pre_batch_size', 4096),
                callbacks=[early_stop],
                verbose=2
            )
        
        # Pre-train DY model
        if hasattr(self, 'X_train_dy') and fold in self.X_train_dy:
            print(f"\nPre-training DY sub-model for {pre_epoch} epochs...")
            X_mass_dy = self.X_train_dy[fold][:, :n_mass]
            X_topo_dy = self.X_train_dy[fold][:, n_mass:]
            submodels['dy'].fit(
                [X_mass_dy, X_topo_dy],
                self.y_train_dy[fold],
                sample_weight=self.wt_train_dy[fold],
                validation_data=(
                    [self.X_val_dy[fold][:, :n_mass], self.X_val_dy[fold][:, n_mass:]],
                    self.y_val_dy[fold],
                    self.wt_val_dy[fold]
                ),
                epochs=pre_epoch,
                batch_size=params.get('batch_size', 8192),
                callbacks=[early_stop],
                verbose=2
            )
        
        # Pre-train no-mass model (topology only)
        print(f"\nPre-training no-mass sub-model for {pre_epoch} epochs...")
        X_mass_train = self.X_train[fold][:, :n_mass]
        X_topo_train = self.X_train[fold][:, n_mass:]
        X_mass_val = self.X_val[fold][:, :n_mass]
        X_topo_val = self.X_val[fold][:, n_mass:]
        
        # Convert 3-class labels to binary (signal vs background)
        y_train_binary = (self.y_train[fold][:, 0] | self.y_train[fold][:, 1]).astype(int)
        y_val_binary = (self.y_val[fold][:, 0] | self.y_val[fold][:, 1]).astype(int)
        
        submodels['no_mass'].fit(
            X_topo_train,
            y_train_binary,
            sample_weight=self.train_wt[fold],
            validation_data=(X_topo_val, y_val_binary, self.val_wt[fold]),
            epochs=pre_epoch,
            batch_size=params.get('batch_size', 8192),
            callbacks=[early_stop],
            verbose=2
        )
        
        # Pre-train mass-only model
        print(f"\nPre-training mass-only sub-model for {pre_epoch} epochs...")
        submodels['mass_only'].fit(
            X_mass_train,
            y_train_binary,
            sample_weight=self.train_wt[fold],
            validation_data=(X_mass_val, y_val_binary, self.val_wt[fold]),
            epochs=pre_epoch,
            batch_size=params.get('batch_size', 8192),
            callbacks=[early_stop],
            verbose=2
        )
        
        # Pre-train ggF-VBF model (signal only - VBF vs ggF)
        print(f"\nPre-training ggF-VBF sub-model for {pre_epoch} epochs...")
        # Only use signal events (VBF and ggF)
        signal_mask_train = (self.y_train[fold][:, 0] == 1) | (self.y_train[fold][:, 1] == 1)
        signal_mask_val = (self.y_val[fold][:, 0] == 1) | (self.y_val[fold][:, 1] == 1)
        
        X_mass_train_sig = X_mass_train[signal_mask_train]
        X_topo_train_sig = X_topo_train[signal_mask_train]
        X_mass_val_sig = X_mass_val[signal_mask_val]
        X_topo_val_sig = X_topo_val[signal_mask_val]
        
        # VBF=1, ggF=0
        y_train_ggf_vbf = self.y_train[fold][signal_mask_train][:, 0].astype(int)
        y_val_ggf_vbf = self.y_val[fold][signal_mask_val][:, 0].astype(int)
        wt_train_sig = self.train_wt[fold][signal_mask_train]
        wt_val_sig = self.val_wt[fold][signal_mask_val]
        
        submodels['ggf_vbf'].fit(
            [X_mass_train_sig, X_topo_train_sig],
            y_train_ggf_vbf,
            sample_weight=wt_train_sig,
            validation_data=(
                [X_mass_val_sig, X_topo_val_sig],
                y_val_ggf_vbf,
                wt_val_sig
            ),
            epochs=pre_epoch,
            batch_size=params.get('pre_batch_size', 4096),
            callbacks=[early_stop],
            verbose=2
        )

        # # Freeze all sub-model layers (lock their parameters)
        # print("\nFreezing sub-model layers...")
        # trainable_layer_names = []
        
        # # Find the final concatenate layer (the one that merges all sub-model outputs)
        # final_concat_idx = -1
        # for idx, layer in enumerate(model.layers):
        #     if 'concatenate' in layer.name:
        #         final_concat_idx = idx
        
        # print(f"Final concatenate layer found at index: {final_concat_idx}")
        
        # # Freeze all layers before and including the final concatenate
        # # Only make the dense layers after concatenation trainable
        # for idx, layer in enumerate(model.layers):
        #     if idx <= final_concat_idx:
        #         layer.trainable = False
        #     else:
        #         # These are the merged dense layers
        #         layer.trainable = True
        #         trainable_layer_names.append(layer.name)
        
        # print(f"Trainable layers (merged layers only): {trainable_layer_names}")
        # print(f"Total trainable parameters: {sum([K.count_params(w) for w in model.trainable_weights])}")
        # print(f"Total non-trainable parameters: {sum([K.count_params(w) for w in model.non_trainable_weights])}")
        
        # Re-compile model after freezing layers
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model.summary()
        
        # Prepare training data
        X_mass_train = self.X_train[fold][:, :n_mass]
        X_topo_train = self.X_train[fold][:, n_mass:]
        X_mass_val = self.X_val[fold][:, :n_mass]
        X_topo_val = self.X_val[fold][:, n_mass:]
        
        # Train merged model (only the final layers are trainable)
        print(f"\nTraining merged 3-class model (merged layers only)...")
        history = model.fit(
            [X_mass_train, X_topo_train],
            self.y_train[fold],
            sample_weight=self.train_wt[fold],
            validation_data=(
                [X_mass_val, X_topo_val],
                self.y_val[fold],
                self.val_wt[fold]
            ),
            epochs=self._epochs,
            batch_size=params.get('batch_size', 8192),
            callbacks=callbacks,
            verbose=2
        )
        
        # Store model and history
        self.models[fold] = model
        self.histories[fold] = history
        
        # Make predictions
        print("\nMaking predictions...")
        self.y_train_pred[fold] = model.predict([X_mass_train, X_topo_train], batch_size=81920, verbose=2)
        self.y_val_pred[fold] = model.predict([X_mass_val, X_topo_val], batch_size=81920, verbose=2)
        
        X_mass_test = self.X_test[fold][:, :n_mass]
        X_topo_test = self.X_test[fold][:, n_mass:]
        self.y_test_pred[fold] = model.predict([X_mass_test, X_topo_test], batch_size=81920, verbose=2)
        
        return model, history
    
    def evaluate_on_val(self, fold):
        """Evaluate model on validation set."""
        
        if fold not in self.models:
            print(f"Model for fold {fold} not found")
            return
        
        model = self.models[fold]
        n_mass = len(self.mass_variables)
        
        X_mass_val = self.X_val[fold][:, :n_mass]
        X_topo_val = self.X_val[fold][:, n_mass:]
        
        val_loss, val_acc = model.evaluate(
            [X_mass_val, X_topo_val],
            self.y_val[fold],
            sample_weight=self.val_wt[fold],
            batch_size=81920,
            verbose=2
        )
        
        print(f"\nFold {fold} Validation Results:")
        print(f"  Loss: {val_loss:.4f}")
        print(f"  Accuracy: {val_acc:.4f}")

        return val_loss, val_acc

    def transformScore(self, fold=0):
        """
        Transform derived scores to uniform distribution using weighted quantile transformer.
        Creates two transformers:
        1. vbf_score_tsf: transforms (VBF - ggF) score on signal only
        2. bdt_score_tsf: transforms (VBF + ggF - bkg) score on signal only
        
        Args:
            fold: fold number
        """
        
        print(f"\nTransforming scores for fold {fold}...")
        
        # Get predictions: shape (n_samples, 3) where columns are [VBF, ggF, bkg]
        y_train_pred = self.y_train_pred[fold]
        y_train_labels = self.y_train[fold]
        
        # Extract VBF, ggF, bkg outputs
        score_vbf = y_train_pred[:, 0]
        score_ggf = y_train_pred[:, 1]
        score_bkg = y_train_pred[:, 2]
        
        # Calculate derived scores
        vbf_score = score_vbf - score_ggf  # VBF - ggF
        bdt_score = (score_vbf + score_ggf) - score_bkg  # (VBF + ggF) - bkg
        
        # Get signal mask (VBF signal or ggF signal)
        signal_mask = (y_train_labels[:, 0] == 1) | (y_train_labels[:, 1] == 1)
        
        # Extract signal events only
        vbf_score_sig = vbf_score[signal_mask]
        bdt_score_sig = bdt_score[signal_mask]
        weights_sig = self.train_wt_raw[fold][signal_mask]  # Use RAW weights
        
        print(f"  Signal events for transformation: {np.sum(signal_mask)}")
        
        # Create and fit transformer for vbf_score (VBF - ggF)
        print("  Fitting transformer for vbf_score (VBF - ggF)...")
        vbf_score_tsf = WeightedQuantileTransformer(n_quantiles=1000, 
                                                      output_distribution='uniform')
        vbf_score_tsf.fit(vbf_score_sig.reshape(-1, 1), sample_weight=weights_sig)
        
        # Create and fit transformer for bdt_score ((VBF + ggF) - bkg)
        print("  Fitting transformer for bdt_score ((VBF + ggF) - bkg)...")
        bdt_score_tsf = WeightedQuantileTransformer(n_quantiles=1000,
                                                      output_distribution='uniform')
        bdt_score_tsf.fit(bdt_score_sig.reshape(-1, 1), sample_weight=weights_sig)
        
        # Store transformers
        if fold not in self.m_tsf:
            self.m_tsf[fold] = {}
        
        self.m_tsf[fold]['vbf_score_tsf'] = vbf_score_tsf
        self.m_tsf[fold]['bdt_score_tsf'] = bdt_score_tsf
        
        print(f"  Score transformation completed for fold {fold}")
        print(f"    - vbf_score range: [{vbf_score_sig.min():.4f}, {vbf_score_sig.max():.4f}]")
        print(f"    - bdt_score range: [{bdt_score_sig.min():.4f}, {bdt_score_sig.max():.4f}]")
    
    def plotROC(self, fold, save=True, show=False):
        """Plot ROC curves for 3-class model. Each class gets its own plot with train/val/test curves."""
        
        if fold not in self.y_val_pred:
            print(f"No predictions available for fold {fold}")
            return
        
        if not os.path.isdir(f"{self._plotFolder}/roc/"):
            os.makedirs(f"{self._plotFolder}/roc/")
        
        # Get predictions and true labels for all datasets
        y_train_pred = self.y_train_pred[fold]
        y_train_true = self.y_train[fold]
        y_val_pred = self.y_val_pred[fold]
        y_val_true = self.y_val[fold]
        y_test_pred = self.y_test_pred[fold]
        y_test_true = self.y_test[fold]
        
        class_names = ['VBF Signal', 'ggF Signal', 'Background']
        colors = {'train': '#1f77b4', 'val': '#ff7f0e', 'test': '#2ca02c'}
        
        # Create a separate figure for each class
        for i, class_name in enumerate(class_names):
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot train ROC
            fpr_train, tpr_train, _ = roc_curve(y_train_true[:, i], y_train_pred[:, i], 
                                                 sample_weight=self.train_wt[fold])
            sorted_indices = np.argsort(fpr_train)
            fpr_train = np.array(fpr_train)[sorted_indices]
            tpr_train = np.array(tpr_train)[sorted_indices]
            auc_train = auc(fpr_train, tpr_train)
            ax.plot(fpr_train, tpr_train, color=colors['train'], 
                   label=f'Train (AUC = {auc_train:.3f})', linewidth=2)
            
            # Plot val ROC
            fpr_val, tpr_val, _ = roc_curve(y_val_true[:, i], y_val_pred[:, i], 
                                            sample_weight=self.val_wt[fold])
            sorted_indices = np.argsort(fpr_val)
            fpr_val = np.array(fpr_val)[sorted_indices]
            tpr_val = np.array(tpr_val)[sorted_indices]
            auc_val = auc(fpr_val, tpr_val)
            ax.plot(fpr_val, tpr_val, color=colors['val'], 
                   label=f'Val (AUC = {auc_val:.3f})', linewidth=2)
            
            # Plot test ROC
            fpr_test, tpr_test, _ = roc_curve(y_test_true[:, i], y_test_pred[:, i], 
                                              sample_weight=self.test_wt[fold])
            sorted_indices = np.argsort(fpr_test)
            fpr_test = np.array(fpr_test)[sorted_indices]
            tpr_test = np.array(tpr_test)[sorted_indices]
            auc_test = auc(fpr_test, tpr_test)
            ax.plot(fpr_test, tpr_test, color=colors['test'], 
                   label=f'Test (AUC = {auc_test:.3f})', linewidth=2)
            
            # Plot diagonal
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title(f'ROC Curve - {class_name} (Fold {fold})', fontsize=14, fontweight='bold')
            ax.legend(loc="lower right", fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save:
                # Save each class as a separate file
                class_suffix = class_name.lower().replace(' ', '_')
                plt.savefig(f"{self._plotFolder}/roc/roc_{self._region}_{class_suffix}_fold{fold}.pdf")
                plt.savefig(f"{self._plotFolder}/roc/roc_{self._region}_{class_suffix}_fold{fold}.png")
            
            if show:
                plt.show()
            else:
                plt.close()
    
    def save(self, fold):
        """Save model, scaler, and transformers for a fold."""
        
        if not os.path.isdir(self._outputFolder):
            os.makedirs(self._outputFolder)
        
        # Save model
        if fold in self.models:
            model_path = f"{self._outputFolder}/model_fold_{fold}.keras"
            self.models[fold].save(model_path)
            print(f"Model saved to {model_path}")
        
        # Save scaler
        if fold in self.scalers:
            scaler_path = f"{self._outputFolder}/scaler_fold_{fold}.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers[fold], f)
            print(f"Scaler saved to {scaler_path}")
        
        # Save the two derived score transformers
        if fold in self.m_tsf:
            # Save vbf_score transformer (VBF - ggF)
            if 'vbf_score_tsf' in self.m_tsf[fold] and self.m_tsf[fold]['vbf_score_tsf'] is not None:
                vbf_tsf_path = f"{self._outputFolder}/vbf_score_tsf_fold_{fold}.pkl"
                with open(vbf_tsf_path, 'wb') as f:
                    pickle.dump(self.m_tsf[fold]['vbf_score_tsf'], f)
                print(f"vbf_score transformer saved to {vbf_tsf_path}")
            
            # Save bdt_score transformer ((VBF + ggF) - bkg)
            if 'bdt_score_tsf' in self.m_tsf[fold] and self.m_tsf[fold]['bdt_score_tsf'] is not None:
                bdt_tsf_path = f"{self._outputFolder}/bdt_score_tsf_fold_{fold}.pkl"
                with open(bdt_tsf_path, 'wb') as f:
                    pickle.dump(self.m_tsf[fold]['bdt_score_tsf'], f)
                print(f"bdt_score transformer saved to {bdt_tsf_path}")
        
        # Save training history
        if fold in self.histories:
            history_path = f"{self._outputFolder}/history_fold_{fold}.pkl"
            with open(history_path, 'wb') as f:
                pickle.dump(self.histories[fold].history, f)
            print(f"Training history saved to {history_path}")
    
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
        exp_dir = f"models/optuna_3class_{self._region}_{suffix}/"
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
                'dense_units_1': trial.suggest_categorical('dense_units_1', [32, 64, 128, 256]),
                'dense_units_2': trial.suggest_categorical('dense_units_2', [16, 32, 64, 128]),
                'dense_units_3': trial.suggest_categorical('dense_units_3', [8, 16, 32, 64]),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.05),
                'mass_dense_units_1': trial.suggest_categorical('mass_dense_units_1', [4, 8, 16, 32]),
                'mass_dense_units_2': trial.suggest_categorical('mass_dense_units_2', [2, 4, 8, 16]),
                'mass_dense_units_3': trial.suggest_categorical('mass_dense_units_3', [2, 4, 8]),
                'mass_dropout_rate': trial.suggest_float('mass_dropout_rate', 0.1, 0.4, step=0.05),
                'ggf_vbf_dense_units_1': trial.suggest_categorical('ggf_vbf_dense_units_1', [16, 32, 64, 128]),
                'ggf_vbf_dense_units_2': trial.suggest_categorical('ggf_vbf_dense_units_2', [8, 16, 32, 64]),
                'ggf_vbf_dense_units_3': trial.suggest_categorical('ggf_vbf_dense_units_3', [4, 8, 16, 32]),
                'ggf_vbf_dropout_rate': trial.suggest_float('ggf_vbf_dropout_rate', 0.1, 0.4, step=0.05),
                'learning_rate': trial.suggest_float('learning_rate', 5e-5, 1e-3, log=True),
                'optimizer': trial.suggest_categorical('optimizer', ['adam', 'nadam']),
                'activation': trial.suggest_categorical('activation', ['relu', 'elu', 'selu', 'tanh']),
                'batch_size': trial.suggest_categorical('batch_size', [2048, 4096, 8192, 12288, 16384, 20480, 32768, 40960]),
                'pre_batch_size': trial.suggest_categorical('pre_batch_size', [1024, 2048, 4096, 8192, 12288]),
                'pre_epoch': trial.suggest_int('pre_epoch', 30, 80, step=5),
                'decay_rate': trial.suggest_float('decay_rate', 0.8, 0.9),
                'decay_steps': trial.suggest_int('decay_steps', 500, 3500, step=200),
                'first_decay_steps': trial.suggest_int('first_decay_steps', 500, 3500, step=200),
                'merged_dense_1': trial.suggest_categorical('merged_dense_1', [32, 64, 128, 256, 384, 512]),
                'merged_dense_2': trial.suggest_categorical('merged_dense_2', [16, 32, 64, 128, 256, 384]),
                'merged_dense_3': trial.suggest_categorical('merged_dense_3', [16, 32, 64, 128, 256]),
            }
            logger.info(f"Trial {trial.number}: {params}")
            
            # Train model with these parameters
            try:
                metrics = []
                for fold in self._folds:
                    model, history = self.trainModel(fold, params)
                    
                    # Get validation accuracy
                    _, val_acc = self.evaluate_on_val(fold)
                    
                    # Get training accuracy from the last epoch
                    train_acc = history.history['accuracy'][-1]
                    
                    # Calculate metric: 2 * val_acc - train_acc
                    # This penalizes overfitting (when train_acc >> val_acc)
                    metric = 2.0 * val_acc - train_acc
                    metrics.append(metric)
                    
                    logger.info(f"Fold {fold}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, metric={metric:.4f}")
                
                avg_metric = float(np.mean(metrics))
                logger.info(f"Average metric: {avg_metric:.4f}")
                return avg_metric
                
            except Exception as e:
                logger.error(f"Trial failed: {e}")
                return -999.0  # Return a very negative value for failed trials
        
        # Create Optuna study
        study_name = f"NN_3class_{self._region}_allfolds"
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
    handler = VBF3ClassDNNHandler(args.config, args.region)
    
    # Set input folder if provided
    if args.inputFolder:
        handler._inputFolder = args.inputFolder
    
    # Read data
    handler.readData()
    
    # Plot correlations if requested
    if args.corr:
        print("\nPlotting correlations...")
        handler.plot_correlation(handler.data_vbf_sig, 'VBF_signal', fold=0)
        handler.plot_correlation(handler.data_ggf_sig, 'ggF_signal', fold=0)
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
            if args.roc:
                handler.plotROC(fold)
            handler.transformScore(fold)
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
            
            # Evaluate on validation set
            handler.evaluate_on_val(fold)
            
            # Plot ROC
            if args.roc:
                handler.plotROC(fold)
        
            # Transform scores (creates vbf_score_tsf and bdt_score_tsf)
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
