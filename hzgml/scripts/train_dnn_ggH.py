#!/usr/bin/env python
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import gc
import hashlib
import importlib.util
import json
import logging
import math
import mmap
import pickle
import random
import resource
import shutil
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
import uproot

try:
    import numpy.core as _numpy_core
    import numpy.core.numeric as _numpy_core_numeric
    sys.modules.setdefault("numpy._core", _numpy_core)
    sys.modules.setdefault("numpy._core.numeric", _numpy_core_numeric)
except Exception:
    pass

from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from weighted_quantile_transformer import WeightedQuantileTransformer
from dnn_reso_training import weight_policy

try:
    import ROOT
    ROOT.gErrorIgnoreLevel = ROOT.kError + 1
except ImportError:
    ROOT = None

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

OPTUNA_EVAL_AUC_DOMAIN_METRIC = "eval_auc_domain"
OPTUNA_EVAL_AUC_DOM_MASS_METRIC = "eval_auc_dom_mass"
PANDAS_DATAFRAME_SHUFFLE_SEED = 123
NPY_CACHE_ARRAY_KEYS = (
    "x_train",
    "x_val",
    "x_test",
    "x_test_sig",
    "x_test_bkg",
    "y_train",
    "y_val",
    "y_test",
    "train_wt",
    "val_wt",
    "test_wt",
    "train_wt_raw",
    "val_wt_raw",
    "test_wt_raw",
    "train_event_weight",
    "val_event_weight",
    "test_event_weight",
    "mass_x_train",
    "mass_x_val",
    "mass_wt_raw_train",
    "mass_wt_raw_val",
    "mass_event_weight_train",
    "mass_event_weight_val",
    "mass_mass_train",
    "mass_mass_val",
    "train_mass",
    "val_mass",
    "test_mass",
    "train_mass_norm",
    "val_mass_norm",
    "test_mass_norm",
    "domain_x_train",
    "domain_x_val",
    "domain_y_train",
    "domain_y_val",
    "domain_wt_train",
    "domain_wt_val",
)

MASS_VARIABLE = "diMufsr_kit_BSC_mass"
SCALED_BY_MASS_VARIABLES = ["diMufsr_kit_BSC_pt", "jet1_pt", "jet2_pt", "dijet_pt"]
MISSING_JET_SENTINEL_MAX = -90.0
SCORE_MASS_MIN = 115.0
SCORE_MASS_MAX = 135.0
RUN2_POINTS = np.array(
    [
        [0.04284412032816773, 0.009954731636669087],
        [0.2014585232452142, 0.06513180342367707],
        [0.09936189608021878, 0.0262847852723399],
        [0.2999088422971741, 0.11328885621920834],
        [0.39927073837739285, 0.17197605418968803],
        [0.5013673655423884, 0.23841847264654056],
        [0.5989061075660893, 0.3187485925972288],
        [0.699179580674567, 0.4146999568238319],
        [0.7994530537830447, 0.5298316906283708],
        [0.9006381039197812, 0.6956079422694142],
        [0.9963536918869644, 0.9731445166958551],
        [0.006381039197812215, 0.001011407356996462],
        [0.14585232452142205, 0.04408947986472283],
        [0.35004557885141296, 0.14343338450581608],
        [0.24886052871467637, 0.0894796214122385],
        [0.5496809480401094, 0.27818581134958453],
        [0.649954421148587, 0.3619267555526089],
        [0.7502278942570647, 0.46662232927030695],
        [0.8477666362807657, 0.6070874171543386],
        [0.95077484047402, 0.8043012047240471],
        [0.4512306289881495, 0.20433597178569418],
    ],
    dtype=np.float64,
)
DOMAIN_LOW_MIN = 110.0
DOMAIN_LOW_MAX = 115.0
DOMAIN_HIGH_MIN = 135.0
DOMAIN_HIGH_MAX = 150.0
DOMAIN_REQUIRED_NJETS = 0
SHAPE_EVENT_WEIGHT_COLUMN = "__shape_event_weight__"
SCORE_BIN_MASS_METRIC_KEYS = (
    "score_bin_mass_mean_variance",
    "score_bin_mass_mean_rms",
    "score_bin_mass_shape_rms",
    "score_bin_mass_shape_max_abs",
)
PHYSICS_FEATURE_WEIGHT_MODES = ("none", "score", "mass", "score_mass")
PHYSICS_FEATURE_MONITOR_KEYS = (
    "physics_feature_cov_loss",
    "physics_feature_cov_loss_none",
    "physics_feature_cov_loss_score",
    "physics_feature_cov_loss_mass",
    "physics_feature_cov_loss_score_mass",
)
EPOCH_RESULT_KEYS = (
    "total_loss",
    "score_loss",
    "domain_loss",
    "domain_bce",
    "domain_bce_no_score_weight",
    "domain_closeness_loss",
    "mass_loss",
    "mass_hsic_loss",
    *PHYSICS_FEATURE_MONITOR_KEYS,
    "mass_pred_mean",
    "mass_pred_std",
    "n_mass_events",
    "asimov_loss",
    "asimov_z",
    "asimov_active",
)
HISTORY_KEYS = (
    "train_total_loss",
    "train_score_loss",
    "train_domain_loss",
    "train_domain_bce",
    "train_domain_bce_no_score_weight",
    "train_domain_closeness_loss",
    "train_mass_loss",
    "train_mass_hsic_loss",
    "train_physics_feature_cov_loss",
    "train_physics_feature_cov_loss_none",
    "train_physics_feature_cov_loss_score",
    "train_physics_feature_cov_loss_mass",
    "train_physics_feature_cov_loss_score_mass",
    "train_mass_pred_mean",
    "train_mass_pred_std",
    "train_n_mass_events",
    "train_asimov_loss",
    "train_asimov_z",
    "train_asimov_active",
    "train_bce_only",
    "train_reg_loss",
    "train_eval_total_loss",
    "train_eval_score_loss",
    "train_eval_domain_loss",
    "train_eval_domain_bce",
    "train_eval_domain_bce_no_score_weight",
    "train_eval_domain_closeness_loss",
    "train_eval_mass_loss",
    "train_eval_mass_hsic_loss",
    "train_eval_physics_feature_cov_loss",
    "train_eval_physics_feature_cov_loss_none",
    "train_eval_physics_feature_cov_loss_score",
    "train_eval_physics_feature_cov_loss_mass",
    "train_eval_physics_feature_cov_loss_score_mass",
    "train_eval_mass_pred_mean",
    "train_eval_mass_pred_std",
    "train_eval_n_mass_events",
    "train_eval_asimov_loss",
    "train_eval_asimov_z",
    "train_eval_asimov_active",
    "train_eval_bce_only",
    "train_eval_reg_loss",
    "val_total_loss",
    "val_score_loss",
    "val_domain_loss",
    "val_domain_bce",
    "val_domain_bce_no_score_weight",
    "val_domain_closeness_loss",
    "val_mass_loss",
    "val_mass_hsic_loss",
    "val_physics_feature_cov_loss",
    "val_physics_feature_cov_loss_none",
    "val_physics_feature_cov_loss_score",
    "val_physics_feature_cov_loss_mass",
    "val_physics_feature_cov_loss_score_mass",
    "val_mass_pred_mean",
    "val_mass_pred_std",
    "val_n_mass_events",
    "val_score_bin_mass_mean_variance",
    "val_score_bin_mass_mean_rms",
    "val_score_bin_mass_shape_rms",
    "val_score_bin_mass_shape_max_abs",
    "val_asimov_loss",
    "val_asimov_z",
    "val_asimov_active",
    "val_bce_only",
    "val_reg_loss",
    "val_auc",
    "early_stop_metric",
    "learning_rate",
    "domain_lambda",
    "epoch_seconds",
    "train_seconds",
    "train_eval_seconds",
    "val_seconds",
    "train_fetch_seconds",
    "train_step_seconds",
    "train_eval_fetch_seconds",
    "train_eval_step_seconds",
    "val_fetch_seconds",
    "val_step_seconds",
)
DOMAIN_METRIC_DEFINITIONS = {
    "domain_reweight_definition": "domain sample weights multiplied by exp(score_reweight_exponent * classifier_score_t), where classifier_score_t is a fixed signal-uniform score after warmup; raw classifier_score is used before the transformer is ready",
    "domain_bce_no_score_weight_definition": "domain train-reference normalized label-balanced weighted BCE without classifier score reweighting; monitor only",
    "domain_loss_definition": "score-reweighted label-balanced domain_bce - ln2; monitor only",
    "domain_closeness_loss_definition": "sqrt((score-reweighted label-balanced domain_bce - ln2)^2 + epsilon^2), epsilon=1e-7; total_loss = score_bce + domain_lambda * domain_closeness_loss + regularization",
}
DOMAIN_CLOSENESS_EPSILON = 1e-7
PHYSICS_SCORE_QUANTILE_PROBS = tf.constant(
    [0.1 * index for index in range(1, 10)] + [0.05 + 0.10 * index for index in range(10)],
    dtype=tf.float32,
)


def physics_mass_features(mass):
    mass = tf.reshape(tf.cast(mass, tf.float32), (-1, 1))
    z = (mass - 125.0) / 20.0
    polynomial = tf.concat([tf.pow(z, power) for power in (1, 2, 3, 4)], axis=1)
    thresholds = tf.constant(np.arange(110.0, 150.0 + 0.1, 2.0), dtype=tf.float32)
    threshold_features = tf.sigmoid((mass - thresholds[None, :]) / 0.5)
    local_centers = tf.constant([118.0, 120.0, 122.0, 124.0, 125.0, 126.0, 128.0, 130.0, 132.0], dtype=tf.float32)
    local_bumps = tf.exp(-0.5 * tf.square((mass - local_centers[None, :]) / 1.5))
    wide_centers = tf.constant([115.0, 120.0, 125.0, 130.0, 135.0], dtype=tf.float32)
    wide_bumps = tf.exp(-0.5 * tf.square((mass - wide_centers[None, :]) / 4.0))
    sideband_features = tf.concat(
        [
            tf.sigmoid((115.0 - mass) / 0.5),
            tf.sigmoid((mass - 135.0) / 0.5),
            tf.sigmoid((120.0 - mass) / 1.0),
            tf.sigmoid((mass - 130.0) / 1.0),
        ],
        axis=1,
    )
    return tf.concat([polynomial, threshold_features, local_bumps, wide_bumps, sideband_features], axis=1)


def physics_score_features(score, score_quantiles):
    score = tf.reshape(tf.cast(score, tf.float32), (-1, 1))
    score_quantiles = tf.reshape(tf.cast(score_quantiles, tf.float32), (1, -1))
    polynomial = tf.concat([tf.pow(score, power) for power in (1, 2, 3, 4)], axis=1)
    cut_features = tf.sigmoid((score - score_quantiles) / 0.02)
    return tf.concat([polynomial, cut_features], axis=1)


def normalized_feature_cov_loss(mass_features, score_features, weights):
    mass_features = tf.cast(mass_features, tf.float32)
    score_features = tf.cast(score_features, tf.float32)
    weights = tf.reshape(tf.maximum(tf.cast(weights, tf.float32), 0.0), (-1,))
    weight_sum = tf.reduce_sum(weights)

    def compute_loss():
        norm_weights = weights / tf.maximum(weight_sum, 1e-12)
        mass_mean = tf.reduce_sum(mass_features * norm_weights[:, None], axis=0)
        score_mean = tf.reduce_sum(score_features * norm_weights[:, None], axis=0)
        mass_centered = mass_features - mass_mean[None, :]
        score_centered = score_features - score_mean[None, :]
        c_ms = tf.matmul(mass_centered, score_centered * norm_weights[:, None], transpose_a=True)
        c_mm = tf.matmul(mass_centered, mass_centered * norm_weights[:, None], transpose_a=True)
        c_ss = tf.matmul(score_centered, score_centered * norm_weights[:, None], transpose_a=True)
        numerator = tf.reduce_sum(tf.square(c_ms))
        denominator = tf.sqrt(tf.reduce_sum(tf.square(c_mm)) * tf.reduce_sum(tf.square(c_ss)) + 1e-12)
        return numerator / tf.maximum(denominator, 1e-12)

    return tf.cond(weight_sum > 0.0, compute_loss, lambda: tf.constant(0.0, dtype=tf.float32))


def _weighted_quantiles_tensor(values, weights, quantile_probs):
    values = tf.reshape(tf.cast(values, tf.float32), (-1,))
    weights = tf.reshape(tf.maximum(tf.cast(weights, tf.float32), 0.0), (-1,))
    quantile_probs = tf.reshape(tf.cast(quantile_probs, tf.float32), (-1,))
    weight_sum = tf.reduce_sum(weights)

    def compute_quantiles():
        order = tf.argsort(values, stable=True)
        sorted_values = tf.gather(values, order)
        sorted_weights = tf.gather(weights, order)
        cumulative_weights = tf.cumsum(sorted_weights)
        targets = tf.clip_by_value(quantile_probs, 0.0, 1.0) * weight_sum
        indices = tf.searchsorted(cumulative_weights, targets, side="left")
        indices = tf.clip_by_value(indices, 0, tf.shape(sorted_values)[0] - 1)
        return tf.stop_gradient(tf.gather(sorted_values, indices))

    return tf.cond(
        (tf.shape(values)[0] > 0) & (weight_sum > 0.0),
        compute_quantiles,
        lambda: tf.zeros_like(quantile_probs),
    )


def getArgs():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        action="store",
        default="hzgml/data/training_config_BDT_ggH_multiyear_template.py",
        help="Region config file",
    )
    parser.add_argument("-i", "--inputFolder", action="store", help="directory of training inputs")
    parser.add_argument("-o", "--outputFolder", action="store", help="directory for outputs")
    parser.add_argument(
        "-r",
        "--region",
        action="store",
        choices=["two_jet", "one_jet", "zero_jet", "zero_to_one_jet", "VH_ttH", "VBF", "all_jet", "ggH", "VBFsep", "ggH_sep"],
        default="ggH",
        help="Region to process",
    )
    parser.add_argument(
        "-f",
        "--fold",
        action="store",
        type=int,
        nargs="+",
        choices=[0, 1, 2, 3],
        default=[0, 1, 2, 3],
        help="Specify the folds for training",
    )
    parser.add_argument("--inputTree", action="store", default="inclusive", help="name of input tree")
    parser.add_argument("-p", "--params", action="store", type=json.loads, default=None, help="JSON string of DNN hyperparameters")
    parser.add_argument("--hyperparams_path", action="store", default=None, help="path of hyperparameters json")
    parser.add_argument("--save", action="store_true", help="save model/scaler/transformer")
    parser.add_argument("--corr", dest="corr", action="store_true", default=None, help="plot correlations")
    parser.add_argument("--no-corr", dest="corr", action="store_false", help="disable high-memory correlation plots")
    parser.add_argument("--roc", action="store_true", default=True, help="plot ROC")
    parser.add_argument("--optuna", action="store_true", default=False, help="run hyperparameter tuning using optuna")
    parser.add_argument(
        "--optuna_metric",
        action="store",
        default=OPTUNA_EVAL_AUC_DOM_MASS_METRIC,
        choices=[
            OPTUNA_EVAL_AUC_DOMAIN_METRIC,
            OPTUNA_EVAL_AUC_DOM_MASS_METRIC,
            "eval_auc",
            "sqrt_eval_auc_minus_train_auc",
            "eval_auc_minus_train_auc",
            "eval_auc_over_train_auc",
        ],
        help="Optuna metric to maximize over all folds",
    )
    parser.add_argument("--n-calls", action="store", type=int, default=36, help="number of optuna trials")
    parser.add_argument("--continue-optuna", action="store", type=int, default=0, help="continue existing optuna study")
    parser.add_argument("--optuna-suffix", action="store", default="", help="suffix for optuna experiment directory")
    parser.add_argument("-s", "--shield", action="store", type=int, default=-1, help="training variable index to remove")
    parser.add_argument("-a", "--add", action="store", type=int, default=-1, help="test variable index to add")
    parser.add_argument("--reweight", action="store_true", default=False, help="flatten background mass spectrum before classification")
    parser.add_argument("--reso", action="store_true", default=False, help="apply resolution weighting to signal")
    parser.add_argument("--reso-bkg", action="store_true", default=False, help="apply resolution weighting to background")
    parser.add_argument("--ptdecor", action="store_true", default=False, help="divide selected pT inputs by diMufsr_kit_BSC_mass before training")
    parser.add_argument("--seed", action="store", type=int, default=20250408, help="global random seed")
    parser.add_argument("--years", action="store", nargs="+", default=None, help="restrict training to selected source years, e.g. --years 2024")
    parser.add_argument("--max-chunks-per-sample", action="store", type=int, default=0, help="for smoke tests: limit number of uproot chunks loaded per sample; 0 means no limit")
    parser.add_argument("--train-cache-dir", action="store", default="/eos/home-j/jiehan/HMuMuShare/train_cache", help="directory for prepared fold cache; existing cache is used before reading ROOT inputs")
    parser.add_argument("--no-train-cache", action="store_true", default=False, help="disable prepared fold cache reads/writes")
    parser.add_argument("--memory-mode", action="store", choices=["streaming", "legacy"], default="streaming", help="prepared-data memory mode; streaming avoids keeping full ROOT inputs in pandas")
    parser.add_argument("--cache-format", action="store", choices=["npy"], default="npy", help="prepared cache storage format")
    parser.add_argument("--strict-memory-cache", dest="strict_memory_cache", action="store_true", default=True, help="require streaming preparation when memory-mode=streaming")
    parser.add_argument("--no-strict-memory-cache", dest="strict_memory_cache", action="store_false", help="allow legacy fallback if streaming preparation is unsupported")
    parser.add_argument("--tensor-prefetch-workers", action="store", type=int, default=None, help="number of threads used to warm mmap cache pages before TensorFlow tensor creation; default is 2 in streaming mode and 0 in legacy mode")
    parser.add_argument("--tensor-prefetch-mb", action="store", type=int, default=64, help="chunk size in MB for mmap tensor prefetch warmup")
    parser.add_argument("--parallel-tensor-convert", action="store_true", default=False, help="experimental: convert epoch tensors in a small thread pool")
    parser.add_argument("--profile-batches", action="store", type=int, default=0, help="synchronize and print timing for the first N train/val batches")
    parser.add_argument("--train-eval-every", action="store", type=int, default=0, help="run full train-eval every N epochs during training; 0 means final best-model train-eval only")
    parser.add_argument("--fuse-dnn-heads", dest="fuse_dnn_heads", action="store_true", default=True, help="reuse one model forward per batch where score/domain heads share inputs")
    parser.add_argument("--no-fuse-dnn-heads", dest="fuse_dnn_heads", action="store_false", help="use the historical separate score/domain forward calls")
    return parser.parse_args()


def fixRandomSeed(seed_value):
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)


def is_jet_feature(name):
    return (
        name == "njets"
        or name.startswith("jet")
        or name.startswith("dijet")
        or name in {"z_star", "R_pT", "delta_eta_jj", "delta_phi_jj"}
        or name.startswith("min_delta_phi_dimu_jets")
        or name.startswith("min_delta_eta_dimu_jets")
    )


def preserves_negative_ptdecor_sentinels(name):
    return is_jet_feature(name)


def is_missing_jet_impute_feature(name):
    return name.startswith("jet") or name.startswith("dijet") or name in {"z_star", "R_pT"}


@tf.keras.utils.register_keras_serializable(package="hzgml")
class GradientReversal(tf.keras.layers.Layer):
    def __init__(self, strength=1.0, **kwargs):
        super().__init__(**kwargs)
        self.strength = float(strength)

    def call(self, inputs):
        strength = tf.constant(self.strength, dtype=tf.float32)

        @tf.custom_gradient
        def reverse_gradient(x):
            def grad(dy):
                return -strength * dy
            return x, grad

        return reverse_gradient(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"strength": self.strength})
        return config


@tf.keras.utils.register_keras_serializable(package="hzgml")
class DecorrelatingDNN(tf.keras.Model):
    def __init__(self, input_dim, hparams=None, seed=20250408, muon_indices=None, jet_indices=None):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hparams = dict(hparams or {})
        self.seed = seed
        self.muon_indices = [] if muon_indices is None else [int(idx) for idx in muon_indices]
        self.jet_indices = [] if jet_indices is None else [int(idx) for idx in jet_indices]
        activation = self.hparams.get("activation", "elu")
        l2_value = float(self.hparams.get("l2", 9.513651681504025e-08))
        regularizer = tf.keras.regularizers.l2(l2_value) if l2_value > 0 else None
        trunk_units = [int(value) for value in self.hparams.get("trunk_units", [512, 128, 128, 16])]
        trunk_dropout = [float(value) for value in self.hparams.get("trunk_dropout", [0.05, 0.15, 0.1, 0.0])]
        output_dense_units = int(self.hparams.get("output_dense_units", 128))
        output_dropout = float(self.hparams.get("output_dropout", 0.1))
        if len(trunk_dropout) < len(trunk_units):
            trunk_dropout = trunk_dropout + [0.0] * (len(trunk_units) - len(trunk_dropout))
        self.trunk_layers = []
        for idx, units in enumerate(trunk_units, start=1):
            self.trunk_layers.append(
                tf.keras.layers.Dense(
                    units,
                    activation=activation,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + idx),
                    kernel_regularizer=regularizer,
                    name=f"trunk_dense_{idx}",
                )
            )
            dropout = trunk_dropout[idx - 1]
            self.trunk_layers.append(tf.keras.layers.Dropout(dropout, seed=seed + 100 + idx, name=f"trunk_dropout_{idx}"))

        self.output_hidden = tf.keras.layers.Dense(
            output_dense_units,
            activation=activation,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 1000),
            kernel_regularizer=regularizer,
            name="output_dense_1",
        )
        self.output_dropout = tf.keras.layers.Dropout(output_dropout, seed=seed + 1100, name="output_dropout_1")
        self.score_domain_output = tf.keras.layers.Dense(
            2,
            activation="sigmoid",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 1001),
            kernel_regularizer=regularizer,
            name="score_domain_output",
        )
        self(tf.zeros((1, self.input_dim), dtype=tf.float32), training=False)
        self.predict_score(tf.zeros((1, self.input_dim), dtype=tf.float32), training=False)
        self.predict_domain(tf.zeros((1, self.input_dim), dtype=tf.float32), training=False)

    def shared_features(self, inputs, training=False):
        x = tf.cast(inputs, tf.float32)
        for layer in self.trunk_layers:
            try:
                x = layer(x, training=training)
            except TypeError:
                x = layer(x)
        return x

    def call(self, inputs, training=False):
        features = self.shared_features(inputs, training=training)
        outputs = self.output_hidden(features)
        outputs = self.output_dropout(outputs, training=training)
        return self.score_domain_output(outputs)

    def predict_score(self, inputs, training=False):
        outputs = self(inputs, training=training)
        return outputs[:, 0:1]

    def predict_domain(self, inputs, training=False):
        outputs = self(inputs, training=training)
        return outputs[:, 1:2]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "hparams": self.hparams,
                "seed": self.seed,
                "muon_indices": self.muon_indices,
                "jet_indices": self.jet_indices,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        input_dim = config.pop("input_dim")
        hparams = config.pop("hparams")
        seed = config.pop("seed")
        muon_indices = config.pop("muon_indices", None)
        jet_indices = config.pop("jet_indices", None)
        return cls(input_dim=input_dim, hparams=hparams, seed=seed, muon_indices=muon_indices, jet_indices=jet_indices)


class DNNHandler(object):
    def __init__(self, configPath, region=""):
        print(
            """
========================================================================
|| **     *     ** ###### $$        ######    ****    #       # %%%%% ||
||  **   ***   **  ##     $$       ##       ***  ***  ##     ## %     ||
||   ** ** ** **   ###### $$      ##       **      ** # #   # # %%%%% ||
||    ***   ***    ##     $$       ##       ***  ***  #  # #  # %     ||
||     *     *     ###### $$$$$$$   ######    ****    #   #   # %%%%% ||
||                                                                    ||
||    Split-branch DNN for ggH                                        ||
========================================================================
            """
        )
        args = getArgs()
        self.args = args
        self.continue_optuna = args.continue_optuna
        self.optuna_metric = args.optuna_metric
        self.optuna_suffix = args.optuna_suffix
        self.n_calls = args.n_calls
        self._shield = args.shield
        self._add = args.add
        self._reweight = args.reweight
        self._reso = args.reso
        self._reso_bkg = args.reso_bkg
        self._ptdecor = args.ptdecor
        self._ptdecor_applied_to_full_data = False
        self._corr_plotted_from_loaded_data = False
        self._bkg_mass_reweight_factors = {}
        self._years = [str(year) for year in args.years] if args.years else []
        self._max_chunks_per_sample = max(0, int(args.max_chunks_per_sample))
        self._train_cache_dir = None if args.no_train_cache else args.train_cache_dir
        self._memory_mode = args.memory_mode
        self._cache_format = args.cache_format
        self._tensor_prefetch_workers = self._resolve_tensor_prefetch_workers(args.tensor_prefetch_workers)
        self._tensor_prefetch_chunk_bytes = max(1, int(args.tensor_prefetch_mb)) * 1024 * 1024
        self._parallel_tensor_convert = bool(args.parallel_tensor_convert)
        if args.corr is None:
            args.corr = self._memory_mode != "streaming"
        if self._memory_mode == "streaming" and args.corr:
            print("DNN WARNING: correlation plots are high-memory in streaming mode; pass --no-corr for 12GB jobs.")
        self._region = region
        self._inputFolder = ""
        self._outputFolder = args.outputFolder if args.outputFolder else "models_dnn"
        self._plotFolder = ("plots_" + args.outputFolder.split("_", 1)[1]) if args.outputFolder and "_" in args.outputFolder else "plots_dnn"
        self._chunksize = 500000
        self._branches = []
        self._sig_branches = []
        self._mc_branches = []
        self._data_branches = []
        self._current_fold = 0

        self.m_data_sig = pd.DataFrame()
        self.m_data_bkg = pd.DataFrame()
        self.m_train_wt = {}
        self.m_val_wt = {}
        self.m_test_wt = {}
        self.m_train_wt_raw = {}
        self.m_val_wt_raw = {}
        self.m_test_wt_raw = {}
        self.m_train_event_weight = {}
        self.m_val_event_weight = {}
        self.m_test_event_weight = {}
        self.m_mass_x_train = {}
        self.m_mass_x_val = {}
        self.m_mass_wt_raw_train = {}
        self.m_mass_wt_raw_val = {}
        self.m_mass_event_weight_train = {}
        self.m_mass_event_weight_val = {}
        self.m_mass_mass_train = {}
        self.m_mass_mass_val = {}
        self.m_train_mass = {}
        self.m_val_mass = {}
        self.m_test_mass = {}
        self.m_train_mass_norm = {}
        self.m_val_mass_norm = {}
        self.m_test_mass_norm = {}
        self.m_mass_norm = {}
        self.m_domain_x_train = {}
        self.m_domain_x_val = {}
        self.m_domain_y_train = {}
        self.m_domain_y_val = {}
        self.m_domain_wt_train = {}
        self.m_domain_wt_val = {}
        self.m_corr_sig = {}
        self.m_corr_bkg = {}
        self.m_y_train = {}
        self.m_y_val = {}
        self.m_y_test = {}
        self.m_x_train = {}
        self.m_x_val = {}
        self.m_x_test = {}
        self.m_x_test_sig = {}
        self.m_x_test_bkg = {}
        self.m_score_train = {}
        self.m_score_val = {}
        self.m_score_test = {}
        self.m_score_test_sig = {}
        self.m_score_test_bkg = {}
        self.m_tsf = {}
        self.m_training_score_reweight = {}
        self.m_scaler = {}
        self.m_missing_jet_impute_values = {}
        self.m_model = {}
        self.m_history = {}
        self.m_metrics = {}

        self.inputTree = args.inputTree
        self.train_signal = []
        self.train_data_background = []
        self.train_mc_background = []
        self.train_dd_background = []
        self.train_variables = []
        self.keep_branches = []
        self.preselections = []
        self.mc_preselections = []
        self.data_preselections = []
        self.signal_preselections = []
        self.background_preselections = []
        self.yearly_inputs = []
        self.paired_year_splitting = {}
        self.paired_year_reports = {}
        self.source_year_map = {}
        self.source_year_inverse_map = {}
        self.randomIndex = ""
        self.reso = "diMu-mass_resolution_BSC"
        self.weight = ""
        self.params = []
        self.dnn_defaults = {}
        self.early_stopping_rounds = 20
        self.num_epochs = 200
        self.SF = -1
        self.readConfig(configPath)
        self.checkConfig()

    def readConfig(self, configPath):
        try:
            member_variables = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("_") and not attr.startswith("m_")]
            if "params" in member_variables:
                member_variables.remove("params")
            if configPath.endswith(".py"):
                spec = importlib.util.spec_from_file_location("training_config", configPath)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                configs = getattr(module, "CONFIG", None)
                if configs is None:
                    raise ValueError(f"Python config {configPath} must define CONFIG = {{...}}")
            else:
                with open(configPath, "r") as stream:
                    configs = json.loads(stream.read())

            config = configs[self._region]
            for member in config.keys():
                if member in member_variables:
                    setattr(self, member, config[member])

            if self._region:
                config = configs[self._region]
                if self._add >= 0 and "test_variables" in config:
                    config["+train_variables"].append(config["test_variables"][self._add])
                for member in config.keys():
                    if member in member_variables:
                        setattr(self, member, config[member])
                if "+train_mc_background" in config:
                    self.train_mc_background += config["+train_mc_background"]
                if "+train_signal" in config:
                    self.train_signal += config["+train_signal"]
                if "+train_variables" in config:
                    self.train_variables += config["+train_variables"]
                if "+preselections" in config:
                    self.preselections += config["+preselections"]
                if "+signal_preselections" in config:
                    self.signal_preselections += config["+signal_preselections"]
                if "+background_preselections" in config:
                    self.background_preselections += config["+background_preselections"]

            if self._shield >= 0:
                self.train_variables.pop(self._shield)

            self._initialize_source_year_mapping()

            metadata_branches = set(self.keep_branches)
            mandatory_metadata = {"diMufsr_kit_BSC_mass", "__sample__", "__source__", "source_year"}
            self._branches = list(
                set(self.train_variables)
                | set([p.split()[0].replace("(", "") for p in self.preselections])
                | set([p.split()[0].replace("(", "") for p in self.background_preselections])
                | set([self.randomIndex, self.weight, self.reso])
                | metadata_branches
                | mandatory_metadata
            )
            self._mc_branches = list(
                set(self.train_variables)
                | set([p.split()[0].replace("(", "") for p in self.preselections])
                | set([p.split()[0].replace("(", "") for p in self.mc_preselections])
                | set([p.split()[0].replace("(", "") for p in self.signal_preselections])
                | set([p.split()[0].replace("(", "") for p in self.background_preselections])
                | set([self.randomIndex, self.weight, self.reso])
                | metadata_branches
                | mandatory_metadata
            )
            self._data_branches = list(
                set(self.train_variables)
                | set([p.split()[0].replace("(", "") for p in self.preselections])
                | set([p.split()[0].replace("(", "") for p in self.data_preselections])
                | set([p.split()[0].replace("(", "") for p in self.signal_preselections])
                | set([p.split()[0].replace("(", "") for p in self.background_preselections])
                | set([self.randomIndex, self.weight, self.reso])
                | metadata_branches
                | mandatory_metadata
            )
            self._sig_branches = list(
                set(self.train_variables)
                | set([p.split()[0].replace("(", "") for p in self.preselections])
                | set([p.split()[0].replace("(", "") for p in self.mc_preselections])
                | set([p.split()[0].replace("(", "") for p in self.signal_preselections])
                | set([self.randomIndex, self.weight, self.reso])
                | metadata_branches
                | mandatory_metadata
            )

            self.train_variables = [x.replace("noexpand:", "") for x in self.train_variables]
            self.preselections = [x.replace("noexpand:", "") for x in self.preselections]
            self.mc_preselections = [x.replace("noexpand:", "") for x in self.mc_preselections]
            self.data_preselections = [x.replace("noexpand:", "") for x in self.data_preselections]
            self.signal_preselections = [x.replace("noexpand:", "") for x in self.signal_preselections]
            self.background_preselections = [x.replace("noexpand:", "") for x in self.background_preselections]
            self.randomIndex = self.randomIndex.replace("noexpand:", "")
            self.weight = self.weight.replace("noexpand:", "")

            if self.preselections:
                self.preselections = ["data." + p for p in self.preselections]
            if self.signal_preselections:
                self.signal_preselections = ["data." + p for p in self.signal_preselections]
            if self.mc_preselections:
                self.mc_preselections = ["data." + p for p in self.mc_preselections]
            if self.data_preselections:
                self.data_preselections = ["data." + p for p in self.data_preselections]
            if self.background_preselections:
                self.background_preselections = ["data." + p for p in self.background_preselections]
        except Exception as e:
            logging.error("Error reading configuration '%s'", configPath)
            logging.error(e)
            raise

    def checkConfig(self):
        if not self.train_signal:
            print("ERROR: no training signal!!")
        if not self.train_mc_background:
            print("ERROR: no MC training background!!")
        if not self.train_variables:
            print("ERROR: no training variables!!")

    def _normalize_year_key(self, year):
        if isinstance(year, str):
            return year.strip()
        if isinstance(year, (np.integer, int)):
            return str(int(year))
        return str(year)

    def _initialize_source_year_mapping(self):
        self.source_year_map = {}
        self.source_year_inverse_map = {}
        if self._years:
            selected_years = set(self._years)
            self.yearly_inputs = [
                year_cfg for year_cfg in self.yearly_inputs
                if self._normalize_year_key(year_cfg.get("year")) in selected_years
            ]
            if not self.yearly_inputs:
                raise ValueError(f"No yearly_inputs left after applying --years {self._years}")
        if not self.yearly_inputs:
            return
        ordered_years = []
        seen = set()
        for year_cfg in self.yearly_inputs:
            year = year_cfg.get("year")
            key = self._normalize_year_key(year)
            if key in seen:
                continue
            seen.add(key)
            ordered_years.append(year)
        self.source_year_map = {self._normalize_year_key(year): index for index, year in enumerate(ordered_years)}
        self.source_year_inverse_map = {index: year for index, year in enumerate(ordered_years)}
        print(f"DNN INFO: source_year encoding: {self.source_year_map}")

    def _encode_source_year(self, year):
        key = self._normalize_year_key(year)
        if key in self.source_year_map:
            return self.source_year_map[key]
        raise KeyError(f'Unknown source_year "{year}". Known years: {list(self.source_year_map.keys())}')

    def setParams(self, params, fold=-1):
        if fold == -1:
            fold = 0
        merged_params = self._merge_dnn_params(params)
        if not self.params:
            self.params = [self._merge_dnn_params()]
        while len(self.params) <= fold:
            self.params.append(self._merge_dnn_params())
        self.params[fold] = merged_params

    def setInputFolder(self, inputFolder):
        self._inputFolder = inputFolder

    def setOutputFolder(self, outputFolder):
        self._outputFolder = outputFolder

    def _train_cache_payload(self):
        return {
            "version": 23,
            "training_definition": "single_2out_domain_closeness_true_mass_rff_hsic_reweight_v1",
            "score_wt_raw_definition": "train/val/test wt_raw store effective raw score weights after reso/reweight, not ROOT raw and not normalized score loss weights",
            "domain_wt_definition": "domain train/val weights use score-style normalization with train data sideband as ref_sig and train MC sideband as ref_bkg; arrays remain [bkg,data]",
            "shape_event_weight_definition": "train/val/test event_weight arrays store original eventWeight after yearly scale and before score, mass, or resolution reweight; used only by score-bin mass-shape epoch metrics",
            "mass_wt_raw_definition": "mass-only train/val wt_raw store background MC physical-mass-flattened weights in mass_reweight_min/max; used only by mass loss",
            "mass_event_weight_definition": "mass-only train/val event weights store original eventWeight after yearly scale and before score, mass, or resolution reweight; used by deterministic physics feature loss monitor modes",
            "region": self._region,
            "years": self._years,
            "max_chunks_per_sample": self._max_chunks_per_sample,
            "reweight": self._reweight,
            "reso": self._reso,
            "reso_bkg": self._reso_bkg,
            "ptdecor": self._ptdecor,
            "scaled_by_mass_variables": SCALED_BY_MASS_VARIABLES,
            "shield": self._shield,
            "add": self._add,
            "inputTree": self.inputTree,
            "train_variables": self.train_variables,
            "randomIndex": self.randomIndex,
            "weight": self.weight,
            "reso_variable": self.reso,
            "source_year_map": self.source_year_map,
            "yearly_inputs": self.yearly_inputs,
            "pandas_dataframe_shuffle_seed": PANDAS_DATAFRAME_SHUFFLE_SEED,
            "mass_reweight_min": float(self._merge_dnn_params().get("mass_reweight_min", self._default_params()["mass_reweight_min"])),
            "mass_reweight_max": float(self._merge_dnn_params().get("mass_reweight_max", self._default_params()["mass_reweight_max"])),
            "mass_reweight_bin_width": float(self._merge_dnn_params().get("mass_reweight_bin_width", self._default_params()["mass_reweight_bin_width"])),
        }

    def _normalized_train_cache_payload(self):
        return json.loads(json.dumps(self._train_cache_payload(), sort_keys=True, default=str))

    def _train_cache_key(self):
        payload = json.dumps(self._normalized_train_cache_payload(), sort_keys=True)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:16]

    def _train_cache_base(self):
        if not self._train_cache_dir:
            return None
        return os.path.join(self._train_cache_dir, f"DNN_{self._region}_{self._train_cache_key()}")

    def _train_cache_paths(self, fold):
        base = self._train_cache_base()
        if not base:
            return None
        return {
            "base": base,
            "arrays_dir": os.path.join(base, f"fold{fold}_arrays"),
            "scaler": os.path.join(base, f"fold{fold}_scaler.pkl"),
            "meta": os.path.join(base, f"fold{fold}_meta.json"),
        }

    def hasPreparedCache(self, fold):
        paths = self._train_cache_paths(fold)
        if not paths or not os.path.exists(paths["scaler"]) or not os.path.exists(paths["meta"]):
            return False
        return os.path.isdir(paths["arrays_dir"]) and all(
            os.path.exists(os.path.join(paths["arrays_dir"], f"{key}.npy")) for key in NPY_CACHE_ARRAY_KEYS
        )

    def hasPreparedCaches(self, folds):
        return bool(self._train_cache_dir) and all(self.hasPreparedCache(fold) for fold in folds)

    def _prepared_cache_arrays(self, fold):
        return {
            "x_train": self.m_x_train[fold],
            "x_val": self.m_x_val[fold],
            "x_test": self.m_x_test[fold],
            "x_test_sig": self.m_x_test_sig[fold],
            "x_test_bkg": self.m_x_test_bkg[fold],
            "y_train": self.m_y_train[fold],
            "y_val": self.m_y_val[fold],
            "y_test": self.m_y_test[fold],
            "train_wt": self.m_train_wt[fold],
            "val_wt": self.m_val_wt[fold],
            "test_wt": self.m_test_wt[fold],
            "train_wt_raw": self.m_train_wt_raw[fold],
            "val_wt_raw": self.m_val_wt_raw[fold],
            "test_wt_raw": self.m_test_wt_raw[fold],
            "train_event_weight": self.m_train_event_weight[fold],
            "val_event_weight": self.m_val_event_weight[fold],
            "test_event_weight": self.m_test_event_weight[fold],
            "mass_x_train": self.m_mass_x_train[fold],
            "mass_x_val": self.m_mass_x_val[fold],
            "mass_wt_raw_train": self.m_mass_wt_raw_train[fold],
            "mass_wt_raw_val": self.m_mass_wt_raw_val[fold],
            "mass_event_weight_train": self.m_mass_event_weight_train[fold],
            "mass_event_weight_val": self.m_mass_event_weight_val[fold],
            "mass_mass_train": self.m_mass_mass_train[fold],
            "mass_mass_val": self.m_mass_mass_val[fold],
            "train_mass": self.m_train_mass[fold],
            "val_mass": self.m_val_mass[fold],
            "test_mass": self.m_test_mass[fold],
            "train_mass_norm": self.m_train_mass_norm[fold],
            "val_mass_norm": self.m_val_mass_norm[fold],
            "test_mass_norm": self.m_test_mass_norm[fold],
            "domain_x_train": self.m_domain_x_train[fold],
            "domain_x_val": self.m_domain_x_val[fold],
            "domain_y_train": self.m_domain_y_train[fold],
            "domain_y_val": self.m_domain_y_val[fold],
            "domain_wt_train": self.m_domain_wt_train[fold],
            "domain_wt_val": self.m_domain_wt_val[fold],
        }

    def _cache_size_gb(self, paths, cache_format="npy"):
        total = 0
        for root, _, files in os.walk(paths["arrays_dir"]):
            for name in files:
                total += os.path.getsize(os.path.join(root, name))
        return total / 1024**3

    def _log_rss(self, label):
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        print(f"DNN MEMORY: {label} maxrss_mb={rss_mb:.1f}")

    def _resolve_tensor_prefetch_workers(self, requested):
        if requested is None:
            requested = 4 if self._memory_mode == "streaming" else 0
        try:
            workers = int(requested)
        except (TypeError, ValueError):
            workers = 0
        return max(0, min(workers, 4))

    def _prefetch_array_pages(self, array, chunk_bytes):
        if not isinstance(array, np.memmap) or array.size == 0:
            return 0
        mmap_obj = getattr(array, "_mmap", None)
        madvise = getattr(mmap_obj, "madvise", None)
        madv_willneed = getattr(mmap, "MADV_WILLNEED", None)
        if callable(madvise) and madv_willneed is not None:
            madvise(madv_willneed)
            return 1
        itemsize = max(1, int(array.dtype.itemsize))
        step = max(1, int(chunk_bytes) // itemsize)
        flat = array.reshape(-1)
        touched = 0
        checksum = 0.0
        for start in range(0, flat.size, step):
            value = flat[start]
            try:
                checksum += float(value)
            except (TypeError, ValueError):
                pass
            touched += 1
        # Touch the last page-sized region too; this keeps the loop cheap but warms sparse mmap pages.
        if flat.size:
            try:
                checksum += float(flat[-1])
            except (TypeError, ValueError):
                pass
        return touched

    def _prefetch_epoch_arrays(self, arrays, label):
        workers = self._tensor_prefetch_workers
        if workers <= 0:
            return 0.0
        candidates = [
            array for array in arrays
            if isinstance(array, np.memmap) and array.nbytes >= 1024 * 1024
        ]
        if not candidates:
            return 0.0
        candidates = sorted(candidates, key=lambda array: int(array.nbytes), reverse=True)
        start = time.perf_counter()
        touched = 0
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(self._prefetch_array_pages, array, self._tensor_prefetch_chunk_bytes)
                    for array in candidates
                ]
                for future in as_completed(futures):
                    touched += int(future.result())
        except Exception as exc:
            print(f"DNN WARNING: tensor prefetch failed for {label}: {exc}. Continue without prefetch.")
            return time.perf_counter() - start
        seconds = time.perf_counter() - start
        print(
            f"DNN PROFILE: {label} tensor_prefetch_s={seconds:.3f} "
            f"workers={workers} arrays={len(candidates)} touched_chunks={touched}"
        )
        self._log_rss(f"{label} tensor prefetch")
        return seconds

    def _convert_epoch_arrays(self, arrays, device, label):
        start = time.perf_counter()
        target_device = device if device and "GPU" in str(device).upper() else "/CPU:0"

        def convert(array):
            with tf.device(target_device):
                return tf.convert_to_tensor(array, dtype=tf.float32)

        if self._parallel_tensor_convert:
            workers = min(4, max(1, self._tensor_prefetch_workers or 1))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                tensors = tuple(executor.map(convert, arrays))
        else:
            tensors = tuple(convert(array) for array in arrays)
        seconds = time.perf_counter() - start
        print(
            f"DNN PROFILE: {label} tensor_convert_s={seconds:.3f} "
            f"parallel={self._parallel_tensor_convert}"
        )
        self._log_rss(f"{label} tensor convert")
        return tensors, seconds

    def releaseFold(self, fold):
        fold_dicts = (
            self.m_train_wt,
            self.m_val_wt,
            self.m_test_wt,
            self.m_train_wt_raw,
            self.m_val_wt_raw,
            self.m_test_wt_raw,
            self.m_train_event_weight,
            self.m_val_event_weight,
            self.m_test_event_weight,
            self.m_mass_x_train,
            self.m_mass_x_val,
            self.m_mass_wt_raw_train,
            self.m_mass_wt_raw_val,
            self.m_mass_event_weight_train,
            self.m_mass_event_weight_val,
            self.m_mass_mass_train,
            self.m_mass_mass_val,
            self.m_train_mass,
            self.m_val_mass,
            self.m_test_mass,
            self.m_train_mass_norm,
            self.m_val_mass_norm,
            self.m_test_mass_norm,
            self.m_mass_norm,
            self.m_domain_x_train,
            self.m_domain_x_val,
            self.m_domain_y_train,
            self.m_domain_y_val,
            self.m_domain_wt_train,
            self.m_domain_wt_val,
            self.m_corr_sig,
            self.m_corr_bkg,
            self.m_y_train,
            self.m_y_val,
            self.m_y_test,
            self.m_x_train,
            self.m_x_val,
            self.m_x_test,
            self.m_x_test_sig,
            self.m_x_test_bkg,
            self.m_score_train,
            self.m_score_val,
            self.m_score_test,
            self.m_score_test_sig,
            self.m_score_test_bkg,
            self.m_tsf,
            self.m_training_score_reweight,
            self.m_scaler,
            self.m_model,
            self.m_history,
            self.m_metrics,
        )
        for mapping in fold_dicts:
            mapping.pop(fold, None)
        tf.keras.backend.clear_session()
        gc.collect()
        self._log_rss(f"released fold={fold}")

    def loadPreparedCache(self, fold):
        paths = self._train_cache_paths(fold)
        if not paths or not self.hasPreparedCache(fold):
            return False
        start = time.perf_counter()
        with open(paths["meta"], "r") as handle:
            meta = json.load(handle)
        expected_payload = self._normalized_train_cache_payload()
        if meta.get("cache_payload") != expected_payload:
            print(f"DNN INFO: prepared cache metadata mismatch for fold {fold}: {paths['meta']}")
            return False
        cache_format = meta.get("cache_format", "npy")
        if cache_format != "npy":
            print(f"DNN INFO: prepared cache format {cache_format!r} is no longer supported for fold {fold}; regenerate cache.")
            return False
        arrays = {
            key: np.load(os.path.join(paths["arrays_dir"], f"{key}.npy"), allow_pickle=False, mmap_mode="r")
            for key in NPY_CACHE_ARRAY_KEYS
        }
        self.m_x_train[fold] = arrays["x_train"]
        self.m_x_val[fold] = arrays["x_val"]
        self.m_x_test[fold] = arrays["x_test"]
        self.m_x_test_sig[fold] = arrays["x_test_sig"]
        self.m_x_test_bkg[fold] = arrays["x_test_bkg"]
        self.m_y_train[fold] = arrays["y_train"]
        self.m_y_val[fold] = arrays["y_val"]
        self.m_y_test[fold] = arrays["y_test"]
        self.m_train_wt[fold] = arrays["train_wt"]
        self.m_val_wt[fold] = arrays["val_wt"]
        self.m_test_wt[fold] = arrays["test_wt"]
        self.m_train_wt_raw[fold] = arrays["train_wt_raw"]
        self.m_val_wt_raw[fold] = arrays["val_wt_raw"]
        self.m_test_wt_raw[fold] = arrays["test_wt_raw"]
        self.m_train_event_weight[fold] = arrays["train_event_weight"]
        self.m_val_event_weight[fold] = arrays["val_event_weight"]
        self.m_test_event_weight[fold] = arrays["test_event_weight"]
        self.m_mass_x_train[fold] = arrays["mass_x_train"]
        self.m_mass_x_val[fold] = arrays["mass_x_val"]
        self.m_mass_wt_raw_train[fold] = arrays["mass_wt_raw_train"]
        self.m_mass_wt_raw_val[fold] = arrays["mass_wt_raw_val"]
        self.m_mass_event_weight_train[fold] = arrays["mass_event_weight_train"]
        self.m_mass_event_weight_val[fold] = arrays["mass_event_weight_val"]
        self.m_mass_mass_train[fold] = arrays["mass_mass_train"]
        self.m_mass_mass_val[fold] = arrays["mass_mass_val"]
        self.m_train_mass[fold] = arrays["train_mass"]
        self.m_val_mass[fold] = arrays["val_mass"]
        self.m_test_mass[fold] = arrays["test_mass"]
        self.m_train_mass_norm[fold] = arrays["train_mass_norm"]
        self.m_val_mass_norm[fold] = arrays["val_mass_norm"]
        self.m_test_mass_norm[fold] = arrays["test_mass_norm"]
        self.m_domain_x_train[fold] = arrays["domain_x_train"]
        self.m_domain_x_val[fold] = arrays["domain_x_val"]
        self.m_domain_y_train[fold] = arrays["domain_y_train"]
        self.m_domain_y_val[fold] = arrays["domain_y_val"]
        self.m_domain_wt_train[fold] = arrays["domain_wt_train"]
        self.m_domain_wt_val[fold] = arrays["domain_wt_val"]
        self.m_mass_norm[fold] = meta["mass_norm"]
        self.m_missing_jet_impute_values[fold] = meta.get("missing_jet_impute_values", {})
        self.SF = float(meta.get("SF", self.SF))
        with open(paths["scaler"], "rb") as handle:
            self.m_scaler[fold] = pickle.load(handle)
        self._log_rss(f"cache loaded fold={fold} format={cache_format}")
        print(f"DNN CACHE: loaded prepared fold={fold} from {paths['base']} format={cache_format} seconds={time.perf_counter() - start:.1f}")
        return True

    def savePreparedCache(self, fold):
        paths = self._train_cache_paths(fold)
        if not paths:
            return
        start = time.perf_counter()
        os.makedirs(paths["base"], exist_ok=True)
        tmp_arrays_dir = paths["arrays_dir"] + ".tmp"
        tmp_scaler = paths["scaler"] + ".tmp"
        tmp_meta = paths["meta"] + ".tmp"
        arrays = self._prepared_cache_arrays(fold)
        cache_format = "npy"
        if os.path.exists(tmp_arrays_dir):
            shutil.rmtree(tmp_arrays_dir)
        os.makedirs(tmp_arrays_dir, exist_ok=True)
        for key, value in arrays.items():
            np.save(os.path.join(tmp_arrays_dir, f"{key}.npy"), value, allow_pickle=False)
        if os.path.exists(paths["arrays_dir"]):
            shutil.rmtree(paths["arrays_dir"])
        os.replace(tmp_arrays_dir, paths["arrays_dir"])
        with open(tmp_scaler, "wb") as handle:
            pickle.dump(self.m_scaler[fold], handle, -1)
        os.replace(tmp_scaler, paths["scaler"])
        meta = {
            "cache_payload": self._normalized_train_cache_payload(),
            "fold": fold,
            "mass_norm": self.m_mass_norm[fold],
            "missing_jet_impute_values": self.m_missing_jet_impute_values.get(fold, {}),
            "SF": self.SF,
            "cache_format": cache_format,
            "shapes": {
                "x_train": list(self.m_x_train[fold].shape),
                "x_val": list(self.m_x_val[fold].shape),
                "x_test": list(self.m_x_test[fold].shape),
                "x_test_sig": list(self.m_x_test_sig[fold].shape),
                "x_test_bkg": list(self.m_x_test_bkg[fold].shape),
                "mass_x_train": list(self.m_mass_x_train[fold].shape),
                "mass_x_val": list(self.m_mass_x_val[fold].shape),
                "domain_x_train": list(self.m_domain_x_train[fold].shape),
                "domain_x_val": list(self.m_domain_x_val[fold].shape),
            },
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(tmp_meta, "w") as handle:
            json.dump(meta, handle, indent=2, sort_keys=True)
        os.replace(tmp_meta, paths["meta"])
        size_gb = self._cache_size_gb(paths, cache_format)
        print(f"DNN CACHE: saved prepared fold={fold} to {paths['base']} format={cache_format} size_gb={size_gb:.3f} seconds={time.perf_counter() - start:.1f}")

    def preselect(self, data, sample=""):
        if sample == "signal":
            for p in self.signal_preselections:
                data = data[eval(p)]
            for p in self.mc_preselections:
                data = data[eval(p)]
        elif sample == "background":
            for p in self.background_preselections:
                data = data[eval(p)]
        elif sample == "data":
            for p in self.background_preselections:
                data = data[eval(p)]
            for p in self.data_preselections:
                data = data[eval(p)]
        elif sample == "mc_background":
            for p in self.background_preselections:
                data = data[eval(p)]
            for p in self.mc_preselections:
                data = data[eval(p)]
        for p in self.preselections:
            data = data[eval(p)]
        return data

    def readData(self):
        if self.yearly_inputs:
            self._readData_multiyear()
            return
        raise NotImplementedError("This DNN trainer currently expects yearly_inputs in the config, matching the ggH setup.")

    def _readData_multiyear(self):
        for year_cfg in self.yearly_inputs:
            year = year_cfg.get("year")
            folder = year_cfg.get("folder", self._inputFolder)
            if not folder:
                raise ValueError(f"Missing folder for yearly input: {year_cfg}")
            sample_map = year_cfg.get("sample_map", {})
            branch_aliases = year_cfg.get("branch_aliases", {})
            event_weight_scale = year_cfg.get("eventWeight_scale", 1.0)
            print("-------------------------------------------------")
            print(f"DNN INFO: Loading yearly inputs for source_year={year} from {folder}")
            self._load_sample_group(
                folder=folder,
                sample_names=year_cfg.get("train_signal", self.train_signal),
                sample_map=sample_map,
                tree_name=year_cfg.get("tree_name", self.inputTree),
                tree_map=year_cfg.get("tree_map", {}),
                branches=self._sig_branches,
                sample_type="signal",
                source_type="mc",
                source_year=year,
                branch_aliases=branch_aliases,
                event_weight_scale=event_weight_scale,
            )
            self._load_sample_group(
                folder=folder,
                sample_names=year_cfg.get("train_mc_background", self.train_mc_background),
                sample_map=sample_map,
                tree_name=year_cfg.get("tree_name", self.inputTree),
                tree_map=year_cfg.get("tree_map", {}),
                branches=self._mc_branches,
                sample_type="mc_background",
                source_type="mc",
                source_year=year,
                branch_aliases=branch_aliases,
                event_weight_scale=event_weight_scale,
            )
            self._load_sample_group(
                folder=folder,
                sample_names=year_cfg.get("train_dd_background", self.train_dd_background),
                sample_map=sample_map,
                tree_name=year_cfg.get("tree_name", self.inputTree),
                tree_map=year_cfg.get("tree_map", {}),
                branches=self._branches,
                sample_type="background",
                source_type="dd",
                source_year=year,
                branch_aliases=branch_aliases,
                event_weight_scale=event_weight_scale,
            )
            self._load_sample_group(
                folder=folder,
                sample_names=year_cfg.get("train_data_background", self.train_data_background),
                sample_map=sample_map,
                tree_name=year_cfg.get("tree_name", self.inputTree),
                tree_map=year_cfg.get("tree_map", {}),
                branches=self._data_branches,
                sample_type="data",
                source_type="data",
                source_year=year,
                branch_aliases=branch_aliases,
                event_weight_scale=event_weight_scale,
            )

        self._balance_gluglu_mc()
        self._apply_paired_year_splitting()

    def _load_sample_group(self, folder, sample_names, sample_map, tree_name, tree_map, branches, sample_type, source_type, source_year, branch_aliases=None, event_weight_scale=1.0):
        if not sample_names:
            return
        branch_aliases = branch_aliases or {}
        for sample_name in sample_names:
            sample_file = sample_map.get(sample_name, sample_name)
            filename = os.path.join(folder, f"{sample_file}.root")
            if not os.path.exists(filename):
                raise FileNotFoundError(f'Cannot find sample "{sample_name}" at {filename}')
            resolved_tree = tree_map.get(sample_name, tree_name)
            sample_start = time.perf_counter()
            print(f"DNN INFO: Adding {sample_type} sample: {filename} (tree={resolved_tree}, source_year={source_year})")
            loaded = self._load_root_sample(
                filename=filename,
                tree_name=resolved_tree,
                branches=branches,
                sample_type=sample_type,
                sample_name=sample_name,
                source_type=source_type,
                source_year=source_year,
                branch_aliases=branch_aliases,
                event_weight_scale=event_weight_scale,
            )
            print(f"DNN PROFILE: loaded sample={sample_name} type={sample_type} rows={len(loaded)} seconds={time.perf_counter() - sample_start:.1f}")
            if sample_type == "signal":
                self.m_data_sig = pd.concat([self.m_data_sig, loaded], ignore_index=True)
            else:
                self.m_data_bkg = pd.concat([self.m_data_bkg, loaded], ignore_index=True)

    def _load_root_sample(self, filename, tree_name, branches, sample_type, sample_name, source_type, source_year, branch_aliases=None, event_weight_scale=1.0):
        file = uproot.open(filename)
        available_trees = [key.split(";")[0] for key in file.keys()]
        if tree_name not in available_trees:
            raise KeyError(f'Tree "{tree_name}" not found in {filename}. Available trees: {available_trees}')
        tree = file[tree_name]
        available_branches = set(tree.keys())
        branch_aliases = branch_aliases or {}
        actual_to_target = {}
        for branch in branches:
            actual_branch = branch_aliases.get(branch, branch)
            if actual_branch in available_branches:
                actual_to_target[actual_branch] = branch
        selected_branches = list(actual_to_target.keys())
        missing_branches = sorted(set(branches) - set(actual_to_target.values()))
        if missing_branches:
            print(f"DNN WARNING: Missing branches in {filename}:{tree_name}: {missing_branches}")

        has_source_year_in_file = "source_year" in available_branches
        if has_source_year_in_file and "source_year" not in selected_branches:
            selected_branches.append("source_year")

        loaded_chunks = []
        chunk_idx = 0
        for data in tree.iterate(selected_branches, library="pd", step_size=self._chunksize):
            data = data.rename(columns=actual_to_target)
            data = self.preselect(data, sample_type)
            data["__sample__"] = sample_name
            data["__source__"] = source_type
            data["__tree__"] = tree_name
            if has_source_year_in_file:
                data["source_year_from_file"] = data["source_year"]
            data["source_year_label"] = self._coerce_year_value(source_year)
            data["source_year"] = self._encode_source_year(source_year)
            if event_weight_scale != 1.0 and "eventWeight" in data.columns:
                data["eventWeight"] = data["eventWeight"] * float(event_weight_scale)
            loaded_chunks.append(data)
            chunk_idx += 1
            if self._max_chunks_per_sample and chunk_idx >= self._max_chunks_per_sample:
                break
        if not loaded_chunks:
            return pd.DataFrame()
        loaded = pd.concat(loaded_chunks, ignore_index=True)
        if has_source_year_in_file and "source_year_from_file" in loaded.columns:
            loaded["source_year_file_matches_config"] = loaded["source_year_from_file"] == loaded["source_year"]
            mismatch_count = (~loaded["source_year_file_matches_config"]).sum()
            print(f"DNN INFO: {filename}:{tree_name} has source_year branch, mismatch with config on {mismatch_count} events.")
        else:
            print(f"DNN INFO: {filename}:{tree_name} has no source_year branch, injected source_year={source_year}.")
        return loaded

    def _streaming_sample_specs(self):
        specs = []
        for year_cfg in self.yearly_inputs:
            year = year_cfg.get("year")
            folder = year_cfg.get("folder", self._inputFolder)
            if not folder:
                raise ValueError(f"Missing folder for yearly input: {year_cfg}")
            sample_map = year_cfg.get("sample_map", {})
            tree_map = year_cfg.get("tree_map", {})
            tree_name = year_cfg.get("tree_name", self.inputTree)
            branch_aliases = year_cfg.get("branch_aliases", {})
            event_weight_scale = year_cfg.get("eventWeight_scale", 1.0)
            groups = (
                ("signal", "mc", year_cfg.get("train_signal", self.train_signal), self._sig_branches),
                ("mc_background", "mc", year_cfg.get("train_mc_background", self.train_mc_background), self._mc_branches),
                ("background", "dd", year_cfg.get("train_dd_background", self.train_dd_background), self._branches),
                ("data", "data", year_cfg.get("train_data_background", self.train_data_background), self._data_branches),
            )
            for sample_type, source_type, sample_names, branches in groups:
                for sample_name in sample_names:
                    sample_file = sample_map.get(sample_name, sample_name)
                    specs.append(
                        {
                            "filename": os.path.join(folder, f"{sample_file}.root"),
                            "tree_name": tree_map.get(sample_name, tree_name),
                            "branches": branches,
                            "sample_type": sample_type,
                            "sample_name": sample_name,
                            "source_type": source_type,
                            "source_year": year,
                            "branch_aliases": branch_aliases,
                            "event_weight_scale": event_weight_scale,
                        }
                    )
        return specs

    def _iter_streaming_sample_chunks(self):
        for spec in self._streaming_sample_specs():
            filename = spec["filename"]
            tree_name = spec["tree_name"]
            if not os.path.exists(filename):
                raise FileNotFoundError(f'Cannot find sample "{spec["sample_name"]}" at {filename}')
            with uproot.open(filename) as file:
                available_trees = [key.split(";")[0] for key in file.keys()]
                if tree_name not in available_trees:
                    raise KeyError(f'Tree "{tree_name}" not found in {filename}. Available trees: {available_trees}')
                tree = file[tree_name]
                available_branches = set(tree.keys())
                actual_to_target = {}
                for branch in spec["branches"]:
                    actual_branch = spec["branch_aliases"].get(branch, branch)
                    if actual_branch in available_branches:
                        actual_to_target[actual_branch] = branch
                selected_branches = list(actual_to_target.keys())
                has_source_year_in_file = "source_year" in available_branches
                if has_source_year_in_file and "source_year" not in selected_branches:
                    selected_branches.append("source_year")
                chunk_idx = 0
                print(
                    "DNN STREAM: "
                    f"sample={spec['sample_name']} type={spec['sample_type']} "
                    f"file={filename} tree={tree_name} source_year={spec['source_year']}"
                )
                for data in tree.iterate(selected_branches, library="pd", step_size=self._chunksize):
                    data = data.rename(columns=actual_to_target)
                    data = self.preselect(data, spec["sample_type"])
                    if data.empty:
                        chunk_idx += 1
                        if self._max_chunks_per_sample and chunk_idx >= self._max_chunks_per_sample:
                            break
                        continue
                    data["__sample__"] = spec["sample_name"]
                    data["__source__"] = spec["source_type"]
                    data["__tree__"] = tree_name
                    if has_source_year_in_file:
                        data["source_year_from_file"] = data["source_year"]
                    data["source_year_label"] = self._coerce_year_value(spec["source_year"])
                    data["source_year"] = self._encode_source_year(spec["source_year"])
                    if spec["event_weight_scale"] != 1.0 and "eventWeight" in data.columns:
                        data["eventWeight"] = data["eventWeight"] * float(spec["event_weight_scale"])
                    yield spec, data
                    chunk_idx += 1
                    if self._max_chunks_per_sample and chunk_idx >= self._max_chunks_per_sample:
                        break

    def _streaming_prepare_supported(self):
        if not self.paired_year_splitting or not self.paired_year_splitting.get("enabled", False):
            return True
        selected = {self._coerce_year_value(year_cfg.get("year")) for year_cfg in self.yearly_inputs}
        left = self._coerce_year_value(self.paired_year_splitting.get("left_year"))
        right = self._coerce_year_value(self.paired_year_splitting.get("right_year"))
        return not ({left, right} <= selected)

    def _split_masks(self, data, fold):
        random_index = data[self.randomIndex].to_numpy(dtype=np.int64, copy=False)
        modulo = random_index % 314159 % 4
        return {
            "train": ((random_index - 2) % 314159 % 4 == fold) | ((random_index - 3) % 314159 % 4 == fold),
            "val": (random_index - 1) % 314159 % 4 == fold,
            "test": modulo == fold,
        }

    def _streaming_mass_window_mask(self, data, low, high, include_high=True):
        mass = data[MASS_VARIABLE].to_numpy(dtype=np.float32, copy=False)
        if include_high:
            return (mass >= low) & (mass <= high)
        return (mass >= low) & (mass < high)

    def _streaming_sideband_mask(self, data):
        mass = data[MASS_VARIABLE].to_numpy(dtype=np.float32, copy=False)
        return ((mass >= DOMAIN_LOW_MIN) & (mass < DOMAIN_LOW_MAX)) | ((mass > DOMAIN_HIGH_MIN) & (mass <= DOMAIN_HIGH_MAX))

    def _streaming_domain_jet_mask(self, data):
        if "njets" not in data.columns:
            raise KeyError("Domain training restricted to X-jet events requires 'njets' in the loaded branches")
        return data["njets"].to_numpy(copy=False) >= DOMAIN_REQUIRED_NJETS

    def _streaming_reweight_edges(self, low, high, bin_width=1.0):
        low = float(low)
        high = float(high)
        bin_width = float(bin_width)
        if high <= low:
            raise ValueError(f"reweight high edge must be larger than low edge: {low} -> {high}")
        if bin_width <= 0:
            raise ValueError(f"reweight bin width must be positive, got {bin_width}")
        n_bins = max(1, int(round((high - low) / bin_width)))
        return np.linspace(low, high, n_bins + 1, dtype=np.float64)

    def _streaming_reweight_factors(self, hist):
        hist = np.asarray(hist, dtype=np.float64)
        if hist.size == 0 or not np.isfinite(hist).all() or float(np.sum(hist)) <= 0.0:
            return np.ones(hist.shape, dtype=np.float32)
        hist_safe = np.where(hist > 0, hist, 1e-10)
        target_count = hist.sum() / max(len(hist), 1)
        return (target_count / hist_safe).astype(np.float32)

    def _streaming_apply_reweight(self, data, mask, edges, factors, weight_column):
        if not np.any(mask):
            return
        masses = data.loc[mask, MASS_VARIABLE].to_numpy(dtype=np.float64, copy=False)
        bins = np.digitize(masses, edges) - 1
        bins = np.clip(bins, 0, len(factors) - 1)
        values = data.loc[mask, weight_column].to_numpy(dtype=np.float32, copy=True)
        data.loc[mask, weight_column] = values * factors[bins]

    def _empty_stats(self):
        return {"count": 0, "sum": 0.0, "sum_positive": 0.0, "count_positive": 0, "sum_mass": 0.0, "sum_mass2": 0.0}

    def _update_weight_stats(self, stats, data, mask, include_mass=False, weight_column=None):
        n = int(np.count_nonzero(mask))
        if n == 0:
            return
        weight_column = weight_column or self.weight
        weights = data.loc[mask, weight_column].to_numpy(dtype=np.float32, copy=False) if weight_column in data.columns else np.ones(n, dtype=np.float32)
        stats["count"] += n
        stats["sum"] += float(np.sum(weights))
        positive = weights > 0.0
        stats["count_positive"] += int(np.count_nonzero(positive))
        stats["sum_positive"] += float(np.sum(weights[positive]))
        if include_mass:
            masses = data.loc[mask, MASS_VARIABLE].to_numpy(dtype=np.float32, copy=False)
            stats["sum_mass"] += float(np.sum(masses))
            stats["sum_mass2"] += float(np.sum(np.square(masses)))

    def _prepare_streaming_chunk(self, data, spec, fold, reweight=None):
        data = self._mark_shape_event_weight(data)
        if self._reso and spec["sample_type"] == "signal":
            self._apply_resolution_weight(data, f"stream score signal {spec['sample_name']}")
        if self._reso_bkg and spec["sample_type"] in {"mc_background", "background", "data"}:
            self._apply_resolution_weight(data, f"stream background {spec['sample_name']}")
        if self._ptdecor:
            data = self._apply_ptdecor(data, f"stream_{spec['sample_type']}_{spec['sample_name']}", fold)

        is_signal = spec["sample_type"] == "signal"
        is_mc_bkg = spec["sample_type"] == "mc_background" and spec["source_type"] == "mc"
        is_data = spec["sample_type"] == "data" and spec["source_type"] == "data"
        score_sig = is_signal & self._streaming_mass_window_mask(data, SCORE_MASS_MIN, SCORE_MASS_MAX)
        score_bkg = is_mc_bkg & self._streaming_mass_window_mask(data, SCORE_MASS_MIN, SCORE_MASS_MAX)
        if reweight and "mass_edges" in reweight:
            mass_low = float(reweight["mass_edges"][0])
            mass_high = float(reweight["mass_edges"][-1])
        else:
            params = self._merge_dnn_params()
            mass_low = float(params.get("mass_reweight_min", self._default_params()["mass_reweight_min"]))
            mass_high = float(params.get("mass_reweight_max", self._default_params()["mass_reweight_max"]))
        mass_bkg = is_mc_bkg & self._streaming_mass_window_mask(data, mass_low, mass_high)
        domain_mc = is_mc_bkg & self._streaming_sideband_mask(data) & self._streaming_domain_jet_mask(data)
        domain_data = is_data & self._streaming_sideband_mask(data) & self._streaming_domain_jet_mask(data)

        if reweight:
            if self._reweight and reweight.get("score_factors") is not None:
                data["__score_weight__"] = data[self.weight].to_numpy(dtype=np.float32, copy=True)
                self._streaming_apply_reweight(data, score_bkg, reweight["score_edges"], reweight["score_factors"], "__score_weight__")
            if reweight.get("mass_factors") is not None:
                data["__mass_weight__"] = data[self.weight].to_numpy(dtype=np.float32, copy=True)
                self._streaming_apply_reweight(data, mass_bkg, reweight["mass_edges"], reweight["mass_factors"], "__mass_weight__")

        split_masks = self._split_masks(data, fold)
        return data, {
            "score_sig": score_sig,
            "score_bkg": score_bkg,
            "mass_bkg": mass_bkg,
            "domain_mc": domain_mc,
            "domain_data": domain_data,
            "splits": split_masks,
        }

    def prepareDataStreaming(self, fold=0):
        self._current_fold = fold
        if self.loadPreparedCache(fold):
            return
        if not self._streaming_prepare_supported():
            message = "Streaming preparation does not support paired-year splitting when both paired years are selected."
            self.args.strict_memory_cache = False
            print(f"DNN WARNING: {message} Falling back to legacy prepareData.")
            self.readData()
            self.prepareData(fold)
            return

        print("----------------------------------------------------------")
        print(f"DNN STREAM: preparing fold={fold} with three-pass chunked ROOT reading")
        start_total = time.perf_counter()
        hparams = self.params[fold] if fold < len(self.params) else self._merge_dnn_params()
        mass_reweight_min = float(hparams.get("mass_reweight_min", self._default_params()["mass_reweight_min"]))
        mass_reweight_max = float(hparams.get("mass_reweight_max", self._default_params()["mass_reweight_max"]))
        mass_reweight_bin_width = float(hparams.get("mass_reweight_bin_width", self._default_params()["mass_reweight_bin_width"]))
        n_features = len(self.train_variables)
        score_edges = self._streaming_reweight_edges(SCORE_MASS_MIN, SCORE_MASS_MAX, bin_width=1.0)
        mass_edges = self._streaming_reweight_edges(mass_reweight_min, mass_reweight_max, bin_width=mass_reweight_bin_width)
        score_hist = np.zeros(len(score_edges) - 1, dtype=np.float64)
        mass_hist = np.zeros(len(mass_edges) - 1, dtype=np.float64)
        scaler = StandardScaler()
        scaler_seen = False
        missing_jet_stats = self._empty_missing_jet_stats()

        counts = {
            "score": {split: {"sig": 0, "bkg": 0} for split in ("train", "val", "test")},
            "mass": {split: {"bkg": 0} for split in ("train", "val")},
            "domain": {split: {"mc": 0, "data": 0} for split in ("train", "val")},
        }
        score_stats = {"train_sig": self._empty_stats(), "train_bkg": self._empty_stats()}
        domain_stats = {
            "train_mc": self._empty_stats(),
            "train_data": self._empty_stats(),
            "val_mc": self._empty_stats(),
            "val_data": self._empty_stats(),
        }

        for spec, data in self._iter_streaming_sample_chunks():
            data, masks = self._prepare_streaming_chunk(data, spec, fold, reweight={"mass_edges": mass_edges})
            if self._reweight and np.any(masks["score_bkg"]):
                masses = data.loc[masks["score_bkg"], MASS_VARIABLE].to_numpy(dtype=np.float64, copy=False)
                weights = data.loc[masks["score_bkg"], self.weight].to_numpy(dtype=np.float64, copy=False)
                score_hist += np.histogram(masses, bins=score_edges, weights=np.clip(weights, 0.0, None))[0]
            if np.any(masks["mass_bkg"]):
                masses = data.loc[masks["mass_bkg"], MASS_VARIABLE].to_numpy(dtype=np.float64, copy=False)
                weights = data.loc[masks["mass_bkg"], self.weight].to_numpy(dtype=np.float64, copy=False)
                mass_hist += np.histogram(masses, bins=mass_edges, weights=np.clip(weights, 0.0, None))[0]
            for split_name, split_mask in masks["splits"].items():
                sig_mask = masks["score_sig"] & split_mask
                bkg_mask = masks["score_bkg"] & split_mask
                counts["score"][split_name]["sig"] += int(np.count_nonzero(sig_mask))
                counts["score"][split_name]["bkg"] += int(np.count_nonzero(bkg_mask))
                if split_name in ("train", "val"):
                    counts["mass"][split_name]["bkg"] += int(np.count_nonzero(masks["mass_bkg"] & split_mask))
                if split_name == "train":
                    train_mask = sig_mask | bkg_mask
                    if np.any(train_mask):
                        self._update_missing_jet_stats(missing_jet_stats, data, train_mask)
                    self._update_weight_stats(score_stats["train_sig"], data, sig_mask)
                    self._update_weight_stats(score_stats["train_bkg"], data, bkg_mask, include_mass=True)
            for split_name in ("train", "val"):
                split_mask = masks["splits"][split_name]
                mc_mask = masks["domain_mc"] & split_mask
                data_mask = masks["domain_data"] & split_mask
                counts["domain"][split_name]["mc"] += int(np.count_nonzero(mc_mask))
                counts["domain"][split_name]["data"] += int(np.count_nonzero(data_mask))
                self._update_weight_stats(domain_stats[f"{split_name}_mc"], data, mc_mask)
                self._update_weight_stats(domain_stats[f"{split_name}_data"], data, data_mask)

        missing_jet_impute_values = self._missing_jet_impute_values(missing_jet_stats)
        self.m_missing_jet_impute_values[fold] = missing_jet_impute_values

        reweight = {
            "score_edges": score_edges,
            "score_factors": self._streaming_reweight_factors(score_hist) if self._reweight else None,
            "mass_edges": mass_edges,
            "mass_factors": self._streaming_reweight_factors(mass_hist),
        }
        mass_nonzero = mass_hist > 0.0
        mass_flat = mass_hist * reweight["mass_factors"]
        if np.any(mass_nonzero):
            print(
                "DNN INFO: mass loss reweight "
                f"fold={fold} window=[{mass_reweight_min:g},{mass_reweight_max:g}] "
                f"bin_width={mass_reweight_bin_width:g} raw_sum={float(np.sum(mass_hist)):.6g} "
                f"raw_bin_minmax=({float(np.min(mass_hist[mass_nonzero])):.6g},{float(np.max(mass_hist[mass_nonzero])):.6g}) "
                f"flat_bin_minmax=({float(np.min(mass_flat[mass_nonzero])):.6g},{float(np.max(mass_flat[mass_nonzero])):.6g})"
            )

        score_stats = {"train_sig": self._empty_stats(), "train_bkg": self._empty_stats()}
        domain_stats = {
            "train_mc": self._empty_stats(),
            "train_data": self._empty_stats(),
            "val_mc": self._empty_stats(),
            "val_data": self._empty_stats(),
        }
        print(
            "DNN STREAM: recomputing weight normalizations after streaming reweight factors "
            f"(score_reweight={self._reweight})"
        )
        for spec, data in self._iter_streaming_sample_chunks():
            data, masks = self._prepare_streaming_chunk(data, spec, fold, reweight=reweight)
            for split_name, split_mask in masks["splits"].items():
                if split_name == "train":
                    train_mask = (masks["score_sig"] | masks["score_bkg"]) & split_mask
                    if np.any(train_mask):
                        features = data.loc[train_mask, self.train_variables].to_numpy(dtype=np.float32, copy=False)
                        scaler.partial_fit(self._impute_missing_jet_features(features, missing_jet_impute_values))
                        scaler_seen = True
                    self._update_weight_stats(score_stats["train_sig"], data, masks["score_sig"] & split_mask)
                    score_weight_column = "__score_weight__" if self._reweight else self.weight
                    self._update_weight_stats(score_stats["train_bkg"], data, masks["score_bkg"] & split_mask, include_mass=True, weight_column=score_weight_column)
            for split_name in ("train", "val"):
                split_mask = masks["splits"][split_name]
                self._update_weight_stats(domain_stats[f"{split_name}_mc"], data, masks["domain_mc"] & split_mask)
                self._update_weight_stats(domain_stats[f"{split_name}_data"], data, masks["domain_data"] & split_mask)
        if not scaler_seen:
            raise ValueError(f"No score training rows found for fold {fold}; cannot fit StandardScaler")
        train_sig_n = max(counts["score"]["train"]["sig"], 1)
        train_bkg_n = max(counts["score"]["train"]["bkg"], 1)
        self.SF = 1.0 * train_sig_n / train_bkg_n
        score_norm_factors = {
            1: 1.0 / max(score_stats["train_sig"]["sum"] / train_sig_n, 1e-12),
            0: self.SF / max(score_stats["train_bkg"]["sum"] / train_bkg_n, 1e-12),
        }
        train_domain_data_count = domain_stats["train_data"]["count"]
        train_domain_bkg_count = domain_stats["train_mc"]["count"]
        train_domain_data_mean = max(
            domain_stats["train_data"]["sum"] / train_domain_data_count if train_domain_data_count else 1.0,
            1e-12,
        )
        train_domain_bkg_mean = max(
            domain_stats["train_mc"]["sum"] / train_domain_bkg_count if train_domain_bkg_count else 1.0,
            1e-12,
        )
        domain_sf = 1.0 * train_domain_data_count / max(train_domain_bkg_count, 1)
        domain_norm_factors = {
            0: domain_sf / train_domain_bkg_mean,
            1: 1.0 / train_domain_data_mean,
        }
        mass_mean = score_stats["train_bkg"]["sum_mass"] / train_bkg_n
        mass_var = score_stats["train_bkg"]["sum_mass2"] / train_bkg_n - mass_mean * mass_mean
        mass_std = math.sqrt(max(mass_var, 0.0)) if train_bkg_n else 1.0
        mass_std = mass_std if mass_std > 0 else 1.0
        self.m_mass_norm[fold] = {"variable": MASS_VARIABLE, "background_train_mean": float(mass_mean), "background_train_std": float(mass_std)}

        paths = self._train_cache_paths(fold)
        if not paths:
            raise RuntimeError("Streaming prepare requires --train-cache-dir; disable streaming for no-cache legacy mode.")
        os.makedirs(paths["base"], exist_ok=True)
        tmp_arrays_dir = paths["arrays_dir"] + ".tmp"
        if os.path.exists(tmp_arrays_dir):
            shutil.rmtree(tmp_arrays_dir)
        os.makedirs(tmp_arrays_dir, exist_ok=True)

        def make_array(name, shape, dtype):
            return np.lib.format.open_memmap(os.path.join(tmp_arrays_dir, f"{name}.npy"), mode="w+", dtype=dtype, shape=shape)

        arrays = {
            "x_train": make_array("x_train", (counts["score"]["train"]["sig"] + counts["score"]["train"]["bkg"], n_features), np.float32),
            "x_val": make_array("x_val", (counts["score"]["val"]["sig"] + counts["score"]["val"]["bkg"], n_features), np.float32),
            "x_test": make_array("x_test", (counts["score"]["test"]["sig"] + counts["score"]["test"]["bkg"], n_features), np.float32),
            "x_test_sig": make_array("x_test_sig", (counts["score"]["test"]["sig"], n_features), np.float32),
            "x_test_bkg": make_array("x_test_bkg", (counts["score"]["test"]["bkg"], n_features), np.float32),
            "y_train": make_array("y_train", (counts["score"]["train"]["sig"] + counts["score"]["train"]["bkg"],), np.uint8),
            "y_val": make_array("y_val", (counts["score"]["val"]["sig"] + counts["score"]["val"]["bkg"],), np.uint8),
            "y_test": make_array("y_test", (counts["score"]["test"]["sig"] + counts["score"]["test"]["bkg"],), np.uint8),
            "train_wt": make_array("train_wt", (counts["score"]["train"]["sig"] + counts["score"]["train"]["bkg"],), np.float32),
            "val_wt": make_array("val_wt", (counts["score"]["val"]["sig"] + counts["score"]["val"]["bkg"],), np.float32),
            "test_wt": make_array("test_wt", (counts["score"]["test"]["sig"] + counts["score"]["test"]["bkg"],), np.float32),
            "train_wt_raw": make_array("train_wt_raw", (counts["score"]["train"]["sig"] + counts["score"]["train"]["bkg"],), np.float32),
            "val_wt_raw": make_array("val_wt_raw", (counts["score"]["val"]["sig"] + counts["score"]["val"]["bkg"],), np.float32),
            "test_wt_raw": make_array("test_wt_raw", (counts["score"]["test"]["sig"] + counts["score"]["test"]["bkg"],), np.float32),
            "train_event_weight": make_array("train_event_weight", (counts["score"]["train"]["sig"] + counts["score"]["train"]["bkg"],), np.float32),
            "val_event_weight": make_array("val_event_weight", (counts["score"]["val"]["sig"] + counts["score"]["val"]["bkg"],), np.float32),
            "test_event_weight": make_array("test_event_weight", (counts["score"]["test"]["sig"] + counts["score"]["test"]["bkg"],), np.float32),
            "mass_x_train": make_array("mass_x_train", (counts["mass"]["train"]["bkg"], n_features), np.float32),
            "mass_x_val": make_array("mass_x_val", (counts["mass"]["val"]["bkg"], n_features), np.float32),
            "mass_wt_raw_train": make_array("mass_wt_raw_train", (counts["mass"]["train"]["bkg"],), np.float32),
            "mass_wt_raw_val": make_array("mass_wt_raw_val", (counts["mass"]["val"]["bkg"],), np.float32),
            "mass_event_weight_train": make_array("mass_event_weight_train", (counts["mass"]["train"]["bkg"],), np.float32),
            "mass_event_weight_val": make_array("mass_event_weight_val", (counts["mass"]["val"]["bkg"],), np.float32),
            "mass_mass_train": make_array("mass_mass_train", (counts["mass"]["train"]["bkg"],), np.float32),
            "mass_mass_val": make_array("mass_mass_val", (counts["mass"]["val"]["bkg"],), np.float32),
            "train_mass": make_array("train_mass", (counts["score"]["train"]["sig"] + counts["score"]["train"]["bkg"],), np.float32),
            "val_mass": make_array("val_mass", (counts["score"]["val"]["sig"] + counts["score"]["val"]["bkg"],), np.float32),
            "test_mass": make_array("test_mass", (counts["score"]["test"]["sig"] + counts["score"]["test"]["bkg"],), np.float32),
            "train_mass_norm": make_array("train_mass_norm", (counts["score"]["train"]["sig"] + counts["score"]["train"]["bkg"],), np.float32),
            "val_mass_norm": make_array("val_mass_norm", (counts["score"]["val"]["sig"] + counts["score"]["val"]["bkg"],), np.float32),
            "test_mass_norm": make_array("test_mass_norm", (counts["score"]["test"]["sig"] + counts["score"]["test"]["bkg"],), np.float32),
            "domain_x_train": make_array("domain_x_train", (counts["domain"]["train"]["mc"] + counts["domain"]["train"]["data"], n_features), np.float32),
            "domain_x_val": make_array("domain_x_val", (counts["domain"]["val"]["mc"] + counts["domain"]["val"]["data"], n_features), np.float32),
            "domain_y_train": make_array("domain_y_train", (counts["domain"]["train"]["mc"] + counts["domain"]["train"]["data"],), np.uint8),
            "domain_y_val": make_array("domain_y_val", (counts["domain"]["val"]["mc"] + counts["domain"]["val"]["data"],), np.uint8),
            "domain_wt_train": make_array("domain_wt_train", (counts["domain"]["train"]["mc"] + counts["domain"]["train"]["data"],), np.float32),
            "domain_wt_val": make_array("domain_wt_val", (counts["domain"]["val"]["mc"] + counts["domain"]["val"]["data"],), np.float32),
        }
        offsets = {key: 0 for key in arrays}

        def write_score(split_name, data, mask, label, norm_factor):
            if not np.any(mask):
                return
            suffix = {"train": "train", "val": "val", "test": "test"}[split_name]
            x_key = f"x_{suffix}"
            n = int(np.count_nonzero(mask))
            start = offsets[x_key]
            stop = start + n
            features = data.loc[mask, self.train_variables].to_numpy(dtype=np.float32, copy=False)
            features = self._impute_missing_jet_features(features, missing_jet_impute_values)
            arrays[x_key][start:stop] = scaler.transform(features).astype(np.float32, copy=False)
            arrays[f"y_{suffix}"][start:stop] = np.uint8(label)
            weight_column = "__score_weight__" if label == 0 and "__score_weight__" in data.columns else self.weight
            raw = data.loc[mask, weight_column].to_numpy(dtype=np.float32, copy=True) if weight_column in data.columns else np.ones(n, dtype=np.float32)
            loss_weight = np.clip(raw * norm_factor if split_name != "test" else raw, 0.0, None)
            arrays[f"{suffix}_wt_raw"][start:stop] = raw
            arrays[f"{suffix}_wt"][start:stop] = loss_weight
            arrays[f"{suffix}_event_weight"][start:stop] = data.loc[mask, SHAPE_EVENT_WEIGHT_COLUMN].to_numpy(dtype=np.float32, copy=True) if SHAPE_EVENT_WEIGHT_COLUMN in data.columns else np.ones(n, dtype=np.float32)
            masses = data.loc[mask, MASS_VARIABLE].to_numpy(dtype=np.float32, copy=False)
            arrays[f"{suffix}_mass"][start:stop] = masses
            arrays[f"{suffix}_mass_norm"][start:stop] = ((masses - mass_mean) / mass_std).astype(np.float32, copy=False)
            offsets[x_key] = stop
            for key in (f"y_{suffix}", f"{suffix}_wt", f"{suffix}_wt_raw", f"{suffix}_event_weight", f"{suffix}_mass", f"{suffix}_mass_norm"):
                offsets[key] = stop
            if split_name == "test":
                tb_key = "x_test_sig" if label == 1 else "x_test_bkg"
                tb_start = offsets[tb_key]
                arrays[tb_key][tb_start:tb_start + n] = arrays[x_key][start:stop]
                offsets[tb_key] = tb_start + n

        def write_mass(split_name, data, mask):
            if not np.any(mask):
                return
            x_key = f"mass_x_{split_name}"
            n = int(np.count_nonzero(mask))
            start = offsets[x_key]
            stop = start + n
            features = data.loc[mask, self.train_variables].to_numpy(dtype=np.float32, copy=False)
            features = self._impute_missing_jet_features(features, missing_jet_impute_values)
            arrays[x_key][start:stop] = scaler.transform(features).astype(np.float32, copy=False)
            weight_column = "__mass_weight__" if "__mass_weight__" in data.columns else self.weight
            arrays[f"mass_wt_raw_{split_name}"][start:stop] = data.loc[mask, weight_column].to_numpy(dtype=np.float32, copy=True)
            arrays[f"mass_event_weight_{split_name}"][start:stop] = data.loc[mask, SHAPE_EVENT_WEIGHT_COLUMN].to_numpy(dtype=np.float32, copy=True) if SHAPE_EVENT_WEIGHT_COLUMN in data.columns else np.ones(n, dtype=np.float32)
            arrays[f"mass_mass_{split_name}"][start:stop] = data.loc[mask, MASS_VARIABLE].to_numpy(dtype=np.float32, copy=False)
            offsets[x_key] = stop
            offsets[f"mass_wt_raw_{split_name}"] = stop
            offsets[f"mass_event_weight_{split_name}"] = stop
            offsets[f"mass_mass_{split_name}"] = stop

        def write_domain(split_name, data, mask, label, norm_factor):
            if not np.any(mask):
                return
            x_key = f"domain_x_{split_name}"
            n = int(np.count_nonzero(mask))
            start = offsets[x_key]
            stop = start + n
            features = data.loc[mask, self.train_variables].to_numpy(dtype=np.float32, copy=False)
            features = self._impute_missing_jet_features(features, missing_jet_impute_values)
            arrays[x_key][start:stop] = scaler.transform(features).astype(np.float32, copy=False)
            arrays[f"domain_y_{split_name}"][start:stop] = np.uint8(label)
            raw = data.loc[mask, self.weight].to_numpy(dtype=np.float32, copy=False) if self.weight in data.columns else np.ones(n, dtype=np.float32)
            arrays[f"domain_wt_{split_name}"][start:stop] = np.clip(raw * norm_factor, 0.0, None)
            offsets[x_key] = stop
            offsets[f"domain_y_{split_name}"] = stop
            offsets[f"domain_wt_{split_name}"] = stop

        for spec, data in self._iter_streaming_sample_chunks():
            data, masks = self._prepare_streaming_chunk(data, spec, fold, reweight=reweight)
            for split_name, split_mask in masks["splits"].items():
                write_score(split_name, data, masks["score_sig"] & split_mask, 1, score_norm_factors[1])
                write_score(split_name, data, masks["score_bkg"] & split_mask, 0, score_norm_factors[0])
                if split_name in ("train", "val"):
                    write_mass(split_name, data, masks["mass_bkg"] & split_mask)
            for split_name in ("train", "val"):
                split_mask = masks["splits"][split_name]
                write_domain(split_name, data, masks["domain_mc"] & split_mask, 0, domain_norm_factors[0])
                write_domain(split_name, data, masks["domain_data"] & split_mask, 1, domain_norm_factors[1])

        for value in arrays.values():
            value.flush()
        del arrays
        if os.path.exists(paths["arrays_dir"]):
            shutil.rmtree(paths["arrays_dir"])
        os.replace(tmp_arrays_dir, paths["arrays_dir"])
        with open(paths["scaler"] + ".tmp", "wb") as handle:
            pickle.dump(scaler, handle, -1)
        os.replace(paths["scaler"] + ".tmp", paths["scaler"])
        meta = {
            "cache_payload": self._normalized_train_cache_payload(),
            "fold": fold,
            "cache_format": "npy",
            "mass_norm": self.m_mass_norm[fold],
            "missing_jet_impute_values": self.m_missing_jet_impute_values.get(fold, {}),
            "SF": self.SF,
            "shapes": {
                "x_train": [counts["score"]["train"]["sig"] + counts["score"]["train"]["bkg"], n_features],
                "x_val": [counts["score"]["val"]["sig"] + counts["score"]["val"]["bkg"], n_features],
                "x_test": [counts["score"]["test"]["sig"] + counts["score"]["test"]["bkg"], n_features],
                "mass_x_train": [counts["mass"]["train"]["bkg"], n_features],
                "mass_x_val": [counts["mass"]["val"]["bkg"], n_features],
                "domain_x_train": [counts["domain"]["train"]["mc"] + counts["domain"]["train"]["data"], n_features],
                "domain_x_val": [counts["domain"]["val"]["mc"] + counts["domain"]["val"]["data"], n_features],
            },
            "mass_reweight": {
                "min": mass_reweight_min,
                "max": mass_reweight_max,
                "bin_width": mass_reweight_bin_width,
            },
            "streaming_counts": counts,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(paths["meta"] + ".tmp", "w") as handle:
            json.dump(meta, handle, indent=2, sort_keys=True)
        os.replace(paths["meta"] + ".tmp", paths["meta"])
        print(tabulate(
            [
                ["Signal", counts["score"]["train"]["sig"] + counts["score"]["val"]["sig"], counts["score"]["train"]["sig"], counts["score"]["val"]["sig"]],
                ["Score background MC", counts["score"]["train"]["bkg"] + counts["score"]["val"]["bkg"], counts["score"]["train"]["bkg"], counts["score"]["val"]["bkg"]],
                ["Domain background MC", counts["domain"]["train"]["mc"] + counts["domain"]["val"]["mc"], counts["domain"]["train"]["mc"], counts["domain"]["val"]["mc"]],
                ["Domain data", counts["domain"]["train"]["data"] + counts["domain"]["val"]["data"], counts["domain"]["train"]["data"], counts["domain"]["val"]["data"]],
            ],
            headers=["Sample", "Total", "Training", "Validation"],
            tablefmt="simple",
        ))
        print(f"DNN CACHE: streaming prepared fold={fold} format=npy size_gb={self._cache_size_gb(paths):.3f} seconds={time.perf_counter() - start_total:.1f}")
        self.loadPreparedCache(fold)

    def _coerce_year_value(self, year):
        if isinstance(year, str) and year.isdigit():
            return int(year)
        return year

    def _build_pair_hash(self, row, key_columns, seed):
        payload = "::".join(str(row[col]) for col in key_columns)
        digest = hashlib.blake2b(f"{seed}::{payload}".encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False)

    def _apply_paired_year_splitting(self):
        if not self.paired_year_splitting or not self.paired_year_splitting.get("enabled", False):
            return
        self.m_data_sig = self._apply_paired_year_splitting_to_frame(self.m_data_sig, "signal")
        self.m_data_bkg = self._apply_paired_year_splitting_to_frame(self.m_data_bkg, "background")

    def _apply_paired_year_splitting_to_frame(self, data, label):
        if data.empty:
            return data
        cfg = self.paired_year_splitting
        left_year = self._coerce_year_value(cfg["left_year"])
        right_year = self._coerce_year_value(cfg["right_year"])
        match_keys = cfg.get("match_keys", [self.randomIndex])
        seed = cfg.get("random_seed", 20250325)
        weight_branch = cfg.get("weight_branch", self.weight)
        check_branches = cfg.get("check_branches", [])
        sample_filter = set(cfg.get("samples", []))
        year_column = "source_year_label" if "source_year_label" in data.columns else "source_year"
        required_columns = ["__sample__", year_column] + match_keys
        missing_required = [col for col in required_columns if col not in data.columns]
        if missing_required:
            print(f"DNN WARNING: Skip paired year splitting for {label}, missing columns: {missing_required}")
            return data

        work = data.copy()
        work["__paired_year_split__"] = 0
        base_mask = work[year_column].isin([left_year, right_year])
        if "__source__" in work.columns:
            base_mask &= work["__source__"] == "mc"
        if sample_filter:
            base_mask &= work["__sample__"].isin(sample_filter)
        if not base_mask.any():
            return work

        key_columns = ["__sample__"] + match_keys
        left = work.loc[base_mask & (work[year_column] == left_year)].copy()
        right = work.loc[base_mask & (work[year_column] == right_year)].copy()
        if left.empty or right.empty:
            return work

        left_counts = left.groupby(key_columns).size().rename("n_left").reset_index()
        right_counts = right.groupby(key_columns).size().rename("n_right").reset_index()
        unique_pairs = left_counts[left_counts["n_left"] == 1].merge(right_counts[right_counts["n_right"] == 1], on=key_columns, how="inner")
        if unique_pairs.empty:
            return work

        left_indexed = left.reset_index().merge(unique_pairs[key_columns], on=key_columns, how="inner")
        right_indexed = right.reset_index().merge(unique_pairs[key_columns], on=key_columns, how="inner")
        matched = left_indexed[key_columns + ["index"]].merge(
            right_indexed[key_columns + ["index"]],
            on=key_columns,
            how="inner",
            suffixes=("_left", "_right"),
        )
        if matched.empty:
            return work

        matched["__pair_hash__"] = matched.apply(lambda row: self._build_pair_hash(row, key_columns, seed), axis=1)
        matched["keep_left"] = (matched["__pair_hash__"] % 2) == 0
        keep_left_idx = matched.loc[matched["keep_left"], "index_left"].to_numpy()
        keep_right_idx = matched.loc[~matched["keep_left"], "index_right"].to_numpy()
        drop_left_idx = matched.loc[~matched["keep_left"], "index_left"].to_numpy()
        drop_right_idx = matched.loc[matched["keep_left"], "index_right"].to_numpy()
        kept_idx = np.concatenate([keep_left_idx, keep_right_idx])
        dropped_idx = np.concatenate([drop_left_idx, drop_right_idx])
        if weight_branch and weight_branch in work.columns and len(kept_idx) > 0:
            work.loc[kept_idx, weight_branch] = work.loc[kept_idx, weight_branch] * 2.0
        work.loc[kept_idx, "__paired_year_split__"] = 1
        work = work.drop(index=dropped_idx)
        report = {
            "matched_pairs": int(len(matched)),
            "kept_left_events": int(len(keep_left_idx)),
            "kept_right_events": int(len(keep_right_idx)),
            "dropped_left_events": int(len(drop_left_idx)),
            "dropped_right_events": int(len(drop_right_idx)),
            "left_year_label": left_year,
            "right_year_label": right_year,
        }
        for branch in check_branches:
            if branch not in work.columns:
                report[f"missing_check_branch_{branch}"] = True
                continue
            matched_kept = work.loc[kept_idx]
            report[f"{left_year}_matched_{branch}_sum"] = float(matched_kept.loc[matched_kept[year_column] == left_year, branch].sum())
            report[f"{right_year}_matched_{branch}_sum"] = float(matched_kept.loc[matched_kept[year_column] == right_year, branch].sum())
        self.paired_year_reports[label] = report
        print(f"DNN INFO: Paired-year splitting report for {label}: {report}")
        return work

    def _balance_gluglu_mc(self):
        if "GluGluHToMuMu_M125" not in self.train_mc_background or len(self.train_mc_background) <= 1:
            return
        if not self.weight or self.m_data_bkg.empty:
            return
        if "__sample__" not in self.m_data_bkg.columns or "__source__" not in self.m_data_bkg.columns:
            return
        mc_mask = self.m_data_bkg["__source__"] == "mc"
        glu_mask = mc_mask & (self.m_data_bkg["__sample__"] == "GluGluHToMuMu_M125")
        other_mc_mask = mc_mask & (self.m_data_bkg["__sample__"] != "GluGluHToMuMu_M125")
        total_glu = self.m_data_bkg.loc[glu_mask, self.weight].sum()
        total_other = self.m_data_bkg.loc[other_mc_mask, self.weight].sum()
        if total_glu <= 0 or total_other <= 0:
            return
        scale = total_other / total_glu
        self.m_data_bkg.loc[glu_mask, self.weight] *= scale
        print(f"DNN INFO: Rescaled GluGluHToMuMu_M125 by {scale:.4f} to match other MC yield.")

    def _corr_columns(self, data):
        wanted = self.train_variables + [self.reso, MASS_VARIABLE, self.weight]
        columns = []
        for name in wanted:
            if name in data.columns and name not in columns:
                columns.append(name)
        return columns

    def _plot_weights(self, data):
        if self.weight in data.columns:
            weights = data[self.weight].to_numpy(dtype=np.float64, copy=True)
            weights[~np.isfinite(weights)] = 0.0
            return np.clip(weights, 0.0, None)
        return np.ones(len(data), dtype=np.float64)

    def _weighted_pearson(self, x, y, weights):
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
        if np.count_nonzero(mask) < 2:
            return np.nan
        x = x[mask].astype(np.float64, copy=False)
        y = y[mask].astype(np.float64, copy=False)
        weights = weights[mask].astype(np.float64, copy=False)
        weight_sum = np.sum(weights)
        if weight_sum <= 0:
            return np.nan
        x_mean = np.sum(weights * x) / weight_sum
        y_mean = np.sum(weights * y) / weight_sum
        x_centered = x - x_mean
        y_centered = y - y_mean
        cov = np.sum(weights * x_centered * y_centered) / weight_sum
        x_var = np.sum(weights * x_centered * x_centered) / weight_sum
        y_var = np.sum(weights * y_centered * y_centered) / weight_sum
        denom = math.sqrt(max(x_var * y_var, 0.0))
        if denom <= 0:
            return np.nan
        return cov / denom

    def _score_bin_mass_shape_metrics(self, scores, masses, weights, mass_min=110.0, mass_max=150.0, mass_bin_width=1.0, n_bins=10):
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        masses = np.asarray(masses, dtype=np.float64).reshape(-1)
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        nan_metrics = {key: float("nan") for key in SCORE_BIN_MASS_METRIC_KEYS}
        valid = (
            np.isfinite(scores)
            & np.isfinite(masses)
            & np.isfinite(weights)
            & (weights > 0.0)
            & (masses >= mass_min)
            & (masses < mass_max)
        )
        if np.count_nonzero(valid) < 2:
            return nan_metrics

        valid_indices = np.where(valid)[0]
        order = np.argsort(scores[valid], kind="mergesort")
        sorted_indices = valid_indices[order]
        sorted_weights = weights[sorted_indices]
        total_weight = float(np.sum(sorted_weights))
        if total_weight <= 0.0:
            return nan_metrics

        cumulative_midpoint = np.cumsum(sorted_weights) - 0.5 * sorted_weights
        percentile = cumulative_midpoint / total_weight
        bin_indices = np.full(len(scores), -1, dtype=np.int32)
        bin_indices[sorted_indices] = np.minimum((percentile * n_bins).astype(np.int32), n_bins - 1)

        mass_bins = np.arange(mass_min, mass_max + mass_bin_width, mass_bin_width, dtype=np.float64)
        inclusive_hist, _ = np.histogram(masses[valid], bins=mass_bins, weights=weights[valid])
        inclusive_total = float(np.sum(inclusive_hist))
        inclusive_shape = np.divide(
            inclusive_hist,
            inclusive_total,
            out=np.zeros_like(inclusive_hist, dtype=np.float64),
            where=inclusive_total > 0.0,
        )

        means = []
        ratio_chunks = []
        for score_bin in range(n_bins):
            mask = bin_indices == score_bin
            if not np.any(mask):
                continue
            bin_masses = masses[mask]
            bin_weights = weights[mask]
            bin_weight_sum = float(np.sum(bin_weights))
            if bin_weight_sum <= 0.0:
                continue
            means.append(float(np.average(bin_masses, weights=bin_weights)))
            hist, _ = np.histogram(bin_masses, bins=mass_bins, weights=bin_weights)
            shape = np.divide(hist, bin_weight_sum, out=np.zeros_like(hist, dtype=np.float64), where=bin_weight_sum > 0.0)
            ratio = np.divide(shape, inclusive_shape, out=np.full_like(shape, np.nan), where=inclusive_shape > 0.0)
            finite_ratio = ratio[np.isfinite(ratio)]
            if finite_ratio.size:
                ratio_chunks.append(finite_ratio)

        finite_means = np.asarray([value for value in means if np.isfinite(value)], dtype=np.float64)
        all_ratios = np.concatenate(ratio_chunks) if ratio_chunks else np.empty(0, dtype=np.float64)
        mean_variance = float(np.var(finite_means)) if finite_means.size else float("nan")
        return {
            "score_bin_mass_mean_variance": mean_variance,
            "score_bin_mass_mean_rms": float(math.sqrt(mean_variance)) if np.isfinite(mean_variance) else float("nan"),
            "score_bin_mass_shape_rms": float(np.sqrt(np.mean(np.square(all_ratios - 1.0)))) if all_ratios.size else float("nan"),
            "score_bin_mass_shape_max_abs": float(np.max(np.abs(all_ratios - 1.0))) if all_ratios.size else float("nan"),
        }

    def _validation_score_bin_mass_shape_metrics(self, fold, scores):
        labels = np.asarray(self.m_y_val[fold]).reshape(-1)
        bkg_mask = labels == 0
        return self._score_bin_mass_shape_metrics(
            np.asarray(scores).reshape(-1)[bkg_mask],
            np.asarray(self.m_val_mass[fold]).reshape(-1)[bkg_mask],
            np.asarray(self.m_val_event_weight[fold]).reshape(-1)[bkg_mask],
        )

    def _weighted_corr_dataframe(self, data, columns, weights):
        corr = np.eye(len(columns), dtype=np.float64)
        arrays = {column: data[column].to_numpy(dtype=np.float64, copy=False) for column in columns}
        for i, x_name in enumerate(columns):
            for j, y_name in enumerate(columns[:i]):
                value = self._weighted_pearson(arrays[x_name], arrays[y_name], weights)
                corr[i, j] = value
                corr[j, i] = value
        return pd.DataFrame(corr, index=columns, columns=columns)

    def _plot_corr_from_dataframe(self, data, data_type):
        corr_dir = os.path.join(self._plotFolder, "corr")
        os.makedirs(corr_dir, exist_ok=True)
        weights = self._plot_weights(data)
        columns = list(data.columns)
        for rem in ("is_vbfhmm", "weight", "event", "trg_single_mu24", "__sample__", "__source__", "__tree__", "source_year_label", "source_year_from_file", "source_year_file_matches_config", "__paired_year_split__", "source_year"):
            if rem in columns:
                columns.remove(rem)
                data = data.drop(rem, axis=1)
        columns = sorted(columns)
        if self.reso in columns:
            columns.remove(self.reso)
            columns.append(self.reso)
        if "diMufsr_kit_BSC_mass" in columns:
            columns.remove("diMufsr_kit_BSC_mass")
            columns.append("diMufsr_kit_BSC_mass")
        data = data[columns]
        for col_name in data.columns:
            if pd.api.types.is_numeric_dtype(data[col_name]) and not pd.api.types.is_bool_dtype(data[col_name]):
                data.loc[data[col_name] < -90, col_name] = np.nan
        data = self._weighted_corr_dataframe(data, columns, weights) * 100
        lower_triangle = np.tril(data, -1)
        mask = np.triu(np.ones_like(data, dtype=bool), k=0)
        lower_triangle[mask] = np.nan
        plt.figure(figsize=(12, 9), dpi=300)
        plt.imshow(lower_triangle, cmap="coolwarm")
        cbar = plt.colorbar()
        cbar.set_label("Weighted correlation (%)", fontsize=16)
        plt.clim(-100, 100)
        for i in range(len(columns)):
            for j in range(len(columns)):
                if not np.isnan(lower_triangle[i, j]):
                    plt.text(j, i, f"{lower_triangle[i, j]:3.0f}", ha="center", va="center", color="black", fontsize=12)
        plt.xticks(np.arange(0, len(columns)), columns, rotation=-45, fontsize=16, ha="left")
        plt.yticks(np.arange(0, len(columns)), columns, fontsize=16)
        plt.tight_layout()
        out_path = os.path.join(corr_dir, f"corr_{self._region}_{data_type}.pdf")
        plt.savefig(out_path)
        plt.close()
        print(f"DNN INFO: saved weighted correlation plot to {out_path}")

    def compute_reweight_factors(self, data, mass_var="diMufsr_kit_BSC_mass", weight_var=None, low=None, high=None, bin_width=1.0, label=None):
        if weight_var is None:
            weight_var = self.weight
        if low is None:
            low = float(data[mass_var].min()) if len(data) else 0.0
        if high is None:
            high = float(data[mass_var].max()) if len(data) else low + float(bin_width)
        bin_edges = self._streaming_reweight_edges(low, high, bin_width=bin_width)
        n_bins = max(1, len(bin_edges) - 1)
        masses = data[mass_var].to_numpy(dtype=np.float64, copy=False)
        weights = data[weight_var].to_numpy(dtype=np.float64, copy=False) if weight_var in data.columns else np.ones(len(data), dtype=np.float64)
        weights_for_sf = np.clip(weights, 0.0, None)
        hist, bin_edges = np.histogram(masses, bins=bin_edges, weights=weights_for_sf)
        reweight_per_bin = self._streaming_reweight_factors(hist).astype(np.float64)
        bin_indices = np.digitize(masses, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        reweight_factors = reweight_per_bin[bin_indices]
        os.makedirs(os.path.join(self._plotFolder, "reweight"), exist_ok=True)
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        axes[0].hist(masses, bins=bin_edges, weights=weights_for_sf, histtype="step", label="Original", color="blue")
        axes[1].hist(masses, bins=bin_edges, weights=weights_for_sf * reweight_factors, histtype="step", label="Reweighted", color="red")
        axes[1].axhline(y=float(np.sum(hist)) / n_bins if n_bins else 0.0, color="green", linestyle="--", label="Target")
        for ax in axes:
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlabel(mass_var)
            ax.set_ylabel("Weighted Events")
        plt.tight_layout()
        suffix = f"_{label}" if label else f"_fold{self._current_fold}"
        plt.savefig(os.path.join(self._plotFolder, "reweight", f"reweight_{mass_var}{suffix}.pdf"))
        plt.close()
        print(f"DNN INFO: computed global reweight factors for {label or f'fold{self._current_fold}'} using {len(data)} events")
        return reweight_factors

    def _apply_background_mass_reweight(self, bkg_all, label, low=None, high=None, bin_width=1.0):
        if label not in self._bkg_mass_reweight_factors:
            factors = self.compute_reweight_factors(bkg_all, low=low, high=high, bin_width=bin_width, label=f"global_{label}")
            self._bkg_mass_reweight_factors[label] = pd.Series(factors, index=bkg_all.index)
        bkg_all = bkg_all.copy()
        bkg_all[self.weight] = bkg_all[self.weight].values * self._bkg_mass_reweight_factors[label].loc[bkg_all.index].to_numpy(dtype=np.float64)
        return bkg_all

    def _shuffle_dataframe_for_training(self, data, split_name, fold):
        if data.empty:
            return data, np.empty(0, dtype=np.int64)
        data = data.reset_index(drop=True)
        shuffled = data.sample(frac=1, random_state=PANDAS_DATAFRAME_SHUFFLE_SEED)
        order = shuffled.index.to_numpy(dtype=np.int64)
        shuffled = shuffled.reset_index(drop=True)
        print(
            f"DNN INFO: shuffled {split_name} pandas dataframe before step training "
            f"(fold={fold}, seed={PANDAS_DATAFRAME_SHUFFLE_SEED}, rows={len(shuffled)})"
        )
        return shuffled, order

    def _apply_ptdecor(self, data, split_name, fold):
        if data.empty:
            return data
        mass_var = "diMufsr_kit_BSC_mass"
        pt_vars = [name for name in SCALED_BY_MASS_VARIABLES if name in self.train_variables and name in data.columns]
        if not pt_vars:
            return data
        if mass_var not in data.columns:
            raise KeyError(f"ptdecor requires '{mass_var}' in the loaded branches")
        mass = data[mass_var].to_numpy(dtype=np.float32, copy=False)
        safe_mass = np.where(np.abs(mass) > 1e-6, mass, np.nan).astype(np.float32, copy=False)
        data = data.copy()
        for pt_var in pt_vars:
            values = data[pt_var].to_numpy(dtype=np.float32, copy=True)
            if preserves_negative_ptdecor_sentinels(pt_var):
                sentinel_mask = values < 0
                valid_mask = ~sentinel_mask
                values[valid_mask] = values[valid_mask] / safe_mass[valid_mask]
                sentinel_count = int(np.count_nonzero(sentinel_mask))
                if sentinel_count:
                    print(f"DNN INFO: preserved {sentinel_count} negative ptdecor sentinels for {pt_var} in {split_name} split (fold={fold})")
            else:
                values = values / safe_mass
            data[pt_var] = values
        invalid_mask = ~np.isfinite(data[pt_vars].to_numpy(dtype=np.float32, copy=False)).all(axis=1)
        dropped = int(np.count_nonzero(invalid_mask))
        if dropped:
            print(f"DNN WARNING: dropped {dropped} rows with invalid ptdecor values in {split_name} split (fold={fold})")
            data = data.loc[~invalid_mask].copy()
        print(f"DNN INFO: applied ptdecor to {pt_vars} using {mass_var} in {split_name} split (fold={fold})")
        return data

    def _missing_jet_variables(self):
        return [name for name in self.train_variables if is_missing_jet_impute_feature(name)]

    def _empty_missing_jet_stats(self):
        return {name: {"sum": 0.0, "count": 0} for name in self._missing_jet_variables()}

    def _update_missing_jet_stats(self, stats, data, mask):
        if not np.any(mask):
            return
        for name in stats:
            if name not in data.columns:
                continue
            values = data.loc[mask, name].to_numpy(dtype=np.float32, copy=False)
            valid = np.isfinite(values) & (values >= MISSING_JET_SENTINEL_MAX)
            if np.any(valid):
                stats[name]["sum"] += float(np.sum(values[valid], dtype=np.float64))
                stats[name]["count"] += int(np.count_nonzero(valid))

    def _missing_jet_impute_values(self, stats):
        return {
            name: values["sum"] / values["count"]
            for name, values in stats.items()
            if values["count"] > 0
        }

    def _impute_missing_jet_dataframe(self, data, impute_values, split_name, fold):
        if data.empty or not impute_values:
            return data
        data = data.copy()
        total = 0
        for name, value in impute_values.items():
            if name not in data.columns:
                continue
            mask = data[name].to_numpy(dtype=np.float32, copy=False) < MISSING_JET_SENTINEL_MAX
            count = int(np.count_nonzero(mask))
            if count:
                data.loc[mask, name] = np.asarray(value, dtype=data[name].dtype).item()
                total += count
        if total:
            print(f"DNN INFO: imputed {total} missing jet sentinel feature values with train means in {split_name} split (fold={fold})")
        return data

    def _impute_missing_jet_features(self, features, impute_values):
        if not impute_values:
            return features
        features = np.array(features, dtype=np.float32, copy=True)
        for index, name in enumerate(self.train_variables):
            if name not in impute_values:
                continue
            mask = features[:, index] < MISSING_JET_SENTINEL_MAX
            if np.any(mask):
                features[mask, index] = impute_values[name]
        return features

    def _mass_window(self, data, low, high, include_high=True):
        if include_high:
            return data[(data[MASS_VARIABLE] >= low) & (data[MASS_VARIABLE] <= high)].copy()
        return data[(data[MASS_VARIABLE] >= low) & (data[MASS_VARIABLE] < high)].copy()

    def _sideband_window(self, data):
        return data[
            ((data[MASS_VARIABLE] >= DOMAIN_LOW_MIN) & (data[MASS_VARIABLE] < DOMAIN_LOW_MAX))
            | ((data[MASS_VARIABLE] > DOMAIN_HIGH_MIN) & (data[MASS_VARIABLE] <= DOMAIN_HIGH_MAX))
        ].copy()

    def _domain_jet_window(self, data):
        if "njets" not in data.columns:
            raise KeyError("Domain training restricted to X-jet events requires 'njets' in the loaded branches")
        return data[data["njets"] >= DOMAIN_REQUIRED_NJETS].copy()

    def _fold_split(self, data, fold):
        test = data[data[self.randomIndex] % 314159 % 4 == fold].copy()
        val = data[(data[self.randomIndex] - 1) % 314159 % 4 == fold].copy()
        train = data[
            ((data[self.randomIndex] - 2) % 314159 % 4 == fold)
            | ((data[self.randomIndex] - 3) % 314159 % 4 == fold)
        ].copy()
        return train, val, test

    def _mark_shape_event_weight(self, data):
        if data.empty or SHAPE_EVENT_WEIGHT_COLUMN in data.columns:
            return data
        data = data.copy()
        if self.weight in data.columns:
            data[SHAPE_EVENT_WEIGHT_COLUMN] = data[self.weight].to_numpy(dtype=np.float32, copy=True)
        else:
            data[SHAPE_EVENT_WEIGHT_COLUMN] = np.ones(len(data), dtype=np.float32)
        return data

    def _shape_event_weights(self, data):
        weight_col = SHAPE_EVENT_WEIGHT_COLUMN if SHAPE_EVENT_WEIGHT_COLUMN in data.columns else self.weight
        if weight_col in data.columns:
            return data[weight_col].to_numpy(dtype=np.float32, copy=True).reshape(-1)
        return np.ones(len(data), dtype=np.float32)

    def _raw_weights(self, data):
        weight_col = "eventWeight" if "eventWeight" in data.columns else self.weight
        if weight_col in data.columns:
            return data[weight_col].to_numpy(dtype=np.float32, copy=True).reshape(-1)
        return np.ones(len(data), dtype=np.float32)

    def _apply_resolution_weight(self, data, sample_name):
        if data.empty:
            return
        if self.reso not in data.columns or self.weight not in data.columns:
            print(f"DNN WARNING: skip resolution weighting for {sample_name}, missing {self.reso} or {self.weight}.")
            return
        reso = data[self.reso].to_numpy(dtype=np.float64, copy=False)
        reso = np.round(reso, 6)
        invalid = (~np.isfinite(reso)) | (reso <= 0)
        weighted = data[self.weight].to_numpy(dtype=np.float64, copy=True) / np.where(invalid, np.nan, reso)
        weighted[~np.isfinite(weighted)] = 0.0
        data[self.weight] = weighted
        if np.any(invalid):
            print(f"DNN WARNING: {sample_name} has {int(np.count_nonzero(invalid))} invalid resolution values; weights set to 0.")

    def _normalized_score_weights(self, sig, bkg, reference_sig, reference_bkg):
        if self.SF == -1:
            self.SF = 1.0 * len(reference_sig) / max(len(reference_bkg), 1)
        sig_wt = sig[self.weight].to_numpy(dtype=np.float32, copy=True) if self.weight in sig.columns else np.ones(len(sig), dtype=np.float32)
        bkg_wt = bkg[self.weight].to_numpy(dtype=np.float32, copy=True) if self.weight in bkg.columns else np.ones(len(bkg), dtype=np.float32)
        ref_sig_wt = reference_sig[self.weight].to_numpy(dtype=np.float32, copy=False) if self.weight in reference_sig.columns else np.ones(len(reference_sig), dtype=np.float32)
        ref_bkg_wt = reference_bkg[self.weight].to_numpy(dtype=np.float32, copy=False) if self.weight in reference_bkg.columns else np.ones(len(reference_bkg), dtype=np.float32)
        sig_mean = max(float(np.mean(ref_sig_wt)) if len(ref_sig_wt) else 1.0, 1e-12)
        bkg_mean = max(float(np.mean(ref_bkg_wt)) if len(ref_bkg_wt) else 1.0, 1e-12)
        return np.concatenate([sig_wt / sig_mean, bkg_wt * self.SF / bkg_mean]).astype(np.float32)

    def _domain_weights(self, bkg, data, reference_data, reference_bkg):
        bkg_wt = bkg[self.weight].to_numpy(dtype=np.float32, copy=True) if self.weight in bkg.columns else np.ones(len(bkg), dtype=np.float32)
        data_wt = data[self.weight].to_numpy(dtype=np.float32, copy=True) if self.weight in data.columns else np.ones(len(data), dtype=np.float32)
        ref_data_wt = reference_data[self.weight].to_numpy(dtype=np.float32, copy=False) if self.weight in reference_data.columns else np.ones(len(reference_data), dtype=np.float32)
        ref_bkg_wt = reference_bkg[self.weight].to_numpy(dtype=np.float32, copy=False) if self.weight in reference_bkg.columns else np.ones(len(reference_bkg), dtype=np.float32)
        domain_sf = 1.0 * len(reference_data) / max(len(reference_bkg), 1)
        data_mean = max(float(np.mean(ref_data_wt)) if len(ref_data_wt) else 1.0, 1e-12)
        bkg_mean = max(float(np.mean(ref_bkg_wt)) if len(ref_bkg_wt) else 1.0, 1e-12)
        return np.concatenate([bkg_wt * domain_sf / bkg_mean, data_wt / data_mean]).astype(np.float32)

    def prepareData(self, fold=0):
        self._current_fold = fold
        if self.loadPreparedCache(fold):
            return
        print("----------------------------------------------------------")
        print("DNN INFO: Splitting samples to training, validation, and test...")
        if MASS_VARIABLE not in self.m_data_sig.columns or MASS_VARIABLE not in self.m_data_bkg.columns:
            raise KeyError(f"Training requires '{MASS_VARIABLE}' in the loaded branches")
        hparams = self.params[fold] if fold < len(self.params) else self._merge_dnn_params()
        mass_reweight_min = float(hparams.get("mass_reweight_min", self._default_params()["mass_reweight_min"]))
        mass_reweight_max = float(hparams.get("mass_reweight_max", self._default_params()["mass_reweight_max"]))
        mass_reweight_bin_width = float(hparams.get("mass_reweight_bin_width", self._default_params()["mass_reweight_bin_width"]))

        score_sig_all = self._mass_window(self.m_data_sig, SCORE_MASS_MIN, SCORE_MASS_MAX)
        mc_bkg_all = self.m_data_bkg[self.m_data_bkg["__source__"] == "mc"].copy()
        data_bkg_all = self.m_data_bkg[self.m_data_bkg["__source__"] == "data"].copy()
        score_bkg_all = self._mass_window(mc_bkg_all, SCORE_MASS_MIN, SCORE_MASS_MAX)
        mass_bkg_all = self._mass_window(mc_bkg_all, mass_reweight_min, mass_reweight_max)
        domain_mc_all = self._sideband_window(mc_bkg_all)
        domain_data_all = self._sideband_window(data_bkg_all)
        domain_mc_all = self._domain_jet_window(domain_mc_all)
        domain_data_all = self._domain_jet_window(domain_data_all)
        score_sig_all = self._mark_shape_event_weight(score_sig_all)
        score_bkg_all = self._mark_shape_event_weight(score_bkg_all)
        mass_bkg_all = self._mark_shape_event_weight(mass_bkg_all)

        if self._reso:
            self._apply_resolution_weight(score_sig_all, "score signal all")
        if self._reso_bkg:
            self._apply_resolution_weight(score_bkg_all, "score background all")
            self._apply_resolution_weight(mass_bkg_all, "mass background all")
            self._apply_resolution_weight(domain_mc_all, "domain MC all")
            self._apply_resolution_weight(domain_data_all, "domain data all")

        if self._reweight:
            score_bkg_all = self._apply_background_mass_reweight(
                score_bkg_all,
                "score_bkg",
                low=SCORE_MASS_MIN,
                high=SCORE_MASS_MAX,
                bin_width=1.0,
            )
        mass_bkg_all = self._apply_background_mass_reweight(
            mass_bkg_all,
            "mass_loss_bkg",
            low=mass_reweight_min,
            high=mass_reweight_max,
            bin_width=mass_reweight_bin_width,
        )
        if self._ptdecor:
            score_sig_all = self._apply_ptdecor(score_sig_all, "score_sig_all", fold)
            score_bkg_all = self._apply_ptdecor(score_bkg_all, "score_bkg_all", fold)
            mass_bkg_all = self._apply_ptdecor(mass_bkg_all, "mass_bkg_all", fold)
            domain_mc_all = self._apply_ptdecor(domain_mc_all, "domain_mc_all", fold)
            domain_data_all = self._apply_ptdecor(domain_data_all, "domain_data_all", fold)

        if self.args.corr and not self._corr_plotted_from_loaded_data:
            self._plot_corr_from_dataframe(score_sig_all[self._corr_columns(score_sig_all)].copy(), "sig")
            self._plot_corr_from_dataframe(score_bkg_all[self._corr_columns(score_bkg_all)].copy(), "bkg")
            self._corr_plotted_from_loaded_data = True

        train_sig, val_sig, test_sig = self._fold_split(score_sig_all, fold)
        train_bkg, val_bkg, test_bkg = self._fold_split(score_bkg_all, fold)
        mass_train_bkg, mass_val_bkg, _ = self._fold_split(mass_bkg_all, fold)
        domain_train_mc, domain_val_mc, _ = self._fold_split(domain_mc_all, fold)
        domain_train_data, domain_val_data, _ = self._fold_split(domain_data_all, fold)
        missing_jet_stats = self._empty_missing_jet_stats()
        self._update_missing_jet_stats(missing_jet_stats, train_sig, np.ones(len(train_sig), dtype=bool))
        self._update_missing_jet_stats(missing_jet_stats, train_bkg, np.ones(len(train_bkg), dtype=bool))
        missing_jet_impute_values = self._missing_jet_impute_values(missing_jet_stats)
        self.m_missing_jet_impute_values[fold] = missing_jet_impute_values
        train_sig = self._impute_missing_jet_dataframe(train_sig, missing_jet_impute_values, "score train signal", fold)
        train_bkg = self._impute_missing_jet_dataframe(train_bkg, missing_jet_impute_values, "score train background", fold)
        mass_train_bkg = self._impute_missing_jet_dataframe(mass_train_bkg, missing_jet_impute_values, "mass train background", fold)
        mass_val_bkg = self._impute_missing_jet_dataframe(mass_val_bkg, missing_jet_impute_values, "mass validation background", fold)
        val_sig = self._impute_missing_jet_dataframe(val_sig, missing_jet_impute_values, "score validation signal", fold)
        val_bkg = self._impute_missing_jet_dataframe(val_bkg, missing_jet_impute_values, "score validation background", fold)
        test_sig = self._impute_missing_jet_dataframe(test_sig, missing_jet_impute_values, "score test signal", fold)
        test_bkg = self._impute_missing_jet_dataframe(test_bkg, missing_jet_impute_values, "score test background", fold)
        domain_train_mc = self._impute_missing_jet_dataframe(domain_train_mc, missing_jet_impute_values, "domain train MC", fold)
        domain_train_data = self._impute_missing_jet_dataframe(domain_train_data, missing_jet_impute_values, "domain train data", fold)
        domain_val_mc = self._impute_missing_jet_dataframe(domain_val_mc, missing_jet_impute_values, "domain validation MC", fold)
        domain_val_data = self._impute_missing_jet_dataframe(domain_val_data, missing_jet_impute_values, "domain validation data", fold)

        sample_size_table = [
            ["Signal", len(train_sig) + len(val_sig), len(train_sig), len(val_sig)],
            ["Score background MC", len(train_bkg) + len(val_bkg), len(train_bkg), len(val_bkg)],
            ["Mass background MC", len(mass_train_bkg) + len(mass_val_bkg), len(mass_train_bkg), len(mass_val_bkg)],
            ["Domain background MC", len(domain_train_mc) + len(domain_val_mc), len(domain_train_mc), len(domain_val_mc)],
            ["Domain data", len(domain_train_data) + len(domain_val_data), len(domain_train_data), len(domain_val_data)],
        ]
        print(tabulate(sample_size_table, headers=["Sample", "Total", "Training", "Validation"], tablefmt="simple"))
        mass_edges = self._streaming_reweight_edges(mass_reweight_min, mass_reweight_max, mass_reweight_bin_width)
        mass_values = mass_bkg_all[MASS_VARIABLE].to_numpy(dtype=np.float64, copy=False)
        mass_weights = mass_bkg_all[self.weight].to_numpy(dtype=np.float64, copy=False) if self.weight in mass_bkg_all.columns else np.ones(len(mass_bkg_all), dtype=np.float64)
        mass_hist = np.histogram(mass_values, bins=mass_edges, weights=np.clip(mass_weights, 0.0, None))[0]
        mass_nonzero = mass_hist > 0.0
        if np.any(mass_nonzero):
            print(
                "DNN INFO: mass loss reweight "
                f"fold={fold} window=[{mass_reweight_min:g},{mass_reweight_max:g}] "
                f"bin_width={mass_reweight_bin_width:g} weighted_sum={float(np.sum(mass_hist)):.6g} "
                f"flat_bin_minmax=({float(np.min(mass_hist[mass_nonzero])):.6g},{float(np.max(mass_hist[mass_nonzero])):.6g})"
            )
        print("DNN INFO: Setting the training arrays...")

        print("DNN INFO: Setting the event weights...")
        train_sig_wt_raw = self._raw_weights(train_sig)
        train_bkg_wt_raw = self._raw_weights(train_bkg)
        val_sig_wt_raw = self._raw_weights(val_sig)
        val_bkg_wt_raw = self._raw_weights(val_bkg)
        test_sig_wt_raw = self._raw_weights(test_sig)
        test_bkg_wt_raw = self._raw_weights(test_bkg)

        x_train = pd.concat([train_sig[self.train_variables], train_bkg[self.train_variables]], ignore_index=True)
        x_val = pd.concat([val_sig[self.train_variables], val_bkg[self.train_variables]], ignore_index=True)
        x_test = pd.concat([test_sig[self.train_variables], test_bkg[self.train_variables]], ignore_index=True)
        x_train, train_order = self._shuffle_dataframe_for_training(x_train, "score train", fold)
        x_val, val_order = self._shuffle_dataframe_for_training(x_val, "score validation", fold)
        x_test, test_order = self._shuffle_dataframe_for_training(x_test, "score test", fold)

        self.m_train_wt[fold] = self._normalized_score_weights(train_sig, train_bkg, train_sig, train_bkg)
        self.m_val_wt[fold] = self._normalized_score_weights(val_sig, val_bkg, train_sig, train_bkg)
        self.m_test_wt[fold] = np.concatenate([test_sig[self.weight].to_numpy(dtype=np.float32), test_bkg[self.weight].to_numpy(dtype=np.float32)])
        self.m_train_wt[fold] = self.m_train_wt[fold][train_order]
        self.m_val_wt[fold] = self.m_val_wt[fold][val_order]
        self.m_test_wt[fold] = self.m_test_wt[fold][test_order]
        self.m_train_wt[fold][self.m_train_wt[fold] < 0] = 0
        self.m_val_wt[fold][self.m_val_wt[fold] < 0] = 0
        self.m_test_wt[fold][self.m_test_wt[fold] < 0] = 0

        self.m_train_wt_raw[fold] = np.concatenate([train_sig_wt_raw, train_bkg_wt_raw]).astype(np.float32)
        self.m_val_wt_raw[fold] = np.concatenate([val_sig_wt_raw, val_bkg_wt_raw]).astype(np.float32)
        self.m_test_wt_raw[fold] = np.concatenate([test_sig_wt_raw, test_bkg_wt_raw]).astype(np.float32)
        self.m_train_event_weight[fold] = np.concatenate([self._shape_event_weights(train_sig), self._shape_event_weights(train_bkg)]).astype(np.float32)
        self.m_val_event_weight[fold] = np.concatenate([self._shape_event_weights(val_sig), self._shape_event_weights(val_bkg)]).astype(np.float32)
        self.m_test_event_weight[fold] = np.concatenate([self._shape_event_weights(test_sig), self._shape_event_weights(test_bkg)]).astype(np.float32)
        self.m_train_wt_raw[fold] = self.m_train_wt_raw[fold][train_order]
        self.m_val_wt_raw[fold] = self.m_val_wt_raw[fold][val_order]
        self.m_test_wt_raw[fold] = self.m_test_wt_raw[fold][test_order]
        self.m_train_event_weight[fold] = self.m_train_event_weight[fold][train_order]
        self.m_val_event_weight[fold] = self.m_val_event_weight[fold][val_order]
        self.m_test_event_weight[fold] = self.m_test_event_weight[fold][test_order]

        self.m_y_train[fold] = np.concatenate((np.ones(len(train_sig), dtype=np.uint8), np.zeros(len(train_bkg), dtype=np.uint8)))
        self.m_y_val[fold] = np.concatenate((np.ones(len(val_sig), dtype=np.uint8), np.zeros(len(val_bkg), dtype=np.uint8)))
        self.m_y_test[fold] = np.concatenate((np.ones(len(test_sig), dtype=np.uint8), np.zeros(len(test_bkg), dtype=np.uint8)))
        self.m_y_train[fold] = self.m_y_train[fold][train_order]
        self.m_y_val[fold] = self.m_y_val[fold][val_order]
        self.m_y_test[fold] = self.m_y_test[fold][test_order]

        self.m_train_mass[fold] = np.concatenate([train_sig[MASS_VARIABLE].values, train_bkg[MASS_VARIABLE].values]).astype(np.float32)
        self.m_val_mass[fold] = np.concatenate([val_sig[MASS_VARIABLE].values, val_bkg[MASS_VARIABLE].values]).astype(np.float32)
        self.m_test_mass[fold] = np.concatenate([test_sig[MASS_VARIABLE].values, test_bkg[MASS_VARIABLE].values]).astype(np.float32)
        self.m_train_mass[fold] = self.m_train_mass[fold][train_order]
        self.m_val_mass[fold] = self.m_val_mass[fold][val_order]
        self.m_test_mass[fold] = self.m_test_mass[fold][test_order]
        train_bkg_mass = train_bkg[MASS_VARIABLE].values.astype(np.float32)
        mass_mean = float(np.mean(train_bkg_mass)) if len(train_bkg_mass) else 0.0
        mass_std = float(np.std(train_bkg_mass)) if len(train_bkg_mass) else 1.0
        mass_std = mass_std if mass_std > 0 else 1.0
        self.m_mass_norm[fold] = {"variable": MASS_VARIABLE, "background_train_mean": mass_mean, "background_train_std": mass_std}
        self.m_train_mass_norm[fold] = ((self.m_train_mass[fold] - mass_mean) / mass_std).astype(np.float32)
        self.m_val_mass_norm[fold] = ((self.m_val_mass[fold] - mass_mean) / mass_std).astype(np.float32)
        self.m_test_mass_norm[fold] = ((self.m_test_mass[fold] - mass_mean) / mass_std).astype(np.float32)

        scaler = StandardScaler()
        self.m_x_train[fold] = scaler.fit_transform(x_train.to_numpy(dtype=np.float32)).astype(np.float32)
        self.m_x_val[fold] = scaler.transform(x_val.to_numpy(dtype=np.float32)).astype(np.float32)
        self.m_x_test[fold] = scaler.transform(x_test.to_numpy(dtype=np.float32)).astype(np.float32)
        self.m_x_test_sig[fold] = scaler.transform(test_sig[self.train_variables].to_numpy(dtype=np.float32)).astype(np.float32)
        self.m_x_test_bkg[fold] = scaler.transform(test_bkg[self.train_variables].to_numpy(dtype=np.float32)).astype(np.float32)
        self.m_mass_x_train[fold] = scaler.transform(mass_train_bkg[self.train_variables].to_numpy(dtype=np.float32)).astype(np.float32)
        self.m_mass_x_val[fold] = scaler.transform(mass_val_bkg[self.train_variables].to_numpy(dtype=np.float32)).astype(np.float32)
        self.m_mass_wt_raw_train[fold] = self._raw_weights(mass_train_bkg)
        self.m_mass_wt_raw_val[fold] = self._raw_weights(mass_val_bkg)
        self.m_mass_event_weight_train[fold] = self._shape_event_weights(mass_train_bkg)
        self.m_mass_event_weight_val[fold] = self._shape_event_weights(mass_val_bkg)
        self.m_mass_mass_train[fold] = mass_train_bkg[MASS_VARIABLE].to_numpy(dtype=np.float32, copy=True)
        self.m_mass_mass_val[fold] = mass_val_bkg[MASS_VARIABLE].to_numpy(dtype=np.float32, copy=True)

        domain_x_train = pd.concat([domain_train_mc[self.train_variables], domain_train_data[self.train_variables]], ignore_index=True)
        domain_x_val = pd.concat([domain_val_mc[self.train_variables], domain_val_data[self.train_variables]], ignore_index=True)
        domain_x_train, domain_train_order = self._shuffle_dataframe_for_training(domain_x_train, "domain train", fold)
        domain_x_val, domain_val_order = self._shuffle_dataframe_for_training(domain_x_val, "domain validation", fold)
        self.m_domain_x_train[fold] = scaler.transform(domain_x_train.to_numpy(dtype=np.float32)).astype(np.float32)
        self.m_domain_x_val[fold] = scaler.transform(domain_x_val.to_numpy(dtype=np.float32)).astype(np.float32)
        self.m_domain_y_train[fold] = np.concatenate([np.zeros(len(domain_train_mc), dtype=np.uint8), np.ones(len(domain_train_data), dtype=np.uint8)])[domain_train_order]
        self.m_domain_y_val[fold] = np.concatenate([np.zeros(len(domain_val_mc), dtype=np.uint8), np.ones(len(domain_val_data), dtype=np.uint8)])[domain_val_order]
        self.m_domain_wt_train[fold] = self._domain_weights(domain_train_mc, domain_train_data, domain_train_data, domain_train_mc)[domain_train_order]
        self.m_domain_wt_val[fold] = self._domain_weights(domain_val_mc, domain_val_data, domain_train_data, domain_train_mc)[domain_val_order]
        self.m_domain_wt_train[fold][self.m_domain_wt_train[fold] < 0] = 0
        self.m_domain_wt_val[fold][self.m_domain_wt_val[fold] < 0] = 0

        self.m_scaler[fold] = scaler
        self.m_corr_sig[fold] = score_sig_all[self._corr_columns(score_sig_all)].copy()
        self.m_corr_bkg[fold] = score_bkg_all[self._corr_columns(score_bkg_all)].copy()
        self.savePreparedCache(fold)

    def _weighted_binary_crossentropy(self, y_true, y_pred, weights):
        y_true = tf.cast(tf.reshape(y_true, (-1, 1)), tf.float32)
        y_pred = tf.clip_by_value(tf.cast(tf.reshape(y_pred, (-1, 1)), tf.float32), 1e-7, 1.0 - 1e-7)
        weights = tf.cast(tf.reshape(weights, (-1,)), tf.float32)
        bce = tf.reshape(-(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred)), (-1,))
        weighted = bce * weights
        return tf.reduce_sum(weighted) / tf.maximum(tf.reduce_sum(weights), 1.0)

    def _weighted_binary_crossentropy_sums(self, y_true, y_pred, weights):
        y_true = tf.cast(tf.reshape(y_true, (-1, 1)), tf.float32)
        y_pred = tf.clip_by_value(tf.cast(tf.reshape(y_pred, (-1, 1)), tf.float32), 1e-7, 1.0 - 1e-7)
        weights = tf.cast(tf.reshape(weights, (-1,)), tf.float32)
        bce = tf.reshape(-(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred)), (-1,))
        return tf.reduce_sum(bce * weights), tf.reduce_sum(weights)

    def _score_reweight_tensor(self, score_pred, score_reweight_state):
        score_pred = tf.stop_gradient(tf.cast(tf.reshape(score_pred, (-1,)), tf.float32))
        exponent = tf.cast(score_reweight_state["exponent"], tf.float32)
        transform_lut = tf.cast(tf.reshape(score_reweight_state["transform_lut"], (-1,)), tf.float32)
        n_lut = tf.shape(transform_lut)[0]
        score_scaled = tf.clip_by_value(score_pred, 0.0, 1.0) * tf.cast(n_lut - 1, tf.float32)
        lower_idx = tf.cast(tf.floor(score_scaled), tf.int32)
        upper_idx = tf.minimum(lower_idx + 1, n_lut - 1)
        fraction = score_scaled - tf.cast(lower_idx, tf.float32)
        y0 = tf.gather(transform_lut, lower_idx)
        y1 = tf.gather(transform_lut, upper_idx)
        score_for_weight = tf.clip_by_value(y0 + fraction * (y1 - y0), 0.0, 1.0)
        score_weight = tf.exp(exponent * score_for_weight)
        return tf.stop_gradient(score_weight)

    def _score_reweight_lut_from_references(self, references, lut_size):
        lut_size = max(2, int(lut_size))
        references = np.asarray(references, dtype=np.float32).reshape(-1)
        if references.size < 2 or not np.all(np.isfinite(references)):
            return np.linspace(0.0, 1.0, lut_size, dtype=np.float32)
        quantiles = np.linspace(0.0, 1.0, references.size, dtype=np.float32)
        order = np.argsort(references, kind="mergesort")
        sorted_refs = references[order]
        sorted_quantiles = quantiles[order]
        unique_refs, first_indices = np.unique(sorted_refs, return_index=True)
        last_indices = np.r_[first_indices[1:] - 1, len(sorted_refs) - 1]
        unique_quantiles = sorted_quantiles[last_indices]
        if unique_refs.size < 2:
            return np.linspace(0.0, 1.0, lut_size, dtype=np.float32)
        grid = np.linspace(0.0, 1.0, lut_size, dtype=np.float32)
        lut = np.interp(grid, unique_refs, unique_quantiles, left=0.0, right=1.0)
        return np.clip(lut, 0.0, 1.0).astype(np.float32, copy=False)

    def _scheduled_domain_lambda(self, epoch_number, target, warmup_epochs, schedule):
        epoch_number = int(epoch_number)
        target = float(target)
        warmup_epochs = max(0, int(warmup_epochs))
        schedule = str(schedule)
        if schedule != "epoch1_0_epoch2to5_0p5_then_5x1mexpminus_epochminus5":
            raise ValueError(f"Unsupported domain_lambda_schedule: {schedule}")
        if epoch_number <= warmup_epochs:
            return 0.0
        ramp_start_epoch = warmup_epochs + 4
        if epoch_number <= ramp_start_epoch:
            return 0.1 * target
        return target * (1.0 - math.exp(-(float(epoch_number) - float(ramp_start_epoch))))

    def _scheduled_mass_shape_lambda(self, epoch_number, target, warmup_epochs, ramp_epochs):
        epoch_number = int(epoch_number)
        target = float(target)
        warmup_epochs = max(0, int(warmup_epochs))
        ramp_epochs = max(0, int(ramp_epochs))
        if epoch_number <= warmup_epochs:
            return 0.0
        if ramp_epochs <= 0:
            return target
        return target * min(1.0, float(epoch_number - warmup_epochs) / float(ramp_epochs))

    def _domain_closeness_loss_tensor(self, domain_bce):
        ln2 = tf.math.log(tf.constant(2.0, dtype=tf.float32))
        epsilon = tf.constant(DOMAIN_CLOSENESS_EPSILON, dtype=tf.float32)
        return tf.sqrt(tf.square(tf.cast(domain_bce, tf.float32) - ln2) + tf.square(epsilon))

    def _make_mass_rff_state(self, fold, hparams):
        mass_min = float(hparams.get("mass_reweight_min", self._default_params()["mass_reweight_min"]))
        mass_max = float(hparams.get("mass_reweight_max", self._default_params()["mass_reweight_max"]))
        n_features = max(1, int(hparams.get("mass_rff_num_features", self._default_params()["mass_rff_num_features"])))
        default_score_sigmas = self._default_params()["mass_decorrelation_score_sigmas"]
        default_mass_sigmas = self._default_params()["mass_decorrelation_mass_sigmas"]
        score_sigmas = [max(1e-6, float(value)) for value in hparams.get("mass_decorrelation_score_sigmas", default_score_sigmas)]
        mass_sigmas = [max(1e-6, float(value)) for value in hparams.get("mass_decorrelation_mass_sigmas", default_mass_sigmas)]
        if not score_sigmas:
            score_sigmas = [0.06]
        if not mass_sigmas:
            mass_sigmas = [0.10]
        rng = np.random.default_rng(int(getattr(self.args, "seed", 20250408)) + 1009 * int(fold) + 17001)
        return {
            "mass_min": tf.constant(mass_min, dtype=tf.float32),
            "mass_max": tf.constant(mass_max, dtype=tf.float32),
            "omega_mass": [
                tf.constant(rng.normal(loc=0.0, scale=1.0 / sigma, size=n_features).astype(np.float32), dtype=tf.float32)
                for sigma in mass_sigmas
            ],
            "bias_mass": [
                tf.constant(rng.uniform(0.0, 2.0 * np.pi, size=n_features).astype(np.float32), dtype=tf.float32)
                for _ in mass_sigmas
            ],
            "omega_score": [
                tf.constant(rng.normal(loc=0.0, scale=1.0 / sigma, size=n_features).astype(np.float32), dtype=tf.float32)
                for sigma in score_sigmas
            ],
            "bias_score": [
                tf.constant(rng.uniform(0.0, 2.0 * np.pi, size=n_features).astype(np.float32), dtype=tf.float32)
                for _ in score_sigmas
            ],
            "n_features": n_features,
            "score_sigmas": score_sigmas,
            "mass_sigmas": mass_sigmas,
        }

    def _rff_features_tensor(self, values, omega, bias):
        values = tf.reshape(tf.cast(values, tf.float32), (-1, 1))
        n_features = tf.cast(tf.shape(omega)[0], tf.float32)
        return tf.sqrt(2.0 / n_features) * tf.cos(values * tf.reshape(omega, (1, -1)) + tf.reshape(bias, (1, -1)))

    def _multi_rff_features_tensor(self, values, omegas, biases):
        return tf.concat(
            [self._rff_features_tensor(values, omega, bias) for omega, bias in zip(omegas, biases)],
            axis=1,
        )

    def _mass_losses_tensor(self, score_pred, score_mass, mass_w_raw, mass_event_weight, hparams, mass_rff_state, score_reweight_state):
        scores = tf.reshape(tf.cast(score_pred, tf.float32), (-1,))
        masses = tf.reshape(tf.cast(score_mass, tf.float32), (-1,))
        mass_weights = tf.reshape(tf.abs(tf.cast(mass_w_raw, tf.float32)), (-1,))
        event_weights = tf.reshape(tf.abs(tf.cast(mass_event_weight, tf.float32)), (-1,))
        min_events = tf.constant(max(1, int(hparams.get("mass_min_events", self._default_params()["mass_min_events"]))), dtype=tf.int32)
        mass_loss_type = str(hparams.get("mass_decorrelation_loss", self._default_params()["mass_decorrelation_loss"]))
        physics_weight_mode = str(hparams.get("physics_feature_weight_mode", self._default_params()["physics_feature_weight_mode"]))
        physics_weight_index = PHYSICS_FEATURE_WEIGHT_MODES.index(physics_weight_mode)

        mask = (
            (masses >= mass_rff_state["mass_min"])
            & (masses <= mass_rff_state["mass_max"])
            & tf.math.is_finite(scores)
            & tf.math.is_finite(masses)
            & tf.math.is_finite(mass_weights)
            & tf.math.is_finite(event_weights)
            & (event_weights > 0.0)
        )
        n_selected = tf.reduce_sum(tf.cast(mask, tf.int32))

        def compute_loss():
            selected_scores = tf.boolean_mask(scores, mask)
            selected_masses = tf.boolean_mask(masses, mask)
            selected_mass_weights = tf.maximum(tf.boolean_mask(mass_weights, mask), 0.0)
            selected_event_weights = tf.maximum(tf.boolean_mask(event_weights, mask), 0.0)
            selected_score_weights = self._score_reweight_tensor(selected_scores, score_reweight_state)
            selected_mass_u = tf.clip_by_value(
                (selected_masses - mass_rff_state["mass_min"])
                / tf.maximum(mass_rff_state["mass_max"] - mass_rff_state["mass_min"], 1e-6),
                0.0,
                1.0,
            )
            rff_weights = tf.maximum(selected_mass_weights * selected_score_weights, 0.0)
            physics_weights = (
                selected_event_weights,
                tf.maximum(selected_event_weights * selected_score_weights, 0.0),
                selected_mass_weights,
                rff_weights,
            )

            def rff_hsic_loss(weights):
                weight_sum = tf.reduce_sum(weights)

                def compute_weighted_loss():
                    norm_weights = weights / tf.maximum(weight_sum, 1e-12)
                    mass_features = self._multi_rff_features_tensor(selected_mass_u, mass_rff_state["omega_mass"], mass_rff_state["bias_mass"])
                    score_features = self._multi_rff_features_tensor(selected_scores, mass_rff_state["omega_score"], mass_rff_state["bias_score"])
                    mass_mean = tf.reduce_sum(mass_features * norm_weights[:, None], axis=0)
                    score_mean = tf.reduce_sum(score_features * norm_weights[:, None], axis=0)
                    centered_mass = mass_features - mass_mean[None, :]
                    centered_score = score_features - score_mean[None, :]
                    covariance = tf.matmul(centered_mass, centered_score * norm_weights[:, None], transpose_a=True)
                    return tf.reduce_sum(tf.square(covariance))

                return tf.cond(
                    weight_sum > 0.0,
                    compute_weighted_loss,
                    lambda: tf.constant(0.0, dtype=tf.float32),
                )

            def physics_loss(weights):
                weight_sum = tf.reduce_sum(weights)

                def compute_weighted_loss():
                    score_quantiles = _weighted_quantiles_tensor(selected_scores, weights, PHYSICS_SCORE_QUANTILE_PROBS)
                    mass_features = physics_mass_features(selected_masses)
                    score_features = physics_score_features(selected_scores, score_quantiles)
                    return normalized_feature_cov_loss(mass_features, score_features, weights)

                return tf.cond(
                    weight_sum > 0.0,
                    compute_weighted_loss,
                    lambda: tf.constant(0.0, dtype=tf.float32),
                )

            mass_hsic_loss = rff_hsic_loss(rff_weights)
            physics_losses = tuple(physics_loss(weights) for weights in physics_weights)
            physics_active = physics_losses[physics_weight_index]
            mass_loss = physics_active if mass_loss_type == "physics_feature_cov" else mass_hsic_loss
            return (
                mass_loss,
                mass_hsic_loss,
                physics_active,
                physics_losses[0],
                physics_losses[1],
                physics_losses[2],
                physics_losses[3],
            )

        zero_losses = (
            tf.constant(0.0, dtype=tf.float32),
            tf.constant(0.0, dtype=tf.float32),
            tf.constant(0.0, dtype=tf.float32),
            tf.constant(0.0, dtype=tf.float32),
            tf.constant(0.0, dtype=tf.float32),
            tf.constant(0.0, dtype=tf.float32),
            tf.constant(0.0, dtype=tf.float32),
        )
        (
            mass_loss,
            mass_hsic_loss,
            physics_active,
            physics_none,
            physics_score,
            physics_mass,
            physics_score_mass,
        ) = tf.cond(
            n_selected >= min_events,
            compute_loss,
            lambda: zero_losses,
        )
        selected_for_stats = tf.boolean_mask(scores, mask)
        score_pred_mean = tf.cond(n_selected > 0, lambda: tf.reduce_mean(selected_for_stats), lambda: tf.constant(0.0, dtype=tf.float32))
        score_pred_std = tf.cond(n_selected > 0, lambda: tf.math.reduce_std(selected_for_stats), lambda: tf.constant(0.0, dtype=tf.float32))
        return (
            mass_loss,
            mass_hsic_loss,
            physics_active,
            physics_none,
            physics_score,
            physics_mass,
            physics_score_mass,
            score_pred_mean,
            score_pred_std,
            tf.cast(n_selected, tf.float32),
        )

    def _asimov_loss_tensor(self, score_y, score_pred, score_w_raw, hparams):
        labels = tf.cast(tf.reshape(score_y, (-1,)), tf.float32)
        scores = tf.cast(tf.reshape(score_pred, (-1,)), tf.float32)
        raw_weights = tf.cast(tf.reshape(score_w_raw, (-1,)), tf.float32)
        quantiles = tf.constant(hparams.get("asimov_quantiles", [0.05, 0.40, 0.80]), dtype=tf.float32)
        tau = tf.constant(float(hparams.get("asimov_tau", 0.02)), dtype=tf.float32)
        b_epsilon = tf.constant(float(hparams.get("asimov_b_epsilon", 1e-6)), dtype=tf.float32)
        mcstat_alpha = tf.constant(float(hparams.get("asimov_mcstat_alpha", 0.5)), dtype=tf.float32)
        tau = tf.maximum(tau, tf.constant(1e-6, dtype=tf.float32))

        signal_mask = labels > 0.5
        signal_scores = tf.boolean_mask(scores, signal_mask)
        signal_quantile_weights = tf.boolean_mask(tf.abs(raw_weights), signal_mask)
        signal_weight_sum = tf.reduce_sum(signal_quantile_weights)

        def compute_loss():
            order = tf.argsort(signal_scores, stable=True)
            sorted_scores = tf.gather(signal_scores, order)
            sorted_weights = tf.gather(signal_quantile_weights, order)
            cumulative_weights = tf.cumsum(sorted_weights)
            target_weights = quantiles * signal_weight_sum
            boundary_indices = tf.searchsorted(cumulative_weights, target_weights, side="left")
            boundary_indices = tf.clip_by_value(boundary_indices, 0, tf.shape(sorted_scores)[0] - 1)
            boundaries = tf.stop_gradient(tf.gather(sorted_scores, boundary_indices))

            lower = tf.sigmoid((scores - boundaries[0]) / tau)
            middle = tf.sigmoid((scores - boundaries[1]) / tau)
            upper = tf.sigmoid((scores - boundaries[2]) / tau)
            category_weights = tf.stack(
                [
                    1.0 - lower,
                    lower - middle,
                    middle - upper,
                    upper,
                ],
                axis=1,
            )
            category_weights = tf.clip_by_value(category_weights, 0.0, 1.0)

            signal_event_mask = tf.cast(labels > 0.5, tf.float32)
            background_event_mask = 1.0 - signal_event_mask
            signal_yields = tf.maximum(tf.reduce_sum(category_weights * raw_weights[:, None] * signal_event_mask[:, None], axis=0), 0.0)
            background_yields = tf.maximum(tf.reduce_sum(category_weights * raw_weights[:, None] * background_event_mask[:, None], axis=0), b_epsilon)
            background_variance = tf.reduce_sum(tf.square(category_weights) * tf.square(raw_weights)[:, None] * background_event_mask[:, None], axis=0)
            background_effective = tf.maximum(background_yields + mcstat_alpha * background_variance, b_epsilon)

            log_argument = tf.maximum(1.0 + signal_yields / background_effective, b_epsilon)
            z2 = 2.0 * ((signal_yields + background_effective) * tf.math.log(log_argument) - signal_yields)
            z2 = tf.where(tf.math.is_finite(z2) & (z2 > 0.0), z2, tf.zeros_like(z2))
            z_total = tf.sqrt(tf.reduce_sum(z2))
            return -z_total, z_total

        return tf.cond(
            (tf.shape(signal_scores)[0] > 0) & (signal_weight_sum > 0.0),
            compute_loss,
            lambda: (tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32)),
        )

    def _resolve_batch_size(self, n_events, hparams):
        min_batch = max(1, int(math.ceil(float(n_events) / 10000.0)))
        max_batch = max(min_batch, int(math.floor(float(n_events) / 100.0)))
        denominator = int(hparams["batch_size_denominator"])
        proposal = max(1, int(round(float(n_events) / float(denominator))))
        batch_size = min(max(proposal, min_batch), max_batch)
        return batch_size, min_batch, max_batch

    def _default_params(self):
        # Defaults follow the 11082226 all-fold JSON best_params.
        active_param_keys = {
            "activation",
            "learning_rate",
            "l2",
            "epochs",
            "batch_size_denominator",
            "early_stopping_min_delta",
            "early_stopping_metric",
            "early_stopping_domain_weight",
            "early_stopping_post_warmup_skip_epochs",
            "reduce_lr_on_plateau",
            "lr_patience",
            "lr_factor",
            "min_learning_rate",
            "domain_lambda",
            "domain_lambda_schedule",
            "domain_lambda_warmup_epochs",
            "trunk_units",
            "trunk_dropout",
            "output_dense_units",
            "output_dense_units_ratio",
            "output_dropout",
            "score_reweight_use_transformer",
            "score_reweight_transformer_warmup_epochs",
            "score_reweight_exponent",
            "score_reweight_reference_sample",
            "score_reweight_reference_set",
            "score_reweight_n_quantiles",
            "score_reweight_lut_size",
            "mass_lambda",
            "mass_decorrelation_loss",
            "physics_feature_weight_mode",
            "mass_rff_num_features",
            "mass_decorrelation_score_sigmas",
            "mass_decorrelation_mass_sigmas",
            "mass_min_events",
            "mass_shape_warmup_epochs",
            "mass_shape_ramp_epochs",
            "mass_reweight_min",
            "mass_reweight_max",
            "mass_reweight_bin_width",
            "asimov_loss_enabled",
            "asimov_lambda",
            "asimov_tau",
            "asimov_quantiles",
            "asimov_b_epsilon",
            "asimov_plateau_threshold",
            "asimov_plateau_patience",
            "asimov_mcstat_alpha",
        }
        defaults = {key: value for key, value in self.dnn_defaults.items() if key in active_param_keys}
        defaults.setdefault("activation", 'swish')
        defaults.setdefault("learning_rate", 0.00043393337239758603)
        defaults.setdefault("l2", 2.8446910598191515e-07)
        defaults.setdefault("epochs", 200)
        defaults.setdefault("batch_size_denominator", 448.70608458588595)
        defaults.setdefault("early_stopping_min_delta", 0.0001)
        defaults.setdefault("early_stopping_metric", 'val_total')
        defaults.setdefault("early_stopping_post_warmup_skip_epochs", 10)
        defaults.setdefault("reduce_lr_on_plateau", True)
        defaults.setdefault("lr_patience", 5)
        defaults.setdefault("lr_factor", 0.5)
        defaults.setdefault("min_learning_rate", 1e-05)
        defaults.setdefault("domain_lambda", 10.0)
        defaults.setdefault("domain_lambda_schedule", 'epoch1_0_epoch2to5_0p5_then_5x1mexpminus_epochminus5')
        defaults.setdefault("trunk_units", [2048, 512, 256, 128])
        defaults.setdefault("trunk_dropout", [0.2, 0.2, 0.15, 0.1])
        defaults.setdefault("output_dense_units", 32)
        output_raw_ratio = float(defaults["output_dense_units"]) / max(float(defaults["trunk_units"][-1]), 1.0)
        defaults.setdefault("output_dense_units_ratio", 0.25)
        defaults.setdefault("output_dropout", 0.2)
        defaults.setdefault("domain_lambda_warmup_epochs", 1)
        defaults.setdefault("score_reweight_use_transformer", True)
        defaults.setdefault("score_reweight_transformer_warmup_epochs", 3)
        defaults.setdefault("score_reweight_exponent", 3.0)
        defaults.setdefault("score_reweight_reference_sample", 'sig')
        defaults.setdefault("score_reweight_reference_set", 'train')
        defaults.setdefault("score_reweight_n_quantiles", 1000)
        defaults.setdefault("score_reweight_lut_size", 16384)
        defaults.setdefault("mass_lambda", 2.0)
        defaults.setdefault("mass_decorrelation_loss", 'rff_hsic')
        defaults.setdefault("physics_feature_weight_mode", "score_mass")
        defaults.setdefault("mass_rff_num_features", 128)
        defaults.setdefault("mass_decorrelation_score_sigmas", [0.03, 0.06, 0.12])
        defaults.setdefault("mass_decorrelation_mass_sigmas", [0.05, 0.1, 0.2])
        defaults.setdefault("mass_min_events", 128)
        defaults.setdefault("mass_shape_warmup_epochs", 5)
        defaults.setdefault("mass_shape_ramp_epochs", 10)
        defaults.setdefault("mass_reweight_min", 110.0)
        defaults.setdefault("mass_reweight_max", 150.0)
        defaults.setdefault("mass_reweight_bin_width", 1.0)
        defaults.setdefault("asimov_loss_enabled", False)
        defaults.setdefault("asimov_lambda", 0.05)
        defaults.setdefault("asimov_tau", 0.02)
        defaults.setdefault("asimov_quantiles", [0.05, 0.4, 0.8])
        defaults.setdefault("asimov_b_epsilon", 1e-06)
        defaults.setdefault("asimov_plateau_threshold", 0.0002)
        defaults.setdefault("asimov_plateau_patience", 5)
        defaults.setdefault("asimov_mcstat_alpha", 0.5)
        return defaults

    def _merge_dnn_params(self, params=None):
        merged = self._default_params()
        if params:
            allowed_keys = set(merged) | {"resolved_batch_size", "_restart_seed", "_restart_seed_offset"}
            ignored_keys = sorted(key for key in params if key not in allowed_keys)
            if ignored_keys:
                print(f"DNN WARNING: ignoring non-DNN hyperparameter keys: {ignored_keys}")
            merged.update({key: value for key, value in params.items() if key in allowed_keys})
        return merged

    def _training_device(self):
        gpus = tf.config.list_logical_devices("GPU")
        return gpus[0].name if gpus else "/CPU:0"

    def _print_dnn_model_summary(self, fold, model):
        print(f"DNN INFO: fold={fold} DNN model structure before training:")
        model.summary(print_fn=lambda line: print(f"DNN MODEL: {line}"))

    def _make_score_reweight_state(self, fold, hparams):
        n_quantiles = max(2, int(hparams.get("score_reweight_n_quantiles", self._default_params()["score_reweight_n_quantiles"])))
        lut_size = max(2, int(hparams.get("score_reweight_lut_size", self._default_params()["score_reweight_lut_size"])))
        references = np.linspace(0.0, 1.0, n_quantiles, dtype=np.float32)
        transform_lut = np.linspace(0.0, 1.0, lut_size, dtype=np.float32)
        return {
            "enabled": tf.Variable(False, trainable=False, dtype=tf.bool, name=f"score_reweight_transformer_enabled_fold{fold}"),
            "references": tf.Variable(references, trainable=False, dtype=tf.float32, name=f"score_reweight_references_fold{fold}"),
            "transform_lut": tf.Variable(transform_lut, trainable=False, dtype=tf.float32, name=f"score_reweight_transform_lut_fold{fold}"),
            "exponent": tf.constant(float(hparams.get("score_reweight_exponent", 3.0)), dtype=tf.float32),
        }

    def _make_train_step(
        self,
        model,
        optimizer,
        domain_lambda,
        hparams,
        fold,
        score_reweight_state,
        mass_rff_state,
        scheduled_mass_lambda_state,
        asimov_active,
        compute_domain_loss=True,
    ):
        step_optimizer = optimizer
        fuse_heads = bool(getattr(self.args, "fuse_dnn_heads", True))
        asimov_enabled = bool(hparams.get("asimov_loss_enabled", True))
        asimov_lambda = tf.constant(float(hparams.get("asimov_lambda", 0.1)), dtype=tf.float32)
        scheduled_mass_lambda = tf.cast(scheduled_mass_lambda_state, tf.float32)

        def score_weighted_bce(y_true, y_pred, weights):
            y_true = tf.cast(tf.reshape(y_true, (-1, 1)), tf.float32)
            y_pred = tf.clip_by_value(tf.cast(tf.reshape(y_pred, (-1, 1)), tf.float32), 1e-7, 1.0 - 1e-7)
            weights = tf.cast(tf.reshape(weights, (-1,)), tf.float32)
            bce = tf.reshape(-(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred)), (-1,))
            loss_sum = tf.reduce_sum(bce * weights)
            weight_sum = tf.reduce_sum(weights)
            return loss_sum / tf.maximum(weight_sum, 1e-12), loss_sum, weight_sum

        def balanced_bce(y_true, y_pred, weights):
            y_true = tf.cast(tf.reshape(y_true, (-1, 1)), tf.float32)
            y_pred = tf.clip_by_value(tf.cast(tf.reshape(y_pred, (-1, 1)), tf.float32), 1e-7, 1.0 - 1e-7)
            weights = tf.cast(tf.reshape(weights, (-1,)), tf.float32)
            bce = tf.reshape(-(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred)), (-1,))
            labels = tf.reshape(y_true, (-1,))
            mask0 = labels < 0.5
            mask1 = labels >= 0.5
            loss0 = tf.reduce_sum(tf.boolean_mask(bce * weights, mask0))
            loss1 = tf.reduce_sum(tf.boolean_mask(bce * weights, mask1))
            weight0 = tf.reduce_sum(tf.boolean_mask(weights, mask0))
            weight1 = tf.reduce_sum(tf.boolean_mask(weights, mask1))
            mean0 = loss0 / tf.maximum(weight0, 1e-12)
            mean1 = loss1 / tf.maximum(weight1, 1e-12)
            active0 = tf.cast(weight0 > 0.0, tf.float32)
            active1 = tf.cast(weight1 > 0.0, tf.float32)
            loss = (mean0 * active0 + mean1 * active1) / tf.maximum(active0 + active1, 1.0)
            return loss, loss, tf.constant(1.0, dtype=tf.float32)

        @tf.function(reduce_retracing=True)
        def train_step(score_x, score_y, score_w, score_w_raw, score_mass, score_mass_norm, mass_x, mass_w_raw, mass_event_weight, mass_mass, domain_x, domain_y, domain_w):
            with tf.GradientTape() as tape:
                if fuse_heads:
                    score_outputs = model(score_x, training=True)
                    score_pred = score_outputs[:, 0:1]
                    mass_outputs = model(mass_x, training=True)
                    mass_score_pred = mass_outputs[:, 0:1]
                    if compute_domain_loss:
                        domain_outputs = model(domain_x, training=True)
                        domain_pred = domain_outputs[:, 1:2]
                        domain_score_pred = tf.stop_gradient(domain_outputs[:, 0:1])
                else:
                    score_pred = model.predict_score(score_x, training=True)
                    mass_score_pred = model.predict_score(mass_x, training=True)
                    if compute_domain_loss:
                        domain_pred = model.predict_domain(domain_x, training=True)
                        with tape.stop_recording():
                            domain_score_pred = model.predict_score(domain_x, training=False)
                score_loss, score_loss_sum, score_weight_sum = score_weighted_bce(score_y, score_pred, score_w)
                if compute_domain_loss:
                    domain_score_weights = self._score_reweight_tensor(domain_score_pred, score_reweight_state)
                    domain_effective_w = tf.cast(tf.reshape(domain_w, (-1,)), tf.float32) * domain_score_weights
                    domain_bce, domain_bce_sum, domain_weight_sum = balanced_bce(domain_y, domain_pred, domain_effective_w)
                    with tape.stop_recording():
                        _, domain_bce_no_score_weight_sum, domain_no_score_weight_sum = balanced_bce(domain_y, domain_pred, domain_w)
                    domain_closeness_loss = self._domain_closeness_loss_tensor(domain_bce)
                else:
                    domain_bce_sum = tf.constant(0.0, dtype=tf.float32)
                    domain_weight_sum = tf.constant(0.0, dtype=tf.float32)
                    domain_bce_no_score_weight_sum = tf.constant(0.0, dtype=tf.float32)
                    domain_no_score_weight_sum = tf.constant(0.0, dtype=tf.float32)
                    domain_closeness_loss = tf.constant(0.0, dtype=tf.float32)
                reg_loss = tf.add_n(model.losses) if model.losses else tf.constant(0.0, dtype=tf.float32)
                (
                    mass_loss,
                    mass_hsic_loss,
                    physics_feature_cov_loss,
                    physics_feature_cov_loss_none,
                    physics_feature_cov_loss_score,
                    physics_feature_cov_loss_mass,
                    physics_feature_cov_loss_score_mass,
                    mass_pred_mean,
                    mass_pred_std,
                    n_mass_events,
                ) = self._mass_losses_tensor(
                    mass_score_pred,
                    mass_mass,
                    mass_w_raw,
                    mass_event_weight,
                    hparams,
                    mass_rff_state,
                    score_reweight_state,
                )
                base_loss = (
                    score_loss
                    + tf.cast(domain_lambda, tf.float32) * domain_closeness_loss
                    + scheduled_mass_lambda * mass_loss
                    + reg_loss
                )
                if asimov_enabled:
                    asimov_loss, asimov_z = tf.cond(
                        asimov_active,
                        lambda: self._asimov_loss_tensor(score_y, score_pred, score_w_raw, hparams),
                        lambda: (tf.constant(0.0, dtype=tf.float32), tf.constant(float("nan"), dtype=tf.float32)),
                    )
                else:
                    asimov_loss = tf.constant(0.0, dtype=tf.float32)
                    asimov_z = tf.constant(float("nan"), dtype=tf.float32)
                total_loss = base_loss + asimov_lambda * asimov_loss
            grads = tape.gradient(total_loss, model.trainable_variables)
            step_optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return (
                score_loss_sum,
                score_weight_sum,
                domain_bce_sum,
                domain_weight_sum,
                domain_bce_no_score_weight_sum,
                domain_no_score_weight_sum,
                reg_loss,
                mass_loss,
                mass_hsic_loss,
                physics_feature_cov_loss,
                physics_feature_cov_loss_none,
                physics_feature_cov_loss_score,
                physics_feature_cov_loss_mass,
                physics_feature_cov_loss_score_mass,
                mass_pred_mean,
                mass_pred_std,
                n_mass_events,
                asimov_loss,
                asimov_z,
                tf.cast(asimov_active, tf.float32),
            )

        return train_step

    def _make_eval_step(
        self,
        model,
        domain_lambda,
        hparams,
        fold,
        score_reweight_state,
        mass_rff_state,
        scheduled_mass_lambda_state,
        asimov_active,
        compute_domain_loss=True,
    ):
        fuse_heads = bool(getattr(self.args, "fuse_dnn_heads", True))
        asimov_enabled = bool(hparams.get("asimov_loss_enabled", True))
        asimov_lambda = tf.constant(float(hparams.get("asimov_lambda", 0.005)), dtype=tf.float32)
        scheduled_mass_lambda = tf.cast(scheduled_mass_lambda_state, tf.float32)

        def score_weighted_bce(y_true, y_pred, weights):
            y_true = tf.cast(tf.reshape(y_true, (-1, 1)), tf.float32)
            y_pred = tf.clip_by_value(tf.cast(tf.reshape(y_pred, (-1, 1)), tf.float32), 1e-7, 1.0 - 1e-7)
            weights = tf.cast(tf.reshape(weights, (-1,)), tf.float32)
            bce = tf.reshape(-(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred)), (-1,))
            loss_sum = tf.reduce_sum(bce * weights)
            weight_sum = tf.reduce_sum(weights)
            return loss_sum / tf.maximum(weight_sum, 1e-12), loss_sum, weight_sum

        def balanced_bce(y_true, y_pred, weights):
            y_true = tf.cast(tf.reshape(y_true, (-1, 1)), tf.float32)
            y_pred = tf.clip_by_value(tf.cast(tf.reshape(y_pred, (-1, 1)), tf.float32), 1e-7, 1.0 - 1e-7)
            weights = tf.cast(tf.reshape(weights, (-1,)), tf.float32)
            bce = tf.reshape(-(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred)), (-1,))
            labels = tf.reshape(y_true, (-1,))
            mask0 = labels < 0.5
            mask1 = labels >= 0.5
            loss0 = tf.reduce_sum(tf.boolean_mask(bce * weights, mask0))
            loss1 = tf.reduce_sum(tf.boolean_mask(bce * weights, mask1))
            weight0 = tf.reduce_sum(tf.boolean_mask(weights, mask0))
            weight1 = tf.reduce_sum(tf.boolean_mask(weights, mask1))
            mean0 = loss0 / tf.maximum(weight0, 1e-12)
            mean1 = loss1 / tf.maximum(weight1, 1e-12)
            active0 = tf.cast(weight0 > 0.0, tf.float32)
            active1 = tf.cast(weight1 > 0.0, tf.float32)
            loss = (mean0 * active0 + mean1 * active1) / tf.maximum(active0 + active1, 1.0)
            return loss, loss, tf.constant(1.0, dtype=tf.float32)

        @tf.function(reduce_retracing=True)
        def eval_step(score_x, score_y, score_w, score_w_raw, score_mass, score_mass_norm, mass_x, mass_w_raw, mass_event_weight, mass_mass, domain_x, domain_y, domain_w):
            if fuse_heads:
                score_outputs = model(score_x, training=False)
                score_pred = score_outputs[:, 0:1]
                mass_outputs = model(mass_x, training=False)
                mass_score_pred = mass_outputs[:, 0:1]
                if compute_domain_loss:
                    domain_outputs = model(domain_x, training=False)
                    domain_pred = domain_outputs[:, 1:2]
                    domain_score_pred = domain_outputs[:, 0:1]
            else:
                score_pred = model.predict_score(score_x, training=False)
                mass_score_pred = model.predict_score(mass_x, training=False)
                if compute_domain_loss:
                    domain_pred = model.predict_domain(domain_x, training=False)
                    domain_score_pred = model.predict_score(domain_x, training=False)
            score_loss, score_loss_sum, score_weight_sum = score_weighted_bce(score_y, score_pred, score_w)
            if compute_domain_loss:
                domain_score_weights = self._score_reweight_tensor(domain_score_pred, score_reweight_state)
                domain_effective_w = tf.cast(tf.reshape(domain_w, (-1,)), tf.float32) * domain_score_weights
                domain_bce, domain_bce_sum, domain_weight_sum = balanced_bce(domain_y, domain_pred, domain_effective_w)
                _, domain_bce_no_score_weight_sum, domain_no_score_weight_sum = balanced_bce(domain_y, domain_pred, domain_w)
                domain_closeness_loss = self._domain_closeness_loss_tensor(domain_bce)
            else:
                domain_bce_sum = tf.constant(0.0, dtype=tf.float32)
                domain_weight_sum = tf.constant(0.0, dtype=tf.float32)
                domain_bce_no_score_weight_sum = tf.constant(0.0, dtype=tf.float32)
                domain_no_score_weight_sum = tf.constant(0.0, dtype=tf.float32)
                domain_closeness_loss = tf.constant(0.0, dtype=tf.float32)
            reg_loss = tf.add_n(model.losses) if model.losses else tf.constant(0.0, dtype=tf.float32)
            (
                mass_loss,
                mass_hsic_loss,
                physics_feature_cov_loss,
                physics_feature_cov_loss_none,
                physics_feature_cov_loss_score,
                physics_feature_cov_loss_mass,
                physics_feature_cov_loss_score_mass,
                mass_pred_mean,
                mass_pred_std,
                n_mass_events,
            ) = self._mass_losses_tensor(
                mass_score_pred,
                mass_mass,
                mass_w_raw,
                mass_event_weight,
                hparams,
                mass_rff_state,
                score_reweight_state,
            )
            base_loss = (
                score_loss
                + tf.cast(domain_lambda, tf.float32) * domain_closeness_loss
                + scheduled_mass_lambda * mass_loss
                + reg_loss
            )
            if asimov_enabled:
                asimov_loss, asimov_z = tf.cond(
                    asimov_active,
                    lambda: self._asimov_loss_tensor(score_y, score_pred, score_w_raw, hparams),
                    lambda: (tf.constant(0.0, dtype=tf.float32), tf.constant(float("nan"), dtype=tf.float32)),
                )
            else:
                asimov_loss = tf.constant(0.0, dtype=tf.float32)
                asimov_z = tf.constant(float("nan"), dtype=tf.float32)
            total_loss = base_loss + asimov_lambda * asimov_loss
            return (
                score_loss_sum,
                score_weight_sum,
                domain_bce_sum,
                domain_weight_sum,
                domain_bce_no_score_weight_sum,
                domain_no_score_weight_sum,
                reg_loss,
                mass_loss,
                mass_hsic_loss,
                physics_feature_cov_loss,
                physics_feature_cov_loss_none,
                physics_feature_cov_loss_score,
                physics_feature_cov_loss_mass,
                physics_feature_cov_loss_score_mass,
                mass_pred_mean,
                mass_pred_std,
                n_mass_events,
                asimov_loss,
                asimov_z,
                tf.cast(asimov_active, tf.float32),
            )

        return eval_step

    def _gpu_epoch_tensors(self, fold, training=True, device=None):
        if training:
            arrays = (
                self.m_x_train[fold],
                self.m_y_train[fold],
                self.m_train_wt[fold],
                self.m_train_wt_raw[fold],
                self.m_train_mass[fold],
                self.m_train_mass_norm[fold],
                self.m_mass_x_train[fold],
                self.m_mass_wt_raw_train[fold],
                self.m_mass_event_weight_train[fold],
                self.m_mass_mass_train[fold],
                self.m_domain_x_train[fold],
                self.m_domain_y_train[fold],
                self.m_domain_wt_train[fold],
            )
        else:
            arrays = (
                self.m_x_val[fold],
                self.m_y_val[fold],
                self.m_val_wt[fold],
                self.m_val_wt_raw[fold],
                self.m_val_mass[fold],
                self.m_val_mass_norm[fold],
                self.m_mass_x_val[fold],
                self.m_mass_wt_raw_val[fold],
                self.m_mass_event_weight_val[fold],
                self.m_mass_mass_val[fold],
                self.m_domain_x_val[fold],
                self.m_domain_y_val[fold],
                self.m_domain_wt_val[fold],
            )

        label = "train" if training else "val"
        self._prefetch_epoch_arrays(arrays, label)
        tensors, _ = self._convert_epoch_arrays(arrays, device, label)
        return tensors

    def _make_gpu_train_epoch(
        self,
        tensors,
        batch_size,
        train_step,
        domain_lambda,
        asimov_lambda,
        scheduled_mass_lambda_state,
        compute_domain_loss=True,
    ):
        score_x, score_y, score_w, score_w_raw, score_mass, score_mass_norm, mass_x, mass_w_raw, mass_event_weight, mass_mass, domain_x, domain_y, domain_w = tensors
        batch_size = tf.constant(int(batch_size), dtype=tf.int32)
        asimov_lambda = tf.constant(float(asimov_lambda), dtype=tf.float32)
        scheduled_mass_lambda = tf.cast(scheduled_mass_lambda_state, tf.float32)

        @tf.function(reduce_retracing=True)
        def train_epoch():
            n_events = tf.shape(score_x)[0]
            n_mass = tf.shape(mass_x)[0]
            n_domain = tf.shape(domain_x)[0]
            indices = tf.random.shuffle(tf.range(n_events))
            mass_indices = tf.random.shuffle(tf.range(n_mass))
            domain_indices = tf.random.shuffle(tf.range(n_domain))
            score_loss_sum = tf.constant(0.0, dtype=tf.float32)
            score_weight_sum = tf.constant(0.0, dtype=tf.float32)
            domain_bce_loss_sum = tf.constant(0.0, dtype=tf.float32)
            domain_weight_sum = tf.constant(0.0, dtype=tf.float32)
            domain_bce_no_score_weight_loss_sum = tf.constant(0.0, dtype=tf.float32)
            domain_no_score_weight_sum = tf.constant(0.0, dtype=tf.float32)
            reg_sum = tf.constant(0.0, dtype=tf.float32)
            mass_loss_sum = tf.constant(0.0, dtype=tf.float32)
            mass_hsic_loss_sum = tf.constant(0.0, dtype=tf.float32)
            physics_feature_cov_loss_sum = tf.constant(0.0, dtype=tf.float32)
            physics_feature_cov_loss_none_sum = tf.constant(0.0, dtype=tf.float32)
            physics_feature_cov_loss_score_sum = tf.constant(0.0, dtype=tf.float32)
            physics_feature_cov_loss_mass_sum = tf.constant(0.0, dtype=tf.float32)
            physics_feature_cov_loss_score_mass_sum = tf.constant(0.0, dtype=tf.float32)
            mass_pred_mean_sum = tf.constant(0.0, dtype=tf.float32)
            mass_pred_std_sum = tf.constant(0.0, dtype=tf.float32)
            n_mass_events_sum = tf.constant(0.0, dtype=tf.float32)
            asimov_loss_sum = tf.constant(0.0, dtype=tf.float32)
            asimov_z_sum = tf.constant(0.0, dtype=tf.float32)
            asimov_active_sum = tf.constant(0.0, dtype=tf.float32)
            n_batches = tf.constant(0.0, dtype=tf.float32)
            start = tf.constant(0, dtype=tf.int32)

            def cond(start, score_loss_sum, score_weight_sum, domain_bce_loss_sum, domain_weight_sum, domain_bce_no_score_weight_loss_sum, domain_no_score_weight_sum, reg_sum, mass_loss_sum, mass_hsic_loss_sum, physics_feature_cov_loss_sum, physics_feature_cov_loss_none_sum, physics_feature_cov_loss_score_sum, physics_feature_cov_loss_mass_sum, physics_feature_cov_loss_score_mass_sum, mass_pred_mean_sum, mass_pred_std_sum, n_mass_events_sum, asimov_loss_sum, asimov_z_sum, asimov_active_sum, n_batches):
                return start < n_events

            def body(start, score_loss_sum, score_weight_sum, domain_bce_loss_sum, domain_weight_sum, domain_bce_no_score_weight_loss_sum, domain_no_score_weight_sum, reg_sum, mass_loss_sum, mass_hsic_loss_sum, physics_feature_cov_loss_sum, physics_feature_cov_loss_none_sum, physics_feature_cov_loss_score_sum, physics_feature_cov_loss_mass_sum, physics_feature_cov_loss_score_mass_sum, mass_pred_mean_sum, mass_pred_std_sum, n_mass_events_sum, asimov_loss_sum, asimov_z_sum, asimov_active_sum, n_batches):
                stop = tf.minimum(start + batch_size, n_events)
                batch_indices = indices[start:stop]
                batch_positions = tf.range(start, stop)
                mass_batch_indices = tf.gather(mass_indices, tf.math.floormod(batch_positions, tf.maximum(n_mass, 1)))
                domain_batch_indices = tf.gather(domain_indices, tf.math.floormod(batch_positions, tf.maximum(n_domain, 1)))
                (
                    batch_score_loss_sum,
                    batch_score_weight_sum,
                    batch_domain_bce_loss_sum,
                    batch_domain_weight_sum,
                    batch_domain_bce_no_score_weight_loss_sum,
                    batch_domain_no_score_weight_sum,
                    batch_reg_loss,
                    batch_mass_loss,
                    batch_mass_hsic_loss,
                    batch_physics_feature_cov_loss,
                    batch_physics_feature_cov_loss_none,
                    batch_physics_feature_cov_loss_score,
                    batch_physics_feature_cov_loss_mass,
                    batch_physics_feature_cov_loss_score_mass,
                    batch_mass_pred_mean,
                    batch_mass_pred_std,
                    batch_n_mass_events,
                    batch_asimov_loss,
                    batch_asimov_z,
                    batch_asimov_active,
                ) = train_step(
                    tf.gather(score_x, batch_indices),
                    tf.gather(score_y, batch_indices),
                    tf.gather(score_w, batch_indices),
                    tf.gather(score_w_raw, batch_indices),
                    tf.gather(score_mass, batch_indices),
                    tf.gather(score_mass_norm, batch_indices),
                    tf.gather(mass_x, mass_batch_indices),
                    tf.gather(mass_w_raw, mass_batch_indices),
                    tf.gather(mass_event_weight, mass_batch_indices),
                    tf.gather(mass_mass, mass_batch_indices),
                    tf.gather(domain_x, domain_batch_indices),
                    tf.gather(domain_y, domain_batch_indices),
                    tf.gather(domain_w, domain_batch_indices),
                )
                return (
                    stop,
                    score_loss_sum + batch_score_loss_sum,
                    score_weight_sum + batch_score_weight_sum,
                    domain_bce_loss_sum + batch_domain_bce_loss_sum,
                    domain_weight_sum + batch_domain_weight_sum,
                    domain_bce_no_score_weight_loss_sum + batch_domain_bce_no_score_weight_loss_sum,
                    domain_no_score_weight_sum + batch_domain_no_score_weight_sum,
                    reg_sum + batch_reg_loss,
                    mass_loss_sum + tf.where(tf.math.is_finite(batch_mass_loss), batch_mass_loss, 0.0),
                    mass_hsic_loss_sum + tf.where(tf.math.is_finite(batch_mass_hsic_loss), batch_mass_hsic_loss, 0.0),
                    physics_feature_cov_loss_sum + tf.where(tf.math.is_finite(batch_physics_feature_cov_loss), batch_physics_feature_cov_loss, 0.0),
                    physics_feature_cov_loss_none_sum + tf.where(tf.math.is_finite(batch_physics_feature_cov_loss_none), batch_physics_feature_cov_loss_none, 0.0),
                    physics_feature_cov_loss_score_sum + tf.where(tf.math.is_finite(batch_physics_feature_cov_loss_score), batch_physics_feature_cov_loss_score, 0.0),
                    physics_feature_cov_loss_mass_sum + tf.where(tf.math.is_finite(batch_physics_feature_cov_loss_mass), batch_physics_feature_cov_loss_mass, 0.0),
                    physics_feature_cov_loss_score_mass_sum + tf.where(tf.math.is_finite(batch_physics_feature_cov_loss_score_mass), batch_physics_feature_cov_loss_score_mass, 0.0),
                    mass_pred_mean_sum + tf.where(tf.math.is_finite(batch_mass_pred_mean), batch_mass_pred_mean, 0.0),
                    mass_pred_std_sum + tf.where(tf.math.is_finite(batch_mass_pred_std), batch_mass_pred_std, 0.0),
                    n_mass_events_sum + batch_n_mass_events,
                    asimov_loss_sum + tf.where(tf.math.is_finite(batch_asimov_loss), batch_asimov_loss, 0.0),
                    asimov_z_sum + tf.where(tf.math.is_finite(batch_asimov_z), batch_asimov_z, 0.0),
                    asimov_active_sum + batch_asimov_active,
                    n_batches + 1.0,
                )

            _, score_loss_sum, score_weight_sum, domain_bce_loss_sum, domain_weight_sum, domain_bce_no_score_weight_loss_sum, domain_no_score_weight_sum, reg_sum, mass_loss_sum, mass_hsic_loss_sum, physics_feature_cov_loss_sum, physics_feature_cov_loss_none_sum, physics_feature_cov_loss_score_sum, physics_feature_cov_loss_mass_sum, physics_feature_cov_loss_score_mass_sum, mass_pred_mean_sum, mass_pred_std_sum, n_mass_events_sum, asimov_loss_sum, asimov_z_sum, asimov_active_sum, n_batches = tf.while_loop(
                cond,
                body,
                loop_vars=(
                    start,
                    score_loss_sum,
                    score_weight_sum,
                    domain_bce_loss_sum,
                    domain_weight_sum,
                    domain_bce_no_score_weight_loss_sum,
                    domain_no_score_weight_sum,
                    reg_sum,
                    mass_loss_sum,
                    mass_hsic_loss_sum,
                    physics_feature_cov_loss_sum,
                    physics_feature_cov_loss_none_sum,
                    physics_feature_cov_loss_score_sum,
                    physics_feature_cov_loss_mass_sum,
                    physics_feature_cov_loss_score_mass_sum,
                    mass_pred_mean_sum,
                    mass_pred_std_sum,
                    n_mass_events_sum,
                    asimov_loss_sum,
                    asimov_z_sum,
                    asimov_active_sum,
                    n_batches,
                ),
            )
            score_loss = score_loss_sum / tf.maximum(score_weight_sum, 1e-12)
            if compute_domain_loss:
                domain_bce = domain_bce_loss_sum / tf.maximum(domain_weight_sum, 1e-12)
                domain_bce_no_score_weight = domain_bce_no_score_weight_loss_sum / tf.maximum(domain_no_score_weight_sum, 1e-12)
                domain_loss = domain_bce - tf.math.log(tf.constant(2.0, dtype=tf.float32))
                domain_closeness_loss = self._domain_closeness_loss_tensor(domain_bce)
            else:
                domain_loss = tf.constant(0.0, dtype=tf.float32)
                domain_bce = tf.constant(0.0, dtype=tf.float32)
                domain_bce_no_score_weight = tf.constant(0.0, dtype=tf.float32)
                domain_closeness_loss = tf.constant(0.0, dtype=tf.float32)
            reg_loss = reg_sum / tf.maximum(n_batches, 1.0)
            mass_loss = mass_loss_sum / tf.maximum(n_batches, 1.0)
            mass_hsic_loss = mass_hsic_loss_sum / tf.maximum(n_batches, 1.0)
            physics_feature_cov_loss = physics_feature_cov_loss_sum / tf.maximum(n_batches, 1.0)
            physics_feature_cov_loss_none = physics_feature_cov_loss_none_sum / tf.maximum(n_batches, 1.0)
            physics_feature_cov_loss_score = physics_feature_cov_loss_score_sum / tf.maximum(n_batches, 1.0)
            physics_feature_cov_loss_mass = physics_feature_cov_loss_mass_sum / tf.maximum(n_batches, 1.0)
            physics_feature_cov_loss_score_mass = physics_feature_cov_loss_score_mass_sum / tf.maximum(n_batches, 1.0)
            mass_pred_mean = mass_pred_mean_sum / tf.maximum(n_batches, 1.0)
            mass_pred_std = mass_pred_std_sum / tf.maximum(n_batches, 1.0)
            asimov_active = asimov_active_sum / tf.maximum(n_batches, 1.0)
            asimov_loss = tf.cond(asimov_active > 0.5, lambda: asimov_loss_sum / tf.maximum(n_batches, 1.0), lambda: tf.constant(float("nan"), dtype=tf.float32))
            asimov_z = tf.cond(asimov_active > 0.5, lambda: asimov_z_sum / tf.maximum(n_batches, 1.0), lambda: tf.constant(float("nan"), dtype=tf.float32))
            total_loss = (
                score_loss
                + tf.cast(domain_lambda, tf.float32) * domain_closeness_loss
                + scheduled_mass_lambda * mass_loss
                + reg_loss
                + asimov_lambda * tf.where(tf.math.is_finite(asimov_loss), asimov_loss, 0.0)
            )
            return (
                total_loss,
                score_loss,
                domain_loss,
                domain_bce,
                domain_bce_no_score_weight,
                domain_closeness_loss,
                mass_loss,
                mass_hsic_loss,
                physics_feature_cov_loss,
                physics_feature_cov_loss_none,
                physics_feature_cov_loss_score,
                physics_feature_cov_loss_mass,
                physics_feature_cov_loss_score_mass,
                mass_pred_mean,
                mass_pred_std,
                n_mass_events_sum,
                asimov_loss,
                asimov_z,
                asimov_active,
            )

        return train_epoch

    def _make_gpu_eval_epoch(
        self,
        tensors,
        batch_size,
        eval_step,
        domain_lambda,
        asimov_lambda,
        scheduled_mass_lambda_state,
        compute_domain_loss=True,
    ):
        score_x, score_y, score_w, score_w_raw, score_mass, score_mass_norm, mass_x, mass_w_raw, mass_event_weight, mass_mass, domain_x, domain_y, domain_w = tensors
        batch_size = tf.constant(int(batch_size), dtype=tf.int32)
        asimov_lambda = tf.constant(float(asimov_lambda), dtype=tf.float32)
        scheduled_mass_lambda = tf.cast(scheduled_mass_lambda_state, tf.float32)

        @tf.function(reduce_retracing=True)
        def eval_epoch():
            n_events = tf.shape(score_x)[0]
            n_mass = tf.shape(mass_x)[0]
            n_domain = tf.shape(domain_x)[0]
            score_loss_sum = tf.constant(0.0, dtype=tf.float32)
            score_weight_sum = tf.constant(0.0, dtype=tf.float32)
            domain_bce_loss_sum = tf.constant(0.0, dtype=tf.float32)
            domain_weight_sum = tf.constant(0.0, dtype=tf.float32)
            domain_bce_no_score_weight_loss_sum = tf.constant(0.0, dtype=tf.float32)
            domain_no_score_weight_sum = tf.constant(0.0, dtype=tf.float32)
            reg_sum = tf.constant(0.0, dtype=tf.float32)
            mass_loss_sum = tf.constant(0.0, dtype=tf.float32)
            mass_hsic_loss_sum = tf.constant(0.0, dtype=tf.float32)
            physics_feature_cov_loss_sum = tf.constant(0.0, dtype=tf.float32)
            physics_feature_cov_loss_none_sum = tf.constant(0.0, dtype=tf.float32)
            physics_feature_cov_loss_score_sum = tf.constant(0.0, dtype=tf.float32)
            physics_feature_cov_loss_mass_sum = tf.constant(0.0, dtype=tf.float32)
            physics_feature_cov_loss_score_mass_sum = tf.constant(0.0, dtype=tf.float32)
            mass_pred_mean_sum = tf.constant(0.0, dtype=tf.float32)
            mass_pred_std_sum = tf.constant(0.0, dtype=tf.float32)
            n_mass_events_sum = tf.constant(0.0, dtype=tf.float32)
            asimov_loss_sum = tf.constant(0.0, dtype=tf.float32)
            asimov_z_sum = tf.constant(0.0, dtype=tf.float32)
            asimov_active_sum = tf.constant(0.0, dtype=tf.float32)
            n_batches = tf.constant(0.0, dtype=tf.float32)
            start = tf.constant(0, dtype=tf.int32)

            def cond(start, score_loss_sum, score_weight_sum, domain_bce_loss_sum, domain_weight_sum, domain_bce_no_score_weight_loss_sum, domain_no_score_weight_sum, reg_sum, mass_loss_sum, mass_hsic_loss_sum, physics_feature_cov_loss_sum, physics_feature_cov_loss_none_sum, physics_feature_cov_loss_score_sum, physics_feature_cov_loss_mass_sum, physics_feature_cov_loss_score_mass_sum, mass_pred_mean_sum, mass_pred_std_sum, n_mass_events_sum, asimov_loss_sum, asimov_z_sum, asimov_active_sum, n_batches):
                return start < n_events

            def body(start, score_loss_sum, score_weight_sum, domain_bce_loss_sum, domain_weight_sum, domain_bce_no_score_weight_loss_sum, domain_no_score_weight_sum, reg_sum, mass_loss_sum, mass_hsic_loss_sum, physics_feature_cov_loss_sum, physics_feature_cov_loss_none_sum, physics_feature_cov_loss_score_sum, physics_feature_cov_loss_mass_sum, physics_feature_cov_loss_score_mass_sum, mass_pred_mean_sum, mass_pred_std_sum, n_mass_events_sum, asimov_loss_sum, asimov_z_sum, asimov_active_sum, n_batches):
                stop = tf.minimum(start + batch_size, n_events)
                batch_positions = tf.range(start, stop)
                mass_positions = tf.math.floormod(batch_positions, tf.maximum(n_mass, 1))
                domain_positions = tf.math.floormod(batch_positions, tf.maximum(n_domain, 1))
                (
                    batch_score_loss_sum,
                    batch_score_weight_sum,
                    batch_domain_bce_loss_sum,
                    batch_domain_weight_sum,
                    batch_domain_bce_no_score_weight_loss_sum,
                    batch_domain_no_score_weight_sum,
                    batch_reg_loss,
                    batch_mass_loss,
                    batch_mass_hsic_loss,
                    batch_physics_feature_cov_loss,
                    batch_physics_feature_cov_loss_none,
                    batch_physics_feature_cov_loss_score,
                    batch_physics_feature_cov_loss_mass,
                    batch_physics_feature_cov_loss_score_mass,
                    batch_mass_pred_mean,
                    batch_mass_pred_std,
                    batch_n_mass_events,
                    batch_asimov_loss,
                    batch_asimov_z,
                    batch_asimov_active,
                ) = eval_step(
                    score_x[start:stop],
                    score_y[start:stop],
                    score_w[start:stop],
                    score_w_raw[start:stop],
                    score_mass[start:stop],
                    score_mass_norm[start:stop],
                    tf.gather(mass_x, mass_positions),
                    tf.gather(mass_w_raw, mass_positions),
                    tf.gather(mass_event_weight, mass_positions),
                    tf.gather(mass_mass, mass_positions),
                    tf.gather(domain_x, domain_positions),
                    tf.gather(domain_y, domain_positions),
                    tf.gather(domain_w, domain_positions),
                )
                return (
                    stop,
                    score_loss_sum + batch_score_loss_sum,
                    score_weight_sum + batch_score_weight_sum,
                    domain_bce_loss_sum + batch_domain_bce_loss_sum,
                    domain_weight_sum + batch_domain_weight_sum,
                    domain_bce_no_score_weight_loss_sum + batch_domain_bce_no_score_weight_loss_sum,
                    domain_no_score_weight_sum + batch_domain_no_score_weight_sum,
                    reg_sum + batch_reg_loss,
                    mass_loss_sum + tf.where(tf.math.is_finite(batch_mass_loss), batch_mass_loss, 0.0),
                    mass_hsic_loss_sum + tf.where(tf.math.is_finite(batch_mass_hsic_loss), batch_mass_hsic_loss, 0.0),
                    physics_feature_cov_loss_sum + tf.where(tf.math.is_finite(batch_physics_feature_cov_loss), batch_physics_feature_cov_loss, 0.0),
                    physics_feature_cov_loss_none_sum + tf.where(tf.math.is_finite(batch_physics_feature_cov_loss_none), batch_physics_feature_cov_loss_none, 0.0),
                    physics_feature_cov_loss_score_sum + tf.where(tf.math.is_finite(batch_physics_feature_cov_loss_score), batch_physics_feature_cov_loss_score, 0.0),
                    physics_feature_cov_loss_mass_sum + tf.where(tf.math.is_finite(batch_physics_feature_cov_loss_mass), batch_physics_feature_cov_loss_mass, 0.0),
                    physics_feature_cov_loss_score_mass_sum + tf.where(tf.math.is_finite(batch_physics_feature_cov_loss_score_mass), batch_physics_feature_cov_loss_score_mass, 0.0),
                    mass_pred_mean_sum + tf.where(tf.math.is_finite(batch_mass_pred_mean), batch_mass_pred_mean, 0.0),
                    mass_pred_std_sum + tf.where(tf.math.is_finite(batch_mass_pred_std), batch_mass_pred_std, 0.0),
                    n_mass_events_sum + batch_n_mass_events,
                    asimov_loss_sum + tf.where(tf.math.is_finite(batch_asimov_loss), batch_asimov_loss, 0.0),
                    asimov_z_sum + tf.where(tf.math.is_finite(batch_asimov_z), batch_asimov_z, 0.0),
                    asimov_active_sum + batch_asimov_active,
                    n_batches + 1.0,
                )

            _, score_loss_sum, score_weight_sum, domain_bce_loss_sum, domain_weight_sum, domain_bce_no_score_weight_loss_sum, domain_no_score_weight_sum, reg_sum, mass_loss_sum, mass_hsic_loss_sum, physics_feature_cov_loss_sum, physics_feature_cov_loss_none_sum, physics_feature_cov_loss_score_sum, physics_feature_cov_loss_mass_sum, physics_feature_cov_loss_score_mass_sum, mass_pred_mean_sum, mass_pred_std_sum, n_mass_events_sum, asimov_loss_sum, asimov_z_sum, asimov_active_sum, n_batches = tf.while_loop(
                cond,
                body,
                loop_vars=(
                    start,
                    score_loss_sum,
                    score_weight_sum,
                    domain_bce_loss_sum,
                    domain_weight_sum,
                    domain_bce_no_score_weight_loss_sum,
                    domain_no_score_weight_sum,
                    reg_sum,
                    mass_loss_sum,
                    mass_hsic_loss_sum,
                    physics_feature_cov_loss_sum,
                    physics_feature_cov_loss_none_sum,
                    physics_feature_cov_loss_score_sum,
                    physics_feature_cov_loss_mass_sum,
                    physics_feature_cov_loss_score_mass_sum,
                    mass_pred_mean_sum,
                    mass_pred_std_sum,
                    n_mass_events_sum,
                    asimov_loss_sum,
                    asimov_z_sum,
                    asimov_active_sum,
                    n_batches,
                ),
            )
            score_loss = score_loss_sum / tf.maximum(score_weight_sum, 1e-12)
            if compute_domain_loss:
                domain_bce = domain_bce_loss_sum / tf.maximum(domain_weight_sum, 1e-12)
                domain_bce_no_score_weight = domain_bce_no_score_weight_loss_sum / tf.maximum(domain_no_score_weight_sum, 1e-12)
                domain_loss = domain_bce - tf.math.log(tf.constant(2.0, dtype=tf.float32))
                domain_closeness_loss = self._domain_closeness_loss_tensor(domain_bce)
            else:
                domain_loss = tf.constant(0.0, dtype=tf.float32)
                domain_bce = tf.constant(0.0, dtype=tf.float32)
                domain_bce_no_score_weight = tf.constant(0.0, dtype=tf.float32)
                domain_closeness_loss = tf.constant(0.0, dtype=tf.float32)
            reg_loss = reg_sum / tf.maximum(n_batches, 1.0)
            mass_loss = mass_loss_sum / tf.maximum(n_batches, 1.0)
            mass_hsic_loss = mass_hsic_loss_sum / tf.maximum(n_batches, 1.0)
            physics_feature_cov_loss = physics_feature_cov_loss_sum / tf.maximum(n_batches, 1.0)
            physics_feature_cov_loss_none = physics_feature_cov_loss_none_sum / tf.maximum(n_batches, 1.0)
            physics_feature_cov_loss_score = physics_feature_cov_loss_score_sum / tf.maximum(n_batches, 1.0)
            physics_feature_cov_loss_mass = physics_feature_cov_loss_mass_sum / tf.maximum(n_batches, 1.0)
            physics_feature_cov_loss_score_mass = physics_feature_cov_loss_score_mass_sum / tf.maximum(n_batches, 1.0)
            mass_pred_mean = mass_pred_mean_sum / tf.maximum(n_batches, 1.0)
            mass_pred_std = mass_pred_std_sum / tf.maximum(n_batches, 1.0)
            asimov_active = asimov_active_sum / tf.maximum(n_batches, 1.0)
            asimov_loss = tf.cond(asimov_active > 0.5, lambda: asimov_loss_sum / tf.maximum(n_batches, 1.0), lambda: tf.constant(float("nan"), dtype=tf.float32))
            asimov_z = tf.cond(asimov_active > 0.5, lambda: asimov_z_sum / tf.maximum(n_batches, 1.0), lambda: tf.constant(float("nan"), dtype=tf.float32))
            total_loss = (
                score_loss
                + tf.cast(domain_lambda, tf.float32) * domain_closeness_loss
                + scheduled_mass_lambda * mass_loss
                + reg_loss
                + asimov_lambda * tf.where(tf.math.is_finite(asimov_loss), asimov_loss, 0.0)
            )
            return (
                total_loss,
                score_loss,
                domain_loss,
                domain_bce,
                domain_bce_no_score_weight,
                domain_closeness_loss,
                mass_loss,
                mass_hsic_loss,
                physics_feature_cov_loss,
                physics_feature_cov_loss_none,
                physics_feature_cov_loss_score,
                physics_feature_cov_loss_mass,
                physics_feature_cov_loss_score_mass,
                mass_pred_mean,
                mass_pred_std,
                n_mass_events_sum,
                asimov_loss,
                asimov_z,
                asimov_active,
            )

        return eval_epoch

    def _new_history(self):
        return {key: [] for key in HISTORY_KEYS}

    def _epoch_result_values(self, result):
        return {key: float(value.numpy()) for key, value in zip(EPOCH_RESULT_KEYS, result)}

    def _nan_epoch_result_values(self):
        return {key: float("nan") for key in EPOCH_RESULT_KEYS}

    def _append_epoch_history(self, history, train, train_eval, val, val_auc, val_shape_metrics, effective_domain_lambda, asimov_lambda, scheduled_mass_lambda, learning_rate, timings, early_stop_value):
        def asimov_contribution(values):
            if values.get("asimov_active", 0.0) > 0.5 and np.isfinite(values.get("asimov_loss", float("nan"))):
                return asimov_lambda * values["asimov_loss"]
            return 0.0

        def mass_contribution(values):
            return scheduled_mass_lambda * values["mass_loss"]

        history["train_total_loss"].append(train["total_loss"])
        history["train_score_loss"].append(train["score_loss"])
        history["train_domain_loss"].append(train["domain_loss"])
        history["train_domain_bce"].append(train["domain_bce"])
        history["train_domain_bce_no_score_weight"].append(train["domain_bce_no_score_weight"])
        history["train_domain_closeness_loss"].append(train["domain_closeness_loss"])
        history["train_mass_loss"].append(train["mass_loss"])
        history["train_mass_hsic_loss"].append(train["mass_hsic_loss"])
        for key in PHYSICS_FEATURE_MONITOR_KEYS:
            history[f"train_{key}"].append(train[key])
        history["train_mass_pred_mean"].append(train["mass_pred_mean"])
        history["train_mass_pred_std"].append(train["mass_pred_std"])
        history["train_n_mass_events"].append(train["n_mass_events"])
        history["train_asimov_loss"].append(train["asimov_loss"])
        history["train_asimov_z"].append(train["asimov_z"])
        history["train_asimov_active"].append(train["asimov_active"])
        history["train_bce_only"].append(train_eval["score_loss"])
        history["train_reg_loss"].append(train["total_loss"] - train["score_loss"] - effective_domain_lambda * train["domain_closeness_loss"] - mass_contribution(train) - asimov_contribution(train))

        history["train_eval_total_loss"].append(train_eval["total_loss"])
        history["train_eval_score_loss"].append(train_eval["score_loss"])
        history["train_eval_domain_loss"].append(train_eval["domain_loss"])
        history["train_eval_domain_bce"].append(train_eval["domain_bce"])
        history["train_eval_domain_bce_no_score_weight"].append(train_eval["domain_bce_no_score_weight"])
        history["train_eval_domain_closeness_loss"].append(train_eval["domain_closeness_loss"])
        history["train_eval_mass_loss"].append(train_eval["mass_loss"])
        history["train_eval_mass_hsic_loss"].append(train_eval["mass_hsic_loss"])
        for key in PHYSICS_FEATURE_MONITOR_KEYS:
            history[f"train_eval_{key}"].append(train_eval[key])
        history["train_eval_mass_pred_mean"].append(train_eval["mass_pred_mean"])
        history["train_eval_mass_pred_std"].append(train_eval["mass_pred_std"])
        history["train_eval_n_mass_events"].append(train_eval["n_mass_events"])
        history["train_eval_asimov_loss"].append(train_eval["asimov_loss"])
        history["train_eval_asimov_z"].append(train_eval["asimov_z"])
        history["train_eval_asimov_active"].append(train_eval["asimov_active"])
        history["train_eval_bce_only"].append(train_eval["score_loss"])
        history["train_eval_reg_loss"].append(train_eval["total_loss"] - train_eval["score_loss"] - effective_domain_lambda * train_eval["domain_closeness_loss"] - mass_contribution(train_eval) - asimov_contribution(train_eval))

        history["val_total_loss"].append(val["total_loss"])
        history["val_score_loss"].append(val["score_loss"])
        history["val_domain_loss"].append(val["domain_loss"])
        history["val_domain_bce"].append(val["domain_bce"])
        history["val_domain_bce_no_score_weight"].append(val["domain_bce_no_score_weight"])
        history["val_domain_closeness_loss"].append(val["domain_closeness_loss"])
        history["val_mass_loss"].append(val["mass_loss"])
        history["val_mass_hsic_loss"].append(val["mass_hsic_loss"])
        for key in PHYSICS_FEATURE_MONITOR_KEYS:
            history[f"val_{key}"].append(val[key])
        history["val_mass_pred_mean"].append(val["mass_pred_mean"])
        history["val_mass_pred_std"].append(val["mass_pred_std"])
        history["val_n_mass_events"].append(val["n_mass_events"])
        for key in SCORE_BIN_MASS_METRIC_KEYS:
            history[f"val_{key}"].append(float(val_shape_metrics.get(key, float("nan"))))
        history["val_asimov_loss"].append(val["asimov_loss"])
        history["val_asimov_z"].append(val["asimov_z"])
        history["val_asimov_active"].append(val["asimov_active"])
        history["val_bce_only"].append(val["score_loss"])
        history["val_reg_loss"].append(val["total_loss"] - val["score_loss"] - effective_domain_lambda * val["domain_closeness_loss"] - mass_contribution(val) - asimov_contribution(val))
        history["val_auc"].append(val_auc)

        history["early_stop_metric"].append(early_stop_value)
        history["learning_rate"].append(learning_rate)
        history["domain_lambda"].append(effective_domain_lambda)
        for key, value in timings.items():
            history[key].append(float(value))

    def _print_epoch_summary(self, fold, epoch, epochs, history, timings):
        print(
            f"DNN EPOCH fold={fold} {epoch + 1}/{epochs} "
            f"s={timings['epoch_seconds']:.2f} "
            f"train={history['train_total_loss'][-1]:.6f} "
            f"val={history['val_total_loss'][-1]:.6f} "
            f"v_score={history['val_score_loss'][-1]:.6f} "
            f"v_close={history['val_domain_closeness_loss'][-1]:.6f} "
            f"v_hsic={history['val_mass_hsic_loss'][-1]:.6f} "
            f"v_phys={history['val_physics_feature_cov_loss'][-1]:.6f} "
            f"v_phys_none={history['val_physics_feature_cov_loss_none'][-1]:.6f} "
            f"v_phys_score={history['val_physics_feature_cov_loss_score'][-1]:.6f} "
            f"v_phys_mass={history['val_physics_feature_cov_loss_mass'][-1]:.6f} "
            f"v_phys_score_mass={history['val_physics_feature_cov_loss_score_mass'][-1]:.6f} "
            f"v_mean_var={history['val_score_bin_mass_mean_variance'][-1]:.6g} "
            f"v_mean_rms={history['val_score_bin_mass_mean_rms'][-1]:.6g} "
            f"v_shape_rms={history['val_score_bin_mass_shape_rms'][-1]:.6g} "
            f"v_shape_max={history['val_score_bin_mass_shape_max_abs'][-1]:.6g} "
            f"v_mass_score_std={history['val_mass_pred_std'][-1]:.6f} "
            f"v_auc={history['val_auc'][-1]:.6f} "
            f"v_stop={history['early_stop_metric'][-1]:.6f} "
            f"asimov={int(history['train_asimov_active'][-1] > 0.5)} "
            f"z={history['train_asimov_z'][-1]:.6f}"
        )

    def _store_fold_params(self, fold, param):
        if not self.params:
            self.params = [param]
        elif len(self.params) == 1:
            self.params[0] = param
        else:
            while len(self.params) <= fold:
                self.params.append(self._merge_dnn_params())
            self.params[fold] = param

    def _training_metrics(self, history, best_epoch, best_metric, domain_lambda_target, domain_lambda_schedule, domain_lambda_warmup_epochs, batch_size, param):
        def best_index(key, minimize=True):
            values = np.asarray(history.get(key, []), dtype=np.float64)
            finite = np.isfinite(values)
            if not np.any(finite):
                return -1, float("nan")
            masked = np.where(finite, values, np.inf if minimize else -np.inf)
            idx = int(np.argmin(masked) if minimize else np.argmax(masked))
            return idx, float(values[idx])

        best_score_epoch, best_score_value = best_index("val_score_loss")
        output_dense_units = int(param.get("output_dense_units", self._default_params()["output_dense_units"]))
        output_dense_units_ratio = float(param.get("output_dense_units_ratio", self._default_params()["output_dense_units_ratio"]))
        output_dropout = float(param.get("output_dropout", self._default_params()["output_dropout"]))
        metrics = {
            "best_epoch": best_epoch + 1,
            "best_metric": float(best_metric),
            "best_by_score_epoch": best_score_epoch + 1 if best_score_epoch >= 0 else None,
            "best_by_score_val_score_loss": best_score_value,
            "best_by_combined_epoch": best_epoch + 1,
            "best_by_combined_metric": float(best_metric),
            "domain_lambda": domain_lambda_target,
            "domain_lambda_target": domain_lambda_target,
            "domain_lambda_schedule": domain_lambda_schedule,
            "domain_lambda_warmup_epochs": domain_lambda_warmup_epochs,
            "domain_lambda_final": float(history["domain_lambda"][-1]) if history["domain_lambda"] else 0.0,
            "domain_bce_indistinguishable_target": float(math.log(2.0)),
            "domain_closeness_epsilon": float(DOMAIN_CLOSENESS_EPSILON),
            "score_reweight_use_transformer": bool(param.get("score_reweight_use_transformer", True)),
            "score_reweight_transformer_warmup_epochs": int(param.get("score_reweight_transformer_warmup_epochs", 3)),
            "score_reweight_exponent": float(param.get("score_reweight_exponent", 3.0)),
            "score_reweight_reference_set": str(param.get("score_reweight_reference_set", "train")),
            "score_reweight_reference_sample": str(param.get("score_reweight_reference_sample", "sig")),
            "score_reweight_n_quantiles": int(param.get("score_reweight_n_quantiles", 1000)),
            "score_reweight_lut_size": int(param.get("score_reweight_lut_size", self._default_params()["score_reweight_lut_size"])),
            "asimov_loss_enabled": bool(param.get("asimov_loss_enabled", True)),
            "asimov_lambda": float(param.get("asimov_lambda", self._default_params()["asimov_lambda"])),
            "asimov_tau": float(param.get("asimov_tau", self._default_params()["asimov_tau"])),
            "asimov_quantiles": [float(value) for value in param.get("asimov_quantiles", self._default_params()["asimov_quantiles"])],
            "asimov_b_epsilon": float(param.get("asimov_b_epsilon", self._default_params()["asimov_b_epsilon"])),
            "asimov_plateau_threshold": float(param.get("asimov_plateau_threshold", self._default_params()["asimov_plateau_threshold"])),
            "asimov_plateau_patience": int(param.get("asimov_plateau_patience", self._default_params()["asimov_plateau_patience"])),
            "asimov_mcstat_alpha": float(param.get("asimov_mcstat_alpha", self._default_params()["asimov_mcstat_alpha"])),
            "mass_lambda": float(param.get("mass_lambda", self._default_params()["mass_lambda"])),
            "mass_decorrelation_loss": str(param.get("mass_decorrelation_loss", self._default_params()["mass_decorrelation_loss"])),
            "physics_feature_weight_mode": str(param.get("physics_feature_weight_mode", self._default_params()["physics_feature_weight_mode"])),
            "mass_min_events": int(param.get("mass_min_events", self._default_params()["mass_min_events"])),
            "mass_shape_warmup_epochs": int(param.get("mass_shape_warmup_epochs", self._default_params()["mass_shape_warmup_epochs"])),
            "mass_shape_ramp_epochs": int(param.get("mass_shape_ramp_epochs", self._default_params()["mass_shape_ramp_epochs"])),
            "mass_rff_num_features": int(param.get("mass_rff_num_features", self._default_params()["mass_rff_num_features"])),
            "mass_decorrelation_score_sigmas": [float(value) for value in param.get("mass_decorrelation_score_sigmas", self._default_params()["mass_decorrelation_score_sigmas"])],
            "mass_decorrelation_mass_sigmas": [float(value) for value in param.get("mass_decorrelation_mass_sigmas", self._default_params()["mass_decorrelation_mass_sigmas"])],
            "mass_reweight_min": float(param.get("mass_reweight_min", self._default_params()["mass_reweight_min"])),
            "mass_reweight_max": float(param.get("mass_reweight_max", self._default_params()["mass_reweight_max"])),
            "mass_reweight_bin_width": float(param.get("mass_reweight_bin_width", self._default_params()["mass_reweight_bin_width"])),
            "resolved_batch_size": batch_size,
            "architecture": f"trunk_{'_'.join(str(value) for value in param['trunk_units'])}_output{output_dense_units}x2",
            "trunk_units": [int(value) for value in param["trunk_units"]],
            "trunk_dropout": [float(value) for value in param["trunk_dropout"]],
            "output_dense_units": output_dense_units,
            "output_dense_units_ratio": output_dense_units_ratio,
            "output_dropout": output_dropout,
            "output_units": [output_dense_units, 2],
            "output_indices": {"score": 0, "domain": 1},
            "mass_decorrelation_definition": "mass_loss is selected by mass_decorrelation_loss; rff_hsic keeps weighted multi-kernel RFF-HSIC, physics_feature_cov uses deterministic mass/score feature normalized covariance",
            "physics_feature_cov_loss_definition": "weighted normalized covariance between fixed physics mass features and classifier score features; monitor modes none=eventWeight, score=eventWeight*score_reweight, mass=mass-reweighted weight, score_mass=mass-reweighted weight*score_reweight",
            "mass_decorrelation_lambda": float(param.get("mass_lambda", self._default_params()["mass_lambda"])),
            "mass_decorrelation_window": [float(param.get("mass_reweight_min", self._default_params()["mass_reweight_min"])), float(param.get("mass_reweight_max", self._default_params()["mass_reweight_max"]))],
            "mass_decorrelation_kernels": {
                "score_sigmas": [float(value) for value in param.get("mass_decorrelation_score_sigmas", self._default_params()["mass_decorrelation_score_sigmas"])],
                "mass_sigmas": [float(value) for value in param.get("mass_decorrelation_mass_sigmas", self._default_params()["mass_decorrelation_mass_sigmas"])],
                "rff_num_features_per_kernel": int(param.get("mass_rff_num_features", self._default_params()["mass_rff_num_features"])),
            },
            "score_bin_mass_shape_epoch_definition": "validation background only; 10 weighted percentile bins by raw classifier score; mass in [110,150) with 1 GeV bins; weights are original eventWeight captured before score, mass, or resolution reweight",
            "mass_variable": MASS_VARIABLE,
            "scaled_by_mass_variables": SCALED_BY_MASS_VARIABLES,
            "val_auc": float("nan"),
        }
        for prefix in ("val", "train", "train_eval"):
            for key in ("domain_loss", "domain_bce", "domain_bce_no_score_weight", "domain_closeness_loss"):
                history_key = f"{prefix}_{key}"
                metrics[history_key] = float(history[history_key][best_epoch])
            for key in ("mass_loss", "mass_hsic_loss", *PHYSICS_FEATURE_MONITOR_KEYS, "mass_pred_mean", "mass_pred_std", "n_mass_events"):
                history_key = f"{prefix}_{key}"
                metrics[history_key] = float(history[history_key][best_epoch])
            for key in ("asimov_loss", "asimov_z", "asimov_active"):
                history_key = f"{prefix}_{key}"
                metrics[history_key] = float(history[history_key][best_epoch])
        for key in SCORE_BIN_MASS_METRIC_KEYS:
            history_key = f"val_{key}"
            metrics[history_key] = float(history[history_key][best_epoch])
        metrics.update(DOMAIN_METRIC_DEFINITIONS)
        metrics.update(param.get("score_reweight_fit_record", {}))
        if 0 <= best_epoch < len(history.get("domain_lambda", [])):
            metrics["best_epoch_domain_lambda"] = float(history["domain_lambda"][best_epoch])
        return metrics

    def _fit_training_score_reweight_transformer(self, fold, model, score_reweight_state, param, epoch_number):
        record = {
            "score_reweight_transformer_enabled": False,
            "score_reweight_transformer_fit_epoch": int(epoch_number),
            "score_reweight_reference_set": str(param.get("score_reweight_reference_set", "train")),
            "score_reweight_reference_sample": str(param.get("score_reweight_reference_sample", "sig")),
            "score_reweight_reference_events": 0,
            "score_reweight_reference_weight_sum": 0.0,
        }
        reference_set = record["score_reweight_reference_set"]
        reference_sample = record["score_reweight_reference_sample"]
        if reference_set != "train":
            print(f"DNN WARNING: training score reweight transformer only supports reference_set='train'; got {reference_set!r}. Falling back to raw score reweight.")
            self.m_training_score_reweight[fold] = record
            return record

        labels = np.asarray(self.m_y_train[fold]).reshape(-1)
        raw_weights = np.asarray(self.m_train_wt_raw[fold], dtype=np.float64).reshape(-1)
        if reference_sample == "sig":
            fit_mask = labels == 1
        elif reference_sample == "bkg":
            fit_mask = labels == 0
        else:
            print(f"DNN WARNING: unsupported score_reweight_reference_sample={reference_sample!r}. Falling back to raw score reweight.")
            self.m_training_score_reweight[fold] = record
            return record

        record["score_reweight_reference_events"] = int(np.count_nonzero(fit_mask))
        if not np.any(fit_mask):
            print(f"DNN WARNING: no {reference_sample} events for training score reweight transformer in fold {fold}. Falling back to raw score reweight.")
            self.m_training_score_reweight[fold] = record
            return record

        scores = self._predict_scores_batched(model, self.m_x_train[fold])
        fit_scores = scores[fit_mask].reshape(-1, 1)
        fit_weights = np.clip(raw_weights[fit_mask], 0.0, None)
        weight_sum = float(np.sum(fit_weights))
        record["score_reweight_reference_weight_sum"] = weight_sum
        if weight_sum <= 0.0:
            print(f"DNN WARNING: non-positive {reference_sample} weight sum for training score reweight transformer in fold {fold}. Falling back to raw score reweight.")
            self.m_training_score_reweight[fold] = record
            return record

        n_quantiles = int(param.get("score_reweight_n_quantiles", self._default_params()["score_reweight_n_quantiles"]))
        transformer = WeightedQuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="uniform",
            subsample=1000000000,
            random_state=0,
        )
        transformer.fit(fit_scores, sample_weight=fit_weights)
        references = np.asarray(transformer.references_, dtype=np.float32).reshape(-1)
        if len(references) != int(score_reweight_state["references"].shape[0]):
            print(
                f"DNN WARNING: fitted score reweight transformer has {len(references)} references, "
                f"expected {int(score_reweight_state['references'].shape[0])}. Falling back to raw score reweight."
            )
            self.m_training_score_reweight[fold] = record
            return record

        score_reweight_state["references"].assign(references)
        lut_size = int(param.get("score_reweight_lut_size", self._default_params()["score_reweight_lut_size"]))
        transform_lut = self._score_reweight_lut_from_references(references, lut_size)
        score_reweight_state["transform_lut"].assign(transform_lut)
        score_reweight_state["enabled"].assign(True)
        record.update(
            {
                "score_reweight_transformer_enabled": True,
                "score_reweight_reference_min": float(np.min(references)),
                "score_reweight_reference_max": float(np.max(references)),
                "score_reweight_lut_size": int(len(transform_lut)),
            }
        )
        self.m_training_score_reweight[fold] = record
        print(
            f"DNN INFO: fold={fold} fitted training score reweight transformer at epoch {epoch_number} "
            f"using {record['score_reweight_reference_events']} {reference_set} {reference_sample} events "
            f"(weight_sum={weight_sum:.6g}); score_reweight=exp({float(param.get('score_reweight_exponent', 3.0)):g}*score_t)"
        )
        return record

    def trainModel(self, fold, param=None):
        base_param = self._merge_dnn_params(param if param else (self.params[0 if len(self.params) == 1 else fold] if self.params else None))
        asimov_started = self._trainModelOnce(fold, base_param)
        if bool(base_param.get("asimov_loss_enabled", True)) and not asimov_started:
            retry_param = dict(base_param)
            retry_param["_restart_seed_offset"] = int(retry_param.get("_restart_seed_offset", 0)) + 10000
            print(
                f"DNN INFO: fold={fold} Asimov loss never activated; "
                f"reinitializing and retraining once with seed_offset={retry_param['_restart_seed_offset']}"
            )
            self._trainModelOnce(fold, retry_param)

    def _trainModelOnce(self, fold, param=None):
        if not param:
            param = self.params[0 if len(self.params) == 1 else fold] if self.params else self._default_params()
        param = self._merge_dnn_params(param)
        param["output_dense_units"] = max(1, int(param.get("output_dense_units", self._default_params()["output_dense_units"])))
        param["output_dropout"] = float(param.get("output_dropout", self._default_params()["output_dropout"]))
        print("DNN INFO: hyperparameters:", param)

        batch_size, min_batch, max_batch = self._resolve_batch_size(len(self.m_x_train[fold]), param)
        param["resolved_batch_size"] = batch_size
        print(f"DNN INFO: batch_size={batch_size} from N_train={len(self.m_x_train[fold])}, allowed range=[{min_batch}, {max_batch}]")

        device = self._training_device()
        print(f"DNN INFO: training device={device}")
        if len(self.m_domain_x_train[fold]) == 0 or len(self.m_domain_x_val[fold]) == 0:
            raise ValueError("Domain training requires non-empty sideband MC and data samples.")
        model_seed = int(param.get("_restart_seed", self.args.seed + fold + int(param.get("_restart_seed_offset", 0))))
        model = DecorrelatingDNN(self.m_x_train[fold].shape[1], param, seed=model_seed)
        self._print_dnn_model_summary(fold, model)
        optimizer = tf.keras.optimizers.Adam(learning_rate=float(param["learning_rate"]))
        domain_lambda_target = float(param.get("domain_lambda", self._default_params()["domain_lambda"]))
        domain_lambda_warmup_epochs = int(param.get("domain_lambda_warmup_epochs", 1))
        domain_lambda_schedule = str(param.get("domain_lambda_schedule", "epoch1_0_epoch2to5_0p5_then_5x1mexpminus_epochminus5"))
        param["domain_lambda"] = domain_lambda_target
        param["domain_lambda_warmup_epochs"] = domain_lambda_warmup_epochs
        param["domain_lambda_schedule"] = domain_lambda_schedule
        domain_lambda = tf.Variable(0.0, trainable=False, dtype=tf.float32, name=f"domain_lambda_fold{fold}")
        compute_domain_loss = domain_lambda_target > 0.0
        score_reweight_use_transformer = bool(param.get("score_reweight_use_transformer", True))
        score_reweight_warmup_epochs = max(0, int(param.get("score_reweight_transformer_warmup_epochs", 3)))
        score_reweight_exponent = float(param.get("score_reweight_exponent", 3.0))
        score_reweight_lut_size = max(2, int(param.get("score_reweight_lut_size", self._default_params()["score_reweight_lut_size"])))
        param["score_reweight_use_transformer"] = score_reweight_use_transformer
        param["score_reweight_transformer_warmup_epochs"] = score_reweight_warmup_epochs
        param["score_reweight_exponent"] = score_reweight_exponent
        param["score_reweight_lut_size"] = score_reweight_lut_size
        asimov_loss_enabled = bool(param.get("asimov_loss_enabled", True))
        asimov_lambda = float(param.get("asimov_lambda", self._default_params()["asimov_lambda"]))
        asimov_tau = float(param.get("asimov_tau", self._default_params()["asimov_tau"]))
        asimov_quantiles = [float(value) for value in param.get("asimov_quantiles", self._default_params()["asimov_quantiles"])]
        asimov_b_epsilon = float(param.get("asimov_b_epsilon", self._default_params()["asimov_b_epsilon"]))
        asimov_plateau_threshold = max(0.0, float(param.get("asimov_plateau_threshold", self._default_params()["asimov_plateau_threshold"])))
        asimov_plateau_patience = max(1, int(param.get("asimov_plateau_patience", self._default_params()["asimov_plateau_patience"])))
        asimov_mcstat_alpha = max(0.0, float(param.get("asimov_mcstat_alpha", self._default_params()["asimov_mcstat_alpha"])))
        if len(asimov_quantiles) != 3:
            raise ValueError(f"asimov_quantiles must contain exactly 3 values for 4 categories, got {asimov_quantiles}")
        if sorted(asimov_quantiles) != asimov_quantiles:
            raise ValueError(f"asimov_quantiles must be sorted ascending, got {asimov_quantiles}")
        param["asimov_loss_enabled"] = asimov_loss_enabled
        param["asimov_lambda"] = asimov_lambda
        param["asimov_tau"] = asimov_tau
        param["asimov_quantiles"] = asimov_quantiles
        param["asimov_b_epsilon"] = asimov_b_epsilon
        param["asimov_plateau_threshold"] = asimov_plateau_threshold
        param["asimov_plateau_patience"] = asimov_plateau_patience
        param["asimov_mcstat_alpha"] = asimov_mcstat_alpha
        asimov_active = tf.Variable(False, trainable=False, dtype=tf.bool, name=f"asimov_active_fold{fold}")
        mass_lambda = float(param.get("mass_lambda", self._default_params()["mass_lambda"]))
        mass_decorrelation_loss = str(param.get("mass_decorrelation_loss", self._default_params()["mass_decorrelation_loss"]))
        if mass_decorrelation_loss not in {"rff_hsic", "physics_feature_cov"}:
            raise ValueError(f"Unsupported mass_decorrelation_loss={mass_decorrelation_loss!r}; use 'rff_hsic' or 'physics_feature_cov'")
        physics_feature_weight_mode = str(param.get("physics_feature_weight_mode", self._default_params()["physics_feature_weight_mode"]))
        if physics_feature_weight_mode not in PHYSICS_FEATURE_WEIGHT_MODES:
            raise ValueError(f"Unsupported physics_feature_weight_mode={physics_feature_weight_mode!r}; use one of {PHYSICS_FEATURE_WEIGHT_MODES}")
        mass_min_events = max(1, int(param.get("mass_min_events", self._default_params()["mass_min_events"])))
        mass_shape_warmup_epochs = max(0, int(param.get("mass_shape_warmup_epochs", self._default_params()["mass_shape_warmup_epochs"])))
        mass_shape_ramp_epochs = max(0, int(param.get("mass_shape_ramp_epochs", self._default_params()["mass_shape_ramp_epochs"])))
        param["mass_lambda"] = mass_lambda
        param["mass_decorrelation_loss"] = mass_decorrelation_loss
        param["physics_feature_weight_mode"] = physics_feature_weight_mode
        param["mass_min_events"] = mass_min_events
        param["mass_shape_warmup_epochs"] = mass_shape_warmup_epochs
        param["mass_shape_ramp_epochs"] = mass_shape_ramp_epochs
        score_reweight_state = self._make_score_reweight_state(fold, param)
        mass_rff_state = self._make_mass_rff_state(fold, param)
        param["mass_rff_num_features"] = int(mass_rff_state["n_features"])
        param["mass_decorrelation_score_sigmas"] = [float(value) for value in mass_rff_state["score_sigmas"]]
        param["mass_decorrelation_mass_sigmas"] = [float(value) for value in mass_rff_state["mass_sigmas"]]
        param["mass_reweight_min"] = float(param.get("mass_reweight_min", self._default_params()["mass_reweight_min"]))
        param["mass_reweight_max"] = float(param.get("mass_reweight_max", self._default_params()["mass_reweight_max"]))
        param["mass_reweight_bin_width"] = float(param.get("mass_reweight_bin_width", self._default_params()["mass_reweight_bin_width"]))
        scheduled_mass_lambda_var = tf.Variable(mass_lambda, trainable=False, dtype=tf.float32, name=f"scheduled_mass_lambda_fold{fold}")
        param["score_reweight_fit_record"] = {
            "score_reweight_transformer_enabled": False,
            "score_reweight_transformer_fit_epoch": None,
            "score_reweight_reference_set": str(param.get("score_reweight_reference_set", "train")),
            "score_reweight_reference_sample": str(param.get("score_reweight_reference_sample", "sig")),
            "score_reweight_reference_events": 0,
            "score_reweight_reference_weight_sum": 0.0,
            "score_reweight_lut_size": score_reweight_lut_size,
        }
        print(
            f"DNN INFO: fold={fold} single 2-output head with domain closeness and true-mass decorrelation loss enabled "
            f"(domain_lambda_target={domain_lambda_target:g}, "
            f"mass_lambda={mass_lambda:g}, "
            f"mass_decorrelation_loss={mass_decorrelation_loss}, "
            f"physics_feature_weight_mode={physics_feature_weight_mode}, "
            f"mass_rff_num_features={param['mass_rff_num_features']}, "
            f"score_sigmas={param['mass_decorrelation_score_sigmas']}, "
            f"mass_sigmas={param['mass_decorrelation_mass_sigmas']}, "
            f"mass_window=[{param['mass_reweight_min']:g},{param['mass_reweight_max']:g}], "
            f"mass_shape_warmup_epochs={mass_shape_warmup_epochs}, "
            f"mass_shape_ramp_epochs={mass_shape_ramp_epochs}, "
            f"domain_lambda_warmup_epochs={domain_lambda_warmup_epochs}, "
            f"domain_lambda_schedule={domain_lambda_schedule}, "
            f"score_reweight=exp({score_reweight_exponent:g}*raw_score) until epoch {score_reweight_warmup_epochs}, "
            f"then exp({score_reweight_exponent:g}*score_t), "
            f"asimov_enabled={asimov_loss_enabled}, "
            f"asimov_lambda={asimov_lambda:g}, "
            f"asimov_tau={asimov_tau:g}, "
            f"asimov_plateau={asimov_plateau_threshold:g}x{asimov_plateau_patience})"
        )
        print(
            "DNN INFO: "
            f"fold={fold} epoch_loss_compute "
            f"domain={compute_domain_loss}; "
            "final_train_eval computes full domain metrics once after best weights are restored"
        )
        try:
            optimizer.build(model.trainable_variables)
        except AttributeError:
            pass
        epochs = int(param.get("epochs", self.num_epochs))
        early_stopping_min_delta = max(0.0, float(param.get("early_stopping_min_delta", 0.0)))
        early_stopping_metric = str(param.get("early_stopping_metric", "val_total"))
        if early_stopping_metric not in {"val_total", "val_bce_plus_domain_loss"}:
            raise ValueError(f"Unsupported early_stopping_metric: {early_stopping_metric}")
        early_stopping_domain_weight = float(param.get("early_stopping_domain_weight", 1.0))
        early_stopping_moving_average_window = 5
        early_stopping_post_warmup_skip_epochs = max(0, int(param.get("early_stopping_post_warmup_skip_epochs", self._default_params()["early_stopping_post_warmup_skip_epochs"])))
        if score_reweight_use_transformer:
            early_stopping_anchor = "score_reweight"
            early_stopping_anchor_epoch = score_reweight_warmup_epochs
        else:
            early_stopping_anchor = "domain_warmup"
            early_stopping_anchor_epoch = domain_lambda_warmup_epochs
        early_stopping_start_epoch = early_stopping_anchor_epoch + early_stopping_post_warmup_skip_epochs + 1
        param["early_stopping_post_warmup_skip_epochs"] = early_stopping_post_warmup_skip_epochs
        train_eval_every = max(0, int(getattr(self.args, "train_eval_every", 0)))
        print(
            "DNN INFO: monitoring "
            f"train_eval_every={train_eval_every} "
            f"early_stopping_anchor={early_stopping_anchor} "
            f"early_stopping_start_epoch={early_stopping_start_epoch} "
            f"fuse_dnn_heads={bool(getattr(self.args, 'fuse_dnn_heads', True))}"
        )
        reduce_lr_on_plateau = bool(param.get("reduce_lr_on_plateau", False))
        lr_patience = max(1, int(param.get("lr_patience", max(1, self.early_stopping_rounds // 3))))
        lr_factor = min(1.0, max(0.0, float(param.get("lr_factor", 0.5))))
        min_learning_rate = max(0.0, float(param.get("min_learning_rate", 1e-6)))

        def current_learning_rate():
            return float(tf.keras.backend.get_value(optimizer.learning_rate))

        def set_learning_rate(value):
            optimizer.learning_rate.assign(float(value))

        train_step = self._make_train_step(
            model,
            optimizer,
            domain_lambda,
            param,
            fold,
            score_reweight_state,
            mass_rff_state,
            scheduled_mass_lambda_var,
            asimov_active,
            compute_domain_loss=compute_domain_loss,
        )
        eval_step = self._make_eval_step(
            model,
            domain_lambda,
            param,
            fold,
            score_reweight_state,
            mass_rff_state,
            scheduled_mass_lambda_var,
            asimov_active,
            compute_domain_loss=compute_domain_loss,
        )
        final_eval_step = self._make_eval_step(
            model,
            domain_lambda,
            param,
            fold,
            score_reweight_state,
            mass_rff_state,
            scheduled_mass_lambda_var,
            asimov_active,
            compute_domain_loss=True,
        )
        tensor_start = time.perf_counter()
        train_tensors = self._gpu_epoch_tensors(fold, training=True, device=device)
        val_tensors = self._gpu_epoch_tensors(fold, training=False, device=device)
        train_epoch = self._make_gpu_train_epoch(
            train_tensors,
            batch_size,
            train_step,
            domain_lambda,
            asimov_lambda,
            scheduled_mass_lambda_var,
            compute_domain_loss=compute_domain_loss,
        )
        train_eval_epoch = self._make_gpu_eval_epoch(
            train_tensors,
            batch_size,
            eval_step,
            domain_lambda,
            asimov_lambda,
            scheduled_mass_lambda_var,
            compute_domain_loss=compute_domain_loss,
        )
        eval_epoch = self._make_gpu_eval_epoch(
            val_tensors,
            batch_size,
            eval_step,
            domain_lambda,
            asimov_lambda,
            scheduled_mass_lambda_var,
            compute_domain_loss=compute_domain_loss,
        )
        final_train_eval_epoch = self._make_gpu_eval_epoch(
            train_tensors,
            batch_size,
            final_eval_step,
            domain_lambda,
            asimov_lambda,
            scheduled_mass_lambda_var,
            compute_domain_loss=True,
        )
        print(f"DNN PROFILE: fold={fold} gpu_tensor_setup_s={time.perf_counter() - tensor_start:.2f}")

        best_weights = None
        best_epoch = -1
        best_metric = np.inf
        patience_counter = 0
        lr_patience_counter = 0
        early_stopping_started = False
        previous_val_score_loss = None
        asimov_plateau_count = 0
        asimov_lr_reduced = False
        history = self._new_history()

        for epoch in range(epochs):
            effective_domain_lambda = self._scheduled_domain_lambda(
                epoch + 1,
                domain_lambda_target,
                domain_lambda_warmup_epochs,
                domain_lambda_schedule,
            )
            domain_lambda.assign(effective_domain_lambda)
            effective_mass_lambda = self._scheduled_mass_shape_lambda(
                epoch + 1,
                mass_lambda,
                mass_shape_warmup_epochs,
                mass_shape_ramp_epochs,
            )
            scheduled_mass_lambda_var.assign(effective_mass_lambda)
            epoch_start = time.perf_counter()
            train_start = epoch_start
            train_fetch_seconds = 0.0
            step_start = time.perf_counter()
            train_values = self._epoch_result_values(train_epoch())
            train_step_seconds = time.perf_counter() - step_start
            train_seconds = time.perf_counter() - train_start

            train_eval_start = time.perf_counter()
            train_eval_fetch_seconds = 0.0
            should_run_train_eval = train_eval_every > 0 and ((epoch + 1) % train_eval_every == 0)
            if should_run_train_eval:
                step_start = time.perf_counter()
                train_eval_values = self._epoch_result_values(train_eval_epoch())
                train_eval_step_seconds = time.perf_counter() - step_start
            else:
                train_eval_values = self._nan_epoch_result_values()
                train_eval_step_seconds = 0.0
            train_eval_seconds = time.perf_counter() - train_eval_start

            val_start = time.perf_counter()
            val_fetch_seconds = 0.0
            step_start = time.perf_counter()
            val_values = self._epoch_result_values(eval_epoch())
            val_step_seconds = time.perf_counter() - step_start
            val_auc_scores = self._predict_scores_batched(model, self.m_x_val[fold])
            _, val_auc = self.getAUC_from_arrays(self.m_y_val[fold], val_auc_scores, self.m_val_wt_raw[fold])
            val_shape_metrics = self._validation_score_bin_mass_shape_metrics(fold, val_auc_scores)
            val_seconds = time.perf_counter() - val_start
            epoch_seconds = time.perf_counter() - epoch_start

            if early_stopping_metric == "val_total":
                recent_val_losses = history["val_total_loss"][-(early_stopping_moving_average_window - 1):] + [val_values["total_loss"]]
                early_stop_value = float(np.mean(recent_val_losses))
            else:
                early_stop_value = float(
                    val_values["score_loss"]
                    + early_stopping_domain_weight * val_values["domain_closeness_loss"]
                )
            timings = {
                "epoch_seconds": epoch_seconds,
                "train_seconds": train_seconds,
                "train_eval_seconds": train_eval_seconds,
                "val_seconds": val_seconds,
                "train_fetch_seconds": train_fetch_seconds,
                "train_step_seconds": train_step_seconds,
                "train_eval_fetch_seconds": train_eval_fetch_seconds,
                "train_eval_step_seconds": train_eval_step_seconds,
                "val_fetch_seconds": val_fetch_seconds,
                "val_step_seconds": val_step_seconds,
            }
            self._append_epoch_history(
                history,
                train_values,
                train_eval_values,
                val_values,
                float(val_auc),
                val_shape_metrics,
                effective_domain_lambda,
                asimov_lambda,
                effective_mass_lambda,
                current_learning_rate(),
                timings,
                early_stop_value,
            )
            self._print_epoch_summary(fold, epoch, epochs, history, timings)
            if previous_val_score_loss is not None and asimov_loss_enabled and not bool(asimov_active.numpy()):
                score_delta = abs(float(val_values["score_loss"]) - previous_val_score_loss)
                if score_delta < asimov_plateau_threshold:
                    asimov_plateau_count += 1
                else:
                    asimov_plateau_count = 0
                if asimov_plateau_count >= asimov_plateau_patience:
                    asimov_active.assign(True)
                    if (not asimov_lr_reduced) and lr_factor < 1.0:
                        old_lr = current_learning_rate()
                        new_lr = max(old_lr * lr_factor, min_learning_rate)
                        if new_lr < old_lr - 1e-12:
                            set_learning_rate(new_lr)
                            asimov_lr_reduced = True
                            print(
                                f"DNN INFO: reduce learning rate for Asimov at epoch {epoch + 1}: "
                                f"{old_lr:.6g} -> {new_lr:.6g}"
                            )
                    print(
                        f"DNN INFO: enable Asimov loss at epoch {epoch + 1} "
                        f"(lambda={asimov_lambda:g}, tau={asimov_tau:g}, "
                        f"val_score_delta={score_delta:.6g})"
                    )
            previous_val_score_loss = float(val_values["score_loss"])
            if self.args.profile_batches > 0 and epoch == 0:
                print(
                    f"DNN PROFILE: fold={fold} first_epoch "
                    f"train_fetch_s={train_fetch_seconds:.3f} train_step_s={train_step_seconds:.3f} "
                    f"val_fetch_s={val_fetch_seconds:.3f} val_step_s={val_step_seconds:.3f} "
                    "gpu_tensors=True"
                )
            if score_reweight_use_transformer and not bool(score_reweight_state["enabled"].numpy()) and (epoch + 1) >= score_reweight_warmup_epochs:
                param["score_reweight_fit_record"] = self._fit_training_score_reweight_transformer(
                    fold,
                    model,
                    score_reweight_state,
                    param,
                    epoch + 1,
                )

            if (epoch + 1) < early_stopping_start_epoch:
                continue

            early_stopping_started = True
            if early_stop_value < best_metric - early_stopping_min_delta:
                best_metric = early_stop_value
                best_epoch = epoch
                best_weights = model.get_weights()
                patience_counter = 0
                lr_patience_counter = 0
            else:
                patience_counter += 1
                lr_patience_counter += 1
                if reduce_lr_on_plateau and lr_factor < 1.0 and lr_patience_counter >= lr_patience:
                    old_lr = current_learning_rate()
                    new_lr = max(old_lr * lr_factor, min_learning_rate)
                    if new_lr < old_lr - 1e-12:
                        set_learning_rate(new_lr)
                        print(
                            f"DNN INFO: reduce learning rate at epoch {epoch + 1}: "
                            f"{old_lr:.6g} -> {new_lr:.6g}"
                        )
                    lr_patience_counter = 0
                if patience_counter >= self.early_stopping_rounds:
                    print(f"DNN INFO: early stopping at epoch {epoch + 1}")
                    break

        if best_weights is None and history["early_stop_metric"]:
            best_epoch = len(history["early_stop_metric"]) - 1
            best_metric = float(history["early_stop_metric"][-1])
            best_weights = model.get_weights()
        if best_weights is not None:
            model.set_weights(best_weights)

        final_train_eval_start = time.perf_counter()
        final_train_eval_values = self._epoch_result_values(final_train_eval_epoch())
        final_train_eval_seconds = time.perf_counter() - final_train_eval_start
        print(f"DNN PROFILE: fold={fold} final_train_eval_s={final_train_eval_seconds:.2f}")

        self.m_model[fold] = model
        self.m_history[fold] = history
        self.m_history[fold]["score_reweight_fit_record"] = param.get("score_reweight_fit_record", {})
        self._store_fold_params(fold, param)
        self.m_metrics[fold] = self._training_metrics(
            history,
            best_epoch,
            best_metric,
            domain_lambda_target,
            domain_lambda_schedule,
            domain_lambda_warmup_epochs,
            batch_size,
            param,
        )
        self.m_metrics[fold].update(
            {
                "early_stopping_metric": early_stopping_metric,
                "early_stopping_min_delta": early_stopping_min_delta,
                "early_stopping_post_warmup_skip_epochs": early_stopping_post_warmup_skip_epochs,
                "early_stopping_anchor": early_stopping_anchor,
                "early_stopping_anchor_epoch": early_stopping_anchor_epoch,
                "early_stopping_start_epoch": early_stopping_start_epoch,
                "early_stopping_started": early_stopping_started,
                "early_stopping_moving_average_window": early_stopping_moving_average_window,
                "early_stopping_domain_weight": early_stopping_domain_weight,
                "early_stopping_metric_definition": (
                    "moving average of last up to 5 val_total_loss values, including warmup epochs"
                    if early_stopping_metric == "val_total"
                    else "val_score_loss + early_stopping_domain_weight * val_domain_closeness_loss"
                ),
                "reduce_lr_on_plateau": reduce_lr_on_plateau,
                "lr_patience": lr_patience,
                "lr_factor": lr_factor,
                "min_learning_rate": min_learning_rate,
                "asimov_loss_enabled": asimov_loss_enabled,
                "asimov_lambda": asimov_lambda,
                "asimov_tau": asimov_tau,
                "asimov_quantiles": asimov_quantiles,
                "asimov_b_epsilon": asimov_b_epsilon,
                "asimov_plateau_threshold": asimov_plateau_threshold,
                "asimov_plateau_patience": asimov_plateau_patience,
                "asimov_mcstat_alpha": asimov_mcstat_alpha,
                "asimov_active_final": bool(asimov_active.numpy()),
                "mass_lambda": mass_lambda,
                "mass_decorrelation_loss": mass_decorrelation_loss,
                "physics_feature_weight_mode": physics_feature_weight_mode,
                "physics_feature_weight_modes": list(PHYSICS_FEATURE_WEIGHT_MODES),
                "mass_min_events": mass_min_events,
                "mass_shape_warmup_epochs": mass_shape_warmup_epochs,
                "mass_shape_ramp_epochs": mass_shape_ramp_epochs,
                "mass_rff_num_features": int(mass_rff_state["n_features"]),
                "mass_decorrelation_score_sigmas": [float(value) for value in param["mass_decorrelation_score_sigmas"]],
                "mass_decorrelation_mass_sigmas": [float(value) for value in param["mass_decorrelation_mass_sigmas"]],
                "mass_decorrelation_definition": "mass_loss is selected by mass_decorrelation_loss; rff_hsic keeps weighted multi-kernel RFF-HSIC, physics_feature_cov uses deterministic mass/score feature normalized covariance",
                "physics_feature_cov_loss_definition": "weighted normalized covariance between fixed physics mass features and classifier score features; monitor modes none=eventWeight, score=eventWeight*score_reweight, mass=mass-reweighted weight, score_mass=mass-reweighted weight*score_reweight",
                "mass_decorrelation_lambda": mass_lambda,
                "mass_decorrelation_window": [float(param["mass_reweight_min"]), float(param["mass_reweight_max"])],
                "mass_decorrelation_kernels": {
                    "score_sigmas": [float(value) for value in param["mass_decorrelation_score_sigmas"]],
                    "mass_sigmas": [float(value) for value in param["mass_decorrelation_mass_sigmas"]],
                    "rff_num_features_per_kernel": int(mass_rff_state["n_features"]),
                },
                "mass_reweight_min": float(param.get("mass_reweight_min", self._default_params()["mass_reweight_min"])),
                "mass_reweight_max": float(param.get("mass_reweight_max", self._default_params()["mass_reweight_max"])),
                "mass_reweight_bin_width": float(param.get("mass_reweight_bin_width", self._default_params()["mass_reweight_bin_width"])),
                "train_eval_every": train_eval_every,
                "fuse_dnn_heads": bool(getattr(self.args, "fuse_dnn_heads", True)),
                "final_train_eval_seconds": final_train_eval_seconds,
                "final_train_eval_total_loss": final_train_eval_values["total_loss"],
                "final_train_eval_score_loss": final_train_eval_values["score_loss"],
                "final_train_eval_domain_loss": final_train_eval_values["domain_loss"],
                "final_train_eval_domain_bce": final_train_eval_values["domain_bce"],
                "final_train_eval_domain_bce_no_score_weight": final_train_eval_values["domain_bce_no_score_weight"],
                "final_train_eval_domain_closeness_loss": final_train_eval_values["domain_closeness_loss"],
                "final_train_eval_mass_loss": final_train_eval_values["mass_loss"],
                "final_train_eval_mass_hsic_loss": final_train_eval_values["mass_hsic_loss"],
                "final_train_eval_physics_feature_cov_loss": final_train_eval_values["physics_feature_cov_loss"],
                "final_train_eval_physics_feature_cov_loss_none": final_train_eval_values["physics_feature_cov_loss_none"],
                "final_train_eval_physics_feature_cov_loss_score": final_train_eval_values["physics_feature_cov_loss_score"],
                "final_train_eval_physics_feature_cov_loss_mass": final_train_eval_values["physics_feature_cov_loss_mass"],
                "final_train_eval_physics_feature_cov_loss_score_mass": final_train_eval_values["physics_feature_cov_loss_score_mass"],
                "final_train_eval_mass_pred_mean": final_train_eval_values["mass_pred_mean"],
                "final_train_eval_mass_pred_std": final_train_eval_values["mass_pred_std"],
                "final_train_eval_n_mass_events": final_train_eval_values["n_mass_events"],
                "final_train_eval_asimov_loss": final_train_eval_values["asimov_loss"],
                "final_train_eval_asimov_z": final_train_eval_values["asimov_z"],
                "final_train_eval_asimov_active": final_train_eval_values["asimov_active"],
            }
        )
        return bool(asimov_active.numpy()) if asimov_loss_enabled else True

    def testModel(self, fold):
        print("DNN INFO: Computing scores for different sample sets...")
        test_start = time.perf_counter()
        model = self.m_model[fold]
        predict_start = time.perf_counter()
        self.m_score_train[fold] = self._predict_scores_batched(model, self.m_x_train[fold])
        self.m_score_val[fold] = self._predict_scores_batched(model, self.m_x_val[fold])
        self.m_score_test[fold] = self._predict_scores_batched(model, self.m_x_test[fold])
        self.m_score_test_sig[fold] = self._predict_scores_batched(model, self.m_x_test_sig[fold])
        self.m_score_test_bkg[fold] = self._predict_scores_batched(model, self.m_x_test_bkg[fold])
        predict_seconds = time.perf_counter() - predict_start

        metric_start = time.perf_counter()
        train_metrics = self._evaluate_split_metrics(
            self.m_y_train[fold],
            self.m_score_train[fold],
            self.m_train_wt_raw[fold],
        )
        val_metrics = self._evaluate_split_metrics(
            self.m_y_val[fold],
            self.m_score_val[fold],
            self.m_val_wt_raw[fold],
        )
        test_metrics = self._evaluate_split_metrics(
            self.m_y_test[fold],
            self.m_score_test[fold],
            self.m_test_wt_raw[fold],
        )
        train_losses = self._evaluate_split_losses(
            model,
            self.m_y_train[fold],
            self.m_score_train[fold],
            self.m_train_wt[fold],
        )
        val_losses = self._evaluate_split_losses(
            model,
            self.m_y_val[fold],
            self.m_score_val[fold],
            self.m_val_wt[fold],
        )
        test_losses = self._evaluate_split_losses(
            model,
            self.m_y_test[fold],
            self.m_score_test[fold],
            self.m_test_wt[fold],
        )
        metric_seconds = time.perf_counter() - metric_start
        test_total_seconds = time.perf_counter() - test_start
        self.m_metrics[fold].update(
            {
                "train_auc": train_metrics["auc"],
                "eval_auc": val_metrics["auc"],
                "val_auc": val_metrics["auc"],
                "test_auc": test_metrics["auc"],
                "train_eval_loss": train_losses["loss"],
                "train_bce_only": train_losses["bce_only"],
                "train_regularization_loss": train_losses["regularization_loss"],
                "val_loss": val_losses["loss"],
                "eval_loss": val_losses["loss"],
                "val_bce_only": val_losses["bce_only"],
                "eval_bce_only": val_losses["bce_only"],
                "val_regularization_loss": val_losses["regularization_loss"],
                "eval_regularization_loss": val_losses["regularization_loss"],
                "test_loss": test_losses["loss"],
                "test_bce_only": test_losses["bce_only"],
                "test_regularization_loss": test_losses["regularization_loss"],
                "score_predict_s": predict_seconds,
                "metrics_auc_s": metric_seconds,
                "test_total_s": test_total_seconds,
            }
        )
        print(
            f"DNN METRIC: fold={fold} "
            f"train_eval_loss={train_losses['loss']:.6f} val_loss={val_losses['loss']:.6f} "
            f"train_bce_only={train_losses['bce_only']:.6f} val_bce_only={val_losses['bce_only']:.6f} "
            f"train_auc={train_metrics['auc']:.6f} val_auc={val_metrics['auc']:.6f} "
            f"test_auc={test_metrics['auc']:.6f}"
        )
        print(f"DNN PROFILE: fold={fold} score_predict_s={predict_seconds:.1f} metrics_auc_s={metric_seconds:.1f} test_total_s={test_total_seconds:.1f}")

    def _predict_scores_batched(self, model, features, batch_size=262144):
        features = np.asarray(features, dtype=np.float32)
        if len(features) == 0:
            return np.empty(0, dtype=np.float32)
        score_batches = []
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            pred = model(features[start:stop], training=False)
            score_batches.append(tf.reshape(pred[:, 0], (-1,)).numpy().astype(np.float32, copy=False))
        return np.concatenate(score_batches)

    def _evaluate_split_metrics(self, labels, scores, raw_weights):
        _, auc_value = self.getAUC_from_arrays(labels, scores, raw_weights)
        return {"auc": float(auc_value)}

    def _evaluate_split_losses(self, model, labels, scores, weights):
        labels = np.asarray(labels, dtype=np.float32).reshape(-1)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        scores = np.clip(scores, 1e-7, 1.0 - 1e-7)
        bce = -(labels * np.log(scores) + (1.0 - labels) * np.log(1.0 - scores))
        bce_loss = float(np.sum(bce * weights) / max(float(np.sum(weights)), 1.0))
        reg_loss = float(tf.add_n(model.losses).numpy()) if model.losses else 0.0
        return {
            "loss": bce_loss + reg_loss,
            "bce_only": bce_loss,
            "regularization_loss": reg_loss,
        }

    def plotROC(self, fold=0, save=True, show=False):
        plt.figure(figsize=(8, 6), dpi=200)
        plt.scatter(RUN2_POINTS[:, 0], 1.0 - RUN2_POINTS[:, 1], color="red", marker="o", s=22, label="Run2 REF", zorder=4)
        for name, labels, scores, weights, color, linestyle in (
            ("Train", self.m_y_train[fold], self.m_score_train[fold], self.m_train_wt_raw[fold], "black", "dotted"),
            ("Val", self.m_y_val[fold], self.m_score_val[fold], self.m_val_wt_raw[fold], "blue", "dashdot"),
            ("Test", self.m_y_test[fold], self.m_score_test[fold], self.m_test_wt_raw[fold], "red", "dashed"),
        ):
            fpr, tpr, _ = roc_curve(labels, scores, sample_weight=weights)
            order = np.argsort(tpr)
            roc_auc = 1.0 - auc(tpr[order], fpr[order])
            plt.plot(tpr, 1.0 - fpr, label=f"{name} set, area = {roc_auc:.6f}", color=color, linestyle=linestyle)
        plt.xlabel("Signal efficiency")
        plt.ylabel("Background rejection")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save:
            roc_dir = os.path.join(self._plotFolder, "roc_curve")
            os.makedirs(roc_dir, exist_ok=True)
            plt.savefig(os.path.join(roc_dir, f"DNN_{self._region}_{fold}.pdf"))
        if show:
            plt.show()
        plt.close()

    def plotScoreDecileMassOverlay(self, fold=0, mass_min=110.0, mass_max=150.0, mass_bin_width=1.0):
        print(f"DNN INFO: plotting train/val/test background mass-shape overlays in score slices for fold {fold}")
        out_dir = os.path.join(self._plotFolder, "mass_score_deciles")
        os.makedirs(out_dir, exist_ok=True)

        mass_bins = np.arange(mass_min, mass_max + mass_bin_width, mass_bin_width, dtype=np.float64)
        centers = 0.5 * (mass_bins[:-1] + mass_bins[1:])
        split_payload = [
            ("train", self.m_score_train[fold][self.m_y_train[fold] == 0], self.m_train_mass[fold][self.m_y_train[fold] == 0], self.m_train_event_weight[fold][self.m_y_train[fold] == 0]),
            ("val", self.m_score_val[fold][self.m_y_val[fold] == 0], self.m_val_mass[fold][self.m_y_val[fold] == 0], self.m_val_event_weight[fold][self.m_y_val[fold] == 0]),
            ("test", self.m_score_test[fold][self.m_y_test[fold] == 0], self.m_test_mass[fold][self.m_y_test[fold] == 0], self.m_test_event_weight[fold][self.m_y_test[fold] == 0]),
        ]
        plot_source_definition = "score-training background arrays with original eventWeight before score, mass, or resolution reweight"

        rows = []
        fig, axes = plt.subplots(3, 1, figsize=(11, 13), dpi=160, sharex=True)
        colors = plt.cm.viridis(np.linspace(0.05, 0.95, 10))
        tail_defs = [
            ("top_10pct", 0.10, "#d62728", 2.2, "--"),
            ("top_1pct", 0.01, "#000000", 2.4, "-."),
            ("top_0p1pct", 0.001, "#ff7f0e", 2.6, ":"),
        ]
        tail_metric_summary = {}

        for ax, (split_name, scores, masses, weights) in zip(axes, split_payload):
            scores = np.asarray(scores, dtype=np.float64).reshape(-1)
            masses = np.asarray(masses, dtype=np.float64).reshape(-1)
            weights = np.asarray(weights, dtype=np.float64).reshape(-1)
            bkg_mask = np.isfinite(scores) & np.isfinite(masses) & np.isfinite(weights)
            bkg_mask &= (masses >= mass_min) & (masses < mass_max)

            split_scores = scores[bkg_mask]
            split_masses = masses[bkg_mask]
            split_weights = weights[bkg_mask]
            if len(split_scores) == 0:
                ax.text(0.5, 0.5, f"{split_name}: no background events", transform=ax.transAxes, ha="center", va="center")
                ax.set_ylabel("slice shape / inclusive")
                continue

            order = np.argsort(split_scores, kind="mergesort")
            ordered_weights = np.clip(split_weights[order], 0.0, None)
            total_ordered_weight = float(np.sum(ordered_weights))
            if total_ordered_weight > 0.0:
                cumulative_weight = np.cumsum(ordered_weights)
                split_points = np.searchsorted(
                    cumulative_weight,
                    total_ordered_weight * np.linspace(0.1, 0.9, 9),
                    side="left",
                )
                decile_bounds = np.concatenate(([0], split_points, [len(order)]))
                decile_indices = [order[decile_bounds[idx]:decile_bounds[idx + 1]].astype(np.int64, copy=False) for idx in range(10)]
            else:
                decile_indices = [indices.astype(np.int64, copy=False) for indices in np.array_split(order, 10)]

            inclusive_hist, _ = np.histogram(split_masses, bins=mass_bins, weights=split_weights)
            inclusive_hist = inclusive_hist.astype(np.float64)
            inclusive_total = float(np.sum(inclusive_hist))
            inclusive_shape = np.divide(
                inclusive_hist,
                inclusive_total,
                out=np.zeros_like(inclusive_hist, dtype=np.float64),
                where=inclusive_total > 0.0,
            )

            def add_slice(indices, slice_label, score_eff_low, score_eff_high, color, linewidth, linestyle, legend_label, score_threshold_low=None, score_threshold_high=None):
                slice_scores = split_scores[indices]
                slice_masses = split_masses[indices]
                slice_weights = split_weights[indices]
                hist, _ = np.histogram(slice_masses, bins=mass_bins, weights=slice_weights)
                hist = hist.astype(np.float64)
                total_weight = float(np.sum(hist))
                shape = np.divide(hist, total_weight, out=np.zeros_like(hist), where=total_weight > 0.0)
                ratio = np.divide(shape, inclusive_shape, out=np.full_like(shape, np.nan), where=inclusive_shape > 0.0)
                score_low = float(np.min(slice_scores)) if len(slice_scores) else float("nan")
                score_high = float(np.max(slice_scores)) if len(slice_scores) else float("nan")
                if score_threshold_low is None:
                    score_threshold_low = score_low
                if score_threshold_high is None:
                    score_threshold_high = score_high
                ax.step(centers, ratio, where="mid", color=color, linewidth=linewidth, linestyle=linestyle, label=legend_label)
                finite = ratio[np.isfinite(ratio)]
                max_abs = float(np.max(np.abs(finite - 1.0))) if finite.size else float("nan")
                rms = float(np.sqrt(np.mean(np.square(finite - 1.0)))) if finite.size else float("nan")
                tail_metric_summary[(split_name, slice_label)] = {
                    "max_abs": max_abs,
                    "rms": rms,
                    "weighted_yield": total_weight,
                    "events": int(len(indices)),
                    "score_threshold_low": float(score_threshold_low) if np.isfinite(score_threshold_low) else None,
                    "score_threshold_high": float(score_threshold_high) if np.isfinite(score_threshold_high) else None,
                }
                for bin_index, value in enumerate(hist):
                    rows.append(
                        {
                            "fold": fold,
                            "split": split_name,
                            "slice_label": slice_label,
                            "score_decile": None,
                            "score_eff_low": float(score_eff_low),
                            "score_eff_high": float(score_eff_high),
                            "score_percentile_low": float(score_eff_low * 100.0),
                            "score_percentile_high": float(score_eff_high * 100.0),
                            "score_low": score_low,
                            "score_high": score_high,
                            "score_threshold_low": score_threshold_low,
                            "score_threshold_high": score_threshold_high,
                            "mass_low": float(mass_bins[bin_index]),
                            "mass_high": float(mass_bins[bin_index + 1]),
                            "weighted_yield": float(value),
                            "inclusive_weighted_yield": float(inclusive_hist[bin_index]),
                            "inclusive_shape": float(inclusive_shape[bin_index]),
                            "slice_shape": float(shape[bin_index]),
                            "ratio_to_inclusive_shape": float(ratio[bin_index]) if np.isfinite(ratio[bin_index]) else None,
                            "ratio_to_mass_bin_mean": float(ratio[bin_index]) if np.isfinite(ratio[bin_index]) else None,
                            "slice_total_weight": total_weight,
                            "slice_events": int(len(indices)),
                            "slice_max_abs_ratio_minus_one": max_abs,
                            "slice_rms_ratio_minus_one": rms,
                        }
                    )
                return ratio

            all_ratios = []
            for decile, indices in enumerate(decile_indices):
                label = f"{decile * 10}-{(decile + 1) * 10}% score"
                ratio = add_slice(
                    indices,
                    f"score_{decile * 10}_{(decile + 1) * 10}",
                    decile / 10.0,
                    (decile + 1) / 10.0,
                    colors[decile],
                    1.05,
                    "-",
                    label,
                )
                all_ratios.append(ratio)

            for slice_label, tail_fraction, color, linewidth, linestyle in tail_defs:
                if len(order) == 0:
                    indices = np.empty(0, dtype=np.int64)
                    threshold = float("nan")
                elif total_ordered_weight > 0.0:
                    threshold_weight = total_ordered_weight * (1.0 - tail_fraction)
                    start = int(np.searchsorted(cumulative_weight, threshold_weight, side="left"))
                    start = min(max(start, 0), len(order) - 1)
                    indices = order[start:].astype(np.int64, copy=False)
                    threshold = float(split_scores[order[start]])
                else:
                    n_tail = max(1, int(math.ceil(len(order) * tail_fraction)))
                    indices = order[-n_tail:].astype(np.int64, copy=False)
                    threshold = float(split_scores[indices[0]]) if len(indices) else float("nan")
                legend_label = slice_label.replace("_", " ")
                ratio = add_slice(
                    indices,
                    slice_label,
                    1.0 - tail_fraction,
                    1.0,
                    color,
                    linewidth,
                    linestyle,
                    legend_label,
                    score_threshold_low=threshold,
                    score_threshold_high=float(np.max(split_scores[indices])) if len(indices) else float("nan"),
                )
                all_ratios.append(ratio)

            ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.set_ylabel("slice shape / inclusive")
            ax.set_title(f"{split_name}: background MC mass shape in score slices")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(mass_min, mass_max)
            finite_ratio_chunks = [ratio[np.isfinite(ratio)] for ratio in all_ratios if np.any(np.isfinite(ratio))]
            finite_ratios = np.concatenate(finite_ratio_chunks) if finite_ratio_chunks else np.empty(0)
            if finite_ratios.size:
                lower = max(0.0, min(0.8, float(np.nanpercentile(finite_ratios, 2)) * 0.9))
                upper = min(3.0, max(1.4, float(np.nanpercentile(finite_ratios, 98)) * 1.15))
                ax.set_ylim(lower, upper)

        axes[-1].set_xlabel("mass [GeV]")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=7, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf_path = os.path.join(out_dir, f"DNN_{self._region}_{fold}_train_val_test_bkg_mass_score_slices_ratio.pdf")
        png_path = os.path.join(out_dir, f"DNN_{self._region}_{fold}_train_val_test_bkg_mass_score_slices_ratio.png")
        fig.savefig(pdf_path)
        fig.savefig(png_path)
        plt.close(fig)

        csv_path = os.path.join(out_dir, f"DNN_{self._region}_{fold}_train_val_test_bkg_mass_score_slices_ratio.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        if fold in self.m_metrics:
            for (split_name, slice_label), metrics in tail_metric_summary.items():
                prefix = f"mass_shape_{split_name}_{slice_label}"
                self.m_metrics[fold][f"{prefix}_max_abs"] = metrics["max_abs"]
                self.m_metrics[fold][f"{prefix}_rms"] = metrics["rms"]
                self.m_metrics[fold][f"{prefix}_weighted_yield"] = metrics["weighted_yield"]
                self.m_metrics[fold][f"{prefix}_events"] = metrics["events"]
                self.m_metrics[fold][f"{prefix}_score_threshold_low"] = metrics["score_threshold_low"]
                self.m_metrics[fold][f"{prefix}_score_threshold_high"] = metrics["score_threshold_high"]
            self.m_metrics[fold].update(
                {
                    "mass_score_decile_overlay_pdf": pdf_path,
                    "mass_score_decile_overlay_png": png_path,
                    "mass_score_decile_overlay_csv": csv_path,
                    "mass_score_slice_overlay_pdf": pdf_path,
                    "mass_score_slice_overlay_png": png_path,
                    "mass_score_slice_overlay_csv": csv_path,
                    "mass_score_decile_overlay_definition": f"{plot_source_definition}; score deciles plus top 10%, 1%, and 0.1% weighted-score tails per split; 1 GeV mass bins; y is each score slice normalized mass shape divided by the inclusive background mass shape in the same split",
                    "mass_score_slice_overlay_source": plot_source_definition,
                }
            )
        print(f"DNN INFO: saved mass score-slice overlay to {pdf_path}")

    def getAUC_from_arrays(self, labels, scores, raw_weights):
        labels = np.asarray(labels)
        if len(np.unique(labels)) < 2:
            return None, float("nan")
        raw_weights = np.asarray(raw_weights)
        if np.all(raw_weights >= 0):
            roc_auc = roc_auc_score(labels, scores, sample_weight=raw_weights)
        else:
            fpr, tpr, _ = roc_curve(labels, scores, sample_weight=raw_weights)
            order = np.argsort(tpr)
            roc_auc = 1 - auc(tpr[order], fpr[order])
        return None, roc_auc

    def getAUC(self, fold=0, sample_set="val"):
        if sample_set == "train":
            return self.getAUC_from_arrays(self.m_y_train[fold], self.m_score_train[fold], self.m_train_wt_raw[fold])
        if sample_set == "test":
            return self.getAUC_from_arrays(self.m_y_test[fold], self.m_score_test[fold], self.m_test_wt_raw[fold])
        return self.getAUC_from_arrays(self.m_y_val[fold], self.m_score_val[fold], self.m_val_wt_raw[fold])

    def transformScore(self, fold=0, sample="sig", reference_set="train"):
        print(f"DNN INFO: transforming scores based on {reference_set} {sample}")
        if reference_set == "train":
            labels = self.m_y_train[fold]
            scores = self.m_score_train[fold]
            weights = self.m_train_wt_raw[fold]
        elif reference_set == "val":
            labels = self.m_y_val[fold]
            scores = self.m_score_val[fold]
            weights = self.m_val_wt_raw[fold]
        elif reference_set == "test":
            labels = self.m_y_test[fold]
            scores = self.m_score_test[fold]
            weights = self.m_test_wt_raw[fold]
        else:
            raise ValueError(f"Unknown transformer reference_set '{reference_set}'")

        self.m_tsf[fold] = WeightedQuantileTransformer(n_quantiles=1000, output_distribution="uniform", subsample=1000000000, random_state=0)
        if sample == "sig":
            fit_mask = labels == 1
        elif sample == "bkg":
            fit_mask = labels == 0
        else:
            raise ValueError(f"Unknown transformer sample '{sample}'")
        if not np.any(fit_mask):
            raise ValueError(f"No {sample} events found in {reference_set} set for fold {fold}")
        fit_scores = scores[fit_mask].reshape(-1, 1)
        fit_weights = np.clip(weights[fit_mask], 0.0, None)
        if np.sum(fit_weights) <= 0:
            fit_weights = None
        self.m_tsf[fold].fit(fit_scores, sample_weight=fit_weights)
        if fold in self.m_metrics:
            self.m_metrics[fold].update(
                {
                    "transformer_reference_set": reference_set,
                    "transformer_reference_sample": sample,
                    "transformer_reference_events": int(np.count_nonzero(fit_mask)),
                    "transformer_reference_weight_source": "raw_weights_clipped_nonnegative",
                    "transformer_reference_weight_sum": float(np.sum(fit_weights)) if fit_weights is not None else None,
                }
            )

    def save(self, fold=0):
        os.makedirs(self._outputFolder, exist_ok=True)
        if fold in self.m_model:
            model_path = os.path.join(self._outputFolder, f"DNN_{self._region}_{fold}.keras")
            self.m_model[fold].save(model_path)
            print(f"DNN INFO: saved independent fold {fold} model to {model_path}")
        if fold in self.m_scaler:
            scaler_path = os.path.join(self._outputFolder, f"DNN_scaler_{self._region}_{fold}.pkl")
            with open(scaler_path, "wb") as handle:
                pickle.dump(self.m_scaler[fold], handle, -1)
        if fold in self.m_tsf:
            transformer_path = os.path.join(self._outputFolder, f"DNN_tsf_{self._region}_{fold}.pkl")
            with open(transformer_path, "wb") as handle:
                pickle.dump(self.m_tsf[fold], handle, -1)
            print(f"DNN INFO: saved independent fold {fold} transformer to {transformer_path}")
        if fold in self.m_history:
            with open(os.path.join(self._outputFolder, f"DNN_history_{self._region}_{fold}.json"), "w") as handle:
                json.dump(self.m_history[fold], handle, indent=2)
        if fold in self.m_metrics:
            with open(os.path.join(self._outputFolder, f"DNN_metrics_{self._region}_{fold}.json"), "w") as handle:
                json.dump(self.m_metrics[fold], handle, indent=2)
        metadata_path = os.path.join(self._outputFolder, f"DNN_metadata_{self._region}.json")
        metadata_fold_path = os.path.join(self._outputFolder, f"DNN_metadata_{self._region}_{fold}.json")
        metadata_params = self.params[0 if len(self.params) == 1 else fold] if self.params else self._default_params()
        default_metadata_params = self._default_params()
        metadata_output_dense_units = int(metadata_params.get("output_dense_units", default_metadata_params["output_dense_units"]))
        metadata_output_dense_units_ratio = float(metadata_params.get("output_dense_units_ratio", default_metadata_params["output_dense_units_ratio"]))
        metadata_output_dropout = float(metadata_params.get("output_dropout", default_metadata_params["output_dropout"]))
        metadata = {
            "training_weights": weight_policy({"training_weights": {"reso": self._reso, "reso_bkg": self._reso_bkg}}),
            "region": self._region,
            "ptdecor": self._ptdecor,
            "ptdecor_variables": SCALED_BY_MASS_VARIABLES,
            "scaled_by_mass_variables": SCALED_BY_MASS_VARIABLES,
            "mass_variable": MASS_VARIABLE,
            "architecture": {
                "trunk_units": metadata_params.get("trunk_units", [256, 256, 64, 64]),
                "trunk_dropout": metadata_params.get("trunk_dropout", [0.05, 0.15, 0.1, 0.1]),
                "output_dense_units": metadata_output_dense_units,
                "output_dense_units_ratio": metadata_output_dense_units_ratio,
                "output_dropout": metadata_output_dropout,
                "output_units": [metadata_output_dense_units, 2],
                "output_indices": {"score": 0, "domain": 1},
                "domain_layer": "closeness",
            },
            "domain_lambda": float(metadata_params.get("domain_lambda", default_metadata_params["domain_lambda"])),
            "domain_lambda_target": float(metadata_params.get("domain_lambda", default_metadata_params["domain_lambda"])),
            "domain_random_bce_target": float(math.log(2.0)),
            "domain_closeness_epsilon": float(DOMAIN_CLOSENESS_EPSILON),
            "score_reweight_use_transformer": bool(metadata_params.get("score_reweight_use_transformer", default_metadata_params["score_reweight_use_transformer"])),
            "score_reweight_transformer_warmup_epochs": int(metadata_params.get("score_reweight_transformer_warmup_epochs", default_metadata_params["score_reweight_transformer_warmup_epochs"])),
            "score_reweight_exponent": float(metadata_params.get("score_reweight_exponent", default_metadata_params["score_reweight_exponent"])),
            "score_reweight_reference_set": metadata_params.get("score_reweight_reference_set", default_metadata_params["score_reweight_reference_set"]),
            "score_reweight_reference_sample": metadata_params.get("score_reweight_reference_sample", default_metadata_params["score_reweight_reference_sample"]),
            "score_reweight_n_quantiles": int(metadata_params.get("score_reweight_n_quantiles", default_metadata_params["score_reweight_n_quantiles"])),
            "score_reweight_lut_size": int(metadata_params.get("score_reweight_lut_size", default_metadata_params["score_reweight_lut_size"])),
            "score_reweight_fit_record": metadata_params.get("score_reweight_fit_record", {}),
            "asimov_loss_enabled": bool(metadata_params.get("asimov_loss_enabled", default_metadata_params["asimov_loss_enabled"])),
            "asimov_lambda": float(metadata_params.get("asimov_lambda", default_metadata_params["asimov_lambda"])),
            "asimov_tau": float(metadata_params.get("asimov_tau", default_metadata_params["asimov_tau"])),
            "asimov_quantiles": [float(value) for value in metadata_params.get("asimov_quantiles", default_metadata_params["asimov_quantiles"])],
            "asimov_b_epsilon": float(metadata_params.get("asimov_b_epsilon", default_metadata_params["asimov_b_epsilon"])),
            "asimov_plateau_threshold": float(metadata_params.get("asimov_plateau_threshold", default_metadata_params["asimov_plateau_threshold"])),
            "asimov_plateau_patience": int(metadata_params.get("asimov_plateau_patience", default_metadata_params["asimov_plateau_patience"])),
            "asimov_mcstat_alpha": float(metadata_params.get("asimov_mcstat_alpha", default_metadata_params["asimov_mcstat_alpha"])),
            "asimov_activation": "val_score_loss_plateau",
            "asimov_yield_weight_definition": "abs wt_raw for signal quantile boundaries; signed wt_raw for S and B; B_eff=max(B_signed, eps)+asimov_mcstat_alpha*sum(w_bkg_raw^2*category_weight^2)",
            "mass_lambda": float(metadata_params.get("mass_lambda", default_metadata_params["mass_lambda"])),
            "mass_decorrelation_loss": str(metadata_params.get("mass_decorrelation_loss", default_metadata_params["mass_decorrelation_loss"])),
            "physics_feature_weight_mode": str(metadata_params.get("physics_feature_weight_mode", default_metadata_params["physics_feature_weight_mode"])),
            "physics_feature_weight_modes": list(PHYSICS_FEATURE_WEIGHT_MODES),
            "mass_min_events": int(metadata_params.get("mass_min_events", default_metadata_params["mass_min_events"])),
            "mass_shape_warmup_epochs": int(metadata_params.get("mass_shape_warmup_epochs", default_metadata_params["mass_shape_warmup_epochs"])),
            "mass_shape_ramp_epochs": int(metadata_params.get("mass_shape_ramp_epochs", default_metadata_params["mass_shape_ramp_epochs"])),
            "mass_rff_num_features": int(metadata_params.get("mass_rff_num_features", default_metadata_params["mass_rff_num_features"])),
            "mass_decorrelation_score_sigmas": [float(value) for value in metadata_params.get("mass_decorrelation_score_sigmas", default_metadata_params["mass_decorrelation_score_sigmas"])],
            "mass_decorrelation_mass_sigmas": [float(value) for value in metadata_params.get("mass_decorrelation_mass_sigmas", default_metadata_params["mass_decorrelation_mass_sigmas"])],
            "mass_reweight_min": float(metadata_params.get("mass_reweight_min", default_metadata_params["mass_reweight_min"])),
            "mass_reweight_max": float(metadata_params.get("mass_reweight_max", default_metadata_params["mass_reweight_max"])),
            "mass_reweight_bin_width": float(metadata_params.get("mass_reweight_bin_width", default_metadata_params["mass_reweight_bin_width"])),
            "mass_decorrelation_definition": "mass_loss is selected by mass_decorrelation_loss; rff_hsic keeps weighted multi-kernel RFF-HSIC, physics_feature_cov uses deterministic mass/score feature normalized covariance",
            "physics_feature_cov_loss_definition": "weighted normalized covariance between fixed physics mass features and classifier score features; monitor modes none=eventWeight, score=eventWeight*score_reweight, mass=mass-reweighted weight, score_mass=mass-reweighted weight*score_reweight",
            "mass_decorrelation_lambda": float(metadata_params.get("mass_lambda", default_metadata_params["mass_lambda"])),
            "mass_decorrelation_window": [
                float(metadata_params.get("mass_reweight_min", default_metadata_params["mass_reweight_min"])),
                float(metadata_params.get("mass_reweight_max", default_metadata_params["mass_reweight_max"])),
            ],
            "mass_decorrelation_kernels": {
                "score_sigmas": [float(value) for value in metadata_params.get("mass_decorrelation_score_sigmas", default_metadata_params["mass_decorrelation_score_sigmas"])],
                "mass_sigmas": [float(value) for value in metadata_params.get("mass_decorrelation_mass_sigmas", default_metadata_params["mass_decorrelation_mass_sigmas"])],
                "rff_num_features_per_kernel": int(metadata_params.get("mass_rff_num_features", default_metadata_params["mass_rff_num_features"])),
            },
            "mass_loss_definition": "total loss uses scheduled mass_lambda * mass_loss; mass_loss is val/train mass_hsic_loss for rff_hsic or val/train physics_feature_cov_loss for physics_feature_cov",
            "score_bin_mass_shape_epoch_definition": "validation background only; 10 weighted percentile bins by raw classifier score; mass in [110,150) with 1 GeV bins; weights are original eventWeight captured before score, mass, or resolution reweight",
            "domain_lambda_schedule": metadata_params.get("domain_lambda_schedule", "epoch1_0_epoch2to5_0p5_then_5x1mexpminus_epochminus5"),
            "domain_lambda_warmup_epochs": int(metadata_params.get("domain_lambda_warmup_epochs", 1)),
            "early_stopping_metric": metadata_params.get("early_stopping_metric", default_metadata_params["early_stopping_metric"]),
            "fold": int(fold),
            "train_variables": self.train_variables,
            "missing_jet_sentinel_max": MISSING_JET_SENTINEL_MAX,
            "missing_jet_impute_definition": "training variables matching jet*, dijet*, z_star, or R_pT below missing_jet_sentinel_max are replaced by the fold training mean computed from finite non-sentinel score-train events before StandardScaler, so missing jets map to scaled value 0",
            "missing_jet_impute_values": self.m_missing_jet_impute_values.get(fold, {}),
        }
        if "early_stopping_domain_weight" in metadata_params:
            metadata["early_stopping_domain_weight"] = float(metadata_params["early_stopping_domain_weight"])
        metadata.update(DOMAIN_METRIC_DEFINITIONS)
        with open(metadata_fold_path, "w") as handle:
            json.dump(metadata, handle, indent=2)
        with open(metadata_path, "w") as handle:
            json.dump(metadata, handle, indent=2)

    def _optuna_default_trial_params(self):
        params = self._default_params()
        ratio_choices = [1.0, 0.5, 0.25, 0.125]
        params["trunk_units_1"] = int(params["trunk_units"][0])
        for idx in range(2, 5):
            raw_ratio = float(params["trunk_units"][idx - 1]) / max(float(params["trunk_units"][idx - 2]), 1.0)
            params[f"trunk_units_ratio_{idx}"] = min(ratio_choices, key=lambda value: abs(value - raw_ratio))
        output_raw_ratio = float(params["output_dense_units"]) / max(float(params["trunk_units"][-1]), 1.0)
        params["output_dense_units_ratio"] = min(ratio_choices, key=lambda value: abs(value - output_raw_ratio))
        for idx, dropout in enumerate(params["trunk_dropout"], start=1):
            params[f"trunk_dropout_{idx}"] = float(dropout)
        return {
            "activation": params["activation"],
            "learning_rate": float(params["learning_rate"]),
            "l2": float(params["l2"]),
            "mass_lambda": float(params["mass_lambda"]),
            "batch_size_denominator": float(params["batch_size_denominator"]),
            "trunk_units_1": params["trunk_units_1"],
            "trunk_units_ratio_2": params["trunk_units_ratio_2"],
            "trunk_units_ratio_3": params["trunk_units_ratio_3"],
            "trunk_units_ratio_4": params["trunk_units_ratio_4"],
            "output_dense_units_ratio": params["output_dense_units_ratio"],
            "trunk_dropout_1": params["trunk_dropout_1"],
            "trunk_dropout_2": params["trunk_dropout_2"],
            "trunk_dropout_3": params["trunk_dropout_3"],
            "trunk_dropout_4": params["trunk_dropout_4"],
            "output_dropout": float(params["output_dropout"]),
        }

    def _suggest_hparams(self, trial, n_features):
        params = self._default_params()
        params["activation"] = trial.suggest_categorical("activation", ["relu", "elu", "swish"])
        params["learning_rate"] = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
        params["l2"] = trial.suggest_float("l2", 1e-8, 1e-6, log=True)
        params["domain_lambda"] = float(params["domain_lambda"])
        params["mass_lambda"] = trial.suggest_float("mass_lambda", 1.0, 100.0, log=True)
        params["batch_size_denominator"] = trial.suggest_float("batch_size_denominator", 200, 500)
        params["trunk_units"] = [int(trial.suggest_categorical("trunk_units_1", [256, 512, 1024, 2048, 4096]))]
        for idx in range(2, 5):
            ratio = float(trial.suggest_categorical(f"trunk_units_ratio_{idx}", [1.0, 0.5, 0.25, 0.125]))
            params["trunk_units"].append(max(1, int(round(params["trunk_units"][-1] * ratio))))
        output_dense_units_ratio = float(trial.suggest_categorical("output_dense_units_ratio", [1.0, 0.5, 0.25, 0.125]))
        params["output_dense_units_ratio"] = output_dense_units_ratio
        params["output_dense_units"] = max(1, int(round(params["trunk_units"][-1] * output_dense_units_ratio)))
        params["trunk_dropout"] = [
            float(trial.suggest_categorical(f"trunk_dropout_{idx}", [0.0, 0.05, 0.1, 0.15]))
            for idx in range(1, 5)
        ]
        params["output_dropout"] = float(trial.suggest_categorical("output_dropout", [0.0, 0.05, 0.1, 0.15]))
        return params

    def optunaHP(self):
        suffix = self.optuna_suffix if self.optuna_suffix else datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = f"models/optuna_dnn_{self._region}_{suffix}/"
        os.makedirs(exp_dir, exist_ok=True)
        db_path = os.path.join(exp_dir, f"optuna_study_{self._region}_all_folds.db")
        best_params_path = os.path.join(exp_dir, f"DNN_region_{self._region}_all_folds.json")
        log_filename = os.path.join(exp_dir, "optuna_optimization_all_folds.log")

        logger = logging.getLogger(f"optuna_dnn_{self._region}_all_folds")
        logger.setLevel(logging.INFO)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        file_handler = logging.FileHandler(log_filename, mode="a")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)
        logger.addHandler(logging.StreamHandler())

        if not self.continue_optuna and os.path.exists(db_path):
            os.remove(db_path)
        storage = f"sqlite:///{db_path}"

        def objective(trial):
            hparams = self._suggest_hparams(trial, n_features=len(self.train_variables))
            trial.set_user_attr("hparams", hparams)
            for fold in range(4):
                self.setParams(hparams, fold)
            logger.info("Trial %d shared params for all folds: %s", trial.number, hparams)
            metrics_per_fold = []
            try:
                for fold in range(4):
                    tf.keras.backend.clear_session()
                    fixRandomSeed(self.args.seed + fold)
                    try:
                        if self._memory_mode == "streaming":
                            self.prepareDataStreaming(fold)
                        self.trainModel(fold, self.params[fold])
                        self.testModel(fold)
                        metrics_per_fold.append(dict(self.m_metrics[fold]))
                    finally:
                        if self._memory_mode == "streaming":
                            self.releaseFold(fold)
                train_auc_by_fold = [m["train_auc"] for m in metrics_per_fold]
                eval_auc_by_fold = [m["eval_auc"] for m in metrics_per_fold]
                domain_loss_by_fold = [m["val_domain_loss"] for m in metrics_per_fold]
                domain_bce_by_fold = [m["val_domain_bce"] for m in metrics_per_fold]
                domain_closeness_loss_by_fold = [m["val_domain_closeness_loss"] for m in metrics_per_fold]
                mass_loss_by_fold = [m["val_mass_loss"] for m in metrics_per_fold]
                mass_hsic_loss_by_fold = [m["val_mass_hsic_loss"] for m in metrics_per_fold]
                physics_metric_by_fold = {
                    key: [float(m.get(f"val_{key}", float("nan"))) for m in metrics_per_fold]
                    for key in PHYSICS_FEATURE_MONITOR_KEYS
                }
                shape_metric_by_fold = {
                    key: [float(m.get(f"val_{key}", float("nan"))) for m in metrics_per_fold]
                    for key in SCORE_BIN_MASS_METRIC_KEYS
                }
                test_auc_by_fold = [m["test_auc"] for m in metrics_per_fold]
                train_auc = float(np.mean(train_auc_by_fold))
                eval_auc = float(np.mean(eval_auc_by_fold))
                domain_loss = float(np.mean(domain_loss_by_fold))
                domain_bce = float(np.mean(domain_bce_by_fold))
                domain_closeness_loss = float(np.mean(domain_closeness_loss_by_fold))
                mass_loss = float(np.mean(mass_loss_by_fold))
                mass_hsic_loss = float(np.mean(mass_hsic_loss_by_fold))
                physics_metrics = {
                    key: float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")
                    for key, values in physics_metric_by_fold.items()
                }
                shape_metrics = {
                    key: float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")
                    for key, values in shape_metric_by_fold.items()
                }
                test_auc = float(np.mean(test_auc_by_fold))
                domain_weight = 5
                domain_loss_objective_weight = 100.0
                mass_weight = 5
                hsic_objective_weight = 10.0
                optuna_eval_auc_max = 0.75
                sqrt_eval_auc_train_auc_term = math.sqrt(max(2.0 * eval_auc * train_auc - train_auc**2, 0.0))
                sqrt_eval_auc_minus_train_auc = sqrt_eval_auc_train_auc_term - hsic_objective_weight * mass_hsic_loss - domain_loss_objective_weight * domain_closeness_loss
                eval_auc_minus_domain_loss = eval_auc - domain_weight * domain_closeness_loss - domain_loss_objective_weight * domain_loss
                eval_auc_minus_domain_mass_loss = eval_auc - domain_weight * domain_closeness_loss - mass_weight * mass_loss - domain_loss_objective_weight * domain_loss
                fold_eval_auc_minus_domain_loss = [
                    eval_auc_value - domain_weight * domain_closeness_loss_value - domain_loss_objective_weight * domain_loss_value
                    for eval_auc_value, domain_closeness_loss_value, domain_loss_value in zip(eval_auc_by_fold, domain_closeness_loss_by_fold, domain_loss_by_fold)
                ]
                fold_eval_auc_minus_domain_mass_loss = [
                    eval_auc_value - domain_weight * domain_closeness_loss_value - mass_weight * mass_loss_value - domain_loss_objective_weight * domain_loss_value
                    for eval_auc_value, domain_closeness_loss_value, mass_loss_value, domain_loss_value in zip(eval_auc_by_fold, domain_closeness_loss_by_fold, mass_loss_by_fold, domain_loss_by_fold)
                ]
                metrics = {
                    "objective": f"mean_{self.optuna_metric}_over_all_folds",
                    "requested_optuna_metric": self.optuna_metric,
                    OPTUNA_EVAL_AUC_DOMAIN_METRIC: eval_auc_minus_domain_loss,
                    OPTUNA_EVAL_AUC_DOM_MASS_METRIC: eval_auc_minus_domain_mass_loss,
                    "eval_auc": eval_auc,
                    "domain": domain_closeness_loss,
                    "domain_loss": domain_loss,
                    "domain_bce": domain_bce,
                    "domain_closeness_loss": domain_closeness_loss,
                    "mass_loss": mass_loss,
                    "mass_hsic_loss": mass_hsic_loss,
                    **{f"val_{key}": value for key, value in physics_metrics.items()},
                    **{f"val_{key}": value for key, value in shape_metrics.items()},
                    "domain_weight": domain_weight,
                    "domain_loss_objective_weight": domain_loss_objective_weight,
                    "mass_weight": mass_weight,
                    "hsic_objective_weight": hsic_objective_weight,
                    "sqrt_eval_auc_train_auc_term": sqrt_eval_auc_train_auc_term,
                    "optuna_eval_auc_max": optuna_eval_auc_max,
                    "optuna_eval_auc_veto": bool(eval_auc >= optuna_eval_auc_max),
                    "train_auc": train_auc,
                    "test_auc": test_auc,
                    f"fold_{OPTUNA_EVAL_AUC_DOMAIN_METRIC}": fold_eval_auc_minus_domain_loss,
                    f"fold_{OPTUNA_EVAL_AUC_DOM_MASS_METRIC}": fold_eval_auc_minus_domain_mass_loss,
                    "fold_eval_auc": eval_auc_by_fold,
                    "fold_domain": domain_closeness_loss_by_fold,
                    "fold_domain_loss": domain_loss_by_fold,
                    "fold_domain_bce": domain_bce_by_fold,
                    "fold_domain_closeness_loss": domain_closeness_loss_by_fold,
                    "fold_mass_loss": mass_loss_by_fold,
                    "fold_mass_hsic_loss": mass_hsic_loss_by_fold,
                    **{f"fold_val_{key}": values for key, values in physics_metric_by_fold.items()},
                    **{f"fold_val_{key}": values for key, values in shape_metric_by_fold.items()},
                    "fold_train_auc": train_auc_by_fold,
                    "fold_test_auc": test_auc_by_fold,
                    "sqrt_eval_auc_minus_train_auc": sqrt_eval_auc_minus_train_auc,
                    "eval_auc_minus_train_auc": eval_auc - train_auc,
                    "eval_auc_over_train_auc": eval_auc / max(train_auc, 1e-6),
                }
                objective_value = metrics.get(self.optuna_metric, eval_auc_minus_domain_loss)
                fold_objective_values = {
                    OPTUNA_EVAL_AUC_DOMAIN_METRIC: fold_eval_auc_minus_domain_loss,
                    OPTUNA_EVAL_AUC_DOM_MASS_METRIC: fold_eval_auc_minus_domain_mass_loss,
                    "eval_auc": eval_auc_by_fold,
                    "sqrt_eval_auc_minus_train_auc": [
                        math.sqrt(max(2.0 * eval_auc_value * train_auc_value - train_auc_value**2, 0.0)) - hsic_objective_weight * mass_hsic_loss_value
                        for eval_auc_value, train_auc_value, mass_hsic_loss_value in zip(eval_auc_by_fold, train_auc_by_fold, mass_hsic_loss_by_fold)
                    ],
                    "eval_auc_minus_train_auc": [
                        eval_auc_value - train_auc_value
                        for eval_auc_value, train_auc_value in zip(eval_auc_by_fold, train_auc_by_fold)
                    ],
                    "eval_auc_over_train_auc": [
                        eval_auc_value / max(train_auc_value, 1e-6)
                        for eval_auc_value, train_auc_value in zip(eval_auc_by_fold, train_auc_by_fold)
                    ],
                }.get(self.optuna_metric, fold_eval_auc_minus_domain_loss)
                if eval_auc >= optuna_eval_auc_max:
                    objective_value = 0.0
                    fold_objective_values = [0.0 for _ in fold_objective_values]
                metrics["fold_optuna_metric"] = fold_objective_values
                metrics["optuna_metric_value"] = objective_value
                trial.set_user_attr("metrics", metrics)
                trial.set_user_attr("fold_metrics", metrics_per_fold)
                logger.info(
                    "Trial %d mean %s over folds: %.6f (eval_auc=%.6f train_auc=%.6f sqrt_auc_term=%.6f domain_closeness=%.6f domain_bce=%.6f domain_loss=%.6f domain_loss_weight=%.1f mass_loss=%.6f mass_hsic_loss=%.6f eval_auc_veto=%s folds=%s)",
                    trial.number,
                    self.optuna_metric,
                    objective_value,
                    eval_auc,
                    train_auc,
                    sqrt_eval_auc_train_auc_term,
                    domain_closeness_loss,
                    domain_bce,
                    domain_loss,
                    domain_loss_objective_weight,
                    mass_loss,
                    mass_hsic_loss,
                    metrics["optuna_eval_auc_veto"],
                    ", ".join(f"{value:.6f}" for value in fold_objective_values),
                )
                return objective_value
            except Exception as exc:
                logger.exception("Trial %d failed", trial.number)
                raise exc

        def save_best_callback(study, trial):
            if study.best_trial.number != trial.number:
                return
            best_data = {
                "objective": f"mean_{self.optuna_metric}_over_all_folds",
                "best_value": study.best_value,
                "best_params": study.best_trial.user_attrs.get("hparams", study.best_params),
                "best_optuna_params": study.best_params,
                "best_trial_number": study.best_trial.number,
                "n_trials_completed": len(study.trials),
                "best_trial_metrics": study.best_trial.user_attrs.get("metrics", {}),
                "best_trial_fold_metrics": study.best_trial.user_attrs.get("fold_metrics", []),
            }
            with open(best_params_path, "w") as handle:
                json.dump(best_data, handle, indent=2)
            logger.info("Updated best parameters at %s", best_params_path)

        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(n_startup_trials=10, multivariate=True, group=True),
            study_name=f"DNN_{self._region}_allfolds",
            storage=storage,
            load_if_exists=self.continue_optuna != 0,
            direction="maximize",
        )
        active_states = {
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.RUNNING,
            optuna.trial.TrialState.WAITING,
        }
        if not any(trial.state in active_states for trial in study.trials):
            default_trial_params = self._optuna_default_trial_params()
            study.enqueue_trial(default_trial_params)
            logger.info("Enqueued default DNN hyperparameters as Optuna reference trial: %s", default_trial_params)
        study.optimize(objective, n_trials=self.n_calls, callbacks=[save_best_callback])

        best_data = {
            "objective": f"mean_{self.optuna_metric}_over_all_folds",
            "best_value": study.best_value,
            "best_params": study.best_trial.user_attrs.get("hparams", study.best_params),
            "best_optuna_params": study.best_params,
            "best_trial_number": study.best_trial.number,
            "n_trials_completed": len(study.trials),
            "best_trial_metrics": study.best_trial.user_attrs.get("metrics", {}),
            "best_trial_fold_metrics": study.best_trial.user_attrs.get("fold_metrics", []),
        }
        with open(best_params_path, "w") as handle:
            json.dump(best_data, handle, indent=2)
        logger.info("Final best params saved to %s", best_params_path)
        return study.best_params


def main():
    args = getArgs()
    fixRandomSeed(args.seed)
    physical_gpus = tf.config.list_physical_devices("GPU")
    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"DNN WARNING: could not set memory growth for {gpu}: {exc}")
    print(f"DNN INFO: TensorFlow version={tf.__version__}")
    print(f"DNN INFO: physical GPUs={physical_gpus}")
    print(f"DNN INFO: logical GPUs={tf.config.list_logical_devices('GPU')}")
    handler = DNNHandler(args.config, args.region)
    if args.inputFolder:
        handler.setInputFolder(args.inputFolder)
    if args.outputFolder:
        handler.setOutputFolder(args.outputFolder)

    if args.params:
        handler.setParams(args.params)

    if args.hyperparams_path and not args.optuna:
        path = os.path.join(args.hyperparams_path, f"DNN_region_{args.region}_all_folds.json")
        if os.path.exists(path):
            with open(path, "r") as stream:
                hyperparams = json.load(stream)
            for fold in range(4):
                handler.setParams(hyperparams["best_params"], fold)
        else:
            logging.warning("Hyperparameter file not found: %s. Continue with config/default parameters.", path)

    folds_to_prepare = range(4) if args.optuna else args.fold

    if args.memory_mode == "legacy":
        caches_available = handler.hasPreparedCaches(folds_to_prepare)
        if caches_available:
            print("DNN CACHE: all requested prepared folds are available; skipping ROOT readData")
        else:
            read_start = time.perf_counter()
            handler.readData()
            print(f"DNN PROFILE: readData_s={time.perf_counter() - read_start:.1f}")

        for fold in folds_to_prepare:
            prepare_start = time.perf_counter()
            handler.prepareData(fold)
            print(f"DNN PROFILE: fold={fold} prepareData_s={time.perf_counter() - prepare_start:.1f}")

    if args.optuna:
        try:
            handler.optunaHP()
        except KeyboardInterrupt:
            print("Finishing on SIGINT.")
        return

    for fold in args.fold:
        print("===================================================")
        print(f" The #{fold} fold of training")
        print("===================================================")
        fold_start = time.perf_counter()
        try:
            if args.memory_mode == "streaming":
                prepare_start = time.perf_counter()
                handler.prepareDataStreaming(fold)
                print(f"DNN PROFILE: fold={fold} prepareDataStreaming_s={time.perf_counter() - prepare_start:.1f}")
            handler.trainModel(fold)
            handler.testModel(fold)
            handler.plotScoreDecileMassOverlay(fold)
            print("param: %s, Val AUC: %f" % handler.getAUC(fold))
            if args.roc:
                handler.plotROC(fold)
            handler.transformScore(fold, sample="sig", reference_set="train")
            if args.save:
                save_start = time.perf_counter()
                handler.save(fold)
                print(f"DNN PROFILE: fold={fold} save_s={time.perf_counter() - save_start:.1f}")
            print(f"DNN PROFILE: fold={fold} train_test_save_total_s={time.perf_counter() - fold_start:.1f}")
        finally:
            if args.memory_mode == "streaming":
                handler.releaseFold(fold)

    print("------------------------------------------------------------------------------")
    print("Finished training.")


if __name__ == "__main__":
    main()
