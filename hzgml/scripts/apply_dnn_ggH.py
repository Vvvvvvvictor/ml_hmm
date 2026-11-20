#!/usr/bin/env python
import gc
import importlib.util
import json
import logging
import os
import pickle
import shutil
import subprocess
import tempfile
import time
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import pandas as pd
try:
    import ROOT
    ROOT.gErrorIgnoreLevel = ROOT.kError + 1
except ImportError:
    ROOT = None
import tensorflow as tf
import uproot
from tqdm import tqdm
from weighted_quantile_transformer import WeightedQuantileTransformer
from dnn_fit_output import fit_output, mapping_record, select_jet_migrations, systematic_plan
from dnn_root_io import open_published_root

tf.config.optimizer.set_jit(False)

try:
    from train_dnn_ggH import DecorrelatingDNN  # noqa: F401
except ImportError:
    DecorrelatingDNN = None

pd.options.mode.chained_assignment = None

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


current_date = datetime.now().strftime("%m%d")
SCALED_BY_MASS_VARIABLES = ["diMufsr_kit_BSC_pt", "jet1_pt", "jet2_pt", "dijet_pt"]
MISSING_JET_SENTINEL_MAX = -90.0
JET_PATCH_ERAS_2223 = {"2022", "2022EE", "2023", "2023BPix"}
JET_PATCH_SENTINEL = -100.0
JET_PATCH_EXTRA_BRANCHES = {
    "jet1_phi",
    "jet1_mass",
    "jet2_phi",
    "jet2_mass",
    "diMufsr_kit_BSC_phi",
}


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


def dataframe_to_root_arrays(dataframe):
    arrays = {}
    for column in dataframe.columns:
        values = dataframe[column].to_numpy(copy=False)
        if values.dtype == object:
            values = values.astype(str)
        arrays[column] = values
    return arrays


def root_branch_types(arrays):
    return {column: values.dtype for column, values in arrays.items()}


def get_memory_usage():
    if not PSUTIL_AVAILABLE:
        return 0.0
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024


def check_hadd_available():
    try:
        result = subprocess.run(["hadd", "-h"], capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False


HADD_AVAILABLE = check_hadd_available()


def eos_xrootd_url(path):
    path = os.fspath(path)
    if path.startswith("/eos/"):
        return f"root://eosuser.cern.ch/{path}"
    return path


def remove_output_path(path):
    eos = shutil.which("eos")
    if path.startswith("/eos/") and eos:
        subprocess.run([eos, "root://eosuser.cern.ch", "rm", path], capture_output=True, text=True)
    if os.path.exists(path):
        os.remove(path)


def move_published_eos_path(temp_output, output_path):
    eos = shutil.which("eos")
    if not output_path.startswith("/eos/") or not eos:
        os.replace(temp_output, output_path)
        return

    remove_output_path(output_path)
    result = subprocess.run(
        [eos, "root://eosuser.cern.ch", "mv", temp_output, output_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"eos mv failed while publishing {output_path}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def ensure_output_dir(output_path):
    output_dir = os.path.dirname(output_path)
    eos = shutil.which("eos")
    if output_path.startswith("/eos/") and eos:
        result = subprocess.run(
            [eos, "root://eosuser.cern.ch", "mkdir", "-p", output_dir],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"eos mkdir failed for {output_dir}\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        return
    os.makedirs(output_dir, exist_ok=True)


def publish_output_file(local_path, output_path):
    ensure_output_dir(output_path)
    temp_output = f"{output_path}.tmp.{os.getpid()}"
    remove_output_path(temp_output)

    try:
        xrdcp = shutil.which("xrdcp")
        if output_path.startswith("/eos/") and xrdcp:
            result = subprocess.run(
                [xrdcp, "-f", local_path, eos_xrootd_url(temp_output)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"xrdcp failed while publishing {output_path}\n"
                    f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                )
        else:
            shutil.copy2(local_path, temp_output)

        move_published_eos_path(temp_output, output_path)
    except Exception:
        try:
            remove_output_path(temp_output)
        finally:
            raise


def load_config(config_path):
    if config_path.endswith(".py"):
        spec = importlib.util.spec_from_file_location("training_config", config_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = getattr(module, "CONFIG", None)
        if config is None:
            raise ValueError(f"Python config {config_path} must define CONFIG = {{...}}")
        return config

    with open(config_path, "r") as stream:
        return json.loads(stream.read())


def getArgs():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        action="store",
        nargs="+",
        default=["hzgml/data/training_config_BDT_ggH_multiyear_template.py"],
        help="Training config path, optionally followed by legacy apply config path",
    )
    parser.add_argument("-i", "--inputFolder", action="store", required=True, help="directory of input ROOT files")
    parser.add_argument(
        "-t",
        "--inputTree",
        action="store",
        default="",
        help="input tree name, empty means auto-resolve from config or file contents",
    )
    parser.add_argument("-m", "--modelFolder", action="store", default="models_dnn", help="directory of DNN models")
    parser.add_argument("-o", "--outputFolder", action="store", default="outputs", help="directory for outputs")
    parser.add_argument(
        "-r",
        "--region",
        action="store",
        choices=["two_jet", "one_jet", "zero_jet", "zero_to_one_jet", "VH_ttH", "VBF", "all_jet", "ggH", "VBFsep", "ggH_sep"],
        default="ggH",
        help="Region to process",
    )
    parser.add_argument("-cat", "--category", action="store", nargs="+", help="apply only for specific categories")
    parser.add_argument("-s", "--shield", action="store", type=int, default=-1, help="Which variable to remove")
    parser.add_argument("-a", "--add", action="store", type=int, default=-1, help="Which variable to add from test_variables")
    parser.add_argument("-F", "--FixSBH125", action="store_true", default=False, help="Fix the H mass to be 125GeV to get the scores")
    parser.add_argument("-y", "--year", action="store", default="", help="optional year tag appended to output directory")
    parser.add_argument("--inputs-config", action="store", default="hzgml/data/inputs_config_DNN_ggH.json", help="sample list config")
    parser.add_argument("--output-branch", action="store", default="", help="override output score branch name")
    parser.add_argument("--tree-name", dest="inputTreeAlias", action="store", default="", help="alias for --inputTree")
    parser.add_argument("--profile-apply", action="store_true", default=False, help="print per-chunk DNN apply timing")
    parser.add_argument("--max-chunks", action="store", type=int, default=0, help="stop after this many input chunks per sample")
    parser.add_argument("--chunk-size", action="store", type=int, default=500000, help="events to process per uproot chunk")
    parser.add_argument("--predict-batch-size", action="store", type=int, default=262144, help="TensorFlow inference batch size, <=0 uses Keras default")
    parser.add_argument("--drop-output-branches", action="store", nargs="+", default=[], help="branches to remove from output after DNN scoring")
    parser.add_argument("--systematics", action="store_true", help="score input systematic variations on the fixed nominal event selection")
    parser.add_argument("--output-profile", choices=["full", "fit"], default="full", help="fit saves only mass, scores, weights and event identifiers")
    parser.add_argument("--include-jet-migrations", action="store_true",
                        help="include signal events selected by any JER/JES variation in fit output")
    parser.add_argument("--exact-output-folder", action="store_true", help="use --outputFolder directly without date/year suffixes")
    parser.add_argument("--disable-2223-jet-patch", action="store_true", help="use the input jet features without the legacy 2022/2023 apply-time patch")
    score_transformers = parser.add_mutually_exclusive_group()
    score_transformers.add_argument("--save-score-transformers", default="", help="save fitted fold score transformers to this directory")
    score_transformers.add_argument("--load-score-transformers", default="", help="load fold score transformers from this directory")
    args = parser.parse_args()
    if args.include_jet_migrations and (args.output_profile != "fit" or not args.systematics):
        parser.error("--include-jet-migrations requires --output-profile fit and --systematics")
    return args


class ApplyDNNHandler(object):
    def __init__(self, args):
        self.args = args
        self._shield = args.shield
        self._add = args.add
        self._FixSBH125 = args.FixSBH125
        self._year = args.year
        self._region = args.region

        self._inputFolder = args.inputFolder
        self._inputTree = args.inputTreeAlias or args.inputTree
        self._modelFolder = args.modelFolder
        self._outputFolder = ""
        self._chunksize = args.chunk_size
        self._profile_apply = args.profile_apply
        self._max_chunks = args.max_chunks
        self._predict_batch_size = args.predict_batch_size
        self._drop_output_branches = args.drop_output_branches

        self.train_variables = []
        self.train_signal = []
        self.train_mc_background = []
        self.keep_branches = []
        self.observables = []
        self.preselections = []
        self.mc_preselections = []
        self.data_preselections = []
        self.signal_preselections = []
        self.background_preselections = []
        self.yearly_inputs = []
        self.randomIndex = ""
        self.weight = ""
        self.reso = "diMu-mass_resolution_BSC_abs"
        self.dnn_defaults = {}
        self.decorrelation = {}

        self.output_branch = args.output_branch or "dnn_score"
        self.m_models = []
        self.m_tsfs = []
        self.m_scalers = []
        self._requested_branches = []
        self._resolved_year_cfg = {}
        self._branch_aliases = {}
        self._event_weight_scale = 1.0
        self.source_year_map = {}
        self.source_year_inverse_map = {}
        self._apply_source_year_label = None
        self._ptdecor = False
        self._mass_variable = "diMufsr_kit_BSC_mass"
        self._scaled_by_mass_variables = SCALED_BY_MASS_VARIABLES
        self._missing_jet_sentinel_max = MISSING_JET_SENTINEL_MAX
        self._missing_jet_impute_values = []
        self._warned_2223_jet_patch_missing = set()

        self._read_configs(args.config)
        self._resolve_year_config()
        self._resolve_apply_source_year_label()
        self._resolve_input_tree()
        self.loadMetadata()
        self._prepare_branches()

    def _read_configs(self, config_paths):
        training_config = config_paths[0]
        self.readTrainConfig(training_config)
        if len(config_paths) > 1 and os.path.exists(config_paths[1]):
            self.readLegacyApplyConfig(config_paths[1])

    def readTrainConfig(self, config_path):
        config = load_config(config_path)[self._region]
        members = {
            name for name in dir(self)
            if not name.startswith(("_", "m_")) and not callable(getattr(self, name))
        }
        for name, value in config.items():
            if name in members:
                setattr(self, name, value)

        for name in ("train_variables", "preselections", "signal_preselections", "background_preselections"):
            key = "+" + name
            if key in config:
                values = config[key]
                if name == "train_variables" and self._add >= 0 and "test_variables" in config:
                    values = [*values, config["test_variables"][self._add]]
                getattr(self, name).extend(values)
        if self._shield >= 0:
            self.train_variables.pop(self._shield)
        self._initialize_source_year_mapping()

    def readLegacyApplyConfig(self, config_path):
        try:
            configs = load_config(config_path)
            common = configs.get("common", {})
            self.observables = common.get("observables", self.observables)
            for presel in common.get("preselections", []):
                if presel not in self.preselections:
                    self.preselections.append(presel)

            region_cfg = configs.get(self._region, {})
            if "+preselections" in region_cfg:
                self.preselections.extend(region_cfg["+preselections"])
        except Exception as e:
            logging.warning("Failed to read legacy apply config '%s': %s", config_path, e)

    def _normalize_year_key(self, year):
        if isinstance(year, str):
            return year.strip()
        if isinstance(year, (np.integer, int)):
            return str(int(year))
        return str(year)

    def _initialize_source_year_mapping(self):
        self.source_year_map = {}
        self.source_year_inverse_map = {}
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

        self.source_year_map = {
            self._normalize_year_key(year): index
            for index, year in enumerate(ordered_years)
        }
        self.source_year_inverse_map = {
            index: year for index, year in enumerate(ordered_years)
        }
        logging.info("source_year encoding: %s", self.source_year_map)

    def _encode_source_year(self, year):
        key = self._normalize_year_key(year)
        if key in self.source_year_map:
            return self.source_year_map[key]
        raise KeyError(f'Unknown source_year "{year}". Known years: {list(self.source_year_map.keys())}')

    def _encode_or_passthrough_source_year(self, year):
        if pd.isna(year):
            if self._apply_source_year_label is None:
                raise KeyError("source_year is NaN and no configured year is available for fallback.")
            return self._encode_source_year(self._apply_source_year_label)

        if isinstance(year, (np.integer, int)) and int(year) in self.source_year_inverse_map:
            return int(year)

        if isinstance(year, str):
            stripped = year.strip()
            if stripped.isdigit():
                numeric = int(stripped)
                if numeric in self.source_year_inverse_map and stripped not in self.source_year_map:
                    return numeric

        return self._encode_source_year(year)

    def _resolve_year_config(self):
        if not self.yearly_inputs:
            self._resolved_year_cfg = {}
            self._branch_aliases = {}
            self._event_weight_scale = 1.0
            return

        normalized_input = os.path.abspath(os.path.normpath(self._inputFolder))
        for year_cfg in self.yearly_inputs:
            folder = year_cfg.get("folder")
            if not folder:
                continue
            normalized_folder = os.path.abspath(os.path.normpath(folder))
            if normalized_folder == normalized_input:
                self._resolved_year_cfg = year_cfg
                self._branch_aliases = year_cfg.get("branch_aliases", {})
                self._event_weight_scale = float(year_cfg.get("eventWeight_scale", 1.0))
                return

        self._resolved_year_cfg = {}
        self._branch_aliases = {}
        self._event_weight_scale = 1.0

    def _resolve_apply_source_year_label(self):
        if self._resolved_year_cfg.get("year") is not None:
            self._apply_source_year_label = self._resolved_year_cfg["year"]
            return
        if self._year:
            self._apply_source_year_label = self._year
            return
        self._apply_source_year_label = None

    def _resolve_input_tree(self):
        if self._inputTree:
            return
        if self._resolved_year_cfg.get("tree_name"):
            self._inputTree = self._resolved_year_cfg["tree_name"]
            return
        self._inputTree = "inclusive"

    def _is_2223_jet_patch_enabled(self):
        if getattr(self.args, "disable_2223_jet_patch", False) or self._apply_source_year_label is None:
            return False
        return self._normalize_year_key(self._apply_source_year_label) in JET_PATCH_ERAS_2223

    def _prepare_branches(self):
        branches = set(self.train_variables)
        branches.update(self.keep_branches)
        branches.update(self.observables)
        branches.update([self.randomIndex, self.weight, self.reso])
        if self._is_2223_jet_patch_enabled():
            branches.update(JET_PATCH_EXTRA_BRANCHES)
        if self._ptdecor:
            branches.add(self._mass_variable)
            branches.update(self._scaled_by_mass_variables)

        for selection_group in (
            self.preselections,
            self.signal_preselections,
            self.background_preselections,
            self.mc_preselections,
            self.data_preselections,
        ):
            branches.update([p.split()[0].replace("(", "") for p in selection_group])

        branches.discard("")
        self._requested_branches = sorted(branches)

        self.train_variables = [x.replace("noexpand:", "") for x in self.train_variables]
        self.preselections = [x.replace("noexpand:", "") for x in self.preselections]
        self.signal_preselections = [x.replace("noexpand:", "") for x in self.signal_preselections]
        self.background_preselections = [x.replace("noexpand:", "") for x in self.background_preselections]
        self.mc_preselections = [x.replace("noexpand:", "") for x in self.mc_preselections]
        self.data_preselections = [x.replace("noexpand:", "") for x in self.data_preselections]
        self.randomIndex = self.randomIndex.replace("noexpand:", "")
        self.weight = self.weight.replace("noexpand:", "")

        if self.preselections:
            self.preselections = ["data." + p for p in self.preselections]

    def setOutputFolder(self, outputFolder):
        if self.args.exact_output_folder:
            self._outputFolder = outputFolder
            return
        self._outputFolder = outputFolder + f"_dnn_{current_date}"
        if self._year:
            self._outputFolder += "_" + self._year
        if self._FixSBH125:
            self._outputFolder += "_SB_HM125"

    def preselect(self, data):
        for p in self.preselections:
            data = data[eval(p)]
        return data

    def loadModels(self):
        logging.info("Loading DNN models")
        for fold in range(4):
            path = f"{self._modelFolder}/DNN_{self._region}_{fold}.keras"
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
            self.m_models.append(tf.keras.models.load_model(path, compile=False))

    def loadTransformer(self):
        logging.info("Loading score transformers")
        for fold in range(4):
            path = f"{self._modelFolder}/DNN_tsf_{self._region}_{fold}.pkl"
            with open(path, "rb") as handle:
                self.m_tsfs.append(pickle.load(handle, encoding="latin1"))

    def loadScalers(self):
        logging.info("Loading feature scalers")
        for fold in range(4):
            path = f"{self._modelFolder}/DNN_scaler_{self._region}_{fold}.pkl"
            with open(path, "rb") as handle:
                self.m_scalers.append(pickle.load(handle))

    def loadMetadata(self):
        metadata_path = os.path.join(self._modelFolder, f"DNN_metadata_{self._region}.json")
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as handle:
                metadata = json.load(handle)
        self._ptdecor = bool(metadata.get("ptdecor", False))
        self._mass_variable = metadata.get("mass_variable", self._mass_variable)
        self._scaled_by_mass_variables = metadata.get("scaled_by_mass_variables", metadata.get("ptdecor_variables", self._scaled_by_mass_variables))
        self._missing_jet_sentinel_max = float(metadata.get("missing_jet_sentinel_max", self._missing_jet_sentinel_max))
        self._missing_jet_impute_values = []
        for fold in range(4):
            fold_metadata = metadata
            fold_metadata_path = os.path.join(self._modelFolder, f"DNN_metadata_{self._region}_{fold}.json")
            if os.path.exists(fold_metadata_path):
                with open(fold_metadata_path, "r") as handle:
                    fold_metadata = json.load(handle)
            values = fold_metadata.get("missing_jet_impute_values", {})
            self._missing_jet_impute_values.append({name: float(value) for name, value in values.items()})
        logging.info(
            "Loaded DNN metadata from %s, ptdecor=%s, scaled_by_mass_variables=%s, missing_jet_impute_folds=%s",
            metadata_path,
            self._ptdecor,
            self._scaled_by_mass_variables,
            sum(1 for values in self._missing_jet_impute_values if values),
        )

    def load_sample_list(self):
        config = load_config(self.args.inputs_config)
        sample_list = config.get("sample_list", {})
        if isinstance(sample_list, dict):
            return list(sample_list.keys())
        if isinstance(sample_list, list):
            return sample_list
        raise ValueError(f"Unsupported sample_list format in {self.args.inputs_config}")

    def _input_names_for_category(self, category):
        sample_map = self._resolved_year_cfg.get("sample_map", {})
        mapped = sample_map.get(category, category)
        if isinstance(mapped, (list, tuple)):
            names = list(mapped)
        else:
            names = [mapped]
        if category not in names:
            names.append(category)
        return names

    def _find_input_files(self, category):
        mapped = self._resolved_year_cfg.get("sample_map", {}).get(category, category)
        mapped = list(mapped) if isinstance(mapped, (list, tuple)) else [mapped]
        if any(os.path.isabs(name) for name in mapped):
            if not all(os.path.isabs(name) for name in mapped):
                raise ValueError(f"Mixed absolute and relative input mapping: {category}")
            paths = [name if name.endswith(".root") else name + ".root" for name in mapped]
            for path in paths:
                if not os.path.isfile(path):
                    raise FileNotFoundError(path)
            return sorted(set(paths))
        if not os.path.isdir(self._inputFolder):
            return []

        input_files = []
        input_names = self._input_names_for_category(category)
        for filename in os.listdir(self._inputFolder):
            if any(filename.endswith(f"{input_name}.root") for input_name in input_names):
                input_files.append(os.path.join(self._inputFolder, filename))
        return sorted(input_files)

    def fitApplyScoreTransformers(self):
        if self.args.load_score_transformers:
            self.m_tsfs = []
            for fold in range(4):
                path = os.path.join(self.args.load_score_transformers, f"DNN_tsf_{self._region}_{fold}.pkl")
                with open(path, "rb") as handle:
                    self.m_tsfs.append(pickle.load(handle, encoding="latin1"))
            print(f"DNN INFO: Loaded four score transformers from {self.args.load_score_transformers}", flush=True)
            return

        signal_categories = list(self.train_signal)
        if not signal_categories:
            raise RuntimeError("No train_signal categories are configured for apply-time score_t fitting")

        scores_by_fold = [[] for _ in range(4)]
        weights_by_fold = [[] for _ in range(4)]
        used_files = []
        year_label = self._apply_source_year_label if self._apply_source_year_label is not None else self._year

        print(
            "DNN INFO: Fitting apply-time score transformers "
            f"for year={year_label} using signal categories: {signal_categories}",
            flush=True,
        )

        for category in signal_categories:
            input_files = self._find_input_files(category)
            if not input_files:
                logging.warning("No signal files matched for score_t fitting: category=%s input=%s", category, self._inputFolder)
                continue
            used_files.extend(input_files)
            for path in input_files:
                print(f"DNN INFO: score_t signal input: category={category} file={path}", flush=True)
            self._collect_signal_scores(category, input_files, scores_by_fold, weights_by_fold)

        if not used_files:
            raise RuntimeError(f"No ggH/VBF signal input files found for apply-time score_t fitting in {self._inputFolder}")

        self.m_tsfs = []
        for fold_idx in range(4):
            if not scores_by_fold[fold_idx]:
                raise RuntimeError(f"No usable test-signal events found for apply-time score_t fitting: year={year_label} fold={fold_idx}")

            fit_scores = np.concatenate(scores_by_fold[fold_idx]).astype(np.float32, copy=False)
            fit_weights = None
            weight_used = False
            if weights_by_fold[fold_idx]:
                candidate_weights = np.concatenate(weights_by_fold[fold_idx]).astype(np.float64, copy=False)
                candidate_weights = np.clip(candidate_weights, 0.0, None)
                if np.isfinite(candidate_weights).all() and np.sum(candidate_weights) > 0:
                    fit_weights = candidate_weights
                    weight_used = True

            transformer = WeightedQuantileTransformer(
                n_quantiles=1000,
                output_distribution="uniform",
                subsample=1000000000,
                random_state=0,
            )
            transformer.fit(fit_scores.reshape(-1, 1), sample_weight=fit_weights)
            self.m_tsfs.append(transformer)
            print(
                "DNN INFO: fitted apply-time score transformer "
                f"year={year_label} fold={fold_idx} events={len(fit_scores)} weights_used={weight_used}",
                flush=True,
            )

        if self.args.save_score_transformers:
            os.makedirs(self.args.save_score_transformers, exist_ok=True)
            for fold, transformer in enumerate(self.m_tsfs):
                path = os.path.join(self.args.save_score_transformers, f"DNN_tsf_{self._region}_{fold}.pkl")
                with open(path, "wb") as handle:
                    pickle.dump(transformer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"DNN INFO: Saved four score transformers to {self.args.save_score_transformers}", flush=True)

    def _collect_signal_scores(self, category, input_files, scores_by_fold, weights_by_fold):
        for file_idx, filename in enumerate(input_files):
            try:
                root_file = uproot.open(filename)
            except Exception as e:
                logging.error("Failed to open signal file %s for score_t fitting: %s", filename, e)
                continue

            tree_name = self._resolve_tree_name(root_file)
            tree = root_file[tree_name]
            actual_to_target = self._resolve_branch_mapping(tree)
            selected_branches = list(actual_to_target.keys())
            if not selected_branches:
                logging.warning("No usable signal branches found in %s", filename)
                root_file.close()
                continue

            chunk_count = 0
            for data_raw in tree.iterate(selected_branches, library="pd", step_size=self._chunksize):
                data_eval = data_raw[selected_branches].rename(columns=actual_to_target)
                data_eval = self._inject_source_year(data_eval)
                data_eval = self.preselect(data_eval)
                data_eval = self._apply_2223_jet_patch(data_eval)
                data_eval = self._apply_ptdecor(data_eval)
                if data_eval.empty:
                    chunk_count += 1
                    if self._max_chunks > 0 and chunk_count >= self._max_chunks:
                        break
                    continue

                data_eval, _ = self._drop_invalid_feature_rows(data_eval, data_eval.copy(), category, chunk_count + 1)
                if data_eval.empty:
                    chunk_count += 1
                    if self._max_chunks > 0 and chunk_count >= self._max_chunks:
                        break
                    continue

                test_fold = (data_eval[self.randomIndex] % 314159 % 4).astype(int)
                weights = None
                if self.weight in data_eval.columns:
                    weights = data_eval[self.weight].to_numpy(dtype=np.float64, copy=False)

                for fold_idx in range(4):
                    mask = (test_fold == fold_idx).to_numpy()
                    if not np.any(mask):
                        continue
                    transformed = self._transform_features(data_eval.loc[mask, self.train_variables], fold_idx)
                    scores = self._predict_scores(transformed, fold_idx)
                    if len(scores) == 0:
                        continue
                    scores_by_fold[fold_idx].append(scores)
                    if weights is not None:
                        weights_by_fold[fold_idx].append(weights[mask])

                chunk_count += 1
                if chunk_count % 4 == 0:
                    gc.collect()
                if self._max_chunks > 0 and chunk_count >= self._max_chunks:
                    break

            root_file.close()
            gc.collect()
            print(
                f"DNN INFO: Collected score_t signal file {file_idx + 1}/{len(input_files)} "
                f"for {category}, Memory: {get_memory_usage():.2f} GB",
                flush=True,
            )

    def applyDNN(self, category):
        if not os.path.isdir(self._inputFolder):
            logging.warning("Skipping sample %s: input folder does not exist: %s", category, self._inputFolder)
            return

        input_files = self._find_input_files(category)

        if not input_files:
            logging.warning("Skipping sample %s: no input files matched in %s", category, self._inputFolder)
            return

        output_container = os.path.join(self._outputFolder, self._region)
        output_path = os.path.join(output_container, f"{category}.root")
        os.makedirs(output_container, exist_ok=True)
        if os.path.isfile(output_path):
            os.remove(output_path)

        print("-------------------------------------------------")
        for path in input_files:
            print("DNN INFO: Including sample:", path)

        self._process_files(sorted(input_files), output_path, category)

    def _resolve_tree_name(self, file):
        if self._inputTree in file:
            return self._inputTree
        tree_names = [name.split(";")[0] for name in file.keys()]
        if len(tree_names) == 1:
            return tree_names[0]
        raise KeyError(f'Tree "{self._inputTree}" not found. Available trees: {tree_names}')

    def _resolve_branch_mapping(self, tree):
        available_branches = set(tree.keys())
        actual_to_target = {}
        for branch in self._requested_branches:
            actual_branch = self._branch_aliases.get(branch, branch)
            if actual_branch in available_branches:
                actual_to_target[actual_branch] = branch

        missing = sorted(set(self._requested_branches) - set(actual_to_target.values()))
        if missing:
            logging.warning("Missing branches in %s: %s", tree.object_path, missing)
        return actual_to_target

    def _warn_2223_jet_patch_missing_once(self, missing):
        key = tuple(sorted(missing))
        if key in self._warned_2223_jet_patch_missing:
            return
        self._warned_2223_jet_patch_missing.add(key)
        logging.warning("Skipping part of 22/23 jet patch because branches are missing: %s", ", ".join(key))

    def _apply_2223_jet_patch(self, data_eval, data_out=None):
        if not self._is_2223_jet_patch_enabled() or data_eval.empty:
            return data_eval

        required = {"njets", "jet1_pt", "jet1_eta", "jet2_pt", "jet2_eta"}
        missing = sorted(required - set(data_eval.columns))
        if missing:
            self._warn_2223_jet_patch_missing_once(missing)
            return data_eval

        data_eval = data_eval.copy()

        def frames():
            yield data_eval
            if data_out is not None:
                yield data_out

        def set_value(column, mask, value):
            actual_column = self._branch_aliases.get(column, column)
            for frame in frames():
                if column in frame.columns:
                    frame.loc[mask, column] = value
                elif actual_column in frame.columns:
                    frame.loc[mask, actual_column] = value

        def set_array(column, mask, values):
            actual_column = self._branch_aliases.get(column, column)
            masked_values = values[mask.to_numpy()]
            for frame in frames():
                if column in frame.columns:
                    frame.loc[mask, column] = masked_values
                elif actual_column in frame.columns:
                    frame.loc[mask, actual_column] = masked_values

        def columns_with_prefix(prefix):
            columns = set(data_eval.columns)
            if data_out is not None:
                columns.update(data_out.columns)
            return sorted(column for column in columns if column.startswith(prefix))

        original_njets = data_eval["njets"].copy()
        jet1_bad = (
            (original_njets >= 1)
            & (data_eval["jet1_eta"].abs() > 3.0)
            & (data_eval["jet1_pt"] < 50.0)
        )
        jet2_bad = (
            (original_njets >= 2)
            & (data_eval["jet2_eta"].abs() > 3.0)
            & (data_eval["jet2_pt"] < 50.0)
        )
        patch_mask = jet1_bad | jet2_bad
        if not patch_mask.any():
            return data_eval

        jet1_only_one = jet1_bad & ~jet2_bad & (original_njets == 1)
        jet1_only_ge2 = jet1_bad & ~jet2_bad & (original_njets >= 2)
        jet2_only = ~jet1_bad & jet2_bad
        both_bad = jet1_bad & jet2_bad

        for frame in frames():
            for jet1_column in sorted(column for column in frame.columns if column.startswith("jet1")):
                jet2_column = "jet2" + jet1_column[len("jet1"):]
                if jet2_column in frame.columns:
                    frame.loc[jet1_only_ge2, jet1_column] = frame.loc[jet1_only_ge2, jet2_column].to_numpy()
        for column in columns_with_prefix("jet1"):
            set_value(column, jet1_only_one | both_bad, JET_PATCH_SENTINEL)
        for column in columns_with_prefix("jet2"):
            set_value(column, jet1_only_ge2 | jet2_only | both_bad, JET_PATCH_SENTINEL)

        zero_jet_mask = jet1_only_one | both_bad
        one_jet_mask = jet1_only_ge2 | jet2_only
        set_value("njets", zero_jet_mask, 0)
        set_value("njets", one_jet_mask, 1)

        fewer_than_two = zero_jet_mask | one_jet_mask
        for column in columns_with_prefix("dijet"):
            set_value(column, fewer_than_two, JET_PATCH_SENTINEL)
        for column in ("delta_eta_jj", "delta_phi_jj", "z_star", "R_pT", "R_pt"):
            set_value(column, fewer_than_two, JET_PATCH_SENTINEL)

        for column in ("min_delta_eta_dimu_jets", "min_delta_phi_dimu_jets"):
            set_value(column, zero_jet_mask, JET_PATCH_SENTINEL)

        eta_column = "diMufsr_kit_BSC_eta" if "diMufsr_kit_BSC_eta" in data_eval.columns else "diMufsr_rc_BSC_eta"
        if eta_column in data_eval.columns and "min_delta_eta_dimu_jets" in data_eval.columns:
            min_delta_eta = (data_eval[eta_column] - data_eval["jet1_eta"]).abs().to_numpy(copy=False)
            set_array("min_delta_eta_dimu_jets", one_jet_mask, min_delta_eta)
        elif "min_delta_eta_dimu_jets" in data_eval.columns:
            self._warn_2223_jet_patch_missing_once([eta_column])

        if {"diMufsr_kit_BSC_phi", "jet1_phi", "min_delta_phi_dimu_jets"}.issubset(data_eval.columns):
            delta_phi = (data_eval["diMufsr_kit_BSC_phi"] - data_eval["jet1_phi"]).abs().to_numpy(copy=False)
            delta_phi = np.where(delta_phi > np.pi, 2.0 * np.pi - delta_phi, delta_phi)
            set_array("min_delta_phi_dimu_jets", one_jet_mask, delta_phi)
        elif "min_delta_phi_dimu_jets" in data_eval.columns:
            self._warn_2223_jet_patch_missing_once(["diMufsr_kit_BSC_phi", "jet1_phi"])

        return data_eval

    def _apply_fix_mass(self, data_eval, data_out):
        mass_branch = "diMufsr_kit_BSC_mass"
        if mass_branch not in data_eval.columns:
            return
        mask = (
            ((data_eval[mass_branch] > 110) & (data_eval[mass_branch] < 115)) |
            ((data_eval[mass_branch] > 135) & (data_eval[mass_branch] < 150))
        )
        data_eval.loc[mask, mass_branch] = 125
        actual_mass_branch = self._branch_aliases.get(mass_branch, mass_branch)
        if actual_mass_branch in data_out.columns:
            data_out.loc[mask, actual_mass_branch] = 125

    def _write_canonical_mass(self, data_eval, data_out):
        mass_branch = "diMufsr_kit_BSC_mass"
        if mass_branch in data_eval.columns:
            data_out[mass_branch] = data_eval[mass_branch].to_numpy()

    def _inject_source_year(self, data_eval):
        if "source_year" not in self.train_variables:
            return data_eval

        if self._apply_source_year_label is not None:
            if "source_year" in data_eval.columns:
                data_eval["source_year_from_file"] = data_eval["source_year"]
            data_eval["source_year"] = np.int32(self._encode_source_year(self._apply_source_year_label))
            return data_eval

        if "source_year" in data_eval.columns:
            data_eval["source_year_from_file"] = data_eval["source_year"]
            data_eval["source_year"] = data_eval["source_year"].apply(self._encode_or_passthrough_source_year).astype(np.int32)
            return data_eval

        raise KeyError("source_year is required by the model but could not be resolved from config, --year, or input branches.")

    def _apply_ptdecor(self, data_eval):
        if not self._ptdecor or data_eval.empty:
            return data_eval
        mass_var = self._mass_variable
        scaled_vars = [name for name in self._scaled_by_mass_variables if name in self.train_variables and name in data_eval.columns]
        if not scaled_vars:
            return data_eval
        if mass_var not in data_eval.columns:
            raise KeyError(f"ptdecor requires '{mass_var}' in the loaded branches")
        data_eval = data_eval.copy()
        mass = data_eval[mass_var].to_numpy(dtype=np.float32, copy=False)
        safe_mass = np.where(np.abs(mass) > 1e-6, mass, np.nan).astype(np.float32, copy=False)
        for var_name in scaled_vars:
            values = data_eval[var_name].to_numpy(dtype=np.float32, copy=True)
            if preserves_negative_ptdecor_sentinels(var_name):
                sentinel_mask = values < 0
                values[~sentinel_mask] = values[~sentinel_mask] / safe_mass[~sentinel_mask]
            else:
                values = values / safe_mass
            data_eval[var_name] = values
        invalid_mask = ~np.isfinite(data_eval[scaled_vars].to_numpy(dtype=np.float32, copy=False)).all(axis=1)
        dropped = int(np.count_nonzero(invalid_mask))
        if dropped:
            logging.warning("Dropped %d rows with invalid ptdecor values during DNN apply", dropped)
            data_eval = data_eval.loc[~invalid_mask].copy()
        return data_eval

    def _drop_invalid_feature_rows(self, data_eval, data_out, category="", chunk_idx=0):
        features = data_eval[self.train_variables]
        original = features.to_numpy(copy=False)
        values = features.to_numpy(dtype=np.float32, copy=True)
        invalid_mask = (~np.isfinite(original)) | (~np.isfinite(values))
        if not np.any(invalid_mask):
            return data_eval, data_out

        bad_rows = np.any(invalid_mask, axis=1)
        bad_columns = np.where(np.any(invalid_mask, axis=0))[0]
        logging.warning(
            "Dropped %d rows with invalid DNN feature values before scaling: category=%s chunk=%s",
            int(np.count_nonzero(bad_rows)),
            category,
            chunk_idx,
        )
        for column_idx in bad_columns:
            logging.warning(
                "Invalid feature summary before drop: name=%s count=%d",
                self.train_variables[column_idx],
                int(np.count_nonzero(invalid_mask[:, column_idx])),
            )

        keep_rows = ~bad_rows
        return data_eval.loc[keep_rows].copy(), data_out.loc[keep_rows].copy()

    def _transform_features(self, features, fold_idx):
        scaler = self.m_scalers[fold_idx]
        values = features.to_numpy(dtype=np.float32, copy=True)
        if fold_idx < len(self._missing_jet_impute_values):
            impute_values = self._missing_jet_impute_values[fold_idx]
            for index, name in enumerate(self.train_variables):
                if name not in impute_values or not is_missing_jet_impute_feature(name):
                    continue
                mask = values[:, index] < self._missing_jet_sentinel_max
                if np.any(mask):
                    values[mask, index] = impute_values[name]
        return scaler.transform(values).astype(np.float32)

    def _predict_scores(self, transformed, fold_idx):
        model = self.m_models[fold_idx]
        if len(transformed) == 0:
            return np.empty(0, dtype=np.float32)
        if self._predict_batch_size > 0:
            score_batches = []
            for start in range(0, len(transformed), self._predict_batch_size):
                stop = start + self._predict_batch_size
                pred = np.asarray(model(transformed[start:stop], training=False).numpy())
                if pred.ndim == 2 and pred.shape[1] >= 2:
                    score_batches.append(pred[:, 0].astype(np.float32, copy=False))
                else:
                    score_batches.append(pred.reshape(-1).astype(np.float32, copy=False))
            return np.concatenate(score_batches)
        pred = np.asarray(model(transformed, training=False).numpy())
        if pred.ndim == 2 and pred.shape[1] >= 2:
            scores = pred[:, 0]
        else:
            scores = pred.reshape(-1)
        return scores.astype(np.float32, copy=False)

    def _predict_domain(self, transformed, fold_idx):
        model = self.m_models[fold_idx]
        domain = model.predict_domain(transformed, training=False)
        return tf.reshape(domain, (-1,)).numpy().astype(np.float32, copy=False)

    def _predict_score_domain(self, transformed, fold_idx):
        model = self.m_models[fold_idx]
        if len(transformed) == 0:
            empty = np.empty(0, dtype=np.float32)
            return empty, empty
        if self._predict_batch_size <= 0:
            pred = np.asarray(model(transformed, training=False).numpy())
            if pred.ndim == 2 and pred.shape[1] >= 2:
                return (
                    pred[:, 0].astype(np.float32, copy=False),
                    pred[:, 1].astype(np.float32, copy=False),
                )
            return self._predict_scores(transformed, fold_idx), self._predict_domain(transformed, fold_idx)

        scores = []
        domains = []
        for start in range(0, len(transformed), self._predict_batch_size):
            stop = start + self._predict_batch_size
            batch_np = transformed[start:stop]
            pred = np.asarray(model(batch_np, training=False).numpy())
            if pred.ndim == 2 and pred.shape[1] >= 2:
                scores.append(pred[:, 0].astype(np.float32, copy=False))
                domains.append(pred[:, 1].astype(np.float32, copy=False))
                continue
            scores.append(pred.reshape(-1).astype(np.float32, copy=False))
            batch = tf.convert_to_tensor(batch_np, dtype=tf.float32)
            domains.append(tf.reshape(model.predict_domain(batch, training=False), (-1,)).numpy().astype(np.float32, copy=False))
        return np.concatenate(scores), np.concatenate(domains)

    def _domain_label(self, category):
        return np.float32(1.0 if category == "data" else 0.0)

    def _domain_bce(self, domain_pred, domain_label):
        pred = np.clip(domain_pred.astype(np.float32, copy=False), 1e-7, 1.0 - 1e-7)
        return (-(domain_label * np.log(pred) + (1.0 - domain_label) * np.log(1.0 - pred))).astype(np.float32, copy=False)

    def _domain_confusion_loss(self, domain_pred):
        pred = domain_pred.astype(np.float32, copy=False)
        return (4.0 * np.square(pred - 0.5)).astype(np.float32, copy=False)

    def _write_temp_tree(self, path, data):
        arrays = dataframe_to_root_arrays(data)
        with uproot.recreate(path) as out_file:
            tree = out_file.mktree("test", root_branch_types(arrays))
            tree.extend(arrays)

    def _profile_log(self, category, file_idx, chunk_idx, rows, timings):
        if not self._profile_apply:
            return
        timing_text = " ".join(f"{name}={value:.3f}s" for name, value in timings)
        print(
            f"DNN PROFILE_APPLY: category={category} file={file_idx + 1} "
            f"chunk={chunk_idx} rows={rows} memory_gb={get_memory_usage():.2f} {timing_text}",
            flush=True,
        )

    def _process_files(self, input_files, output_path, category):
        if self.args.output_profile == "fit":
            return self._process_fit_files(input_files, output_path, category)
        temp_files = []
        print(f"DNN INFO: Initial memory usage: {get_memory_usage():.2f} GB")

        for file_idx, filename in enumerate(
            tqdm(input_files, desc=f"DNN INFO: Applying DNNs to {category}", bar_format="{desc}: {percentage:3.0f}%|{bar:20}{r_bar}")
        ):
            try:
                root_file = uproot.open(filename)
            except Exception as e:
                logging.error("Failed to open file %s: %s", filename, e)
                continue

            tree_name = self._resolve_tree_name(root_file)
            tree = root_file[tree_name]
            actual_to_target = self._resolve_branch_mapping(tree)
            variations = (systematic_plan(tree.keys(), self.train_variables, self._branch_aliases, self.weight,
                                       allow_nominal_only=category in self.train_mc_background)
                          if self.args.systematics and category != "data" else {})
            selected_branches = list(actual_to_target.keys())
            if not selected_branches:
                logging.warning("No usable branches found in %s", filename)
                root_file.close()
                continue

            chunk_count = 0
            read_start = time.perf_counter()
            for data_raw in tree.iterate(list(tree.keys()), library="pd", step_size=self._chunksize):
                read_s = time.perf_counter() - read_start
                chunk_total_start = time.perf_counter()

                select_start = time.perf_counter()
                data_eval = data_raw[selected_branches].rename(columns=actual_to_target)
                data_eval = self._inject_source_year(data_eval)
                data_eval = self.preselect(data_eval)
                if data_eval.empty:
                    select_s = time.perf_counter() - select_start
                    chunk_count += 1
                    self._profile_log(
                        category,
                        file_idx,
                        chunk_count,
                        0,
                        [
                            ("read", read_s),
                            ("select_preprocess", select_s),
                            ("chunk_total", time.perf_counter() - chunk_total_start),
                        ],
                    )
                    if self._max_chunks > 0 and chunk_count >= self._max_chunks:
                        break
                    read_start = time.perf_counter()
                    continue

                data_out = data_raw.loc[data_eval.index].copy()
                data_eval = data_eval.copy()
                if "source_year" in data_eval.columns:
                    data_out["source_year"] = data_eval["source_year"].to_numpy()

                data_eval = self._apply_2223_jet_patch(data_eval, data_out)
                data_eval = self._apply_ptdecor(data_eval)
                if not data_out.index.equals(data_eval.index):
                    data_out = data_out.loc[data_eval.index].copy()
                select_s = time.perf_counter() - select_start
                if data_eval.empty:
                    chunk_count += 1
                    self._profile_log(
                        category,
                        file_idx,
                        chunk_count,
                        0,
                        [
                            ("read", read_s),
                            ("select_preprocess", select_s),
                            ("chunk_total", time.perf_counter() - chunk_total_start),
                        ],
                    )
                    if self._max_chunks > 0 and chunk_count >= self._max_chunks:
                        break
                    read_start = time.perf_counter()
                    continue

                assemble_start = time.perf_counter()

                if self._FixSBH125:
                    self._apply_fix_mass(data_eval, data_out)

                self._write_canonical_mass(data_eval, data_out)

                data_eval, data_out = self._drop_invalid_feature_rows(data_eval, data_out, category, chunk_count + 1)
                if data_eval.empty:
                    chunk_count += 1
                    self._profile_log(
                        category,
                        file_idx,
                        chunk_count,
                        0,
                        [
                            ("read", read_s),
                            ("select_preprocess", select_s),
                            ("assemble", time.perf_counter() - assemble_start),
                            ("chunk_total", time.perf_counter() - chunk_total_start),
                        ],
                    )
                    if self._max_chunks > 0 and chunk_count >= self._max_chunks:
                        break
                    read_start = time.perf_counter()
                    continue

                # Temporary 2025 MC normalization patch; keep data unchanged.
                if category != "data" and self._event_weight_scale != 1.0 and self.weight in data_out.columns:
                    data_out[self.weight] = data_out[self.weight] * self._event_weight_scale

                test_fold = (data_eval[self.randomIndex] % 314159 % 4).astype(int)
                assemble_s = time.perf_counter() - assemble_start

                transform_s = 0.0
                score_domain_predict_s = 0.0
                score_transform_s = 0.0
                domain_label = self._domain_label(category)
                for fold_idx in range(4):
                    transform_start = time.perf_counter()
                    transformed = self._transform_features(data_eval[self.train_variables], fold_idx)
                    transform_s += time.perf_counter() - transform_start
                    predict_start = time.perf_counter()
                    scores, domain = self._predict_score_domain(transformed, fold_idx)
                    score_domain_predict_s += time.perf_counter() - predict_start
                    score_transform_start = time.perf_counter()
                    if len(scores) > 0:
                        scores_t = self.m_tsfs[fold_idx].transform(scores.reshape(-1, 1)).reshape(-1)
                    else:
                        scores_t = scores
                    score_transform_s += time.perf_counter() - score_transform_start
                    data_out[f"dnn_{fold_idx}_score"] = scores
                    data_out[f"dnn_{fold_idx}_score_t"] = scores_t
                    data_out[f"dnn_{fold_idx}_domain"] = domain
                    data_out[f"dnn_{fold_idx}_domain_bce"] = self._domain_bce(domain, domain_label)
                    data_out[f"dnn_{fold_idx}_domain_confusion_loss"] = self._domain_confusion_loss(domain)

                data_out[self.output_branch] = data_out["dnn_0_score"].copy()
                data_out[self.output_branch + "_t"] = data_out["dnn_0_score_t"].copy()
                data_out[self.output_branch + "_domain"] = data_out["dnn_0_domain"].copy()
                data_out[self.output_branch + "_domain_bce"] = data_out["dnn_0_domain_bce"].copy()
                data_out[self.output_branch + "_domain_confusion_loss"] = data_out["dnn_0_domain_confusion_loss"].copy()
                for fold_idx in range(4):
                    mask = test_fold == fold_idx
                    if mask.any():
                        data_out.loc[mask, self.output_branch] = data_out.loc[mask, f"dnn_{fold_idx}_score"]
                        data_out.loc[mask, self.output_branch + "_t"] = data_out.loc[mask, f"dnn_{fold_idx}_score_t"]
                        data_out.loc[mask, self.output_branch + "_domain"] = data_out.loc[mask, f"dnn_{fold_idx}_domain"]
                        data_out.loc[mask, self.output_branch + "_domain_bce"] = data_out.loc[mask, f"dnn_{fold_idx}_domain_bce"]
                        data_out.loc[mask, self.output_branch + "_domain_confusion_loss"] = data_out.loc[mask, f"dnn_{fold_idx}_domain_confusion_loss"]

                if "index" in data_out.columns:
                    data_out = data_out.drop(columns=["index"])

                if self.args.systematics:
                    compact = fit_output(self, data_raw.loc[data_out.index], actual_to_target, variations, category)
                    for column in compact:
                        data_out[column] = compact[column]

                if self._drop_output_branches:
                    drop_columns = [column for column in self._drop_output_branches if column in data_out.columns]
                    if drop_columns:
                        data_out = data_out.drop(columns=drop_columns)

                temp_file = tempfile.NamedTemporaryFile(suffix=".root", delete=False)
                temp_file.close()
                write_start = time.perf_counter()
                self._write_temp_tree(temp_file.name, data_out)
                write_s = time.perf_counter() - write_start
                temp_files.append(temp_file.name)

                chunk_count += 1
                self._profile_log(
                    category,
                    file_idx,
                    chunk_count,
                    len(data_out),
                    [
                        ("read", read_s),
                        ("select_preprocess", select_s),
                        ("assemble", assemble_s),
                        ("transform", transform_s),
                        ("score_domain_predict", score_domain_predict_s),
                        ("score_transform", score_transform_s),
                        ("temp_write", write_s),
                        ("chunk_total", time.perf_counter() - chunk_total_start),
                    ],
                )

                del data_raw, data_eval, data_out
                if chunk_count % 4 == 0:
                    gc.collect()
                if self._max_chunks > 0 and chunk_count >= self._max_chunks:
                    break
                read_start = time.perf_counter()

            root_file.close()
            gc.collect()
            print(f"DNN INFO: Processed file {file_idx + 1}/{len(input_files)}, Memory: {get_memory_usage():.2f} GB, Temp files: {len(temp_files)}")

        self._combine_temp_files(temp_files, output_path)

        if self.args.systematics:
            with open(output_path + ".systematics.json", "w") as stream:
                json.dump({"inputs": input_files, "systematics": mapping_record(variations)}, stream, indent=2)

        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except OSError:
                pass

        print(f"DNN INFO: Final memory usage: {get_memory_usage():.2f} GB")

    def _process_fit_files(self, input_files, output_path, category):
        temp_files = []
        required = set(self.train_variables) | {
            "run", "lumi", self.randomIndex, self.weight, self._mass_variable,
        }
        required.discard("source_year")
        background = category in self.train_mc_background
        systematics = self.args.systematics and category != "data"
        signal_migrations = self.args.include_jet_migrations and category != "data" and not background
        summary = {"category": category, "year": self._apply_source_year_label,
                   "output": output_path, "entries": 0, "sumw": {}, "inputs": [],
                   "selection": self.preselections, "systematics": {}, "profile": "fit"}
        expected_columns = None
        try:
            for filename in input_files:
                with uproot.open(filename) as root_file:
                    tree = root_file[self._resolve_tree_name(root_file)]
                    mapping = self._resolve_branch_mapping(tree)
                    missing = required - set(mapping.values())
                    if missing:
                        raise KeyError(f"Missing required fit inputs in {filename}: {sorted(missing)}")
                    variations = (systematic_plan(tree.keys(), self.train_variables, self._branch_aliases, self.weight,
                                       allow_nominal_only=background, include_migrations=signal_migrations)
                                  if systematics else {})
                    migration_branches = {v["category"] for v in variations.values() if "category" in v}
                    if signal_migrations and not migration_branches:
                        raise ValueError(f"No shifted category branches for {category}")
                    record = mapping_record(variations)
                    if summary["inputs"] and record != summary["systematics"]:
                        raise ValueError(f"Systematic schema changed across files for {category}")
                    summary["systematics"] = record
                    summary["systematics_status"] = ("data_nominal" if category == "data" else
                        "not_requested" if not self.args.systematics else
                        "available_variations" if variations else "nominal_only_background_input")
                    summary["inputs"].append(filename)
                    branches = set(mapping) | migration_branches
                    for variation in variations.values():
                        branches.add(variation["weight"])
                        branches.update(variation["features"].values())
                    for chunk_idx, raw in enumerate(tree.iterate(sorted(branches), library="pd", step_size=self._chunksize), 1):
                        canonical = self._inject_source_year(raw[list(mapping)].rename(columns=mapping))
                        if signal_migrations:
                            union, nominal_mask = select_jet_migrations(self, canonical, raw, variations)
                            output = fit_output(self, raw.loc[union], mapping, variations, category,
                                                nominal_mask=nominal_mask)
                        else:
                            selected = self.preselect(canonical)
                            output = fit_output(self, raw.loc[selected.index], mapping, variations, category)
                        if expected_columns is None:
                            expected_columns = list(output)
                        elif list(output) != expected_columns:
                            raise ValueError(f"Output schema changed for {category}")
                        summary["entries"] += len(output)
                        for column in output:
                            if column.startswith("eventWeight"):
                                summary["sumw"][column] = summary["sumw"].get(column, 0.0) + float(output[column].sum())
                        # Write an empty tree as well when no nominal events pass.
                        temp_file = tempfile.NamedTemporaryFile(suffix=".root", delete=False)
                        temp_file.close()
                        temp_files.append(temp_file.name)
                        self._write_temp_tree(temp_file.name, output)
                        print(f"DNN FIT: category={category} chunk={chunk_idx} selected={len(output)} variations={len(variations)}", flush=True)
                        if self._max_chunks > 0 and chunk_idx >= self._max_chunks:
                            break
            if not temp_files:
                raise RuntimeError(f"No input chunks found for {category}")
            self._combine_temp_files(temp_files, output_path)
            # EOS CLI publication can leave the FUSE mount's negative cache stale.
            # Stage the published object through xrdcp before checking its size and tree.
            with open_published_root(output_path) as root_file:
                if root_file["test"].num_entries != summary["entries"]:
                    raise RuntimeError(f"Published entry count mismatch for {category}")
                summary["bytes"] = root_file.file.source.num_bytes
            summary["branches"] = expected_columns
            with open(output_path + ".summary.json", "w") as stream:
                json.dump(summary, stream, indent=2)
        finally:
            for path in temp_files:
                if os.path.exists(path):
                    os.remove(path)

    def _combine_temp_files(self, temp_files, output_path):
        if not temp_files:
            logging.warning("No temporary files were produced for %s", output_path)
            return

        if len(temp_files) == 1:
            combine_start = time.perf_counter()
            publish_output_file(temp_files[0], output_path)
            combine_s = time.perf_counter() - combine_start
            print(f"DNN INFO: Published single DNN chunk to {output_path}")
            if self._profile_apply:
                print(f"DNN PROFILE_APPLY: combine method=single_copy files=1 seconds={combine_s:.3f}", flush=True)
            return

        if HADD_AVAILABLE:
            with tempfile.TemporaryDirectory(prefix="dnn_apply_hadd_") as work_dir:
                local_output = os.path.join(work_dir, os.path.basename(output_path))
                command = ["hadd", "-f", local_output] + temp_files
                combine_start = time.perf_counter()
                result = subprocess.run(command, capture_output=True, text=True)
                combine_s = time.perf_counter() - combine_start
                if result.returncode == 0:
                    publish_output_file(local_output, output_path)
                    print(f"DNN INFO: Successfully combined {len(temp_files)} files using hadd")
                    if self._profile_apply:
                        print(f"DNN PROFILE_APPLY: combine method=hadd files={len(temp_files)} seconds={combine_s:.3f}", flush=True)
                    return
                raise RuntimeError(
                    f"hadd failed while combining {len(temp_files)} DNN chunks into {output_path}\n"
                    f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                )

        raise RuntimeError("hadd is required to combine DNN apply chunks, but it is not available in PATH")


def main():
    args = getArgs()
    handler = ApplyDNNHandler(args)
    handler.setOutputFolder(args.outputFolder)
    handler.loadModels()
    handler.loadScalers()
    handler.fitApplyScoreTransformers()

    sample_list = handler.load_sample_list()
    for category in sample_list:
        if args.category and category not in args.category:
            continue
        handler.applyDNN(category)


if __name__ == "__main__":
    main()
