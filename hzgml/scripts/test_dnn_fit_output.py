"""Regression tests for fixed-selection systematic scoring and compact ROOT I/O."""

import tempfile
import json
import sys
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import uproot

from dnn_fit_output import FIT_COLUMNS, fit_output, select_jet_migrations, systematic_plan, systematic_suffix


class AffineTransformer:
    def __init__(self, fold):
        self.fold = fold

    def transform(self, values):
        return values * 0.5 + self.fold * 0.01


class FitOutputTests(unittest.TestCase):
    def setUp(self):
        # Event 314159 checks the modulo order; non-contiguous index checks alignment.
        self.raw = pd.DataFrame({
            "run": [1] * 5, "lumi": [2] * 5, "event": [0, 1, 2, 3, 314159],
            "source_year": [4] * 5, "diMufsr_kit_BSC_mass": [125.] * 5,
            "eventWeight": [1., -2., 3., 0., 5.], "jet1_pt": [10., 20., 30., 40., 50.],
            "jet1_pt__JERUp": [11., 22., 33., 44., 55.],
            "eventWeight__JERUp": [1.1, -2.2, 3.3, 0., 5.5],
            "eventWeight__PDFdown": [.9, -1.8, 2.7, 0., 4.5],
            "cate_index": [4] * 5, "cate_index__JERUp": [5] * 5,
            "unused": [999] * 5,
        }, index=[2, 4, 7, 9, 13])
        self.mapping = {c: c for c in self.raw if "__" not in c and c != "unused"}
        self.handler = SimpleNamespace(
            train_variables=["jet1_pt", "source_year"], randomIndex="event", weight="eventWeight",
            _event_weight_scale=2., _FixSBH125=False, m_tsfs=[AffineTransformer(i) for i in range(4)],
            _inject_source_year=lambda data: data,
            _apply_2223_jet_patch=lambda data: data.copy(),
            _apply_ptdecor=lambda data: data,
            _transform_features=lambda data, fold: data.to_numpy(),
            _predict_scores=lambda values, fold: values[:, 0] / 100 + fold * .1,
            _write_canonical_mass=lambda data, out: out.__setitem__("diMufsr_kit_BSC_mass", data["diMufsr_kit_BSC_mass"].to_numpy()),
        )

    def plan(self, raw=None):
        return systematic_plan((self.raw if raw is None else raw).columns,
                               self.handler.train_variables, {})

    def test_shifted_scores_weights_and_fixed_event_membership(self):
        result = fit_output(self.handler, self.raw, self.mapping, self.plan(), "signal")
        folds = np.array([0, 1, 2, 3, 0])
        # Independently evaluate the affine models and transformers.
        np.testing.assert_allclose(result.score_t, self.raw.jet1_pt / 200 + folds * .06)
        np.testing.assert_allclose(result.score_t_jer_up, self.raw.jet1_pt__JERUp / 200 + folds * .06)
        np.testing.assert_array_equal(result.score_t_pdf_down, result.score_t)
        np.testing.assert_array_equal(result.dnn_score_t, result.score_t)
        np.testing.assert_array_equal(result.eventWeight, self.raw.eventWeight * 2)
        np.testing.assert_array_equal(result.eventWeight_jer_up, self.raw.eventWeight__JERUp * 2)
        np.testing.assert_array_equal(result.eventWeight_pdf_down, self.raw.eventWeight__PDFdown * 2)
        np.testing.assert_array_equal(result.event, self.raw.event)
        self.assertEqual(list(result.index), list(self.raw.index))
        self.assertEqual(set(result), set(FIT_COLUMNS + ["score_t_jer_up", "eventWeight_jer_up", "score_t_pdf_down", "eventWeight_pdf_down"]))

    def test_data_has_nominal_only_and_no_mc_normalization(self):
        result = fit_output(self.handler, self.raw, self.mapping, {}, "data")
        self.assertEqual(list(result), FIT_COLUMNS)
        np.testing.assert_array_equal(result.eventWeight, self.raw.eventWeight)

    def test_jet_variation_includes_migrations_in_both_directions(self):
        raw = self.raw.assign(cate_index=[4, 3, 4, 3, 4], cate_index__JERUp=[3, 4, 4, 3, 4])
        raw.iloc[4, raw.columns.get_loc("diMufsr_kit_BSC_mass")] = 155.
        self.handler.preselect = lambda d: d[(d.cate_index == 4) & d.diMufsr_kit_BSC_mass.between(110, 150)]
        plan = systematic_plan(raw.columns, self.handler.train_variables, {}, include_migrations=True)
        selected, nominal = select_jet_migrations(self.handler, raw, raw, plan)
        result = fit_output(self.handler, raw.loc[selected], self.mapping, plan, "signal", nominal_mask=nominal)
        np.testing.assert_array_equal(result.event, [0, 1, 2])
        np.testing.assert_allclose(result.eventWeight, [2., 0., 6.])
        np.testing.assert_allclose(result.eventWeight_jer_up, [0., -4.4, 6.6])
        np.testing.assert_allclose(result.eventWeight_pdf_down, [1.8, 0., 5.4])
        np.testing.assert_array_equal(result.score_t_pdf_down, result.score_t)
        with self.assertRaisesRegex(KeyError, "Missing systematic category"):
            systematic_plan(raw.drop(columns="cate_index__JERUp").columns, self.handler.train_variables, {},
                            include_migrations=True)

    def test_missing_shifted_input_or_weight_fails(self):
        for column in ["jet1_pt__JERUp", "eventWeight__JERUp"]:
            with self.subTest(column=column), self.assertRaises(KeyError):
                self.plan(self.raw.drop(columns=column))
        with self.assertRaises(ValueError):
            systematic_plan(["eventWeight"], ["jet1_pt"], {})

    def test_names_preserve_years_and_reject_collisions(self):
        self.assertEqual(systematic_suffix("jesAbsolute_2024Up"), "jesabsolute_2024_up")
        self.assertEqual(systematic_suffix("MuonEffdown"), "muoneff_down")
        with self.assertRaises(ValueError):
            systematic_plan(["eventWeight__JERUp", "eventWeight__jerup"], [], {})

    def test_nominal_background_requires_opt_in_and_partial_systematics_still_fail(self):
        nominal = [c for c in self.raw if "__" not in c]
        with self.assertRaisesRegex(ValueError, "no systematic branches"):
            systematic_plan(nominal, self.handler.train_variables, {})
        self.assertEqual(systematic_plan(nominal, self.handler.train_variables, {}, allow_nominal_only=True), {})
        with self.assertRaisesRegex(KeyError, "Missing systematic feature"):
            systematic_plan(nominal + ["eventWeight__JERUp"], self.handler.train_variables, {}, allow_nominal_only=True)

    def test_aliases(self):
        plan = systematic_plan(["pt__JERUp", "w__JERUp"], ["jet1_pt"],
                               {"jet1_pt": "pt", "eventWeight": "w"})
        self.assertEqual(plan["JERUp"]["features"], {"jet1_pt": "pt__JERUp"})

    def test_nonfinite_inputs_weights_and_scores_fail(self):
        for column in ["jet1_pt", "jet1_pt__JERUp", "eventWeight__PDFdown"]:
            bad = self.raw.copy()
            bad.loc[2, column] = np.nan
            with self.subTest(column=column), self.assertRaises(ValueError):
                fit_output(self.handler, bad, self.mapping, self.plan(), "signal")
        self.handler._predict_scores = lambda values, fold: np.full(len(values), np.nan)
        with self.assertRaises(ValueError):
            fit_output(self.handler, self.raw, self.mapping, {}, "signal")

    def test_preprocessing_cannot_drop_shifted_events(self):
        self.handler._apply_ptdecor = lambda data: data.iloc[1:]
        with self.assertRaisesRegex(ValueError, "dropped events"):
            fit_output(self.handler, self.raw, self.mapping, self.plan(), "signal")

    def test_empty_selection_and_root_categorization_interface(self):
        for raw in [self.raw, self.raw.iloc[:0]]:
            result = fit_output(self.handler, raw, self.mapping, self.plan(), "signal")
            arrays = {c: result[c].to_numpy() for c in result}
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "sample.root"
                with uproot.recreate(path) as root_file:
                    tree = root_file.mktree("test", {c: a.dtype for c, a in arrays.items()})
                    tree.extend(arrays)
                with uproot.open(path) as root_file:
                    tree = root_file["test"]
                    # Same branch interface as the existing categorization/fit readers.
                    read = tree.arrays(["dnn_score_t", "diMufsr_kit_BSC_mass", "eventWeight"], library="np")
                    self.assertEqual(len(read["dnn_score_t"]), len(raw))
                    self.assertEqual(set(tree.keys()), set(result))


class ApplyIntegrationTests(unittest.TestCase):
    def test_optional_jet_patch_preserves_training_inputs(self):
        from apply_dnn_ggH import ApplyDNNHandler
        handler = object.__new__(ApplyDNNHandler)
        handler._apply_source_year_label = "2022"
        handler.args = SimpleNamespace(disable_2223_jet_patch=True)
        handler._branch_aliases = {}
        handler._warned_2223_jet_patch_missing = set()
        data = pd.DataFrame({"njets": [2], "jet1_eta": [3.3], "jet1_pt": [40.],
                             "jet2_eta": [0.5], "jet2_pt": [80.], "dijet_mass": [500.]})
        pd.testing.assert_frame_equal(handler._apply_2223_jet_patch(data), data)
        handler.args.disable_2223_jet_patch = False
        patched = handler._apply_2223_jet_patch(data)
        self.assertEqual(patched["njets"].iloc[0], 1)
        self.assertEqual(patched["jet1_pt"].iloc[0], 80.)

    setUp = FitOutputTests.setUp

    def test_absolute_prepared_input_is_authoritative(self):
        import apply_dnn_ggH as apply
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "data.root").touch()
            prepared = folder / "prepared.root"
            prepared.touch()
            proxy = SimpleNamespace(_resolved_year_cfg={"sample_map": {"data": str(prepared.with_suffix(""))}}, _inputFolder=str(folder))
            self.assertEqual(apply.ApplyDNNHandler._find_input_files(proxy, "data"), [str(prepared)])
            prepared.unlink()
            with self.assertRaises(FileNotFoundError):
                apply.ApplyDNNHandler._find_input_files(proxy, "data")

    def test_multichunk_apply_and_legacy_nominal_agree(self):
        import apply_dnn_ggH as apply

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            raw = self.raw.copy()
            raw.loc[4, "cate_index"] = 5
            raw.loc[7, "diMufsr_kit_BSC_mass"] = 151.
            # The first chunk has no selected events; the second has a single fold.
            raw.loc[2, "cate_index"] = 5
            path = directory / "signal.root"
            arrays = {c: raw[c].to_numpy() for c in raw}
            with uproot.recreate(path) as root_file:
                tree = root_file.mktree("Events", {c: a.dtype for c, a in arrays.items()})
                tree.extend(arrays)
            config = directory / "config.py"
            config.write_text("CONFIG = " + repr({"ggH": {
                "train_variables": self.handler.train_variables, "randomIndex": "event", "weight": "eventWeight",
                "keep_branches": ["run", "lumi", "cate_index"], "train_mc_background": ["background"],
                "yearly_inputs": [{"year": "2024", "folder": str(directory), "tree_name": "Events", "eventWeight_scale": 2.}],
                "preselections": ["diMufsr_kit_BSC_mass >= 110", "diMufsr_kit_BSC_mass <= 150", "cate_index == 4"],
            }}))
            for profile in ["full", "fit"]:
                argv = ["apply", "-c", str(config), "-i", str(directory), "--inputTree", "Events",
                        "--modelFolder", str(directory), "--output-profile", profile, "--chunk-size", "2"]
                if profile == "fit":
                    argv += ["--systematics"]
                with patch.object(sys, "argv", argv):
                    handler = apply.ApplyDNNHandler(apply.getArgs())
                handler.m_tsfs = self.handler.m_tsfs
                handler._transform_features = self.handler._transform_features
                handler._predict_scores = self.handler._predict_scores
                handler._predict_score_domain = lambda values, fold: (
                    self.handler._predict_scores(values, fold), np.full(len(values), .5, dtype=np.float32))
                handler._event_weight_scale = 2.
                handler._process_files([str(path)], str(directory / (profile + ".root")), "signal")
            with uproot.open(directory / "fit.root") as fit_file, uproot.open(directory / "full.root") as full_file:
                compact = fit_file["test"].arrays(library="np")
                full = full_file["test"].arrays(library="np")
                np.testing.assert_array_equal(compact["event"], [3, 314159])
                for column in ["dnn_score_t", "diMufsr_kit_BSC_mass", "eventWeight"]:
                    np.testing.assert_array_equal(compact[column], full[column])
                np.testing.assert_array_equal(compact["eventWeight_jer_up"], [0., 11.])
            summary = json.loads((directory / "fit.root.summary.json").read_text())
            self.assertEqual(summary["entries"], 2)
            self.assertEqual(summary["sumw"]["eventWeight"], 10.)
            self.assertEqual(summary["systematics"]["JERUp"]["score_output"], "score_t_jer_up")
            # The upstream skim intentionally has no variations for backgrounds.
            nominal_path = directory / "background_input.root"
            nominal_arrays = {c: raw[c].to_numpy() for c in raw if "__" not in c}
            with uproot.recreate(nominal_path) as root_file:
                tree = root_file.mktree("Events", {c: a.dtype for c, a in nominal_arrays.items()})
                tree.extend(nominal_arrays)
            handler._process_files([str(nominal_path)], str(directory / "background.root"), "background")
            background_summary = json.loads((directory / "background.root.summary.json").read_text())
            self.assertEqual(background_summary["systematics"], {})
            self.assertEqual(background_summary["systematics_status"], "nominal_only_background_input")
            self.assertEqual(background_summary["entries"], 2)
            from dnn_fit_campaign import report
            final = directory / "outputs/2024/ggH/signal.root"
            final.parent.mkdir(parents=True)
            shutil.copyfile(directory / "fit.root", final)
            shutil.copyfile(directory / "fit.root.summary.json", str(final) + ".summary.json")
            shutil.copyfile(directory / "background.root", final.parent / "background.root")
            shutil.copyfile(directory / "background.root.summary.json", final.parent / "background.root.summary.json")
            report({"config": str(config), "artifacts": str(directory), "years": ["2024"],
                    "categories": ["signal", "background"], "model_dir": "fixture"}, require_complete=True)
            self.assertTrue(json.loads((directory / "report.json").read_text())["complete"])


class CampaignParameterTests(unittest.TestCase):
    def test_explicit_parameters_are_applied_or_rejected(self):
        from dnn_fit_campaign import resolve_parameters
        scripts = Path(__file__).parent
        requested = {"learning_rate": 0.00043393337239758603, "trunk_units": [2048, 512, 256, 128],
                     "trunk_dropout": [0.2, 0.2, 0.15, 0.1], "rff_auc_guard_enabled": True}
        resolved = resolve_parameters(scripts / "train_dnn_ggH_true_mass_decor.py", {}, requested)
        for key, value in requested.items():
            self.assertEqual(resolved[key], value)
        with self.assertRaisesRegex(ValueError, "rff_auc_guard_enabled"):
            resolve_parameters(scripts / "train_dnn_ggH.py", {}, requested)



class PublishedRootReadbackTests(unittest.TestCase):
    def test_remote_readback_stages_and_cleans_up(self):
        import dnn_root_io as io
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "input.root"
            with uproot.recreate(original) as f:
                tree = f.mktree("test", {"value": np.dtype("float64")})
                tree.extend({"value": np.array([1., 2.])})
            staged = []
            def transfer(argv, **kwargs):
                self.assertEqual(argv[2], "root://eosuser.cern.ch//eos/test/output.root")
                staged.append(Path(argv[3]))
                shutil.copy2(original, argv[3])
                return subprocess.CompletedProcess(argv, 0, "", "")
            with patch.object(io.shutil, "which", return_value="xrdcp"), patch.object(io.subprocess, "run", side_effect=transfer):
                with io.open_published_root("/eos/test/output.root") as f:
                    self.assertEqual(f["test"].num_entries, 2)
                    self.assertTrue(staged[0].exists())
                self.assertFalse(staged[0].exists())

    def test_transfer_failure_is_explicit_and_cleans_up(self):
        import dnn_root_io as io
        import subprocess
        staged = []
        def fail(argv, **kwargs):
            staged.append(Path(argv[3]))
            return subprocess.CompletedProcess(argv, 1, "", "permission denied")
        with patch.object(io.shutil, "which", return_value="xrdcp"), patch.object(io.subprocess, "run", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "permission denied"):
                with io.open_published_root("/eos/test/output.root"):
                    self.fail("Failed transfer must not yield a ROOT file")
        self.assertFalse(staged[0].parent.exists())

if __name__ == "__main__":
    unittest.main()
