"""Tests against actual trainer methods without importing TensorFlow."""
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from dnn_reso_training import RESOLUTION, resolution_method, training_flags
from merge_dnn_fit_eras import systematic_columns

TRAINER = Path(__file__).with_name("train_dnn_ggH_true_mass_decor.py")


def trainer_method(name):
    tree = ast.parse(TRAINER.read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "DNNHandler")
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == name)
    namespace = {"np": np, "SCORE_MASS_MIN": 110., "SCORE_MASS_MAX": 150.}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(TRAINER), "exec"), namespace)
    return namespace[name]


class ResolutionTrainingTests(unittest.TestCase):
    def test_flags_are_explicit_and_background_remains_off(self):
        self.assertEqual(training_flags({"training_weights": {"reso": True, "reso_bkg": False}}), ["--reso"])
        self.assertEqual(training_flags({"defaults": {"reso": False}}), [])

    def test_division_negative_weights_and_invalid_resolution(self):
        data = pd.DataFrame({"eventWeight": [6., -8., 3., 4., 5., 6., 7.],
                             RESOLUTION: [2., 4., 0., -1., np.nan, np.inf, 1.e-8]})
        resolution_method(TRAINER)(SimpleNamespace(reso=RESOLUTION, weight="eventWeight"), data, "test")
        np.testing.assert_array_equal(data.eventWeight, [3., -2., 0., 0., 0., 0., 0.])

    def test_missing_resolution_fails(self):
        with self.assertRaises(KeyError):
            resolution_method(TRAINER)(SimpleNamespace(reso=RESOLUTION, weight="eventWeight"),
                                       pd.DataFrame({"eventWeight": [1.]}), "test")

    def test_streaming_divides_only_signal_and_keeps_physical_weight(self):
        proxy = SimpleNamespace(_reso=True, _reso_bkg=False, _ptdecor=False)
        def mark(data):
            data["physical"] = data["eventWeight"].copy()
            return data
        proxy._mark_shape_event_weight = mark
        proxy._apply_resolution_weight = lambda data,label: resolution_method(TRAINER)(SimpleNamespace(reso=RESOLUTION,weight="eventWeight"),data,label)
        proxy._streaming_mass_window_mask = lambda data,*a: np.ones(len(data),dtype=bool)
        proxy._streaming_sideband_mask = lambda data: np.ones(len(data),dtype=bool)
        proxy._streaming_domain_jet_mask = proxy._streaming_sideband_mask
        proxy._split_masks = lambda data,fold: {}
        proxy._default_params = lambda: {"mass_reweight_min":110.,"mass_reweight_max":150.}
        proxy._merge_dnn_params = proxy._default_params
        method = trainer_method("_prepare_streaming_chunk")
        for sample_type,source in [("signal","mc"),("mc_background","mc"),("data","data")]:
            data = pd.DataFrame({"eventWeight":[6.,-8.],RESOLUTION:[2.,4.]})
            result,_ = method(proxy,data,{"sample_type":sample_type,"source_type":source,"sample_name":"sample"},0)
            np.testing.assert_array_equal(result.eventWeight,[3.,-2.] if sample_type=="signal" else [6.,-8.])
            np.testing.assert_array_equal(result.physical,[6.,-8.])

    def test_class_normalization_uses_resolution_weight(self):
        proxy=SimpleNamespace(SF=-1.,weight="eventWeight")
        proxy._signed_weight_mean=lambda weights,label: float(np.mean(weights))
        sig=pd.DataFrame({"eventWeight":[2.,6.]})
        bkg=pd.DataFrame({"eventWeight":[2.,4.,6.,8.]})
        actual=trainer_method("_normalized_score_weights")(proxy,sig,bkg,sig,bkg)
        np.testing.assert_allclose(actual,[.5,1.5,.2,.4,.6,.8])

    def test_era_specific_union_and_common_missing_error(self):
        def meta(names):
            return {"systematics":{n:{"suffix":n,"score_output":"score_t_"+n,
                                       "weight_output":"eventWeight_"+n} for n in names}}
        years=["2022","2024"]
        summaries=[meta(["jer_up","jesabsolute_2022_up"]),meta(["jer_up","jesabsolute_2024_up"])]
        variations,_=systematic_columns(summaries,years,"signal")
        self.assertEqual(set(variations),{"jer_up","jesabsolute_2022_up","jesabsolute_2024_up"})
        with self.assertRaises(ValueError):
            systematic_columns([summaries[0],meta(["jesabsolute_2024_up"])],years,"signal")


class PromotionTests(unittest.TestCase):
    def exercise(self, fail=False):
        import promote_dnn_reso as promotion
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); target=root/"published"; stage=target/"staging"; run=root/"run"
            run.mkdir();stage.mkdir(parents=True)
            for folder in [target/"models_dnn",target/"outputs",stage/"models_dnn",stage/"outputs"]:
                folder.mkdir()
            (target/"models_dnn"/"version").write_text("old")
            (stage/"models_dnn"/"version").write_text("new")
            (target/"report.json").write_text('{"old":true}')
            for index in range(22): (stage/"outputs"/f"sample{index}.root").touch()
            report={"complete":True,"cleanup":"complete","training_provenance":{"reso":True},
                    "samples":[{"full_branch_readback":True,"output":str(stage/"outputs"/f"sample{i}.root")} for i in range(22)]}
            (stage/"report.json").write_text(json.dumps(report))
            manifest={"artifacts":str(stage),"promotion_target":str(target),"run_dir":str(run),"created_utc":"test"}
            path=run/"manifest.json";path.write_text(json.dumps(manifest))
            failed=False
            def fake_eos(*args,check=True):
                nonlocal failed
                if args[0]=="mkdir": Path(args[-1]).mkdir(parents=True,exist_ok=True)
                elif args[0]=="mv":
                    source,destination=map(Path,args[1:])
                    if fail and source==stage/"outputs" and not failed:
                        failed=True
                        raise RuntimeError("injected publication failure")
                    source.rename(destination)
                return SimpleNamespace(returncode=0,stdout="",stderr="")
            with patch.object(promotion,"exists",lambda p:Path(p).exists()),patch.object(promotion,"eos",fake_eos),patch.object(promotion,"validate_model_policy"):
                if fail:
                    with self.assertRaises(RuntimeError): promotion.promote(path)
                    self.assertEqual((target/"models_dnn"/"version").read_text(),"old")
                    self.assertEqual((stage/"models_dnn"/"version").read_text(),"new")
                    self.assertEqual(json.loads((target/"report.json").read_text()),{"old":True})
                    self.assertEqual(json.loads((run/"promotion.json").read_text())["status"],"rolled_back")
                else:
                    promotion.promote(path)
                    self.assertEqual((target/"models_dnn"/"version").read_text(),"new")
                    self.assertEqual((target/"backups/pre_reso_test/models_dnn/version").read_text(),"old")
                    got=json.loads((target/"report.json").read_text())
                    self.assertTrue(got["samples"][0]["output"].startswith(str(target/"outputs")))
                    self.assertEqual(json.loads((run/"promotion.json").read_text())["status"],"complete")

    def test_success_archives_previous_version(self):
        self.exercise()

    def test_failure_restores_previous_version(self):
        self.exercise(fail=True)


if __name__ == "__main__":
    unittest.main()
