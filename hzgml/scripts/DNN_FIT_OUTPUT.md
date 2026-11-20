# Default DNN and compact ggH outputs

From the repository root, run `source hzgml/scripts/setup_dnn_gpu.sh` to activate
the CERN DNN environment. The pt100 campaign scripts use the repository's
[forward-pT100 recoil workflow](../../workflows/JetRecoilCorrectionWorkflow_forward_pt100_20261118/README.md).

Prepare and submit an isolated run from the repository root:

```bash
python3 hzgml/scripts/dnn_fit_campaign.py prepare --submit
```

The command checks all configured inputs, snapshots the scripts and configs, and
prints the manifest, DAG and artifact paths. Training uses the current script and
config defaults for all four folds, with no Optuna, resolution weighting or
additional pT decorrelation flags. Existing runs are never reused or overwritten.
The DAG runs training, a CPU smoke check against the legacy inference, one apply
job per year, and a final validation/report job. Failed stages stop their children.

To train with an Optuna result instead of defaults, pass `--hyperparams-file`
pointing to its `DNN_region_ggH_all_folds.json`. The complete `best_params` object
is snapshotted and applied to every fold. For parameters including `rff_auc_guard_*`,
also pass `--training-script train_dnn_ggH_true_mass_decor.py`. Unsupported parameters
fail preparation instead of being silently ignored. The manifest records requested
and resolved parameters, and saved Keras model parameters are checked before apply.

Models are saved under `<artifacts>/models_dnn`. ROOT files are saved under
`<artifacts>/outputs/<year>/ggH/<sample>.root`, with one `test` tree per file.
Each file contains event identifiers, source year, dimuon mass, `score_t`,
`eventWeight`, and the `dnn_score_t` compatibility alias. Signal MC also contains
`score_t_<variation>` and `eventWeight_<variation>`. For example, `JERUp` becomes
`jer_up` and `jesAbsolute_2024Up` becomes `jesabsolute_2024_up`. Each
Backgrounds listed in `train_mc_background` may have nominal inputs only, as
required by this skim contract; their summary records
`systematics_status=nominal_only_background_input`. Any available background
variations are still scored, and incomplete variations or missing signal
variations remain errors.
`.root.summary.json` records input files, the name mapping, entries, weight sums
and file size. Pure weight variations reuse the nominal score.
Published EOS ROOT files are validated by downloading one file at a time with
`xrdcp` to the job scratch directory, reading it locally, and removing the
temporary copy. This avoids EOS FUSE metadata delays and does not require
Python XRootD or fsspec-xrootd packages.

The nominal selection is fixed to mass 110–150 GeV and `cate_index == 4`.
Systematics do not change the event membership. JER/JES inputs are shifted before
preprocessing, and use the same event test fold and nominal score transformer.
Year-specific nuisances are kept in their corresponding year files; do not merge
these files using a common-branch intersection.

The final `report.json` and `report.tsv` list all expected samples, entries,
weight sums and sizes. Validation also compares every output event and systematic
weight against the nominal-selected source events, and checks finite scores and
the categorization branch interface. `smoke/validation.json` records the real-input
nominal and JER/JES comparisons for 2022 and 2024.

To use compact output independently, add these options to the existing apply CLI:

```text
--systematics --output-profile fit --exact-output-folder
```

`--exact-output-folder` suppresses the legacy date/year suffix. Without the new
options, the original full output remains the default.

Run regression checks in the DNN Python environment:

```bash
/eos/user/j/jiehan/ml-gpu-env/bin/python -m unittest discover \
  -s hzgml/scripts -p test_dnn_fit_output.py -v
```
