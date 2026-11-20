# Run3 jet recoil correction

One self-contained campaign fits 2022, 2022EE, 2023, 2023BPix, 2024, 2025 and 2026 independently. **ZEE + WCR are fitted; ZCR and SRSB are independent validation regions.** SRSB uses 110–115 and 135–150 GeV. No central mass-window data are exposed.

The numerical reference is:
`/eos/user/j/jiehan/HMuMuShare/JetRecoilCorrection_2024/20261023_ngenjet10_mcstat_matched030_unmatched150_sf_prior_wsf_njet_mumetpt_zcr20_wcr_muon40_no_constraint`.
The deliberate differences are the ZEE fit channel and current era inputs/luminosities. The fit kernels, objective, covariance, binning, initialization, priors and convergence settings are retained.

## Configuration and inputs

- `model.json`: common fit definition. 560 jet + 130 DY + 104 W positive parameters, fitted in log coordinates. Matched/unmatched priors are Gaussian in linear SF: 1±0.30 and 1±1.50. Nominal JER, pT>25 GeV, |eta|<4.7; seven pT × 40 signed eta cells. No generator-level C/LHE/NGenJet constraint. Zero selected-jet events remain in the model. After the existing event selection/b-tag veto, all jet-based calculations use only the first 32 selected jets in stable descending corrected-pT order. Events are retained; discarded jet counts and affected event counts are recorded. Maximum 300 iterations; scaled gradient maximum <1e-5.
- `eras.json`: seven explicit luminosities, production fingerprints and input adapters.
- `samples.json`: exact nominal samples with required `process_family` labels, cross sections, merge-report hashes, all report-listed data parts, ROOT provenance/schema and file identities. Exclusions record the reference treatment of alternative EWK/tZq and Higgs signals; the nominal Data/background comparison excludes Higgs signals from its normalization. The native 2023/2023BPix productions have no EWK-Z sample; this is visible in the frozen catalog, without substitution from another era.
- `variables.json`: ordinary regional variables, 23 ggF input variables, four additional mass-divided pT inputs, and the three display normalizations.
- `plots.json`: fixed fit-diagnostic templates. The complete per-run figure list is generated from the variable catalog.

The inputs come from the checked `2223` and `2426` production merge reports. Native 2022–2023 use `conditions/genEventSumw`; sampled 2024–2026 require `conditions/normalization_sumw`. The sampling fraction is already in that denominator and is not applied twice. Stored selection and b-tag flags are read through the compact-schema nominal jet adapter. No legacy tree/path/weight/SF fallback is used.

`preflight_inputs.py` checks every catalog file against its report: event counts, required logical branches, fixed provenance and schema. It reads up to 256 entries per file and exercises selection and weights. It refreshes the frozen metadata in `samples.json` and writes `input_preflight.json`. Missing or inconsistent required inputs fail explicitly. EOS reads use XRootD; only opening is retried, at most three times. A failed iterator is never restarted after yielding events.

## Run

Use the same environment as the reference:

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc15-opt/setup.sh
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export X509_USER_PROXY=/afs/cern.ch/user/j/jiehan/x509up_u152815
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/eos/user/j/jiehan/ml-gpu-env/bin/python3 -B run.py \
  --output /absolute/new/result/directory \
  --control-dir /absolute/new/afs/control/directory
```

Both directories must be new unless `--replace` explicitly replaces an existing stopped campaign in place without creating backups. Replacement verifies the old queue is empty and refuses to remove any declared source input. `--era` optionally selects a subset; all seven are the default. `--prepare-only` generates and validates the full DAG without submission. Each campaign freezes its complete code/configuration/definitions/tests, a SHA256 tool ID, an immutable toolkit tarball and all input identities. Previous campaigns and ROOT outputs are not modified.

The single master DAG limits running nodes to **1000**, and fits to **one**. All workflow nodes request 10 CPUs and 30,000 MB of memory by default. Workers constrain BLAS threading to the reference setting. Per-node stdout/stderr, `jobs.log`, `pipeline.status`, the DAG and rescue files live in the AFS control directory. `jobs.json` records resources and dependencies.

The chain is scans → preflight → fit → export/metrics → plot setup → regional/fit-bin/ggF histograms → aggregation and all plots → gallery and completion. A ggF histogram depends on its corresponding regional histogram and cross-checks event yields and pT(ll). The final node depends on **every era's** export and plot stages. Invalid fits retain diagnostics but cannot export corrections or mark completion true.

Use the master DAG rescue file to resume. Successfully completed nodes remain complete; scan and histogram nodes additionally reuse artifacts only after checking the package ID, input identity, fit hash and SHA256 of both the numerical cache and its metadata. Source ROOT files are never deleted.

## Agreement products

Each region displays nominal, jet SF only, DY/W SF only and combined corrections. The DY SF is applied to direct DY in ZEE/ZCR/SRSB, excluding DYTau; W SF is applied to the W group in WCR. Every ordinary and ggF agreement has:

- `physical`: luminosity × cross-section yield, with no Data/MC rescaling;
- `absolute`: the reference's fixed nominal regional Data/MC factor, unchanged across SF stages;
- `shape`: each stage normalized to the valid Data population of that observable.

These display conventions do not enter the fit. Ordinary per-stage yields, variance, ratio, pulls and marginal statistical chi-square are stored in `plot_metrics.json` and `plot_values.npz`. Raw per-sample moments remain available. Statistical diagnostics exclude fitted-SF uncertainty and other systematics, and are not a final goodness-of-fit result. Ratios outside the displayed 0–2 range have boundary markers; numerical values remain unchanged.

Matched/unmatched jet pT and eta are shown prefit/postfit in all four regions and all three normalizations, with inclusive Data. Their sum and full event covariance, including cross terms, must reproduce inclusive MC. Weighted fractions are saved; negative weights can place fractions outside [0,1].

The ggF variables are evaluated in ZCR/SRSB with the original inclusive validation selections, without an extra category cut or DNN inference. Frozen training metadata and skim definitions live in `definitions/`. Missing jets and undefined values are excluded and counted; finite tails are folded and counted. Four pT/mass plots precede model scaling. The historical `pTMu2oMmm` muon-1 numerator and Collins–Soper conventions are preserved explicitly. The raw `source_year` display uses the analysis year declared by the era; model encoding is not applied.

`index.html` filters all plots by era, region, category and normalization. `gallery.json`, `agreement_summary.json`, multipage PDFs, fit status and correction paths form the final overview. No missing plot or invalid fit is silently accepted.

## Corrections and verification

Correctionlib schema-v2 retains four objects: `jet_sf`, `jet_sf_error`, `dy_sf`, `dy_sf_error`, in `corrections.json` and `.json.gz`. W SF remains in fit results/maps. Each export checks every cell with the C++ evaluator at rtol 1e-14. Full log-SF covariance/correlation are retained separately. Sparse bins keep the reference binning and are reported; no agreement-dependent tuning is performed.

`tests/verify_reconstructed_fit.py --reference-core <archived-code-snapshot>` checks the archived C-disabled numerical kernels, signed weights, repeated jets, zero jets, finite-difference Jacobian, full covariance, prior rows, strict denominators and exports. `tests/test_plot_integration.py` reads real Data/DY events, evaluates all ggF features, renders the three normalization modes and validates matched/unmatched covariance.

`tests/test_rerun_repair.py` additionally covers 0/1/31/32/33/40 jets, aligned stable truncation, unchanged event selection/SF products, the formerly failing 2026 WCR data interval, and bounded actual scans plus cache loading/preflight for all 154 fit-sample records across all seven eras. Temporary cache products are removed on exit.

SF uncertainties use `normalized_gauss_newton_v2`: the final information matrix is
normalized by its strictly positive diagonal before the relative rank test.
The central-value optimizer is unchanged. `jet_sf_error`, `dy_sf_error` and
`w_sf_error` remain absolute, symmetric local errors including Gaussian SF priors.
`*_sf_near_boundary` marks cells with `SF - error < 0`; these errors are not
calibrated confidence intervals near zero. `covariance.npy` is in log-SF coordinates;
`covariance_sf.npy` is in linear SF coordinates, ordered jet, DY, W using the
fit's parameter slices and C-order flattening. Propagate a derived quantity using
its full gradient and covariance, including cross terms. No profile intervals or
stat-up/down variations are produced.

For an existing converged fit, run `sf_covariance.py --fit-dir ERA/E/fit`, then
export corrections again with `sf_export.py --run-dir ERA/E`. Isolate writers first;
recomputing errors changes the fit checksum and invalidates dependent plot receipts.

New matching comparisons display physical matched/unmatched MC yields and matched fraction before and after jet SF, with correlated statistical percent changes, for all, leading and subleading jets in pT and eta.
