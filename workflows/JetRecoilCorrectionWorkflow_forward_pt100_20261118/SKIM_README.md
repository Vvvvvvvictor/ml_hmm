# Minimal Run3 corrected skim

`skim_run3.py` freezes the input catalog, repairs invalid/changed fits, and writes
one ROOT TTree `Events` per era, region and sample. The generated production DAG
limits the simultaneous running nodes to four. No training/inference is run.

## Event and branch contract

The 23 inputs come from the frozen reference training config. Additional columns
are `diMufsr_kit_BSC_mass`, `diMu-mass_resolution_BSC`, `run`, `lumi`, `event`,
`weight` (lumi*xsec/original production denominator), `eventWeight` (complete
corrected signed weight), and `cate_index` (4 ggH, 3 VBF, 0 failed selection).
Physical `source_year` is stored; the training loader supplies its configured
era encoding, keeping 2022/2022EE and 2023/2023BPix distinct.

Select `cate_index == 4` for ggH training. A signal row is retained if any nominal
or shifted selection passes. Therefore rows with nominal `cate_index == 0` and
zero nominal weight can carry nonzero shifted weights. Each `__JER...`/`__jes...`
suffix carries its own category, event weight and all 14 affected training inputs,
including the varied `njets`. The
kinematic shifts are evaluated independently, without constructing combinations
of distinct nuisance variations. Missing jet features use -100.

Data and backgrounds contain central values only. Higgs signal has the available
SF norm/up/down components and complete event weights. MuonEff is the original
shared nuisance: both ID and ISO, for both muons, vary together. PDF/QCD up/down
are original multiplicative ratios, with nominal factor 1. PU has nominal only
because these source ROOT files do not contain PU variations.

## Correlated Jet SF errors

`jet_sf_norm` is the ordinary event product. `jet_sf_up/down = norm +/- sigma`
are symmetric local propagated errors, not confidence bounds; down can be
negative and is not clipped. There is no stochastic thinning or jet deletion
based on fitted SF. Zero-jet events have product 1 and error 0.

Each signal row has sparse `sf_index`/`sf_count` columns. An index can occur once
in that representation with count greater than one. The seven per-era correction
directories save the full covariance in log-SF coordinates (jet, DY, W order),
the linear-SF covariance and parameter slices. The jet block comes from the full
joint inverse and thus retains DY/W profiling. Higgs weights have zero direct
derivative with respect to DY/W. Each era uses its own fitted covariance.

For a signed bin yield, `G[a] = sum_events weight * count[a]` in log coordinates.
Use `skim_covariance.accumulate` to add gradients across all chunks/samples that
share the same fit. Then `bin_covariance(G, covariance.npy)` returns the complete
bin-to-bin covariance. Summing event errors in quadrature or treating the single
event envelope as one globally correlated nuisance does not reproduce this.
MC statistical `sumw2` remains separate.

The skim stores raw pT inputs. Mass division, missing-jet imputation and model
standardization remain in the training/inference workflow. The historical
`pTMu2oMmm` convention is intentionally retained for compatibility.

## Publication and completion

Intermediate ROOT/cache files live in worker scratch. Only a closed, verified
ROOT is copied to EOS, checksum-checked and renamed into place. JSON receipts
record source identity, correction hash, branch list, event/yield totals and
checksums. A matching receipt allows a verified retry to reuse a final file.
`completion.json` becomes true only after reopening and checking every final
ROOT and checking all seven correction fits. Original source/fit files remain.
