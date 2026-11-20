"""Build one searchable, era/region/category/normalization-filtered agreement index."""

import html
from pathlib import Path
from runtime import RUN, ERAS, read, write


def build():
    entries = []
    for name in read(RUN / "plots.json")["figures"]:
        if not name.endswith(".png"):
            continue
        parts = Path(name).parts
        era = parts[0]
        region = next((r for r in ("zee", "wcr", "zcr", "srsb") if r in parts), "fit")
        category = (
            "ggF inputs"
            if "ggf_dnn_inputs" in parts
            else "matched / unmatched"
            if "match_split" in parts
            else "agreement"
            if "validation_candidate" in parts
            else "fit diagnostics"
        )
        norm = next(
            (
                n
                for n in ("physical", "shape", "absolute")
                if Path(name).stem.endswith("_" + n)
            ),
            "absolute" if "match_split" in parts else "diagnostic",
        )
        entries.append(
            dict(
                era=era,
                region=region,
                category=category,
                normalization=norm,
                png=name,
                pdf=str(Path(name).with_suffix(".pdf")),
            )
        )
    write(
        RUN / "gallery.json",
        dict(
            plots=entries,
            normalizations=dict(
                physical="Luminosity × cross section",
                absolute="Fixed prefit regional Data/MC",
                shape="Each stage normalized to Data",
            ),
        ),
    )
    rows = []
    for item in entries:
        attrs = " ".join(
            f'data-{k}="{html.escape(item[k])}"'
            for k in ("era", "region", "category", "normalization")
        )
        rows.append(
            f'<section {attrs}><h3>{html.escape(item["png"])}</h3><a href="{item["pdf"]}">PDF</a> <a href="{item["png"]}">PNG</a><br><img loading="lazy" src="{item["png"]}"></section>'
        )
    controls = []
    for key in ("era", "region", "category", "normalization"):
        options = "".join(
            f"<option>{html.escape(value)}</option>"
            for value in sorted({e[key] for e in entries})
        )
        controls.append(
            f'<label>{key} <select id="{key}"><option value="">All</option>{options}</select></label>'
        )
    (RUN / "index.html").write_text(
        """<!doctype html><meta charset="utf-8"><title>Run3 jet recoil agreement</title>
<style>body{font:15px sans-serif;margin:24px}label{margin-right:12px}section{border-top:1px solid #ccc;margin-top:20px}img{width:100%;max-width:1000px}h3{overflow-wrap:anywhere}</style>
<h1>Run3 jet recoil: ZEE + WCR fit; ZCR + SRSB validation</h1>
<p><a href="REPORT.md">Fit report</a> · <a href="agreement_summary.json">Agreement metrics</a> · <a href="gallery.json">Plot catalog</a></p>
<p>Nominal / jet only / DY or W only / combined. Statistical errors only; fitted-SF uncertainty and other systematic uncertainties are not propagated. SRSB: 110–115 and 135–150 GeV.</p>
"""
        + "".join(controls)
        + "".join(rows)
        + """<script>
const keys=['era','region','category','normalization'];
for(const key of keys)document.getElementById(key).onchange=()=>{
for(const section of document.querySelectorAll('section'))section.hidden=keys.some(k=>{const v=document.getElementById(k).value;return v&&section.dataset[k]!==v;});};
</script>"""
    )
    write(
        RUN / "agreement_summary.json",
        {
            era: dict(
                fit=read(RUN / era / "E/fit/fit_results.json")["preflight"],
                regions=read(RUN / "validation_metrics" / f"{era}.json"),
                ggf={
                    r: read(
                        RUN
                        / era
                        / "E/validation_candidate"
                        / era
                        / "ggf_dnn_inputs"
                        / r
                        / "metrics.json"
                    )
                    for r in ("zcr", "srsb")
                },
                corrections=str(RUN / era / "E/fit/corrections.json.gz"),
                sf_uncertainty={key: read(RUN / era / "E/fit/fit_results.json")[key] for key in ("covariance_method", "uncertainty_note", "boundary_parameter_indices")},
                covariance_sf=str(RUN / era / "E/fit/covariance_sf.npy"),
            )
            for era in ERAS
        },
    )
