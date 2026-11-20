"""Logical selected-jet view of the compact dual-JER CROWN schema."""

import os

import awkward as ak
import numpy as np


def jet_pt_mode():
    mode = os.environ.get("CROWN_JET_PT_MODE", "nominal")
    if mode != "nominal":
        raise ValueError("This workflow requires nominal JER")
    return mode


ALIASES = {
    "genWeight": "Generator_weight",
    "Jet_nanoEta": "Jet_corrEta",
    "Jet_nanoPhi": "Jet_corrPhi",
    "Jet_rawEta": "Jet_corrEta",
    "Jet_rawPhi": "Jet_corrPhi",
    "good_jet_collection": "Jet_originalIndex",
}
SUMS = {
    "Jet_emEF": ("Jet_selectedChEmEF", "Jet_selectedNeEmEF"),
    "Jet_hadEF": ("Jet_selectedChHEF", "Jet_selectedNeHEF"),
    "Jet_neutralEF": ("Jet_selectedNeHEF", "Jet_selectedNeEmEF"),
    "Jet_chargedEF": ("Jet_selectedChHEF", "Jet_selectedChEmEF", "Jet_selectedMuEF"),
}
CORE = (
    "Jet_corrPt",
    "Jet_corrEta",
    "Jet_corrPhi",
    "Jet_corrMass",
    "Jet_originalIndex",
    "Jet_selectionMask",
    "Jet_btagLoose",
    "Jet_btagMedium",
)
HIDDEN = (
    "Jet_selectionMask",
    "Jet_corrPt_oldJER",
    "Jet_corrMass_oldJER",
    "Jet_btagLoose_oldJER",
    "Jet_btagMedium_oldJER",
)


def jet_tree(tree, mode=None):
    if isinstance(tree, JetTree):
        return tree
    if "Jet_selectionMask" not in tree.keys():
        raise ValueError(
            "Expected the declared compact CROWN schema with Jet_selectionMask"
        )
    return JetTree(tree, mode or jet_pt_mode())


def numpy_column(values):
    if ak.to_layout(values).purelist_depth == 1:
        return ak.to_numpy(values)
    result = np.empty(len(values), dtype=object)
    result[:] = [ak.to_numpy(row) for row in values]
    return result


class JetTree:
    def __init__(self, tree, mode):
        if mode not in ("nominal", "oldJER"):
            raise ValueError("Unknown jet pT mode: " + mode)
        self.tree, self.mode = tree, mode

    def __getattr__(self, name):
        return getattr(self.tree, name)

    def __contains__(self, name):
        return name in self.keys()

    def __getitem__(self, name):
        return JetBranch(self, name)

    def keys(self):
        keys = set(self.tree.keys())
        suffixes = {
            k[len("Jet_selectionMask") :]
            for k in keys
            if k.startswith("Jet_selectionMask")
        }
        for suffix in suffixes:
            keys.update(
                name + suffix
                for name, source in ALIASES.items()
                if source + suffix in keys
            )
            keys.update(
                name + suffix
                for name, sources in SUMS.items()
                if all(k + suffix in keys for k in sources)
            )
            keys.add("Jet_absEta" + suffix)
            keys.update(
                name + suffix
                for name in (
                    "njets",
                    "nbjets_loose",
                    "nbjets_medium",
                    "dijet_mass",
                    "dijet_eta",
                )
            )
            for rank in (1, 2):
                keys.update(
                    f"jet{rank}_{name}{suffix}"
                    for name in ("pt", "eta", "phi", "mass", "btagDeepFlavQG")
                )
            keys.difference_update(name + suffix for name in HIDDEN)
        return sorted(keys)

    def _read_names(self, requested):
        physical = set(self.tree.keys())
        wanted = set(requested)
        suffixes = {name.split("__", 1)[1] for name in wanted if "__" in name}
        suffixes = [""] + ["__" + name for name in sorted(suffixes)]
        for suffix in suffixes:
            if "Jet_selectionMask" + suffix not in physical:
                continue
            wanted.update(name + suffix for name in CORE)
            wanted.update(
                name + suffix
                for name in ("Jet_btagDeepFlavQG", "Jet_puIdDisc", "Jet_isGenMatched")
                if name + suffix in physical
            )
            if self.mode == "oldJER":
                wanted.update(name + suffix for name in HIDDEN[1:])
            for name, source in ALIASES.items():
                if name + suffix in wanted:
                    wanted.add(source + suffix)
            if "Jet_absEta" + suffix in wanted:
                wanted.add("Jet_corrEta" + suffix)
            for name, sources in SUMS.items():
                if name + suffix in wanted:
                    wanted.update(k + suffix for k in sources)
        missing = wanted - physical - set(self.keys())
        if missing:
            raise KeyError("Missing CROWN inputs: " + ", ".join(sorted(missing)))
        return sorted(wanted & physical)

    def _select(self, arrays):
        out = {name: arrays[name] for name in arrays.fields}
        suffixes = [
            k[len("Jet_selectionMask") :]
            for k in arrays.fields
            if k.startswith("Jet_selectionMask")
        ]
        for suffix in suffixes:
            saved_scalars = {
                k: v
                for k, v in out.items()
                if k in arrays.fields
                and (
                    k.startswith(
                        ("jet1_", "jet2_", "dijet_", "nbjets_", "n_jets_matched_genjet")
                    )
                    or k == "njets"
                )
            }
            bit = 1 if self.mode == "nominal" else 2
            keep = (out["Jet_selectionMask" + suffix] & bit) != 0
            ptname = "Jet_corrPt" + ("_oldJER" if bit == 2 else "") + suffix
            order = ak.argsort(out[ptname][keep], ascending=False, stable=True)
            for name in list(out):
                if suffix and not name.endswith(suffix):
                    continue
                base = name.removesuffix(suffix) if suffix else name
                if "__" in base or base == "Jet_genJetIdx":
                    continue
                if base.startswith("Jet_"):
                    out[name] = out[name][keep][order]
            if bit == 2:
                for name in (
                    "Jet_corrPt",
                    "Jet_corrMass",
                    "Jet_btagLoose",
                    "Jet_btagMedium",
                ):
                    out[name + suffix] = out[name + "_oldJER" + suffix]
            for name, source in ALIASES.items():
                if source + suffix in out:
                    out[name + suffix] = out[source + suffix]
            out["Jet_absEta" + suffix] = abs(out["Jet_corrEta" + suffix])
            for name, sources in SUMS.items():
                if all(k + suffix in out for k in sources):
                    value = out[sources[0] + suffix]
                    for source in sources[1:]:
                        value = value + out[source + suffix]
                    out[name + suffix] = value
            out["njets" + suffix] = ak.num(out["Jet_corrPt" + suffix], axis=1)
            for wp in ("loose", "medium"):
                out["nbjets_" + wp + suffix] = ak.sum(
                    out["Jet_btag" + wp.title() + suffix], axis=1
                )
            for rank in (1, 2):
                for name, source in [
                    ("pt", "Pt"),
                    ("eta", "Eta"),
                    ("phi", "Phi"),
                    ("mass", "Mass"),
                ]:
                    values = ak.pad_none(out["Jet_corr" + source + suffix], rank)
                    out[f"jet{rank}_{name}{suffix}"] = ak.values_astype(
                        ak.fill_none(values[:, rank - 1], -10.0), np.float32
                    )
                for source, name in [
                    ("Jet_btagDeepFlavQG", "btagDeepFlavQG"),
                    ("Jet_puIdDisc", "puIdDisc"),
                ]:
                    if source + suffix in out:
                        out[f"jet{rank}_{name}{suffix}"] = ak.values_astype(
                            ak.fill_none(
                                ak.pad_none(out[source + suffix], rank)[:, rank - 1],
                                -10.0,
                            ),
                            np.float32,
                        )
            pt, eta, phi, mass = [
                out["Jet_corr" + name + suffix] for name in ("Pt", "Eta", "Phi", "Mass")
            ]
            pt, eta, phi, mass = [
                ak.values_astype(
                    ak.fill_none(ak.pad_none(v, 2)[:, :2], 0.0), np.float64
                )
                for v in (pt, eta, phi, mass)
            ]
            px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
            energy = np.sqrt(px * px + py * py + pz * pz + mass * mass)
            x, y, z, e = [ak.sum(v, axis=1) for v in (px, py, pz, energy)]
            two = out["njets" + suffix] >= 2
            out["dijet_mass" + suffix] = ak.where(
                two, np.sqrt(np.maximum(e * e - x * x - y * y - z * z, 0.0)), -10.0
            )
            out["dijet_eta" + suffix] = abs(
                out["jet1_eta" + suffix] - out["jet2_eta" + suffix]
            )
            if "Jet_isGenMatched" + suffix in out:
                out["n_jets_matched_genjet" + suffix] = ak.sum(
                    out["Jet_isGenMatched" + suffix][:, :2], axis=1
                )
            if bit == 1:
                for name, value in saved_scalars.items():
                    if name.startswith("n_jets_matched_genjet"):
                        continue
                    if (suffix and name.endswith(suffix)) or (
                        not suffix and "__" not in name
                    ):
                        out[name] = value
            for name in HIDDEN:
                out.pop(name + suffix, None)
        return out

    def arrays(self, expressions=None, library="ak", **kwargs):
        requested = self.keys() if expressions is None else list(expressions)
        out = self._select(
            self.tree.arrays(self._read_names(requested), library="ak", **kwargs)
        )
        return (
            {k: numpy_column(out[k]) for k in requested}
            if library == "np"
            else ak.zip({k: out[k] for k in requested}, depth_limit=1)
        )

    def iterate(self, expressions=None, library="ak", **kwargs):
        requested = self.keys() if expressions is None else list(expressions)
        for arrays in self.tree.iterate(
            self._read_names(requested), library="ak", **kwargs
        ):
            out = self._select(arrays)
            yield (
                {k: numpy_column(out[k]) for k in requested}
                if library == "np"
                else ak.zip({k: out[k] for k in requested}, depth_limit=1)
            )


class JetBranch:
    def __init__(self, tree, name):
        self.tree, self.name = tree, name

    def __getattr__(self, attribute):
        source = ALIASES.get(self.name, self.name)
        return getattr(self.tree.tree[source], attribute)

    def array(self, library="ak", **kwargs):
        values = self.tree.arrays([self.name], library=library, **kwargs)
        return values[self.name]
