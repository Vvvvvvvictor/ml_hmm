"""Frozen ggH feature definitions, without plotting dependencies."""
import numpy as np
import awkward as ak
PTDECOR = ()
REQUIRED_JETS = dict(jet1_pt=1, jet1_eta=1, jet2_pt=2, jet2_eta=2, dijet_mass=2, dijet_pt=2, dijet_eta=2, delta_eta_jj=2, delta_phi_jj=2, z_star=2, R_pT=2, min_delta_phi_dimu_jets=1, min_delta_eta_dimu_jets=1)

def vector(pt, eta, phi, mass):
    (px, py, pz) = (pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta))
    energy = np.sqrt(px * px + py * py + pz * pz + mass * mass)
    return np.column_stack((px, py, pz, energy))

def rapidity(v):
    return np.arctanh(v[:, 2] / v[:, 3])

def boost(v, beta):
    b2 = np.sum(beta * beta, axis=1)
    gamma = 1 / np.sqrt(1 - b2)
    dot = np.sum(beta * v[:, :3], axis=1)
    factor = np.divide(gamma - 1, b2, out=np.zeros_like(b2), where=b2 > 0) * dot + gamma * v[:, 3]
    return np.column_stack((v[:, :3] + factor[:, None] * beta, gamma * (v[:, 3] + dot)))

def collins_soper(mu1, mu2):
    """Reproduce calc_collins_soper_angles, including its Mu2/beam convention."""
    dimu = mu1 + mu2
    beta = -dimu[:, :3] / dimu[:, 3, None]
    beam = dimu[:, 3] / 2
    zero = np.zeros_like(beam)
    a = boost(np.column_stack((zero, zero, beam, beam)), beta)[:, :3]
    b = boost(np.column_stack((zero, zero, -beam, beam)), beta)[:, :3]

    def unit(v):
        return np.divide(v, np.linalg.norm(v, axis=1)[:, None], out=np.full_like(v, np.nan), where=np.linalg.norm(v, axis=1)[:, None] > 0)
    z = unit(a - b)
    y = unit(np.cross(a, b))
    x = unit(np.cross(y, z))
    mu = boost(mu2, beta)[:, :3]
    phi = np.arctan2(np.sum(mu * y, axis=1), np.sum(mu * x, axis=1))
    phi = (phi + np.pi) % (2 * np.pi) - np.pi
    costheta = np.divide(np.sum(mu * z, axis=1), np.linalg.norm(mu, axis=1), out=np.full(len(mu), np.nan), where=np.linalg.norm(mu, axis=1) > 0)
    return (phi, costheta)

def features(a, jets, era):
    mu = [vector(np.asarray(a[f'pt_kit_bsc_{i}'], float), np.asarray(a[f'mu{i}_fromH_eta'], float), np.asarray(a[f'mu{i}_fromH_phi'], float), 0.1056584) for i in (1, 2)]
    bare = mu[0] + mu[1]
    fsr = bare.copy()
    for i in (1, 2):
        pt = np.asarray(a[f'fsrPhoton_pt_{i}'], float)
        good = (pt > 0) & (np.asarray(a[f'fsrPhoton_dROverEt2_{i}']) < 0.012)
        fsr += vector(np.where(good, pt, 0), np.where(good, np.asarray(a[f'fsrPhoton_eta_{i}'], float), 0), np.where(good, np.asarray(a[f'fsrPhoton_phi_{i}'], float), 0), 0.0)
    ptll = np.hypot(fsr[:, 0], fsr[:, 1])
    mass = np.sqrt(np.maximum(fsr[:, 3] ** 2 - np.sum(fsr[:, :3] ** 2, axis=1), 0))
    etall = np.arcsinh(np.divide(fsr[:, 2], ptll, out=np.zeros_like(ptll), where=ptll > 0))
    phill = np.arctan2(fsr[:, 1], fsr[:, 0])
    barept = np.hypot(bare[:, 0], bare[:, 1])
    bareeta = np.arcsinh(np.divide(bare[:, 2], barept, out=np.zeros_like(ptll), where=barept > 0))
    barephi = np.arctan2(bare[:, 1], bare[:, 0])
    njet = ak.to_numpy(ak.num(jets[0]))
    order = ak.argsort(jets[0], axis=1, ascending=False)[:, :2]
    padded = [ak.to_numpy(ak.fill_none(ak.pad_none(v[order], 2, axis=1, clip=True), 0.0)) for v in jets[:4]]
    (j1, j2) = [vector(*(v[:, i] for v in padded)) for i in (0, 1)]
    dijet = j1 + j2
    dijetpt = np.hypot(dijet[:, 0], dijet[:, 1])
    dijetmass = np.sqrt(np.maximum(dijet[:, 3] ** 2 - np.sum(dijet[:, :3] ** 2, axis=1), 0))

    def dphi(x, y):
        return np.abs((x - y + np.pi) % (2 * np.pi) - np.pi)
    with np.errstate(divide='ignore', invalid='ignore'):
        zstar = (rapidity(bare) - (rapidity(j1) + rapidity(j2)) / 2) / np.abs(rapidity(j1) - rapidity(j2))
        balance = np.hypot(bare[:, 0] + dijet[:, 0], bare[:, 1] + dijet[:, 1]) / (barept + padded[0].sum(axis=1))
    (phi, costheta) = collins_soper(*mu)
    result = dict(diMufsr_kit_BSC_pt=ptll, diMufsr_kit_BSC_eta=etall, mu1_fromH_eta=np.asarray(a.mu1_fromH_eta), mu2_fromH_eta=np.asarray(a.mu2_fromH_eta), phi_CS=phi, cos_theta_CS=costheta, pTMu1oMmm=np.asarray(a.pt_kit_bsc_1) / mass, pTMu2oMmm=np.asarray(a.pt_kit_bsc_1) / mass, njets=njet, z_star=zstar, R_pT=balance, dijet_mass=dijetmass, dijet_pt=dijetpt, dijet_eta=np.arcsinh(np.divide(dijet[:, 2], dijetpt, out=np.zeros_like(ptll), where=dijetpt > 0)), delta_eta_jj=np.abs(padded[1][:, 0] - padded[1][:, 1]), delta_phi_jj=dphi(padded[2][:, 0], padded[2][:, 1]), jet1_pt=padded[0][:, 0], jet1_eta=padded[1][:, 0], jet2_pt=padded[0][:, 1], jet2_eta=padded[1][:, 1], min_delta_phi_dimu_jets=np.where(njet == 1, dphi(phill, padded[2][:, 0]), np.minimum(dphi(barephi, padded[2][:, 0]), dphi(barephi, padded[2][:, 1]))), min_delta_eta_dimu_jets=np.where(njet == 1, np.abs(etall - padded[1][:, 0]), np.minimum(np.abs(bareeta - padded[1][:, 0]), np.abs(bareeta - padded[1][:, 1]))), source_year=np.full(len(njet), int(era[:4])))
    valid = {name: np.isfinite(value) & (njet >= REQUIRED_JETS.get(name, 0)) for (name, value) in result.items()}
    for name in PTDECOR:
        result[name + '__over_mass'] = result[name] / mass
        valid[name + '__over_mass'] = valid[name] & np.isfinite(mass) & (mass > 0)
    return (result, valid)
