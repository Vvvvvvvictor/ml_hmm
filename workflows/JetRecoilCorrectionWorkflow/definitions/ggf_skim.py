import numpy as np
import awkward as ak
#import uproot_methods
import uproot3_methods
####import pandas as pd
####import joblib
#from tensorflow.keras.models import load_model
#from tensorflow.keras.backend import clear_session
#from sklearn.ensemble import RandomForestClassifier

def calculate_y_star(mu1, mu2, jet1, jet2):
    y1 = jet1.rapidity
    y2 = jet2.rapidity
    y_H = (mu1 + mu2).rapidity  # Rapidity of the dimuon system
    return y_H - (y1 + y2) / 2

def calculate_z_star(y_star, jet1, jet2):
    y1 = jet1.rapidity
    y2 = jet2.rapidity
    return y_star / abs(y1 - y2)

def calculate_R_pT(mu1, mu2, jet1, jet2):
    pT_mumujj = (mu1 + mu2 + jet1 + jet2).pt
    pT_mu_mu = (mu1 + mu2).pt
    return pT_mumujj / (jet1.pt + jet2.pt + pT_mu_mu)

def calculate_R_pT_Mag(mu1, mu2, jet1, jet2):
    # Calculate the transverse momentum vectors manually
    #pTj1 = jet1.to_vector().set_z(0)
    #pTj2 = jet2.to_vector().set_z(0)
    #pT_dimuon = (mu1 + mu2).to_vector().set_z(0)
    jet1_vec = uproot3_methods.TVector3Array(jet1.x, jet1.y, np.zeros_like(jet1.x))
    jet2_vec = uproot3_methods.TVector3Array(jet2.x, jet2.y, np.zeros_like(jet2.x))
    dimuon_vec = uproot3_methods.TVector3Array((mu1+mu2).x, (mu1+mu2).y, np.zeros_like((mu1+mu2).x))

    pT_combined = jet1_vec + jet2_vec + dimuon_vec
    #magnitude_manual = np.sqrt(pT_combined.x**2 + pT_combined.y**2)

    # Calculate the combined transverse momentum vector
    #pT_combined = pTj1 + pTj2 + pT_di-mu
    magnitude_manual = pT_combined.mag
    pT_mumujj = (mu1 + mu2 + jet1 + jet2).pt
    #pT_mu_mu = (mu1 + mu2).pt
    #return pT_mumujj / (jet1.pt + jet2.pt + pT_mu_mu)
    return pT_mumujj / magnitude_manual

def calculate_relative_mass_uncertainty(delta_pT_mu1, pT_mu1, delta_pT_mu2, pT_mu2):
    relative_uncertainty = (1/2) * ((delta_pT_mu1 / pT_mu1)**2 + (delta_pT_mu2 / pT_mu2)**2)**0.5
    return relative_uncertainty

def min_delta_eta(mu1, mu2, jet1, jet2):

    dimuon = mu1 + mu2
    eta_dimuon = dimuon.eta

    # Initialize the minimum pseudorapidity difference with a large number
    #min_delta_eta = float('inf')
    min_delta_eta = np.full(mu1.size, np.inf)

    # Calculate the pseudorapidity differences
    delta_eta_jet1 = np.abs(eta_dimuon - jet1.eta)
    delta_eta_jet2 = np.abs(eta_dimuon - jet2.eta)

    # Loop over the jets to calculate the pseudorapidity differences
    #for jet in jets:
    #    delta_eta = abs(eta_dimuon - jet.Eta())
    #    if delta_eta < min_delta_eta:
    #        min_delta_eta = delta_eta
    min_delta_eta = np.minimum(delta_eta_jet1, delta_eta_jet2)

    return min_delta_eta


def min_delta_phi(mu1, mu2, jet1, jet2):
    dimuon = mu1 + mu2
    phi_dimuon = dimuon.phi

    delta_phi_jet1 = np.abs(phi_dimuon - jet1.phi)
    delta_phi_jet1 = np.where(delta_phi_jet1 > np.pi, 2*np.pi - delta_phi_jet1, delta_phi_jet1)

    delta_phi_jet2 = np.abs(phi_dimuon - jet2.phi)
    delta_phi_jet2 = np.where(delta_phi_jet2 > np.pi, 2*np.pi - delta_phi_jet2, delta_phi_jet2)

    # Minimum of the two
    min_delta_phi = np.minimum(delta_phi_jet1, delta_phi_jet2)
    return min_delta_phi

def calc_collins_soper_angles(Mu1, Mu2):
    dimu = Mu1 + Mu2

    beta_vectors = dimu.p3 / dimu.energy
    neg_betas = uproot3_methods.TVector3Array(
        -beta_vectors.x,
        -beta_vectors.y,
        -beta_vectors.z
    )

    Mu1_rf = Mu1.boost(neg_betas)
    Mu2_rf = Mu2.boost(neg_betas)

    zeros = np.zeros(len(dimu))
    beamE = dimu.energy / 2

    pA_lab = uproot3_methods.TLorentzVectorArray.from_xyzm(zeros, zeros, +beamE, zeros)
    pB_lab = uproot3_methods.TLorentzVectorArray.from_xyzm(zeros, zeros, -beamE, zeros)

    # Explicitly set energy component
    pA_lab._content['fT'] = beamE
    pB_lab._content['fT'] = beamE

    pA_rf = pA_lab.boost(neg_betas)
    pB_rf = pB_lab.boost(neg_betas)

    def vec(x, y, z):
        return np.stack([x, y, z], axis=-1)

    # Explicitly get x,y,z from _content after boosts
    pA_v = vec(pA_rf._content["fX"], pA_rf._content["fY"], pA_rf._content["fZ"])
    pB_v = vec(pB_rf._content["fX"], pB_rf._content["fY"], pB_rf._content["fZ"])

    z_cs = pA_v - pB_v
    z_cs /= np.linalg.norm(z_cs, axis=-1, keepdims=True)
    y_cs = np.cross(pA_v, pB_v)
    y_cs /= np.linalg.norm(y_cs, axis=-1, keepdims=True)
    x_cs = np.cross(y_cs, z_cs)
    x_cs /= np.linalg.norm(x_cs, axis=-1, keepdims=True)

    # Also explicitly get muon components after boost
    p_mu_minus = vec(Mu2_rf._content["fX"], Mu2_rf._content["fY"], Mu2_rf._content["fZ"])
    p_mu_minus_mag = np.linalg.norm(p_mu_minus, axis=-1)

    phi_CS = np.arctan2(
        np.sum(p_mu_minus * y_cs, axis=-1),
        np.sum(p_mu_minus * x_cs, axis=-1)
    )
    cos_theta_CS = np.sum(p_mu_minus * z_cs, axis=-1) / p_mu_minus_mag

    #phi_CS = np.mod(phi_CS, 2 * np.pi)
    phi_CS = (phi_CS + np.pi) % (2 * np.pi) - np.pi

    return phi_CS, cos_theta_CS


def select(data):

	data['muon_mass']=0.1056584

	Mu1  = uproot3_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(data['mu1_fromH_pt'],data['mu1_fromH_eta'],data['mu1_fromH_phi'],data['muon_mass'])
	Mu2  = uproot3_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(data['mu2_fromH_pt'],data['mu2_fromH_eta'],data['mu2_fromH_phi'],data['muon_mass'])
	jet1  = uproot3_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(data['jet1_pt'],data['jet1_eta'],data['jet1_phi'],data['jet1_mass'])
	jet2  = uproot3_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(data['jet2_pt'],data['jet2_eta'],data['jet2_phi'],data['jet2_mass'])

	BSC_1_ptErr = np.where(data['mu1_fromH_bsConstrainedChi2'] < 30, data['mu1_fromH_bsConstrainedPtErr'], data['mu1_fromH_ptErr'])
	BSC_2_ptErr = np.where(data['mu2_fromH_bsConstrainedChi2'] < 30, data['mu2_fromH_bsConstrainedPtErr'], data['mu2_fromH_ptErr'])

	##BSC_1_pt_rc = np.where(data['mu1_fromH_bsConstrainedChi2'] < 30, data['pt_rc_bsc_1'], data['pt_rc_1'])
	##BSC_2_pt_rc = np.where(data['mu2_fromH_bsConstrainedChi2'] < 30, data['pt_rc_bsc_2'], data['pt_rc_2'])

	##Mu1_rc_BSC  = uproot3_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(BSC_1_pt_rc,data['mu1_fromH_eta'],data['mu1_fromH_phi'],data['muon_mass'])
	##Mu2_rc_BSC  = uproot3_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(BSC_2_pt_rc,data['mu2_fromH_eta'],data['mu2_fromH_phi'],data['muon_mass'])

	BSC_1_pt = np.where(data['mu1_fromH_bsConstrainedChi2'] < 30, data['mu1_fromH_bsConstrainedPt'], data['mu1_fromH_pt'])
	BSC_2_pt = np.where(data['mu2_fromH_bsConstrainedChi2'] < 30, data['mu2_fromH_bsConstrainedPt'], data['mu2_fromH_pt'])
	Mu1_BSC  = uproot3_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(BSC_1_pt,data['mu1_fromH_eta'],data['mu1_fromH_phi'],data['muon_mass'])
	Mu2_BSC  = uproot3_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(BSC_2_pt,data['mu2_fromH_eta'],data['mu2_fromH_phi'],data['muon_mass'])

	#
	# mu1_formH_pt is original
	# rc is rocc correction; kit is KIT correction
	##BSC_1_pt_kit = np.where(data['mu1_fromH_bsConstrainedChi2'] < 30, data['pt_kit_bsc_1'], data['pt_kit_1'])
	##BSC_2_pt_kit = np.where(data['mu2_fromH_bsConstrainedChi2'] < 30, data['pt_kit_bsc_2'], data['pt_kit_2'])
	BSC_1_pt_kit = data['pt_kit_bsc_1']
	BSC_2_pt_kit = data['pt_kit_bsc_2']
	Mu1_kit_BSC  = uproot3_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(BSC_1_pt_kit,data['mu1_fromH_eta'],data['mu1_fromH_phi'],data['muon_mass'])
	Mu2_kit_BSC  = uproot3_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(BSC_2_pt_kit,data['mu2_fromH_eta'],data['mu2_fromH_phi'],data['muon_mass'])

	##Mu1_kit  = uproot3_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(data['pt_kit_1'],data['mu1_fromH_eta'],data['mu1_fromH_phi'],data['muon_mass'])
	##Mu2_kit  = uproot3_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(data['pt_kit_2'],data['mu2_fromH_eta'],data['mu2_fromH_phi'],data['muon_mass'])

	##data['BSC_pt_rc_1']=BSC_1_pt_rc
	##data['BSC_pt_rc_2']=BSC_2_pt_rc
	data['BSC_pt_1']=BSC_1_pt
	data['BSC_pt_2']=BSC_2_pt
	data['BSC_pt_kit_1']=BSC_1_pt_kit
	data['BSC_pt_kit_2']=BSC_2_pt_kit
	data['BSC_ptErr_1']=BSC_1_ptErr
	data['BSC_ptErr_2']=BSC_2_ptErr

        # Extract px and py components and create TVector3 objects
        #jet1_vec = uproot3_methods.TVector3Array(jet1.px, jet1.py, np.zeros_like(jet1.px))
        #jet2_vec = uproot3_methods.TVector3Array(jet2.px, jet2.py, np.zeros_like(jet2.px))
        #dimuon_vec = uproot3_methods.TVector3Array((Mu1+Mu2).px, (Mu1+Mu2).py, np.zeros_like((Mu1+Mu2).px))

	##mask1 = (data['jet1_pt'] < 50) & (abs(data['jet1_eta'])>2.5) & (abs(data['jet1_eta'])<3)
	##mask2 = (data['jet2_pt'] < 50) & (abs(data['jet2_eta'])>2.5) & (abs(data['jet2_eta'])<3)
	##data['jet1_pt']   = np.where(mask1, -100, data['jet1_pt'])
	##data['jet1_eta']  = np.where(mask1, -100, data['jet1_eta'])
	##data['jet1_phi']  = np.where(mask1, -100, data['jet1_phi'])
	##data['jet1_mass'] = np.where(mask1, -100, data['jet1_mass'])
	##data['jet2_pt']   = np.where(mask1, -100, data['jet2_pt'])
	##data['jet2_eta']  = np.where(mask1, -100, data['jet2_eta'])
	##data['jet2_phi']  = np.where(mask1, -100, data['jet2_phi'])
	##data['jet2_mass'] = np.where(mask1, -100, data['jet2_mass'])
	##data['jet2_pt']   = np.where(mask2, -100, data['jet2_pt'])
	##data['jet2_eta']  = np.where(mask2, -100, data['jet2_eta'])
	##data['jet2_phi']  = np.where(mask2, -100, data['jet2_phi'])
	##data['jet2_mass'] = np.where(mask2, -100, data['jet2_mass'])
	##data['njets'] = np.where(mask1, 0, data['njets'])
	###data['njets'] = np.where(mask2, 0, data['njets'])
	##data['njets'] = np.where((~mask1) & mask2, 1, data['njets'])
	###


	dijet = jet1 + jet2
	data['dijet_pt'] = dijet.pt
	data['dijet_eta'] = dijet.eta
	data['dijet_phi'] = dijet.phi
	data['dijet_mass'] = dijet.mass
	data['dijet_pt'] = data['dijet_pt'] * (data['njets'] > 1)
	data['dijet_eta'] = data['dijet_eta'] * (data['njets'] > 1)
	data['dijet_phi'] = data['dijet_phi'] * (data['njets'] > 1)
	data['dijet_mass'] = data['dijet_mass'] * (data['njets'] > 1)
	data['dijet_eta'] = np.where(data['njets'] < 2, -100, data['dijet_eta'])
	data['dijet_phi'] = np.where(data['njets'] < 2, -100, data['dijet_phi'])
	#data['delta_eta_dijet'] = abs(data['jet1_eta']-data['jet2_eta'])
	#data['delta_eta_dijet'] = np.where(data['njets'] < 2, -100, data['delta_eta_dijet'])
	data['delta_eta_jj'] = abs(data['jet1_eta']-data['jet2_eta'])
	data['delta_eta_jj'] = np.where(data['njets'] < 2, -100, data['delta_eta_jj'])
	delta_phi_jj = np.abs(jet1.phi - jet2.phi)
	data['delta_phi_jj'] = np.where(delta_phi_jj > np.pi, 2*np.pi - delta_phi_jj, delta_phi_jj)
	data['delta_phi_jj'] = np.where(data['njets'] < 2, -100, data['delta_phi_jj'])

	### raw is not in v1
	##jet1_raw  = uproot3_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(data['jet1_rawpT'],data['jet1_eta'],data['jet1_phi'],data['jet1_rawMass'])
	##jet2_raw  = uproot3_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(data['jet2_rawpT'],data['jet2_eta'],data['jet2_phi'],data['jet2_rawMass'])
	##dijet_raw = jet1_raw + jet2_raw
	##data['dijet_raw_pt'] = dijet_raw.pt
	##data['dijet_raw_eta'] = dijet_raw.eta
	##data['dijet_raw_phi'] = dijet_raw.phi
	##data['dijet_raw_mass'] = dijet_raw.mass
	##data['dijet_raw_pt'] = data['dijet_raw_pt'] * (data['njets'] > 1)
	##data['dijet_raw_eta'] = data['dijet_raw_eta'] * (data['njets'] > 1)
	##data['dijet_raw_phi'] = data['dijet_raw_phi'] * (data['njets'] > 1)
	##data['dijet_raw_mass'] = data['dijet_raw_mass'] * (data['njets'] > 1)

	##data['dijet_raw_eta'] = np.where(data['njets'] < 2, -100, data['dijet_raw_eta'])
	##data['dijet_raw_phi'] = np.where(data['njets'] < 2, -100, data['dijet_raw_phi'])
	###data.loc[mask1, ['dijet_raw_eta', 'dijet_raw_phi']] = -100
	###data.loc[mask1, ['dijet_raw_pt', 'dijet_raw_mass']] = 0
	###data.loc[mask2, ['dijet_raw_eta', 'dijet_raw_phi']] = -100
	###data.loc[mask2, ['dijet_raw_pt', 'dijet_raw_mass']] = 0

	###fsrPhoton1  = uproot3_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(data['fsrPhoton_pt_1'],data['fsrPhoton_eta_1'],data['fsrPhoton_phi_1'],0)
	###fsrPhoton2  = uproot3_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(data['fsrPhoton_pt_2'],data['fsrPhoton_eta_2'],data['fsrPhoton_phi_2'],0)

	#data["y_star"] = calculate_y_star(Mu1,Mu2,jet1,jet2)
	#data["z_star"] = calculate_z_star(data["y_star"],jet1,jet2)
	#data["log_z_star"] = np.log(data["z_star"])
	#data["R_pT_Mag"]   = calculate_R_pT_Mag(Mu1,Mu2,jet1,jet2)
	#data["R_pT"]   = calculate_R_pT(Mu1,Mu2,jet1,jet2)
	#data["diMu-mass_resolution"]   = calculate_relative_mass_uncertainty(data["mu1_fromH_ptErr"], data["mu1_fromH_pt"], data["mu2_fromH_ptErr"], data["mu2_fromH_pt"])
	#data["diMu-mass_resolution_BS"]   = calculate_relative_mass_uncertainty(BSC_1_ptErr, BSC_1_pt, BSC_2_ptErr, BSC_2_pt)

	##Mu1_rc  = uproot3_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(data['pt_rc_1'],data['mu1_fromH_eta'],data['mu1_fromH_phi'],data['muon_mass'])
	##Mu2_rc  = uproot3_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(data['pt_rc_2'],data['mu2_fromH_eta'],data['mu2_fromH_phi'],data['muon_mass'])
	##diMu_rc = Mu1_rc + Mu2_rc
	##data['diMu_pt_rc'] = diMu_rc.pt
	##data['diMu_eta_rc'] = diMu_rc.eta
	##data['diMu_phi_rc'] = diMu_rc.phi
	##data['diMu_mass_rc'] = diMu_rc.mass
	##data['mu1_pt_diff'] = data['pt_rc_1'] - data['mu1_fromH_pt']
	##data['mu2_pt_diff'] = data['pt_rc_2'] - data['mu2_fromH_pt']

	##diMu_kit = Mu1_kit + Mu2_kit
	##data['diMu_pt_kit'] = diMu_kit.pt
	##data['diMu_eta_kit'] = diMu_kit.eta
	##data['diMu_phi_kit'] = diMu_kit.phi
	##data['diMu_mass_kit'] = diMu_kit.mass


	data['fsrPhoton1_tight'] = ( (data['fsrPhoton_dROverEt2_1'] < 0.012) & (data['fsrPhoton_pt_1'] > 0) )
	data['fsrPhoton2_tight'] = ( (data['fsrPhoton_dROverEt2_2'] < 0.012) & (data['fsrPhoton_pt_2'] > 0) )
	#data['fsrPhoton_pt_1'] = data['fsrPhoton_pt_1'] & np.logical_not(data['fsrPhoton1_tight'])
	#data['fsrPhoton_eta_1'] = data['fsrPhoton_eta_1'] & np.logical_not(data['fsrPhoton1_tight'])
	#data['fsrPhoton_phi_1'] = data['fsrPhoton_phi_1'] & np.logical_not(data['fsrPhoton1_tight'])
	#data['fsrPhoton_pt_2'] = data['fsrPhoton_pt_2'] & np.logical_not(data['fsrPhoton2_tight'])
	#data['fsrPhoton_eta_2'] = data['fsrPhoton_eta_2'] & np.logical_not(data['fsrPhoton2_tight'])
	#data['fsrPhoton_phi_2'] = data['fsrPhoton_phi_2'] & np.logical_not(data['fsrPhoton2_tight'])
	#data['fsrPhoton_pt_1'] = np.where(np.logical_not(data['fsrPhoton1_tight']), data['fsrPhoton_pt_1'], 0)
	#data['fsrPhoton_eta_1'] = np.where(np.logical_not(data['fsrPhoton1_tight']), data['fsrPhoton_eta_1'], 0)
	#data['fsrPhoton_phi_1'] = np.where(np.logical_not(data['fsrPhoton1_tight']), data['fsrPhoton_phi_1'], 0)
	#data['fsrPhoton_pt_2'] = np.where(np.logical_not(data['fsrPhoton2_tight']), data['fsrPhoton_pt_2'], 0)
	#data['fsrPhoton_eta_2'] = np.where(np.logical_not(data['fsrPhoton2_tight']), data['fsrPhoton_eta_2'], 0)
	#data['fsrPhoton_phi_2'] = np.where(np.logical_not(data['fsrPhoton2_tight']), data['fsrPhoton_phi_2'], 0)
	data['fsrPhoton_pt_1']  = data['fsrPhoton1_tight'] * data['fsrPhoton_pt_1']
	data['fsrPhoton_eta_1'] = data['fsrPhoton1_tight'] * data['fsrPhoton_eta_1']
	data['fsrPhoton_phi_1'] = data['fsrPhoton1_tight'] * data['fsrPhoton_phi_1']
	data['fsrPhoton_pt_2']  = data['fsrPhoton2_tight'] * data['fsrPhoton_pt_2']
	data['fsrPhoton_eta_2'] = data['fsrPhoton2_tight'] * data['fsrPhoton_eta_2']
	data['fsrPhoton_phi_2'] = data['fsrPhoton2_tight'] * data['fsrPhoton_phi_2']


	fsrPhoton1  = uproot3_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(data['fsrPhoton_pt_1'],data['fsrPhoton_eta_1'],data['fsrPhoton_phi_1'],0)
	fsrPhoton2  = uproot3_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(data['fsrPhoton_pt_2'],data['fsrPhoton_eta_2'],data['fsrPhoton_phi_2'],0)

	muonfsr1 = fsrPhoton1 + Mu1
	##muonfsr1_rc = fsrPhoton1 + Mu1_rc
	muonfsr2 = fsrPhoton2 + Mu2
	##muonfsr2_rc = fsrPhoton2 + Mu2_rc

	diMuonfsr = muonfsr1 + muonfsr2
	##diMuonfsr_rc = muonfsr1_rc + muonfsr2_rc

	data['Mu1_fsr_pt']   = (muonfsr1).pt
	data['Mu2_fsr_pt']   = (muonfsr2).pt
	data['diMufsr_pt']   = (diMuonfsr).pt
	data['diMufsr_eta']  = (diMuonfsr).eta
	data['diMufsr_phi']  = (diMuonfsr).phi
	data['diMufsr_mass'] = (diMuonfsr).mass

	##data['diMufsr_rc_pt']   = (diMuonfsr_rc).pt
	##data['diMufsr_rc_eta']  = (diMuonfsr_rc).eta
	##data['diMufsr_rc_phi']  = (diMuonfsr_rc).phi
	##data['diMufsr_rc_mass'] = (diMuonfsr_rc).mass

	##muonfsr1_rc_BSC = fsrPhoton1 + Mu1_rc_BSC
	##muonfsr2_rc_BSC = fsrPhoton2 + Mu2_rc_BSC
	##diMuonfsr_rc_BSC = muonfsr1_rc_BSC + muonfsr2_rc_BSC
	##data['Mu1_fsr_rc_BSC_pt']   = (muonfsr1_rc_BSC).pt
	##data['Mu2_fsr_rc_BSC_pt']   = (muonfsr2_rc_BSC).pt
	##data['diMufsr_rc_BSC_pt']   = (diMuonfsr_rc_BSC).pt
	##data['diMufsr_rc_BSC_eta']  = (diMuonfsr_rc_BSC).eta
	##data['diMufsr_rc_BSC_phi']  = (diMuonfsr_rc_BSC).phi
	##data['diMufsr_rc_BSC_mass'] = (diMuonfsr_rc_BSC).mass

	muonfsr1_BSC = fsrPhoton1 + Mu1_BSC
	muonfsr2_BSC = fsrPhoton2 + Mu2_BSC
	diMuonfsr_BSC = muonfsr1_BSC + muonfsr2_BSC
	data['Mu1_fsr_BSC_pt']   = (muonfsr1_BSC).pt
	data['Mu2_fsr_BSC_pt']   = (muonfsr2_BSC).pt
	data['diMufsr_BSC_pt']   = (diMuonfsr_BSC).pt
	data['diMufsr_BSC_eta']  = (diMuonfsr_BSC).eta
	data['diMufsr_BSC_phi']  = (diMuonfsr_BSC).phi
	data['diMufsr_BSC_mass'] = (diMuonfsr_BSC).mass

	muonfsr1_kit_BSC = fsrPhoton1 + Mu1_kit_BSC
	muonfsr2_kit_BSC = fsrPhoton2 + Mu2_kit_BSC
	diMuonfsr_kit_BSC = muonfsr1_kit_BSC + muonfsr2_kit_BSC
	data['Mu1_fsr_kit_BSC_pt']   = (muonfsr1_kit_BSC).pt
	data['Mu2_fsr_kit_BSC_pt']   = (muonfsr2_kit_BSC).pt
	data['diMufsr_kit_BSC_pt']   = (diMuonfsr_kit_BSC).pt
	data['diMufsr_kit_BSC_eta']  = (diMuonfsr_kit_BSC).eta
	data['diMufsr_kit_BSC_phi']  = (diMuonfsr_kit_BSC).phi
	data['diMufsr_kit_BSC_mass'] = (diMuonfsr_kit_BSC).mass

	##muonfsr1_kit = fsrPhoton1 + Mu1_kit
	##muonfsr2_kit = fsrPhoton2 + Mu2_kit
	##diMuonfsr_kit = muonfsr1_kit + muonfsr2_kit
	##data['Mu1_fsr_kit_pt']   = (muonfsr1_kit).pt
	##data['Mu2_fsr_kit_pt']   = (muonfsr2_kit).pt
	##data['diMufsr_kit_pt']   = (diMuonfsr_kit).pt
	##data['diMufsr_kit_eta']  = (diMuonfsr_kit).eta
	##data['diMufsr_kit_phi']  = (diMuonfsr_kit).phi
	##data['diMufsr_kit_mass'] = (diMuonfsr_kit).mass

	data["y_star"] = calculate_y_star(Mu1_kit_BSC,Mu2_kit_BSC,jet1,jet2)
	data["z_star"] = calculate_z_star(data["y_star"],jet1,jet2)
	data["log_z_star"] = np.log(data["z_star"])
	data["R_pT_Mag"]   = calculate_R_pT_Mag(Mu1_kit_BSC,Mu2_kit_BSC,jet1,jet2)
	data["R_pT"]   = calculate_R_pT(Mu1_kit_BSC,Mu2_kit_BSC,jet1,jet2)
	data["diMu-mass_resolution"]   = calculate_relative_mass_uncertainty(data["mu1_fromH_ptErr"], data["mu1_fromH_pt"], data["mu2_fromH_ptErr"], data["mu2_fromH_pt"])
	data["diMu-mass_resolution_BSC"]   = calculate_relative_mass_uncertainty(BSC_1_ptErr, BSC_1_pt, BSC_2_ptErr, BSC_2_pt)
	#data["diMu-mass_resolution_abs"] = data["diMu-mass_resolution"] * data['diMufsr_kit_mass']
	data["diMu-mass_resolution_abs"] = data["diMu-mass_resolution"] * data['diMufsr_mass']
	data["diMu-mass_resolution_BSC_abs"] = data["diMu-mass_resolution_BSC"] * data['diMufsr_kit_BSC_mass']
	data["log_dijet_mass"] = np.log(data["dijet_mass"])
	#3#data["log_diMufsr_kit_pt"] = np.log(data["diMufsr_kit_pt"])
	data["log_diMufsr_kit_BSC_pt"] = np.log(data["diMufsr_kit_BSC_pt"])
	data["min_delta_eta_dimu_jets"] = min_delta_eta(Mu1_kit_BSC, Mu2_kit_BSC, jet1, jet2)

	data['phi_CS'], data['cos_theta_CS'] = calc_collins_soper_angles(Mu1_kit_BSC, Mu2_kit_BSC)
	data['pTMu1oMmm'] = (Mu1_kit_BSC).pt/(diMuonfsr_kit_BSC).mass
	data['pTMu2oMmm'] = (Mu1_kit_BSC).pt/(diMuonfsr_kit_BSC).mass
	data['min_delta_phi_dimu_jets'] = min_delta_phi(Mu1_kit_BSC, Mu2_kit_BSC, jet1, jet2)
	data['1oDiMuResolution'] = 1/data["diMu-mass_resolution"]
	data['1oDiMuResolution_BSC'] = 1/data["diMu-mass_resolution_BSC"]

	dimu =Mu1 + Mu2
	delta_phi = np.abs(diMuonfsr_kit_BSC.phi - jet1.phi)
	data['delta_phi_dimu_jet1'] = np.where(delta_phi > np.pi, 2*np.pi - delta_phi, delta_phi)
	data['delta_eta_dimu_jet1'] = np.abs(diMuonfsr_kit_BSC.eta - jet1.eta)

	data['delta_eta_jj'] = np.where(data['njets']<2, -100, data['delta_eta_jj'])
	data['delta_phi_jj'] = np.where(data['njets']<2, -100, data['delta_phi_jj'])
	#
	#data['id_MuonEffup'] = (data["id_wgt_mu_1__MuonEffup"]*data["id_wgt_mu_2__MuonEffup"])/(data["id_wgt_mu_1"]*data["id_wgt_mu_2"])
	#data['id_MuonEffdown'] = (data["id_wgt_mu_1__MuonEffdown"]*data["id_wgt_mu_2__MuonEffdown"])/(data["id_wgt_mu_1"]*data["id_wgt_mu_2"])
	#data['iso_MuonEffup'] = (data["iso_wgt_mu_1__MuonEffup"]*data["iso_wgt_mu_2__MuonEffup"])/(data["iso_wgt_mu_1"]*data["iso_wgt_mu_2"])
	#data['iso_MuonEffdown'] = (data["iso_wgt_mu_1__MuonEffdown"]*data["iso_wgt_mu_2__MuonEffdown"])/(data["iso_wgt_mu_1"]*data["iso_wgt_mu_2"])

	# Set all jet variables to 0 if there are no jets
	for key in ['jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass', 'delta_eta_dimu_jet1', 'delta_phi_dimu_jet1', 'min_delta_eta_dimu_jets', 'min_delta_phi_dimu_jets']:
		data[key] = np.where(data['njets'] == 0, -100, data[key])

	# Set all jet2 variables to 0 if there are less than 2 jets
	for key in ['jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass', 'dijet_mass', 'dijet_eta', 'dijet_pt', 'dijet_phi', 'y_star', 'z_star', 'R_pT', 'log_dijet_mass']:
		data[key] = np.where(data['njets'] < 2, -100, data[key])

	data['min_delta_eta_dimu_jets'] = np.where(data['njets'] == 1, data['delta_eta_dimu_jet1'], data['min_delta_eta_dimu_jets'])
	data['min_delta_phi_dimu_jets'] = np.where(data['njets'] == 1, data['delta_phi_dimu_jet1'], data['min_delta_phi_dimu_jets'])


	data['TTH'] = ((data['nbjets_loose'] >= 2) | (data['nbjets_medium'] >= 1))
	data['VH'] = (~data['TTH']) & ((data['nmuons'] + data['nelectrons']) >= 3)

	data['VBF'] = (
			(~data['TTH']) &
			(~data['VH']) &
			(data["njets"] >= 2) &
			(data["jet1_pt"] >= 35) &
			(data["jet2_pt"] >= 25) &
			(data["delta_eta_jj"] >= 2.5) &
			(data["dijet_mass"] >= 400)
		)
	data['GGH'] = (~data['TTH']) & (~data['VH']) & (~data['VBF'])
	conditions = [data['TTH'], data['VH'], data['VBF'], data['GGH']]
	choices = [1, 2, 3, 4]
	data['cate_index'] = np.select(conditions, choices, default=0)

#
#
#	# --------- Predict discriminator from NN -----------------------
#	#clear_session()
#	#ZModel = load_model("/orange/avery/nikmenendez/Wto3l/Optimizer/MVA/ZSelector_model_alt.h5")
#	##Selector_vars = ["dxyL1", "dzL1", "etaL1", "ip3dL1", "phiL1", "sip3dL1",
#    ##        		 "dxyL2", "dzL2", "etaL2", "ip3dL2", "phiL2", "sip3dL2",
#    ##         		 "dxyL3", "dzL3", "etaL3", "ip3dL3", "phiL3", "sip3dL3",
#    ##         		 "dR12", "dR13", "dR23", "dRM0", "m3l", "mt", "met", "nJets",
#	##				 "M0", "m3l_pt"]
#	#df = pd.DataFrame.from_dict(data)[Selector_vars]
#	#df["sType"], df["Weights"] = 1, 1
#	#maxW = pd.read_pickle("/orange/avery/nikmenendez/Wto3l/Optimizer/MVA/maxes.pkl")
#	#minW = pd.read_pickle("/orange/avery/nikmenendez/Wto3l/Optimizer/MVA/mins.pkl")
#	#df = (df-minW)/(maxW-minW)
#
#	#data["discriminator"] = (ZModel.predict(df[Selector_vars])).ravel()
#	## ---------------------------------------------------------------
#
#	## --------- Predict Class from Random Forest --------------------
#	#rf = joblib.load("/orange/avery/nikmenendez/Wto3l/Optimizer/MVA/forest_model.joblib")
#	##Selector_vars = ["etaL1", "phiL1",
#    ##                 "etaL2", "phiL2",
#    ##                 "etaL3", "phiL3",
#    ##                 "dR12", "dR13", "dR23", "dRM0", "m3l", "mt", "met", "nJets",
#    ##                 "M0", "m3l_pt"]
#	#df  = pd.DataFrame.from_dict(data)[Selector_vars]
#	#data["forestguess"] = rf.predict(df)
#	## ---------------------------------------------------------------

	return data
