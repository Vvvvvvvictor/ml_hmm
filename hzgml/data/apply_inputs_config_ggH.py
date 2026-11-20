"""Apply-time-only sample map for the four non-training Higgs processes."""
from copy import deepcopy
from pathlib import Path
import runpy

_base = Path(__file__).with_name('training_config_BDT_ggH_multiyear_template.py')
CONFIG = deepcopy(runpy.run_path(str(_base))['CONFIG'])
_extra = {'2022': {'ttHToMuMu_M125': 'TTH_Hto2Mu_M-125_TuneCP5_13p6TeV_powheg-pythia8',
          'WplusHToMuMu_M125': 'WplusH_Hto2Mu_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8',
          'WminusHToMuMu_M125': 'WminusH_Hto2Mu_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8',
          'ZHToMuMu_M125': 'ZH_Hto2Mu_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8'},
 '2022EE': {'ttHToMuMu_M125': 'TTH_Hto2Mu_M-125_TuneCP5_13p6TeV_powheg-pythia8',
            'WplusHToMuMu_M125': 'WplusH_Hto2Mu_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8',
            'WminusHToMuMu_M125': 'WminusH_Hto2Mu_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8',
            'ZHToMuMu_M125': 'ZH_Hto2Mu_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8'},
 '2023': {'ttHToMuMu_M125': 'TTH_Hto2Mu_M-125_TuneCP5_13p6TeV_powheg-pythia8',
          'WplusHToMuMu_M125': 'WplusH_Hto2Mu_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8',
          'WminusHToMuMu_M125': 'WminusH_Hto2Mu_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8',
          'ZHToMuMu_M125': 'ZH_Hto2Mu_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8'},
 '2023BPix': {'ttHToMuMu_M125': 'TTH_Hto2Mu_M-125_TuneCP5_13p6TeV_powheg-pythia8',
              'WplusHToMuMu_M125': 'WplusH_Hto2Mu_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8',
              'WminusHToMuMu_M125': 'WminusH_Hto2Mu_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8',
              'ZHToMuMu_M125': 'ZH_Hto2Mu_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8'},
 '2024': {'ttHToMuMu_M125': 'TTH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8',
          'WplusHToMuMu_M125': 'WplusH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8',
          'WminusHToMuMu_M125': 'WminusH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8',
          'ZHToMuMu_M125': 'ZH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8'},
 '2025': {'ttHToMuMu_M125': 'TTH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8',
          'WplusHToMuMu_M125': 'WplusH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8',
          'WminusHToMuMu_M125': 'WminusH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8',
          'ZHToMuMu_M125': 'ZH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8'},
 '2026': {'ttHToMuMu_M125': 'TTH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8',
          'WplusHToMuMu_M125': 'WplusH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8',
          'WminusHToMuMu_M125': 'WminusH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8',
          'ZHToMuMu_M125': 'ZH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8'}}
for _year in CONFIG['ggH']['yearly_inputs']:
    _year['sample_map'].update(_extra[_year['year']])
