# include ROOT

source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
source /cvmfs/sft.cern.ch/lcg/releases/LCG_108/ROOT/6.36.02/x86_64-el9-gcc13-opt/bin/thisroot.sh

python3 -m venv /eos/user/${USER::1}/$USER/ml-gpu-env/
source /eos/user/${USER::1}/$USER/ml-gpu-env/bin/activate

export PYTHONPATH="/eos/user/${USER::1}/$USER/ml-gpu-env/lib/python3.12/site-packages/:$PYTHONPATH"

# pip install --upgrade pip
# pip install -r requirements.txt
pip install xgboost==3.1.0
pip install optuna==4.6.0

# pip install -U --ignore-installed scikit-learn tensorboardx servicex numba httpstan coffea

# # setting python path
# export PATH="`pwd`/scripts:${PATH}"
# export PYTHONPATH="`pwd`/scripts:${PYTHONPATH}"
# export PATH="`pwd`/hzgml:$PATH"
# export PYTHONPATH="`pwd`/hzgml:$PYTHONPATH"
