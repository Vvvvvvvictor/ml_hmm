#!/bin/bash
# import ROOT

echo "Setting up GPU environment..."
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
echo "Setting up ROOT..."
source /cvmfs/sft.cern.ch/lcg/releases/LCG_108/ROOT/6.36.02/x86_64-el9-gcc13-opt/bin/thisroot.sh
echo "Activating Python virtual environment..."
source /eos/user/${USER::1}/$USER/ml-gpu-env/bin/activate

echo "Configuring environment variables..."
export PATH="`pwd`:${PATH}"
export PYTHONPATH="`pwd`:${PYTHONPATH}"
export THEANO_FLAGS="gcc.cxxflags='-march=core2'"

export PATH="`pwd`/scripts:${PATH}"
export PYTHONPATH="`pwd`/scripts:${PYTHONPATH}"

export PATH="`pwd`/hzgml:$PATH"
export PYTHONPATH="`pwd`/hzgml:$PYTHONPATH"

export PYTHONPATH="/eos/user/${USER::1}/$USER/ml-gpu-env/lib/python3.12/site-packages/:$PYTHONPATH"
echo "Environment setup complete."

# source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
# sourcre /cvmfs/sft.cern.ch/lcg/releases/LCG_108/ROOT/6.36.02/x86_64-el9-gcc13-opt/bin/thisroot.sh

