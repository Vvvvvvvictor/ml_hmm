# Run 3 DNN and jet recoil workflows

- [DNN training and Apply](../hzgml/scripts/DNN_FIT_OUTPUT.md) describes the ggH training, scoring and ROOT output. The pt100 campaign entry points are `pt100_ggH_campaign.py`, `pt100_grl_campaign.py` and `pt100_jme50_campaign.py` in `hzgml/scripts`.
- [JetRecoilCorrectionWorkflow](JetRecoilCorrectionWorkflow/README.md) contains the nominal recoil workflow.
- [JetRecoilCorrectionWorkflow_forward_pt100_20261118](JetRecoilCorrectionWorkflow_forward_pt100_20261118/README.md) contains the forward-pT100 variant used by the pt100 DNN campaigns.

Both recoil variants include the source, fit and plotting definitions, and the input catalogs needed by `run.py`. The workflows read ROOT inputs and prior fit products from CERN EOS and use CERN CVMFS and HTCondor.
