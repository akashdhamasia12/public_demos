cd $PBS_O_WORKDIR
source /opt/intel/oneapi/setvars.sh
conda activate pytorch
echo "########## Executing the run"
python3 TP_inference_devcloud.py
echo "########## Done with the run"
