# condatoolkit 없으면 bitsandbytes가 CPU 버전으로 설치되니까 설치해줘야함
conda install -y -c conda-forge cudatoolkit=11.8
pip uninstall -y bitsandbytes
pip install bitsandbytes==0.37.0
# 