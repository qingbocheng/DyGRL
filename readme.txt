DyGRL:
git clone https://github.com/microsoft/qlib.git && cd qlib
python setup.py install
cd ..
pip install -r requirements.txt

RUN:
python DyGRL/tools/train_mask_sac_cap.py

