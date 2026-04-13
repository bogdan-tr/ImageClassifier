#!/bin/bash
set -e

echo "####### ACTIVATING VENV ########"
tar -xzf torch_venv2.tar.gz
source torch_venv/bin/activate

# echo "ACTIVATED VENV:"
# which python
# python -c "import sys; print(sys.executable)"
# python -c "import torch; print(torch.__version__)"
#
#
# echo "Checking python binaries:"
# ls -l torch_venv/bin/python*
# file torch_venv/bin/python3
# file torch_venv/bin/python
#
# echo "DIR STRUCTURE:"
# pwd
# tree

echo "###### RUNNING CODE: #########"
echo "###### baseline_cnn_lab4.py ######" 
python3 baseline_cnn_lab4.py 
echo "####### DONE #########"

