#!/bin/bash
set -e

echo "####### ACTIVATING VENV ########"
tar -xzf torch_venv2.tar.gz
source torch_venv/bin/activate

echo "###### RUNNING CODE: #########"

echo "###### RUN 1: #########"
python3 mesonet_imp_v3.py mesonet_imp_v3_1 >> /home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/output_txts/mesonet_imp_v3_1.txt

echo "###### RUN 2: #########"
python3 mesonet_imp_v3.py mesonet_imp_v3_2 >> /home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/output_txts/mesonet_imp_v3_2.txt

echo "###### RUN 3: #########"
python3 mesonet_imp_v3.py mesonet_imp_v3_3 >> /home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/output_txts/mesonet_imp_v3_3.txt

echo "###### RUN 4: #########"
python3 mesonet_imp_v3.py mesonet_imp_v3_4 >> /home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/output_txts/mesonet_imp_v3_4.txt

echo "###### RUN 5: #########"
python3 mesonet_imp_v3.py mesonet_imp_v3_5 >> /home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/output_txts/mesonet_imp_v3_5.txt

echo "###### RUN 6: #########"
python3 mesonet_imp_v3.py mesonet_imp_v3_6 >> /home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/output_txts/mesonet_imp_v3_6.txt

echo "###### RUN 7: #########"
python3 mesonet_imp_v3.py mesonet_imp_v3_7 >> /home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/output_txts/mesonet_imp_v3_7.txt

echo "###### RUN 8: #########"
python3 mesonet_imp_v3.py mesonet_imp_v3_8 >> /home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/output_txts/mesonet_imp_v3_8.txt

echo "###### RUN 9: #########"
python3 mesonet_imp_v3.py mesonet_imp_v3_9 >> /home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/output_txts/mesonet_imp_v3_9.txt

echo "###### RUN 10: #########"
python3 mesonet_imp_v3.py mesonet_imp_v3_10 >> /home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/output_txts/mesonet_imp_v3_10.txt
echo "###### DONE #########"
tree
