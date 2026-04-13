#!/bin/bash
set -e

echo "####### ACTIVATING VENV ########"
tar -xzf torch_venv2.tar.gz
source torch_venv/bin/activate

echo "###### RUNNING CODE: #########"

echo "###### RUN 1: #########"
python3 baseline_cnn_lab4_frequent_metrics.py baseleline_cnn_lab4_fm_1 >> /home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/output_txts/baseline_cnn_lab4_fm_1.txt

echo "###### RUN 2: #########"
python3 baseline_cnn_lab4_frequent_metrics.py baseleline_cnn_lab4_fm_2 >> /home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/output_txts/baseline_cnn_lab4_fm_2.txt

echo "###### RUN 3: #########"
python3 baseline_cnn_lab4_frequent_metrics.py baseleline_cnn_lab4_fm_3 >> /home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/output_txts/baseline_cnn_lab4_fm_3.txt

echo "###### RUN 4: #########"
python3 baseline_cnn_lab4_frequent_metrics.py baseleline_cnn_lab4_fm_4 >> /home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/output_txts/baseline_cnn_lab4_fm_4.txt

echo "###### RUN 5: #########"
python3 baseline_cnn_lab4_frequent_metrics.py baseleline_cnn_lab4_fm_5 >> /home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/output_txts/baseline_cnn_lab4_fm_5.txt

echo "###### RUN 6: #########"
python3 baseline_cnn_lab4_frequent_metrics.py baseleline_cnn_lab4_fm_6 >> /home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/output_txts/baseline_cnn_lab4_fm_6.txt

echo "###### RUN 7: #########"
python3 baseline_cnn_lab4_frequent_metrics.py baseleline_cnn_lab4_fm_7 >> /home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/output_txts/baseline_cnn_lab4_fm_7.txt

echo "###### RUN 8: #########"
python3 baseline_cnn_lab4_frequent_metrics.py baseleline_cnn_lab4_fm_8 >> /home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/output_txts/baseline_cnn_lab4_fm_8.txt

echo "###### RUN 9: #########"
python3 baseline_cnn_lab4_frequent_metrics.py baseleline_cnn_lab4_fm_9 >> /home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/output_txts/baseline_cnn_lab4_fm_9.txt

echo "###### RUN 10: #########"
python3 baseline_cnn_lab4_frequent_metrics.py baseleline_cnn_lab4_fm_10 >> /home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/output_txts/baseline_cnn_lab4_fm_10.txt

echo "###### DONE #########"
tree
