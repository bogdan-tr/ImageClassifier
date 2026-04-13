#!/bin/bash
set -e

echo "####### ACTIVATING VENV ########"
tar -xzf torch_venv2.tar.gz
source torch_venv/bin/activate

echo "###### RUNNING CODE: #########"

echo "###### RUN 1: #########"
python3 vision_trans.py vision_trans_1 >> /home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/output_txts/vision_trans_1.txt

echo "###### RUN 2: #########"
python3 vision_trans.py vision_trans_2 >> /home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/output_txts/vision_trans_2.txt

echo "###### RUN 3: #########"
python3 vision_trans.py vision_trans_3 >> /home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/output_txts/vision_trans_3.txt

echo "###### RUN 4: #########"
python3 vision_trans.py vision_trans_4 >> /home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/output_txts/vision_trans_4.txt

echo "###### RUN 5: #########"
python3 vision_trans.py vision_trans_5 >> /home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/output_txts/vision_trans_5.txt

echo "###### RUN 6: #########"
python3 vision_trans.py vision_trans_6 >> /home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/output_txts/vision_trans_6.txt
#
echo "###### RUN 7: #########"
python3 vision_trans.py vision_trans_7 >> /home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/output_txts/vision_trans_7.txt

echo "###### RUN 8: #########"
python3 vision_trans.py vision_trans_8 >> /home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/output_txts/vision_trans_8.txt

echo "###### RUN 9: #########"
python3 vision_trans.py vision_trans_9 >> /home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/output_txts/vision_trans_9.txt

echo "###### RUN 10: #########"
python3 vision_trans.py vision_trans_10 >> /home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/output_txts/vision_trans_10.txt
echo "###### DONE #########"
tree
