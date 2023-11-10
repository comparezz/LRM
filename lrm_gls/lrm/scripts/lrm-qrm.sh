cd ../src
for i in `seq 1 30`; 
do
	python3 run.py --agent="lrm-qrm" --world="cookie" --seed=$i --workers=16 &&
	python3 run.py --agent="lrm-qrm" --world="symbol" --seed=$i --workers=16 &&
	python3 run.py --agent="lrm-qrm" --world="keys" --seed=$i --workers=16 
done