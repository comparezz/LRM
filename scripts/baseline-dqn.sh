cd ../src
for i in `seq 1 5`; 
do
	python3 run.py --agent="dqn" --world="cookie" --seed=$i &&
	python3 run.py --agent="dqn" --world="symbol" --seed=$i &&
	python3 run.py --agent="dqn" --world="keys" --seed=$i 
done