cd ../models
for j in LUAD SCH
do
python train.py --model CoADS --mode graph --gpu 3 --data $j 
done