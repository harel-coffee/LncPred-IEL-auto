b=('KFruit_fly_coding_RNA' 'KFruit_fly_ncrna' 'S.cerevisiae_coding_RNA' 'S.cerevisiae_ncrna' 'Zebrafish_coding_RNA' 'Zebrafish_ncrna')
for i in ${b[@]}
do
	python ./pse.py -out $i.PseDNC -lamada 7 -w 0.6  ./input/$i.input DNA PseDNC
	python ./pse.py -out $i.PC-PseDNC-General -lamada 7 -w 0.6 ./input/$i.input DNA PC-PseDNC-General
	python ./pse.py -out $i.PC-PseTNC-General -lamada 7 -w 0.6 ./input/$i.input DNA PC-PseTNC-General
	python ./pse.py -out $i.SC-PseDNC-General -lamada 7 -w 0.6 ./input/$i.input DNA SC-PseDNC-General
	python ./pse.py -out $i.SC-PseTNC-General -lamada 7 -w 0.6 ./input/$i.input DNA SC-PseTNC-General
done

