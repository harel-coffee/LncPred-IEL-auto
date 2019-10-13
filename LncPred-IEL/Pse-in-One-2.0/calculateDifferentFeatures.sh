#!/bin/bash
while getopts ":i:" opt 
do
	case $opt in
		i)
		#pse
		echo $OPTARG
		python ./pse.py -out $OPTARG.PseDNC -lamada 7 -w 0.6 $OPTARG DNA PseDNC
		python ./pse.py -out $OPTARG.PC-PseDNC-General -lamada 7 -w 0.6 $OPTARG DNA PC-PseDNC-General
		python ./pse.py -out $OPTARG.PC-PseTNC-General -lamada 7 -w 0.6 $OPTARG DNA PC-PseTNC-General
		python ./pse.py -out $OPTARG.SC-PseDNC-General -lamada 7 -w 0.6 $OPTARG DNA SC-PseDNC-General
		python ./pse.py -out $OPTARG.SC-PseTNC-General -lamada 7 -w 0.6 $OPTARG DNA SC-PseTNC-General
		# kmer rckmer mismatch
		python ./nac.pyc -out $OPTARG.1mer -k 1 -f tab $OPTARG DNA kmer
		python ./nac.pyc -out $OPTARG.2mer -k 2 -f tab $OPTARG DNA kmer
		python ./nac.pyc -out $OPTARG.3mer -k 3 -f tab $OPTARG DNA kmer
		python ./nac.pyc -out $OPTARG.4mer -k 4 -f tab $OPTARG DNA kmer
		python ./nac.pyc -out $OPTARG.5mer -k 5 -f tab $OPTARG DNA kmer
		python ./nac.pyc -out $OPTARG.1rckmer -k 1 -r 1 -f tab $OPTARG DNA kmer
		python ./nac.pyc -out $OPTARG.2rckmer -k 2 -r 1 -f tab $OPTARG DNA kmer
		python ./nac.pyc -out $OPTARG.3rckmer -k 3 -r 1 -f tab $OPTARG DNA kmer
		python ./nac.pyc -out $OPTARG.4rckmer -k 4 -r 1 -f tab $OPTARG DNA kmer
		python ./nac.pyc -out $OPTARG.5rckmer -k 5 -r 1 -f tab $OPTARG DNA kmer
		python ./nac.pyc -out $OPTARG.3mismatch -k 3 -m 1 -f tab $OPTARG DNA Mismatch
		python ./nac.pyc -out $OPTARG.4mismatch -k 4 -m 1 -f tab $OPTARG DNA Mismatch
		python ./nac.pyc -out $OPTARG.5mismatch -k 5 -m 1 -f tab $OPTARG DNA Mismatch
		# ACC
		python ./acc.pyc -out $OPTARG.DACC -lag 7 -f tab $OPTARG DNA DACC
		python ./acc.pyc -out $OPTARG.TACC -lag 7 -f tab $OPTARG DNA TACC
		python CPPred.py -i $OPTARG -hex ./Hexamer/Integrated_Hexamer.tsv -spe Integrated
		;;
		?)
		echo "unknown parameter"
		exit 1;;
	esac
done
