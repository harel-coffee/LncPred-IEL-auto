# LncPred-IEL
This computational tool is designed for long non-coding RNA prediction, aim to accurately predict lncRNAs, and namely LncPred-IEL( **'lncRNA prediction using iterative ensemble learning'**).

## Usage

First, the input files (DNA sequence) format should be 'fasta', 'fa' or 'fastq'.

Then feature vector should be calculated as below :

```shell
# This step should run under python2 enviroment
# change the directory to ./Pse-in-one-2.0
bash calculateDifferentFeatures.sh -i positive_sequence
bash calculateDifferentFeatures.sh -i negative_sequence
```

Now we go back to the LncPred-IEL main directory

```shell
# # This step should run under python2 enviroment
# the args(positive_feature and negative_feature) can be any one of the feature
python FindOptimalIterationRound.py positive_feature negative_feature
```
