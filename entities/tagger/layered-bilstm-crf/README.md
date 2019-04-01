# Nested-NER
Nested-NER is an implementation of [A Neural Layered Model for Nested Named Entity Recognition] (http://aclweb.org/anthology/N18-1131).

# Requirements
* Ubuntu 16.04
* chainer 3.3.0
* python 3.5.2
* numpy 1.14.1
* cupy 2.4.0
* cuda 9.1
* cudnn 7.0

# Data format
Each line has multiple columns separated by a tab key. 
Each line contains
```
word	label1	label2	label3	...	labelN
```
The number of labels (`N`) for each word is determined by the maximum nested level in the data set. `N=maximum nested level + 1`
Each sentence is separated by an empty line.
For example, for these two sentences, `John killed Mary's husband. He was arrested last night` , they contain four entities: John (`PER`), Mary(`PER`), Mary's husband(`PER`),He (`PER`).
The format for these two sentences is listed as following:
```
John    B-PER   O   O
killed  O   O   O
Mary    B-PER   B-PER   O
's  O   I-PER   O
husband O   I-PER   O
.   O   O   O

He    B-PER   O   O
was  O   O   O
arrested  O   O   O
last  O   O   O
night  O   O   O
.  O   O   O
```

# Pretrained word embeddings
* Pretrained word embeddings used in GENIA: https://drive.google.com/open?id=0BzMCqpcgEJgiUWs0ZnU0NlFTam8
* Pretrained word embeddings used in ACE2005:http://tti-coin.jp/data/wikipedia200.bin

# Configuration
Parameters are listed in the config file which is located in the layered-bilstm-crf/src folder.
Before running the codes, please change the parameters with specific values.

# Usage
## Training

```
cd layered-bilstm-crf/src/
python3 train.py
```
## Testing
```
cd layered-bilstm-crf/src
python3 test.py
```


Please cite our NAACL paper when using this code.

* Meizhi Ju, Makoto Miwa, Sophia Ananiadou. [A Neural Layered Model for Nested Named Entity Recognition](http://aclweb.org/anthology/N18-1131) In the Proceedings of NAACL-HLT2018, pp. 1446--1459.
