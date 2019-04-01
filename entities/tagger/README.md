# Tag entities in new text

This code uses a layered bilstm-CRF model from [Ju et al. 2018](https://aclweb.org/anthology/N18-1131), trained on LitBank, to make predictions for a new text.  Code for the layered-bilstm-crf comes from [https://github.com/meizhiju/layered-bilstm-crf](https://github.com/meizhiju/layered-bilstm-crf) (&copy; Meizhi Ju, 2018).

## 0. Setup
Python 3 is required, along with the `chainer`, `texttable`, `gensim` and `yaml` libraries.  Code is tested with Python 3.6 with the following versions, installable via pip:

```sh
pip install chainer==5.3.0
pip install texttable==1.6.1
pip install gensim==3.6.0
pip install pyyaml==5.1
```

Run the following script to download pretrained word embeddings (`wikipedia200.txt`) and place them in the trained_model/ directory.

```sh
./scripts/get_word_embeddings.sh
```

## 1. BookNLP

For an input text `my_antonia_1.txt`, run it through [BookNLP](https://github.com/dbamman/book-nlp) to tokenize; this will yield a tokens file `my_antonia_1.tokens`

Following instructions from the BookNLP repo, execute:

```sh
./runjava novels/BookNLP -doc my_antonia_1.txt -p data/output/cather -tok tokens/my_antonia_1.tokens -f
```

`my_antonia_1.tokens` can then be found in the BookNLP `tokens` directory.

## 2. Tag entities

To tag the entities (PER, LOC, GPE, FAC, ORG, VEH) in the BookNLP-processed tokens file, execute the following command:

```sh
python layered-bilstm-crf/src/predict.py -m trained_model/ -i sample_data/my_antonia_1.tokens -o sample_data/my_antonia_1.entities -g -1
```

Flags:

* -m: path to pretrained model directory (`trained_model/`)
* -i: input .tokens file to process
* -o: output to write entities to
* -g: GPU ID; set to 0 if using a GPU (otherwise -1).  Using a GPU will speed up runtime by 3-5x or so.

This will produce a list of tagged entities in the text, indexed by their BookNLP token ID:

|start token|end token|label|text|
|---|---|---|---|
|4|4|PER|Antonia|
|19|20|GPE|North America|
|14|20|LOC|the great midland plain of North America|
|33|34|PER|my father|
|36|36|PER|mother|
|43|44|PER|Virginia relatives|
|50|51|PER|my grandparents|
|56|56|GPE|Nebraska|
|64|66|PER|a mountain boy|
|68|69|PER|Jake Marpole|
|78|79|PER|my father|
|84|86|FAC|the Blue Ridge|
|92|92|LOC|West|
|96|97|PER|my grandfather|
|78|86|FAC|my father ’s old farm under the Blue Ridge|
|103|104|LOC|the world|
|117|119|VEH|a railway train|
|133|135|LOC|a new world|
|183|184|PER|Jesse James|
|203|203|GPE|Chicago|
|210|212|PER|a friendly passenger|
|219|220|LOC|the country|
|...|...|...|...|

