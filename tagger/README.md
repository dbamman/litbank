# Tag entities/events in new text

This code identifies the entities and events in new text using a bidirectional LSTM (for events) and a layered bidirectional LSTM-CRF (for entities).  Both models use BERT contextual embeddings, which yield a substantial performance increase over equivalent non-BERT neural models (reported in [Bamman et al. 2019] (http://people.ischool.berkeley.edu/~dbamman/pubs/pdf/naacl2019_literary_entities.pdf) and [Sims et al. 2019] (http://people.ischool.berkeley.edu/~dbamman/pubs/pdf/acl2019_literary_events.pdf)).

||Entities|Events|
|---|---|---|
|BERT|77.6 F|73.6 F|
|Non-BERT|68.3 F|66.9 F|

The performance increase does come at a computational cost; it's worth running this code on a GPU or multi-core machine.
Timing on the full text of Willa Cather's *My Antonia* (ca. 100K words):

|Machine|Minutes:seconds|
|---|---|
|Tesla K40 GPU|1:01 |
|10-core 2.4 GHz server | 2:09|
|2-core 2.6 GHz Macbook pro|12:52|



## 0. Setup
Python 3 is required, along with the libraries specified in the included requirements file.  Install via pip with:

```sh
pip install -r requirements.txt
```

Download a model that has been trained on Litbank with:

```sh
./get_trained_model.sh
```


The first time this code is used to tag texts (section 2 below), it will also download the `bert-base-cased` model.


## 1. BookNLP

For an input text `my_antonia_1.txt`, run it through [BookNLP](https://github.com/dbamman/book-nlp) to tokenize; this will yield a tokens file `my_antonia_1.tokens`

Following instructions from the BookNLP repo, execute:

```sh
./runjava novels/BookNLP -doc my_antonia_1.txt -p data/output/cather -tok tokens/my_antonia_1.tokens -f
```

`my_antonia_1.tokens` can then be found in the BookNLP `tokens` directory.

## 2. Tag entities

To tag the events and entities (PER, LOC, GPE, FAC, ORG, VEH) in the BookNLP-processed tokens file, execute the following command:

```sh
python run_tagger.py --mode predict -i sample_data/my_antonia_1.tokens -o sample_data/my_antonia_1.tagged
```

This will produce a list of tagged entities and events in the text, indexed by their BookNLP token ID:

|start token|end token|label|text|
|---|---|---|---|
|14|20|LOC|the great midland plain of North America|
|19|20|GPE|North America|
|31|31|EVENT|lost|
|33|34|PER|my father|
|36|36|PER|mother|
|42|44|PER|my Virginia relatives|
|46|46|EVENT|sending|
|50|51|PER|my grandparents|
|50|56|PER|my grandparents , who lived in Nebraska|
|56|56|GPE|Nebraska|
|59|59|EVENT|travelled|
|64|66|PER|a mountain boy|
|68|69|PER|Jake Marpole|
|71|97|PER|one of the ‘ hands ’ on my father ’s old farm under the Blue Ridge , who was now going West to work for my grandfather|
|78|79|PER|my father|
|78|86|FAC|my father ’s old farm under the Blue Ridge|
|84|86|LOC|the Blue Ridge|
|92|92|LOC|West|
|96|97|PER|my grandfather|
|99|99|PER|Jake|
|103|104|LOC|the world|
|117|119|VEH|a railway train|
|...|...|...|...|

