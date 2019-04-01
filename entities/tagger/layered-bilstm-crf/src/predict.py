import os, re, argparse
import sys
os.environ['CHAINER_SEED'] = '0'
import random
random.seed(0)
import numpy as np
np.random.seed(0)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pickle

import chainer.functions as F
from chainer import iterators
from chainer import cuda
from chainer import serializers

from src.model.layered_model import Model, Evaluator, Updater
from src.model.loader import zero_digits
from src.model.loader import prepare_dataset

import yaml

def parse_config(config_path):
    """
    Read configuration from config file and returns a dictionary.
    """
    with open(config_path, 'r') as stream:
        try:
            config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return config

def read_booknlp(path, zero_digits):

    sentences=[]
    original_sentences=[]

    sentence=[]
    orig_sentence=[]

    with open(path, encoding="utf-8") as file:
        header=file.readline().split("\t")

        s_idx=header.index("sentenceID")
        t_idx=header.index("tokenId")
        w_idx=header.index("originalWord")
        
        lastSentence=None

        for line in file:
            cols=line.rstrip().split("\t")
            s_id=cols[s_idx]
            t_id=cols[t_idx]
            w=cols[w_idx]
            n=w

            if zero_digits:
                n=re.sub('\d', '0', w)

            if s_id != lastSentence:
                if len(sentence) > 0:
                    sentences.append(sentence)
                    sentence=[]

                    original_sentences.append(orig_sentence)
                    orig_sentence=[]

            sentence.append((n, "O", "O", "O", "O"))
            orig_sentence.append((t_id, w))

            lastSentence=s_id

    if len(sentence) > 0:
        sentences.append(sentence)
        original_sentences.append(orig_sentence)

    return sentences, original_sentences

def predict(data_iter, model, mode):

    for batch in data_iter:
        raw_words = [x['str_words'] for x in batch]
        words = [model.xp.array(x['words']).astype('i') for x in batch]
        chars = [model.xp.array(y).astype('i') for x in batch for y in x['chars']]
        tags = model.xp.vstack([model.xp.array(x['tags']).astype('i') for x in batch])

        # Init index to keep track of words
        index_start = model.xp.arange(F.hstack(words).shape[0])
        index_end = index_start + 1
        index = model.xp.column_stack((index_start, index_end))

        # Maximum number of hidden layers = maximum nested level + 1
        max_depth = len(batch[0]['tags'][0])
        sentence_len = np.array([x.shape[0] for x in words])
        section = np.cumsum(sentence_len[:-1])
        predicts_depths = model.xp.empty((0, int(model.xp.sum(sentence_len)))).astype('i')

        for depth in range(max_depth):
            next, index, extend_predicts, words, chars = model.predict(chars, words, tags[:, depth], index, mode)
            predicts_depths = model.xp.vstack((predicts_depths, extend_predicts))
            if not next:
                break

        predicts_depths = model.xp.split(predicts_depths, section, axis=1)
        ts_depths = model.xp.split(model.xp.transpose(tags), section, axis=1)
        yield ts_depths, predicts_depths, raw_words


def load_mappings(mappings_path):
    """
    Load mappings of:
      + id_to_word
      + id_to_tag
      + id_to_char
    """
    with open(mappings_path, 'rb') as f:
        mappings = pickle.load(f)
        id_to_word = mappings['id_to_word']
        id_to_char = mappings['id_to_char']
        id_to_tag = mappings['id_to_tag']

    return id_to_word, id_to_char, id_to_tag


def get_entities(xs, ys, id_to_tag):

    all_entities={}

    for k in range(int(len(ys))):

        start=None
        this_type=None

        for j in range(len(xs)):
            tag0=id_to_tag[int(ys[k][j])]
            parts=tag0.split("-")
            p_type=None
            p_position=tag0
            if len(parts) == 2:
                p_type=parts[1]
                p_position=parts[0]

            if p_position == "B" or p_position == "O" or (p_position == "I" and p_type != last_p_type):
                if start != None:
                    end=j
                    ent=(0,start, end, this_type)
                    all_entities[ent]=1
                start=None
                this_type=None

            if p_position == "B":
                start=j
                this_type=p_type

            last_p_type=p_type

        if start != None:
            ent=(0,start, len(xs), this_type)
            all_entities[ent]=1

    return all_entities

def main(path, outputPath, gpu, model_path):
    config_path=os.path.join(model_path, "predict.config")
    args = parse_config(config_path)
    batchSize=100
    
    sentences, original=read_booknlp(path, args["replace_digit"])

    # Load mappings from disk
    mappings_path=os.path.join(model_path, "lit_mappings.pkl")
    path_model=os.path.join(model_path, "nested_entities.model")
    embedding_path=os.path.join(model_path, "wikipedia200.txt")

    args["mode"]="test"
    args["path_pre_emb"]=embedding_path
    args["path_model"]=path_model
    args["mappings_path"]=mappings_path

    id_to_word, id_to_char, id_to_tag = load_mappings(mappings_path)
    word_to_id = {v: k for k, v in id_to_word.items()}
    char_to_id = {v: k for k, v in id_to_char.items()}
    tag_to_id  = {v: k for k, v in id_to_tag.items()}

    # Index data
    test_data = prepare_dataset(sentences, word_to_id, char_to_id, tag_to_id, None, args["lowercase"])
    test_iter = iterators.SerialIterator(test_data, batchSize, repeat=False, shuffle=False)

    model = Model(len(word_to_id), len(char_to_id), len(tag_to_id), args)

    serializers.load_npz(path_model, model)

    model.id_to_tag = id_to_tag
    model.parameters = args

    useGPU=False
    args['gpus']={}
    args['gpus']['main']=gpu
    if gpu >= 0:
        cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    pred_tags = []
    gold_tags = []
    words = []

    # Collect predictions
    out=open(outputPath, "w", encoding="utf-8")

    all_true={}
    all_pred={}
    idx=0
    for ts, ys, xs in predict(test_iter, model, "test"):
        
        # for sentence in batch size
        for i in range(len(xs)):
            pred_entities=get_entities(xs[i], ys[i], id_to_tag)

            sentence_tokens=original[idx]
            rectified_preds=[]
            for v in pred_entities:
                start=v[1]
                end=v[2]
                cat=v[3]
                t_start=sentence_tokens[start][0]
                t_end=sentence_tokens[end-1][0]
                text=' '.join(xs[i][int(v[1]):int(v[2])])
                rectified_preds.append((t_start, t_end, cat, text))

            for v in rectified_preds:
                out.write("%s %s %s %s\n" % (v[0], v[1], v[2], v[3]))

            idx+=1
    out.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', help='input file to tag', required=True)
    parser.add_argument('-o','--output', help='output file to write to', required=True)
    parser.add_argument('-g','--gpu', help='gpu id (probably "0" if using GPU)', required=False)
    parser.add_argument('-m','--model_path', help='path_to_model; should contain predict.config, lit_mappings.pkl, nested_entities.model, wikipedia200.txt', required=False)
    

    args = vars(parser.parse_args())
    
    inputFile=args["input"]
    outputFile=args["output"]
    gpu=args["gpu"]
    model_path=args["model_path"]

    gpu = int(gpu) if gpu is not None else -1

    main(inputFile, outputFile, gpu, model_path)

