import os
import re
import sys
import codecs

from src.model.utils import create_dico, create_mapping
from src.model.utils import zero_digits,is_float
from src.model.utils import iob2, iob_iobes,insert_singletons
from src.model.utils import cost_matrix

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

    assert os.path.isfile(config["path_train"])
    assert os.path.isfile(config["path_dev"])
    assert os.path.isfile(config["path_test"])
    assert config["char_embedding_dim"] > 0
    assert config["word_embedding_dim"] > 0
    assert config["tag_embedding_dim"]  > 0
    assert 0. <= config["dropout_ratio"] < 1.0
    assert config['tag_scheme'] in ['iob', 'iobes']
    if config['gpus']['main'] < 0 and len(config['gpus']) > 1:
        sys.exit('CPU mode does not allow multi GPU mode')

    if not os.path.exists(config['path_eval_result']):
        os.makedirs(config['path_eval_result'])

    return config


def load_sentences(path, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf-8'):
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)

    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)

    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging shceme to IOB2.
    Only IOB1 and IOB2 schemes are accepted
    """
    for i, s in enumerate(sentences):
        index = []
        if len(s[0]) < 3:
            index.append(1)
        else:
            index = range(1,len(s[0]))

        for j in index:
            tags = [w[j] for w in s]
            if not iob2(tags):
                s_str = '\n'.join(' '.join(w) for w in s)
                raise Exception('Sentences should be given in IOB format!' +
                                'Please check sentence %i:\n%s' % (i, s_str))

            if tag_scheme == 'iob':
                # If format was IOB1, we convert to IOB2
                for word, new_tag in zip(s, tags):
                    word[j] = new_tag

            elif tag_scheme == 'iobes':
                new_tags = iob_iobes(tags)
                for word, new_tag in zip(s, new_tags):
                    word[1] = new_tag

            else:
                raise Exception('Unknown tagging scheme!')


def word_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(words)
    dico['<UNK>'] = 10000000
    word_to_id, id_to_word = create_mapping(dico)

    print("Found %i unique words in (%i in total)" % (len(dico), sum(len(x) for x in words)))

    return dico, word_to_id, id_to_word


def char_mapping(sentences):
    """
    Create a dictionary and mapping of characters, sorted by frequency
    """
    chars = ["".join([w[0] for w in s]) for s in sentences]
    dico = create_dico(chars)
    char_to_id, id_to_char = create_mapping(dico)

    print("Found %i unique characters" % len(dico))

    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    def f(x):
        return [1] if len(x) < 3 else range(1, len(x))

    tags = []
    for s in sentences:
        for word in s:
            tags.append([word[j]for j in f(word)])

    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)

    return dico, tag_to_id, id_to_tag


def entity_mapping(sentences):
    """
    Create a dictionary and a mapping of entities, sorted by frequency.
    """
    def f(x):
        return [1] if len(x) < 3 else range(1, len(x))

    tags=[]
    for s in sentences:
        for word in s:
            tags.append([word[j]for j in f(word)])

    dico = create_dico(tags)
    dico = dict((k.split('-')[1],v) for k, v in dico.items() if k.split('-')[0] == 'B')
    print("Found %i unique named entites tags" % len(dico))
    entity_to_id, id_to_entity = create_mapping(dico)

    return dico,entity_to_id,id_to_entity


def entity_tags(dico):
    """
    Create a dictionary and a mapping of tags
    """
    id_to_tag = {0: 'O'}
    for i, (k, v) in enumerate(dico.items()):
        id_to_tag[2*i + 1] = 'I-' + v
        id_to_tag[2*i + 2] = 'B-' + v

    tag_to_id = {v: k for k, v in id_to_tag.items()}

    return id_to_tag, tag_to_id


def prepare_dataset(sentences, word_to_id, char_to_id, tag_to_id, singletons, lower=False):
    """
    Prepare the dataset. Return a list of dictionaries containing:
        - word indexes
        - word char indexed
        - tag indexed
    """

    def f(x):
        return x.lower() if lower else x

    def char_id(char_to_id, char):
        if f(char) in char_to_id:
            return char_to_id[f(char)]
        else:
            # Unknown char shared the same id with .
            return char_to_id['.']

    data = []
    for s in sentences:
        str_words = [w[0] for w in s]
        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>'] for w in str_words]

        if singletons is not None:
            words = insert_singletons(words, singletons)

        # Characters that are not in the train set are set 1000
        chars = [[char_id(char_to_id, c) for c in w] for w in str_words]

        if len(s[0]) < 3:
            index = [1]
        else:
            index = range(1, len(s[0]))

        tags = [[tag_to_id[w[j]] for j in index] for w in s]

        data.append({
            'str_words': str_words,
            'words': words,
            'chars': chars,
            'tags': tags
        })

    return data


def augment_with_pretrained(dictionary, ext_emb_path, words):
    """
    Augment the dictionary with words that have a pretrained a embedding.
    If 'words' is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by 'words'
    (typically the words in the development and test sets.)
    """
    assert os.path.isfile(ext_emb_path)
    pretrained = []

    with open(ext_emb_path, 'r', encoding='utf-8') as f:
        print("Pre-trained word embeddings shape: {}".format(f.readline().strip()))
        for i, line in enumerate(f):
            pretrained.append(line.strip().split(" ")[0])

    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [ word, word.lower(), re.sub('\d', '0', word.lower()) ]) and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)

    return dictionary, word_to_id, id_to_word, pretrained


def load_cost_matrix(dico, cost):
    """
    Load cost matrix for CRF layer to restricit illegal labels.
    """
    return cost_matrix(dico, cost)

