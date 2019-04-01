import os
os.environ['CHAINER_SEED'] = '0'
import sys
import numpy as np
import random
random.seed(0)
np.random.seed(0)
import copy
import numpy as xp
import pickle

import chainer
import chainer.links as L
import chainer.functions as F
from chainer.functions.math import basic_math as Fmat
from chainer import reporter
from chainer.training import extensions
from chainer import training
from chainer import cuda

from src.model.utils import permutate_list
from src.model.utils import evaluate
from src.model.base_model import ModelBase

to_cpu = chainer.cuda.to_cpu


class Updater(training.StandardUpdater):

    def __init__(self, iterator, optimizer, device):
        super(Updater, self).__init__(iterator=iterator, optimizer=optimizer)

        if device['main'] >= 0:
            self.xp = cuda.cupy
        else:
            self.xp = xp

    def update_core(self):
        batch = self._iterators['main'].next()
        optimizer = self._optimizers['main']

        words = [self.xp.array(x['words']).astype('i') for x in batch]
        chars = [self.xp.array(y, dtype=self.xp.int32) for x in batch for y in x['chars']]
        tags = self.xp.vstack([self.xp.array(x['tags']).astype('i') for x in batch])

        optimizer.target.cleargrads()

        # Init index to keep track of words
        index_start = self.xp.arange(F.hstack(words).shape[0])
        index_end = index_start + 1
        index = self.xp.column_stack((index_start, index_end))

        # Nest level + 1
        max_depth = len(batch[0]['tags'][0])
        batch_loss = 0

        for depth in range(max_depth):
            accuracy, loss, next, index, _, words, chars = optimizer.target(chars, words, tags[:, depth], index, True)
            batch_loss += loss

            if not next:
                break
    
        batch_loss.backward()
        optimizer.update()


class Model(ModelBase):
    """
    Network architecture
    """
    def __init__(self, n_vocab, n_char, n_tag, args):
        feature_dim = args['word_embedding_dim']+ 2 * args['char_embedding_dim']
        super(ModelBase, self).__init__(
            char_embed=L.EmbedID(n_char, args['char_embedding_dim'], ignore_label=-1),
            bi_char=L.NStepBiLSTM(1, args['char_embedding_dim'], args['char_embedding_dim'], 0),
            word_embed=L.EmbedID(n_vocab, args['word_embedding_dim'], ignore_label=-1),
            bi_word=L.NStepBiLSTM(1, feature_dim, int(feature_dim/2), 0),
            l=L.Linear(feature_dim, n_tag),
            crf=L.CRF1d(n_tag))
       
        # Initialize value for hyper parameters
        self.word_embedding_dim = args['word_embedding_dim']
        self.char_embedding_dim = args['char_embedding_dim']
        self.tag_embedding_dim = args['tag_embedding_dim']
        self.dropout_ratio = args['dropout_ratio']
        self.lr_param = args['lr_param']
        self.threshold = args['threshold']
        self.decay_rate = args['decay_rate']
        self.batch_size = args['batch_size']
        print("""Initialized hyper parameters:
                                   - word_embedding_dim = {}
                                   - char_embedding_dim = {}
                                   - dropout_ratio = {}
                                   - lr_param = {}
                                   - threshold = {}
                                   - decay_rate = {}
                                   - batch_size = {}
                                """.format(self.word_embedding_dim,
                                           self.char_embedding_dim,
                                           self.dropout_ratio,
                                           self.lr_param,
                                           self.threshold,
                                           self.decay_rate,
                                           self.batch_size))
        if args['mode'] == 'train':
            for w in self.bi_char:
               w.b1.data[:] = 1.0
               w.b5.data[:] = 1.0
            for w in self.bi_word:
               w.b1.data[:] = 1.0
               w.b5.data[:] = 1.0

    def __call__(self, chars, words, tags, index, train, *args):
        """
        Network workflow
        """
        word_embed = self.embedding_layer(chars, words, train)
        section = self.sentence_section(words)
        accuracy, loss, predicts, ys = self.hidden_layer(word_embed, section, tags, index)
        next, index, ys, extend_predicts = self.start_next(predicts, section, index, ys)

        return accuracy, loss, next, index, extend_predicts, ys, None

    def sentence_section(self, words):
        """
        Get section for a list of sentences.
        Note:section is a numpy array
        """
        words_len = xp.array([x.shape[0] for x in words]).astype('i')
        section = xp.cumsum(words_len[:-1])

        return section

    def hidden_layer(self, word_embed, section, tags, index):
        """
        + xs: word embeddings of sentences
        + ts: gold labels
        + section: sentence boundry
        + index: index of each word
        """
        xs = F.split_axis(word_embed, section, axis=0)
        _, __, ys = self.bi_word(None, None, xs)

        ysl = self.l(F.concat(ys, axis=0))
        ysl = F.split_axis(ysl, section, 0)

        inds = xp.argsort(xp.array([-x.shape[0] for x in ysl]).astype('i'))
        ysdes = permutate_list(ysl, inds, inv=False)

        batch_ts = tags[index[:, 0]]
        ts = F.split_axis(batch_ts, section, 0)

        tsdes = permutate_list(ts, inds, inv=False)
        ysdes = F.transpose_sequence(ysdes)
        tsdes = F.transpose_sequence(tsdes)

        loss = self.crf(ysdes, tsdes)
        reporter.report({'loss': loss}, self)

        _, predicts = self.crf.argmax(ysdes)
        predicts = F.transpose_sequence(predicts)
        predicts = permutate_list(predicts, inds, inv=True)
        concat_predicts = F.concat(predicts, axis=0)

        correct = self.xp.sum(batch_ts == concat_predicts.data)
        accuracy = correct * 1.0 / batch_ts.shape[0]
        reporter.report({'accuracy': accuracy}, self)

        return accuracy, loss, concat_predicts, ys

    def start_next(self, predicts, section, index, ys):
        """
        Start next layer if the predictions contain entities.
        + predicts: BIO tags
        + index: index for each word in sentences
        + ys: context representation, output of bi_word_tag BiLSTM layer
        """
        predicts = self.correct_predict(predicts, section)
        track_index, merge_index = self.construct_merge_index(predicts.data, index)
        extend_predicts = self.extend_tags(predicts.data, index)
        merge_section = self.construct_merge_section(section, predicts)
        ys = self.merge_representation(merge_index, merge_section, ys)
        count = self.xp.sum(predicts.data)
        next = count > 0
        return next, track_index, ys, extend_predicts

    def construct_merge_section(self, section, predicts):
        """
        Construct the merge section for merged sequence.
        """
        if len(section) == 0:
            return section

        start = xp.matrix(xp.insert(section, 0, 0)).transpose()
        end = xp.insert(section, section.shape[0], predicts.shape[0])
        end = xp.matrix(end).transpose()

        if self.parameters['gpus']['main'] >= 0:
            BO_indices = xp.where(self.xp.asnumpy(predicts.data) % 2 == 0)[0]
        else:
            BO_indices = xp.where(predicts.data % 2 == 0)[0]

        BO_indices = np.matrix(BO_indices)
        BO_nums = xp.sum(xp.logical_and(BO_indices >= start, BO_indices < end), axis=1)
        BO_nums = xp.squeeze(xp.asarray(BO_nums))
        merge_section = xp.cumsum(BO_nums[:-1])

        return merge_section

    def merge_representation(self, index, section, ys):
        """
        Merge and average the context representation to prepare the input
        for next layer. If the prediction is 'O', its corresponding row of
        context representation of xs will be used as the input for next  
        layer, otherwise its corresponding row of ys will be seleted as the
        input for next layer.
        
        + index: merge index for predicts
        + ys: context representation, output of bi_word_tag BiLSTM layer
        e.g. predicts: B-Gene, I-Gene, O,B-protein,B-DNA
          index array:
          [ 1,  0,  0,  0
            1,  0,  0,  0
            0,  1,  0,  0
            0,  0,  1,  0
            0,  0,  0,  1 ]

        """
        ys = F.matmul(index.astype('f'), F.vstack(ys), transa=True)

        # Average word vectors for entity representation
        sum_index = F.sum(index.astype('f'), axis=0)
        sum_index = F.tile(sum_index, (ys.shape[1], 1))
        sum_index = F.transpose(sum_index)
        ys = Fmat.div(ys, sum_index)
        ys = F.split_axis(ys, section, axis=0)

        return ys

    def construct_merge_index(self, predicts, index):
        """
        Construct index for merge presentation.Meanwhile,
        correct first predict of each sentence if it's illegal.
        + index: index for current predicts
        + predicts: current predicts output by CRF
        predicts: B-Gene, I-Gene, O,B-protein,B-DNA
        merge_index:
          [ 1, 0, 0, 0
            1, 0, 0, 0
            0, 1, 0, 0
            0, 0, 1, 0
            0, 0, 0, 1 ]
        index:       [0,1],[1,2],[2,3],[3,4],[4,5]
        track_index: [0,2],[2,3],[3,4],[4,5] 
        """
        # Get BO, B, O, and I tags indice
        BO_indice = self.xp.where(predicts % 2 == 0)[0]
        I_indice = self.xp.where(predicts % 2 == 1)[0]

        # Count O and B tags (their ids are even)
        row_shape = predicts.shape[0]
        col_shape = BO_indice.shape[0]
        # Get merge index for merging tokens
        merge_index = self.xp.full((row_shape, col_shape), 0)
        merge_index[BO_indice, self.xp.arange(BO_indice.shape[0])] = 1

        # Fill I tags
        subarr = [predicts[0:item] for item in I_indice]
        entity_count = self.xp.array([self.xp.where(item % 2 == 0)[0].shape[0] for item in subarr]).astype('i')

        # Column index
        merge_index[I_indice, entity_count - 1] = 1

        # Get all the start indice for each word
        track_index_start = self.xp.zeros(merge_index.shape).astype('i')
        irow_start = self.xp.argmax(merge_index, axis=0)
        track_index_start[irow_start, self.xp.arange(col_shape)] = 1
        track_index_start = self.xp.matmul(index[:, 0], track_index_start)

        # Get all the end indice for each word
        merge_index_flip = self.xp.flipud(merge_index)
        irow_end = row_shape - 1 - self.xp.argmax(merge_index_flip, axis=0)
        track_index_end = self.xp.zeros(merge_index.shape).astype('i')
        track_index_end[irow_end, self.xp.arange(col_shape)] = 1
        track_index_end = self.xp.matmul(index[:, 1], track_index_end)
        track_index = self.xp.column_stack((track_index_start, track_index_end))

        return track_index, merge_index

    def correct_predict(self, predicts, section):
        """
        + predict: current predictions for one sequence
        + index  : track index of previous layer predictions
        Correct the prediction of the first word if it's
        illegal. e.g. IOOBIII->BOOBIII
        Illegal labels will disappear as the training
        process continues.
        """
        section = xp.insert(section, 0, 0)
        first_preds = predicts[section]

        # id(B) - id(I) = 1
        rep_first_preds = first_preds + 1

        # Replace I with B
        cond = first_preds.data % 2 == 0
        predicts.data[section] = F.where(cond, first_preds, rep_first_preds).data

        return predicts
              
    def extend_tags(self, predicts, index):
        """
        Restore current predicts to original sentence length
        predcicts.
        e.g. index:         [0,3],[3,4],[4,5]
             predicts:      B-gene,I-gene,O
             restored_tags: B-Gene,I-gene,I-gene,I-gene,O
        """
        times = index[:, 1] - index[:, 0]
        width = int(self.xp.amax(times) - 1)

        # There are no entities containing multiple tokens
        if width == 0:
            return predicts

        padding = self.xp.tile(predicts, [width, 1])
        padding = self.xp.transpose(padding)
        B_id = self.xp.where(self.xp.logical_and(predicts > 0, predicts % 2 == 0))[0]

        # For label B, when restoring it, use I [id(B)-id(I)=1]
        padding[B_id] -= 1
        predicts = self.xp.column_stack((predicts, padding))
        temp_id = self.xp.arange(predicts.shape[1])
        restored_tags = predicts[temp_id < times.reshape(len(times), 1)]

        return restored_tags

    def embedding_layer(self, chars, words, train):
        """
        word embeddings = word embedding + character embedding
        """
        if chars is not None:
            chars_len = xp.array([x.shape[0] for x in chars]).astype('i')
            chars_section = xp.cumsum(chars_len[:-1])
            chars = self.char_embed(F.concat(chars, axis=0))
            chars = F.split_axis(chars, chars_section, axis=0)
            _, __, chars_encs = self.bi_char(None, None, chars)
            chars = F.get_item(F.vstack(chars_encs), xp.cumsum(chars_len) - 1)
            words = self.word_embed(F.concat(words, axis=0))
            words = F.concat((words, chars), axis=1)
        else:
            words = F.concat(words, axis=0)

        if train:
            words = F.dropout(words, self.dropout_ratio)

        return words

    def save_mappings(self, id_to_word, id_to_char, id_to_tag, parameters):
        """
        Save word dict, character dict and tag dict.
        Save initial input hyper parameters.
        """
        self.id_to_word = id_to_word
        self.id_to_char = id_to_char
        self.id_to_tag = id_to_tag
        self.parameters = parameters

        # Save the parameters to disk
        with open(parameters["mappings_path"].replace("mappings", "mappings"), 'wb') as f:
            mappings = {
                'id_to_word': self.id_to_word,
                'id_to_char': self.id_to_char,
                'id_to_tag': self.id_to_tag }

            pickle.dump(mappings, f)

    def predict(self, chars, words, tags, index, mode):
        """
        Predict labels for a list of sequences
        """
        if mode == 'test':
            train = False
        else:
            sys.exit('Mode should be set as test')

        word_embed = self.embedding_layer(chars, words, train)
        section = self.sentence_section(words)
        _, __, predicts, ys = self.hidden_layer(word_embed, section, tags, index)
        next, index, ys, extend_predicts = self.start_next(predicts, section, index, ys)

        return next, index, extend_predicts, ys, None


class Evaluator(extensions.Evaluator):
    def __init__(self, iterator, target, device):
        super(Evaluator, self).__init__(iterator=iterator, target=target, device=device)

        if device['main'] >= 0:
            self.xp = cuda.cupy
        else:
            self.xp = xp

        self.default_name = 'dev'
        self.device = device

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']

        it = copy.copy(iterator)
        summary = reporter.DictSummary()
        ys_final, ts_final, raw_xs = [], [], []
        
        for batch in it:
            # Read batch data and sort sentences in descending order for CRF layer
            observation = {}

            raw_words = [x['str_words'] for x in batch]
            words = [self.xp.array(x['words']).astype('i') for x in batch]
            chars = [self.xp.array(y, dtype=self.xp.int32) for x in batch for y in x['chars']]
            tags = self.xp.vstack([self.xp.array(x['tags']).astype('i') for x in batch])

            # Init index to keep track of words
            index_start = self.xp.arange(F.hstack(words).shape[0])
            index_end = index_start + 1
            index = self.xp.column_stack((index_start, index_end))

            # Nest level + 1
            max_depth = len(batch[0]['tags'][0])
            sentence_len = xp.array([x.shape[0] for x in words])
            section = xp.cumsum(sentence_len[:-1])

            # Init
            predicts_depths = self.xp.empty((0, self.xp.sum(sentence_len))).astype('i')

            with reporter.report_scope(observation):
                for depth in range(max_depth):
                    accuracy, loss, next, index, extend_predicts,words,chars = target(chars, words, tags[:, depth], index, False)
                    predicts_depths = self.xp.vstack((predicts_depths, extend_predicts))

                    if not next:
                        break
          
            summary.add(observation)
            predicts_depths = self.xp.split(predicts_depths, section, axis=1)
            ts_depths = self.xp.split(self.xp.transpose(tags), section, axis=1)
            ys_final.extend(predicts_depths)
            ts_final.extend(ts_depths)
            raw_xs.extend(raw_words)

        fmeasure = summary.compute_mean()

        fmeasure['dev/main/fscore'] = evaluate(target, ys_final, ts_final, raw_xs)

        return fmeasure

