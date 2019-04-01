import os
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import re
import operator
import uuid
import texttable as tt


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1

    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID/ID to item) from a dictionary.
    Items are ordered by decresaing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}

    return item_to_id, id_to_item


def zero_digits(s):
    """
    Replace every digit in a string by a zero
    """
    return re.sub('\d', '0', s)


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:
            tags[i] = 'B' + tag[1:]

    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')

    return new_tags


def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')

    return new_tags


def insert_singletons(x, id2singleton):
    """
    Replace word id with 0 if its frequency = 1.
    + id2singleton: a dict of words whose frequency = 1
    """
    x_array = np.array(x)
    is_singleton = np.array([True if idx in id2singleton else False for idx in x], dtype=np.bool)
    bool_mask = np.random.randint(0, 2, size=is_singleton[is_singleton].shape).astype(np.bool)
    is_singleton[is_singleton] = bool_mask
    r = np.zeros(x_array.shape)
    x_array[is_singleton] = r[is_singleton]

    return x_array


def match(p_entities, r_entities, type):
    """
    Match predicted entities with gold entities.
    """
    p_entities = [tuple(entity) for entity in p_entities if entity[-1] == type]
    r_entities = [tuple(entity) for entity in r_entities if entity[-1] == type]
    pcount = len(p_entities)
    rcount = len(r_entities)
    correct = len(list(set(p_entities) & set(r_entities)))
    return [pcount, rcount, correct]


def is_float(value):
    """
    Check in value is of type float()
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def evaluate(model, y_preds, y_reals, raw_xs):
    """
    Evaluate predictions.
    """
    parameters = model.parameters
    id_to_tag = model.id_to_tag
    assert len(y_preds) == len(y_reals)

    eval_id = str(uuid.uuid4())
    output_path = os.path.join(parameters['path_eval_result'], "eval.%s.output" % eval_id)
    scores_path = os.path.join(parameters['path_eval_result'], "eval.%s.scores" % eval_id)
    file = open(output_path, 'w')
    file_score = open(scores_path, 'w')

    print("Evaluating model with eval_id = " + eval_id)

    # Init result list[precision, recall, F] for each entity type
    elist = [v.split('-', 1)[1] for k, v in id_to_tag.items() if v.split('-', 1)[0] == 'B']
    result = [[0] * 3] * len(elist)

    for i, sentence in enumerate(raw_xs):
        file.write(' '.join(sentence) + '\n')

        p_tags = [[id_to_tag[int(y)] for y in y_pred] for y_pred in y_preds[i]]
        r_tags = [[id_to_tag[int(y)] for y in y_real] for y_real in y_reals[i]]
        p_entities = collect_entity(p_tags)
        r_entities = collect_entity(r_tags)

        file.write('predict|' + '|'.join(
            [','.join([str(entity[0]), str(entity[1]), entity[2]]) for entity in p_entities]) + '\n')
        file.write('gold|' + '|'.join(
            [','.join([str(entity[0]), str(entity[1]), entity[2]]) for entity in r_entities]) + '\n\n')

        result, p_entities, r_entities = eval_level(r_entities, p_entities, result, elist)

    tab = tt.Texttable()
    tab.set_cols_width([35, 10, 10, 10, 10, 10, 10])
    tab.set_cols_align(["l"] * 7)
    headings = ['Category', 'Precision', 'Recall', 'F-score', 'Predicts', 'Golds', 'Correct']
    tab.header(headings)
    f = measures(result, elist, tab)

    file.close()
    file_score.write(tab.draw())
    file_score.close()

    print(tab.draw())

    os.remove(output_path)
    os.remove(scores_path)
    return round(f, 2)


def permutate_list(lst, indices, inv):
    """
    Sort lst by indices.
    """
    ret = [None] * len(lst)
    if inv:
        for i, ind in enumerate(indices):
            ret[ind] = lst[i]
    else:
        for i, ind in enumerate(indices):
            ret[i] = lst[ind]
    return ret


def cost_matrix(dico, cost):
    """
    Reset a cost matrix for CRF to restrict illegal labels.
    Dico is the label dictionary (id_to_tag),cost matrix
    is the transition between two labels
    """
    infi_value = -10000

    for key, value in dico.items():
        tag = value.split('-')[0]
        if tag == 'O':
            iindex = [k for (k, v) in dico.items() if v.split('-')[0] == 'I']
            for i in iindex:
                cost[key][i] = infi_value

        elif tag == 'B':
            sem = value.split('-')[1]
            iindex = [k for (k, v) in dico.items() if v.split('-')[0] == 'I' and v.split('-')[1] != sem]
            for i in iindex:
                cost[key][i] = infi_value

        elif tag == 'I':
            sem = value.split('-')[1]
            iindex = [k for (k, v) in dico.items() if v.split('-')[0] == 'I' and v.split('-')[1] != sem]
            for i in iindex:
                cost[key][i] = infi_value

        elif tag == 'BOS':
            iindex = [k for (k, v) in dico.items() if v.split('-')[0] == 'I']
            for i in iindex:
                cost[key][i] = infi_value

        elif tag == 'EOS':
            cost[key] = infi_value

    return cost


def collect_entity(lst):
    """
    Collect predicted tags(e.g. BIO)
    in order to get entities including nested ones
    """
    entities = []
    for itemlst in lst:
        for i, tag in enumerate(itemlst):
            if tag.split('-', 1)[0] == 'B':
                entities.append([i, i + 1, tag.split('-', 1)[1]])
            elif tag.split('-', 1)[0] == 'I':
                entities[-1][1] += 1

    entities = remove_dul(entities)

    return entities


def remove_dul(entitylst):
    """
    Remove duplicate entities in one sequence.
    """
    entitylst = [tuple(entity) for entity in entitylst]
    entitylst = set(entitylst)
    entitylst = [list(entity) for entity in entitylst]

    return entitylst


def eval_level(golds, preds, result, elist):
    """
    Evaluate overall entities.
    """

    result = [list(map(operator.add, result[j], match(preds, golds, etype)))
              for j, etype in enumerate(elist)]

    return result, preds, golds


def measures(lst, e_lst, table):
    """
    Calculate precision, recall and F-score.
    """

    table.set_cols_dtype(['t', 'f', 'f', 'f', 'i', 'i', 'i'])
    p_count = 0
    r_count = 0
    correct = 0

    for j, etype in enumerate(e_lst):
        p, r, f = p_r_f(lst[j])
        p_count += lst[j][0]
        r_count += lst[j][1]
        correct += lst[j][-1]
        table.add_row(tuple([etype, p, r, f, lst[j][0], lst[j][1], lst[j][2]]))

    overall_p, overall_r, overall_f = p_r_f([p_count, r_count, correct])

    table.add_row(tuple(['Overall', overall_p, overall_r, overall_f,
                         p_count, r_count, correct]))

    return overall_f


def p_r_f(lst):
    p_count, r_count, correct = lst[0], lst[1], lst[-1]

    if correct == 0 and r_count != 0:
        p = r = f = 0
    elif correct == 0 and r_count == 0:
        p = r = f = 100
    else:
        p = correct / p_count * 100
        r = correct / r_count * 100
        f = 2 * p * r / (p + r)

    return p, r, f

