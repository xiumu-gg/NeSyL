import torch
import logging
import numpy as np
from tqdm import tqdm
from model_eval import *
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def evaluation_with_LLM(entity_set, relation_set, triple):
    result = []
    for batch_idx, batch_data in enumerate(tqdm(triple)):
        eval_h = batch_data[0][0]
        eval_t = batch_data[0][1]
        eval_r = batch_data[0][2]
        result_text = id2text([eval_h, eval_t, eval_r], entity_set, relation_set)
        instruction = text2instr(result_text)
        prompts = get_prompts(eval_h, eval_t, eval_r, relation_set)
        ans, response = model_eval(instruction, prompts)
        result.append(
            {
                "answer": ans,
                "predict": response
            }
        )
    answer = []
    predict = []
    for data in result:
        for ite in data["answer"]:
            if "True" in ite:
                answer.append(1)
            else:
                answer.append(0)
        for ite in data["predict"]:
            if "True" in ite:
                predict.append(1)
            else:
                predict.append(0)
    acc = accuracy_score(y_true=answer, y_pred=predict)
    p = precision_score(y_true=answer, y_pred=predict)
    r = recall_score(y_true=answer, y_pred=predict)
    f1 = f1_score(y_true=answer, y_pred=predict)
    eval_show([acc, p, r, f1])




def get_prompts(h_id, t_id, r_id, relation):
    res = []
    for i in range(len(h_id)):
        if h_id[i] in relation[r_id[i]]['detected']:
            if t_id[i] in relation[r_id[i]]['detected'][h_id[i]]:
                res.append(True)
            else:
                res.append(False)
        else:
            res.append(False)
    return res

def id2text(triple_id, entity_set, relation_set):
    entity1 = []
    relation = []
    entity2 = []
    for id in triple_id[0]:
        ans = entity_set[int(id)]['label']
        entity1.append(ans)
    for id in triple_id[2]:
        ans = relation_set[int(id)]['freebase_id']
        relation.append(ans)
    for id in triple_id[1]:
        ans = entity_set[int(id)]['label']
        entity2.append(ans)
    return [entity1, relation, entity2]

def text2instr(text):
    sentence_template = "Help me to determine if the triad entity '{}', relationship '{}', entity '{}' is real, if it is real, please generate true, if it is not real, please. Generate false."
    result = []
    for id in range(len(text[0])):
        sentence = sentence_template.format(text[0][id], text[1][id], text[2][id])
        result.append(sentence)
    return result


def eval_show(result):
    print('For %s data: acc=%.4f - p=%.4f - r=%.4f - f1=%.4f' % ('FB15K237O', result[0], result[1], result[2], result[3]))

