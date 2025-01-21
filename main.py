import logging
import torch
import time
import argparse
from torch.utils.data import DataLoader
import numpy as np
# from train import *
from evaluation import *
from prediction import *
from models import *
# from utils import *
from detector import *
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# eval_dict = {
#     'eval_for_tail': eval_for_tail
# }

class DataPath():
    def __init__(self):

        self.entity_path = "data/FB15K237/entity2id.txt"
        self.relation_path = "data/FB15K237/relation2id.txt"
        self.train_path = "data/FB15K237/train2id.txt"
        self.valid_path = "data/FB15K237/valid2id.txt"
        self.test_path = "data/FB15K237/test2id.txt"
        self.train_class_path = "data/FB15K237/class_train.txt"
        self.valid_class_path = "data/FB15K237/class_valid.txt"
        self.test_class_path = "data/FB15K237/class_test.txt"
        self.ontology_path = "data/FB15K237/output_data.json"
        self.classlabel_path = "data/FB15K237/classlabel2id.txt"
        self.relation_path_path = "data/FB15K237/relation_path.txt"
        self.inheritance_path = "data/FB15K237/inheritance.txt"


def train_without_label(data, model, optimizer):
    model.train()
    for batch_data in tqdm(data):
        h = batch_data[0][0]
        t = batch_data[0][1]
        r = batch_data[0][2]
        ch = batch_data[1][0]
        ct = batch_data[0][1]

        optimizer.zero_grad()
        loss, _ = model(h, r, ch, t)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        l2 = loss.item()

    return l2



class CustomDataset(Dataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        return self.data1[idx], self.data2[idx]

class Experiment:
    def __init__(self, config):
        self.model_name = config.get('model_name')
        self.inter_model = config.get('inter_model')
        self.entity_num = config.get('entity_num')
        self.relation_num = config.get('relation_num')
        self.embedding_dim = config.get('embedding_dim')
        self.batch_size = config.get('batch_size')
        self.dataset = Detector(DataPath())
        self.model = Ontology(self.inter_model, self.entity_num, self.relation_num, 128)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.003, weight_decay=0)
        self.train_func = train_without_label
        self.do_validate = config.get('do_validate')
        self.device = torch.device("cpu")


    def train_and_eval(self):
        dataset = CustomDataset(self.dataset.train, self.dataset.corrupt_train)
        train_loader = DataLoader(dataset, self.batch_size, shuffle=True, drop_last=False)


        for epoch in range(300):
            logging.info('Start training epoch: %d' % (epoch + 1))
            start_time = time.time()
            evaluation_with_LLM(self.dataset.entity, self.dataset.relation, train_loader)

            if self.do_validate:
                dataset = CustomDataset(self.dataset.valid, self.dataset.class_valid)
                valid_loader = DataLoader(dataset, self.batch_size, shuffle=True, drop_last=False)
                evaluation_with_LLM(self.dataset.entity, self.dataset.relation, valid_loader)

            end_time = time.time()
            print('[Epoch #%d] training time: %.2f seconds' % (epoch + 1, end_time - start_time))


        logging.info('Finished! Model saved')



def load_json_config(config_path):
    logging.info(' Loading configuration '.center(100, '-'))
    if not os.path.exists(config_path):
        logging.warning(f'File {config_path} does not exist, empty list is returned.')
    with open(config_path, 'r') as f:
        config = json.load(f)
    config['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return config



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Knowledge graph inference arguments.')
    parser.add_argument('-c', '--config', dest='config_file', default='configs/Ontology.json',
                        help='The path of configuration json file.')
    args = parser.parse_args()
    # print(args)

    config = load_json_config(args.config_file)
    print(config)

    experiment = Experiment(config)

    experiment.train_and_eval()

