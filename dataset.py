import torch
import time
import argparse
from torch.utils.data import DataLoader
import types
import json
import yaml
import logging
from sample import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class DataHandler:
    def __init__(self, data_path):
        self.entity = self.load_concept(data_path.entity_path)
        self.relation = self.load_concept(data_path.relation_path)
        self.train = self.load_triple(data_path.train_path)
        self.valid = self.load_triple(data_path.valid_path)
        self.test = self.load_triple(data_path.test_path)
        self.class_train = self.load_triple(data_path.train_class_path) # [entity_class, entity_class, relation]
        self.class_valid = self.load_triple(data_path.valid_class_path)
        self.class_test = self.load_triple(data_path.test_class_path)

        self.load_ontology(data_path.ontology_path) # ans

        self.classlabel = self.load_classlabel(data_path.classlabel_path)
        self.entity_relation = {}
        self.gen_entity_relation_multidata()

        self.classid_process(self.entity, self.classlabel) # ans

        self.corrupt_train = generate_negative_samples(self.train, len(self.entity), len(self.train), False)
        self.corrupt_test = generate_negative_samples(self.test, len(self.entity), len(self.test), False)
        self.corrupt_valid = generate_negative_samples(self.valid, len(self.entity), len(self.valid), False)

    def load_concept(self, file_path):
        data_list = []
        with open(file_path, 'r') as file:
            total_lines = int(file.readline().strip())
            for line in file:
                line = line.strip()
                if line:
                    key, value = line.split()
                    data_list.append({'freebase_id': key, 'value': int(value)})

        return data_list

    def load_classlabel(self, file_path):
        data_list = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                label, id = line.strip().split('\t')
                data_list.append({'classlabel': label, 'id': int(id)})
        return data_list

    def load_triple(self, path):
        data = []
        with open(path, 'r') as file:
            total_lines = int(file.readline().strip())
            for _ in range(total_lines):
                line = file.readline().strip()
                if line:
                    parts = list(map(int, line.split()))
                    data.append(parts)
        return data

    def load_ontology(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            self.ontology = json.load(file)
        self.num_relation = len(self.relation)
        for ans in self.ontology:
            self.entity[int(ans["value"])]["label"] = ans["label"]
            if ans["classlabel"] == []:
                self.entity[int(ans["value"])]["classname"] = "none"
            else:
                self.entity[int(ans["value"])]["classname"] = ans["classlabel"][0]

    def classid_process(self, data1, data2):

        classlabel_to_id = {item['classlabel']: item['id'] for item in data2}


        ans = 0
        for item in data1:
            classname = item['classname']
            if classname in classlabel_to_id:
                item['classid'] = classlabel_to_id[classname]
            else:
                item['classid'] = -1

    def gen_entity_relation_multidata(self):
        logging.info(' Generating entity-relation dictionaries to accelerate evaluation process '.center(100, '-'))
        full_data = self.train + self.valid + self.test
        self.entity_relation['as_head'] = {}
        self.entity_relation['as_tail'] = {}
        for i in range(len(self.entity)):
            self.entity_relation['as_head'][i] = {}
            self.entity_relation['as_tail'][i] = {}
            for j in range(len(self.relation)):
                self.entity_relation['as_head'][i][j] = []
                self.entity_relation['as_tail'][i][j] = []
        for triple in full_data:
            h, t, r = triple
            self.entity_relation['as_head'][t][r].append(h)
            self.entity_relation['as_tail'][h][r].append(t)
