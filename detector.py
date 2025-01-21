import torch
import time
import argparse
from torch.utils.data import DataLoader
from dataset import *
import types
import json
import yaml
import logging
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info(' Ontology conceptual layer testing started ... '.center(100, '-'))



class Detector(DataHandler):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.detector_set = [{} for _ in range(self.num_relation)]
        self.axiomatic = self.axio_process()
        self.entity_class_detection()
        self.axiomatic_addition = self.load_relation_path(data_path.relation_path_path)
        self.path_generation()
        self.relation_level_detection()
        self.inheritance_relation = self.load_inheritance(data_path.inheritance_path)
        self.class_inheritance_detection()

    def axio_process(self):
        combined_axio = self.class_train + self.class_test + self.class_valid
        return list(set(tuple(item) for item in combined_axio))

    def entity_class_detection(self):

        for item in self.axiomatic:
            if item[0] not in self.detector_set[item[2]]:
                self.detector_set[item[2]][int(item[0])] = []
            self.detector_set[item[2]][int(item[0])].append(int(item[1]))

        for i, d in enumerate(self.detector_set):
            self.relation[i]['detected'] = d

    def relation_level_detection(self): # a-b->a | a-b->b | a-b->c
        # [entity_class 1, relation 1, entity_class 2, relation 2, entity_class 3] -> [entity_class 1, ?, entity_class 3]
        for item in self.axiomatic_addition:
            if item['entity_1'] in self.detector_set[item['relation_1']]:
                if item['entity_2'] in self.detector_set[item['relation_1']][item['entity_1']]:
                    if item['entity_2'] in self.detector_set[item['relation_2']]:
                        if item['entity_3'] in self.detector_set[item['relation_2']][item['entity_2']]:
                            if item['entity_1'] not in self.detector_set[item['res']]:
                                self.detector_set[item['res']][int(item['entity_1'])] = []
                            self.detector_set[item['res']][int(item['entity_1'])].append(int(item['entity3']))

        for i, d in enumerate(self.detector_set):
            self.relation[i]['detected'] = d

    def class_inheritance_detection(self):
        ans_aaa = 0
        for item in self.detector_set:
            for key, value in item.items():
                for id in value:
                    if not any(id == row[0] for row in self.inheritance_relation):
                        ans = 1
                    else:
                        for row in self.inheritance_relation:
                            if id == row[0]:
                                if is_public_property(row[0], row[1], key):
                                    if row[1] not in value:
                                        self.detector_set[ans_aaa][key].append(row[1])
            ans_aaa = ans_aaa + 1

        for i, d in enumerate(self.detector_set):
            self.relation[i]['detected'] = d

    def load_relation_path(self, relation_path_path):
        data_list = []
        with open(relation_path_path, 'r') as file:
            total_lines = int(file.readline().strip())
            for line in file:
                line = line.strip()
                if line:
                    entity_class_1, relation_1, entity_class_2, relation_2, entity_class_3, res = line.split()  # 分割每行为6部分
                    data_list.append({'entity_class_1': int(entity_class_1), 'relation_1': int(relation_1),
                                     'entity_class_2': int(entity_class_2), 'relation_2': int(relation_2),
                                     'entity_class_3': int(entity_class_3), 'res': int(res)})
        return data_list

    def path_generation(self):
        queue = deque()
        for item in self.axiomatic_addition:
            queue.append(item)

        # print("Initial queue:", list(queue))

        while queue:
            current_dict = queue.popleft()
            print("Processing:", current_dict)

            for item in list(queue):
                if item['entity_class_1'] == current_dict['entity_class_3']:
                    ans = {'entity_class_1': current_dict['entity_class_1'], 'relation_1': current_dict['res'],
                           'entity_class_2': current_dict['entity_class_3'], 'relation_2': item['res'],
                           'entity_class_3': item['entity_class_3'], 'res': 237}
                    queue.append(ans)
                    self.axiomatic_addition.append(ans)
                if item['entity_class_1'] == current_dict['entity_class_1']:
                    ans = {'entity_class_1': item['entity_class_1'], 'relation_1': item['res'],
                           'entity_class_2': current_dict['entity_class_1'], 'relation_2': current_dict['res'],
                           'entity_class_3': current_dict['entity_class_3'], 'res': 237}
                    queue.append(ans)
                    self.axiomatic_addition.append(ans)

            # if len(queue) > 0 and len(queue) % 2 == 0:
            #     new_dict = {'new_key': 'new_value'}
            #     queue.append(new_dict)
            #     print("Added new dict to queue:", new_dict)
        # print("Queue after processing:", list(queue))

    def load_inheritance(self, inhertance_path):    # [Subclass, Parent Class]
        data_list = []
        with open(inhertance_path, 'r') as file:
            total_lines = int(file.readline().strip())
            for line in file:
                line = line.strip()
                if line:
                    entity_class_1, entity_class_2 = line.split(',')
                    data_list.append([entity_class_1, entity_class_2])
        return data_list

def is_public_property(subclass, parent_class, property3):
    return True
    pass
