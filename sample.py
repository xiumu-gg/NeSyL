import numpy as np
import random


def generate_negative_samples(triple, num_entities, num_neg_samples=1, corrupt_head=True):

    neg_samples = []

    for i in range(num_neg_samples):
        if corrupt_head:
            while True:
                new_head_id = random.randint(0, num_entities - 1)
                if new_head_id != triple[i][0]:
                    break
            neg_sample = (new_head_id, triple[i][1], triple[i][2])
        else:
            while True:
                new_tail_id = random.randint(0, num_entities - 1)
                if new_tail_id != triple[i][1]:
                    break
            neg_sample = (triple[i][0], new_tail_id, triple[i][2])

        neg_samples.append(neg_sample)

    return neg_samples


num_entities = 100
triple = (1, 2, 3)

neg_samples = generate_negative_samples(triple, num_entities, num_neg_samples=3, corrupt_head=True)
print(neg_samples)