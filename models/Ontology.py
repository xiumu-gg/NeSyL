import torch
from torch import nn, optim
from transformers import AutoModel
from .BaseModel import *
from transformers.models.bert.modeling_bert import BertEncoder, BertConfig
import torch.nn.functional as F


class Ontology(nn.Module):
    def __init__(self, model_name='bert-base-uncased', entity_vocab_size=14541, relation_vocab_size=237, embedding_dim=100):
        super(Ontology, self).__init__()
        self.transformer = AutoModel.from_pretrained('bert-base-uncased')
        self.entity_embedding = nn.Embedding(num_embeddings=entity_vocab_size, embedding_dim=embedding_dim)
        self.relation_embedding = nn.Embedding(num_embeddings=relation_vocab_size, embedding_dim=embedding_dim)
        self.class_entity_embedding = nn.Embedding(num_embeddings=1267, embedding_dim=embedding_dim)

        config = BertConfig(
            vocab_size=0,
            hidden_size=embedding_dim,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=256 * 4
        )

        self.encoder = BertEncoder(config)

        # A linear layer to project transformer output to entity embedding space
        self.output_projection = nn.Linear(384, 128)
        self.classifier = nn.Linear(128, entity_vocab_size)  # Classifier to predict tail entity
        self.loss = Loss(0.1, 14541)

    def forward(self, head_ids, relation_ids, head_class, tail_ids=None):
        # Embed head entities and relations
        head_embeddings = self.entity_embedding(torch.tensor(head_ids))  # (batch_size, embedding_dim)
        relation_embeddings = self.relation_embedding(torch.tensor(relation_ids))  # (batch_size, embedding_dim)
        class_embeddings = self.class_entity_embedding(torch.tensor(head_class))

        # Combine head and relation embeddings
        # combined_embeddings = head_embeddings + relation_embeddings  # (batch_size, embedding_dim)
        # combined_embeddings = combined_embeddings.unsqueeze(1)  # Add sequence dimension for transformer input

        combined_embeddings = torch.cat([head_embeddings.unsqueeze(1), relation_embeddings.unsqueeze(1), class_embeddings.unsqueeze(1)], dim=1)

        # Use Transformer to process the combined embeddings
        # transformer_output = self.transformer(inputs_embeds=combined_embeddings).last_hidden_state
        transformer_output = self.encoder(combined_embeddings).last_hidden_state
        transformer_output = transformer_output.view(32, -1)
        if transformer_output.shape != (32, 384):
            transformer_output = F.pad(transformer_output, (0, 384 - transformer_output.shape[1]), 'constant', 0)

        # Project the transformer output to predict the tail entity embedding
        tail_embedding_prediction = self.output_projection(transformer_output)  # (batch_size, embedding_dim)
        # print(tail_embedding_prediction)
        logit = self.classifier(tail_embedding_prediction)  # (batch_size, entity_vocab_size)
        y = torch.sigmoid(logit)
        return self.loss(y, tail_ids), y


class Loss(BaseModel):
    def __init__(self, label_smoothing, entity_cnt):
        super().__init__()
        self.loss = torch.nn.BCELoss(reduction='sum')
        self.label_smoothing = label_smoothing
        self.entity_cnt = entity_cnt

    def forward(self, batch_p, batch_t=None):
        batch_size = batch_p.shape[0]
        loss = None
        if batch_t is not None:
            batch_e = torch.zeros(batch_size, self.entity_cnt).scatter_(1, batch_t.view(-1, 1), 1)
            batch_e = (1.0 - self.label_smoothing) * batch_e + self.label_smoothing / self.entity_cnt
            loss = self.loss(batch_p, batch_e) / batch_size
        return loss
