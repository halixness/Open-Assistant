import torch
import numpy as np
from transformers import AutoModel, AutoModelForSeq2SeqLM, T5EncoderModel

class FlanT5Model(torch.nn.Module):
    def __init__(self, model_name, embedding_size):
        super().__init__()
        
        self.flant5_model = model_name
        
        assert model_name in [
            "google/flan-t5-base",
            "google/flan-t5-xl",
            "google/flan-t5-xxl",
        ]

        self.model = T5EncoderModel.from_pretrained(self.flant5_model, trust_remote_code=True)
        for p in self.model.parameters():
            p.requires_grad = False

        self.mlp = torch.nn.ModuleList([
            torch.nn.Linear(embedding_size, 256),
            torch.nn.Linear(256, 1),
        ])

    def forward(self, prefixes, suffixes):
        """
            Yields a "positiveness" score for a given pair of sentences.
        """

        embedded_prefixes = self.model(**prefixes).last_hidden_state
        embedded_suffixes = self.model(**suffixes).last_hidden_state

        # projecting to spaces
        for i, l in enumerate(self.mlp):
            embedded_prefixes = l(embedded_prefixes)

        for i, l in enumerate(self.mlp):
            embedded_suffixes = l(embedded_suffixes)

        # [n, 400, 1] => [n, 400]
        embedded_prefixes = torch.squeeze(embedded_prefixes / embedded_prefixes.norm(dim=1, keepdim=True))
        embedded_suffixes = torch.squeeze(embedded_suffixes / embedded_suffixes.norm(dim=1, keepdim=True))

        # [n, 400]*[n, 400] => [n, 1] in [-1, 1]
        cosine_dot_product = torch.sum(embedded_prefixes * embedded_suffixes, dim=1)

        return cosine_dot_product
