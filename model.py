import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel


class Model(nn.Module):
    def forward(self, X: torch.Tensor, return_embedding=False):
        N = X.shape[0]
        h_0 = torch.zeros((self.num_layers, N, self.hidden_size))
        y, _ = self.rnn(X, h_0)
        output = self.sigmoid(self.linear(y[:, -1, :]))
        if return_embedding:
            return output, y[:, -1, :]
        else:
            return output

    def embed_texts(self, texts):
        tokenized_texts = self.tokenizer(texts, padding=True, return_tensors="pt")
        tokenized_texts['attention_mask'][0]*=2
        tokenized_texts['attention_mask'][-1]*=2
        model_output = self.pretrained_model(input_ids=tokenized_texts['input_ids'],
                                             attention_mask=tokenized_texts['attention_mask'])
        embeddings = model_output["last_hidden_state"]
        return embeddings
