import numpy as np
import torch
from nltk.tokenize import RegexpTokenizer
import json
from torchvision import ops

class Tokenizer(object):
    def __init__(self):
        with open('word2ind.json', 'r', encoding='utf-8') as f:
            self.word2ind = json.load(f)
        
        self.tokenizer = RegexpTokenizer(r'[а-яА-Я]+|[^\w\s]|\d+')
    
    def __call__(self, sentences, max_length=128, pad_to_max_length=False):
        if not sentences:
            return torch.tensor([])
        
        tokens = self.tokenizer.tokenize_sents(sentences)
        if not pad_to_max_length:
            max_length = min(max_length, max(map(len, tokens)))
        
        processed_tokens = []
        for s in tokens:
            if len(s) < max_length:
                tokenized = ['[CLS]'] + s + ['[SEP]'] + ['[PAD]'] * (max_length - len(s))
            else:
                tokenized = ['[CLS]'] + s[:max_length] + ['[SEP]']
            processed_tokens.append(tokenized)
        
        ids = [[self.word2ind.get(w, self.word2ind['[UNK]']) for w in sent] for sent in processed_tokens]
        return torch.tensor(ids)
    
    def tokenize(self, sentence):
        return self.tokenizer.tokenize(sentence)

class Encoder(torch.nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(self,
                 vocab_dim=2000004, # размерность словаря
                 emb_dim = 300, # размерность векторов embedding
                 hidden_dim = 300, # размер скрытого состояния LSTM
                 num_layers = 3, # количество слоёв в модели LSTM
                 bidirectional = True):
        super().__init__()

        self.num_direction = int(bidirectional + 1)
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.embedding = torch.nn.Embedding(vocab_dim, emb_dim)
        self.encoder = torch.nn.LSTM(emb_dim, hidden_dim, num_layers, bidirectional = bidirectional)

    def forward(self, input):
        input = self.embedding(input)
        input = torch.transpose(input, 0, 1)
        d, (h, c) = self.encoder(input) # encoder возвращает d, h, c
        return d, torch.transpose(h, 0, 1), torch.transpose(c, 0, 1)

# Классификатор - энкодер + многослойный перцептрон.
class NERClassifier(torch.nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(
            self,
            vocab_dim=2000004, # размерность словаря
            emb_dim = 300, # размерность векторов embedding
            hidden_dim = 300, # размер скрытого состояния LSTM
            num_layers = 3, # количество слоёв в модели LSTM
            bidirectional = True,
            number_of_classes = 8,
            dropout=0
        ):
        super().__init__()

        self.encoder = Encoder(vocab_dim, emb_dim, hidden_dim, num_layers, bidirectional)
        bidirect_mult = 2 if bidirectional else 1
        hidden_dim = hidden_dim * bidirect_mult
        self.MLP = ops.MLP(in_channels = hidden_dim, hidden_channels = [hidden_dim // 2, hidden_dim // 4, number_of_classes], dropout=dropout)

        self.tokenizer = Tokenizer()

        with open('idx2tag.json', 'r', encoding='utf-8') as f:
            loaded_dict = json.load(f)
            self.idx2tag = {int(k): v for k, v in loaded_dict.items()}
    
    def forward(self, input):
        d, _, _ = self.encoder(self.tokenizer(input))
        return self.MLP(d)
    
    def predict(self, input):
        d, _, _ = self.encoder(self.tokenizer([input]))
        out = self.MLP(d)
        pred_tags = torch.argmax(out, dim=-1).cpu().numpy().flatten()
        return [
            (token, self.idx2tag[tag]) for token, tag in zip(self.tokenizer.tokenize(input), pred_tags)
        ]