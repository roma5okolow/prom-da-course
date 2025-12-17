from nltk.tokenize import RegexpTokenizer
import json
import numpy as np

class Tokenizer(object):
    def __init__(self):
        with open('word2ind.json', 'r', encoding='utf-8') as f:
            self.word2ind = json.load(f)
        
        self.tokenizer = RegexpTokenizer(r'[а-яА-Я]+|[^\w\s]|\d+')
    
    def encode(
        self,
        text: str,
        max_length: int = 128,
        pad_to_max_length: bool = True
    ) -> np.ndarray:
        tokens = self.tokenize(text)

        ids = [self.cls_id]
        for t in tokens[:max_length]:
            ids.append(self.word2ind.get(t, self.unk_id))
        ids.append(self.sep_id)

        if pad_to_max_length:
            ids = ids[:max_length]
            ids += [self.pad_id] * (max_length - len(ids))

        return np.array(ids, dtype=np.int64)

    def encode_batch(
        self,
        texts,
        max_length: int = 128
    ) -> np.ndarray:
        batch = [self.encode(t, max_length) for t in texts]
        return np.stack(batch, axis=0)
    
    def tokenize(self, sentence):
        return self.tokenizer.tokenize(sentence)
