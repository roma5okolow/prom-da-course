from nltk.tokenize import RegexpTokenizer
import json
import numpy as np
import os


class Tokenizer(object):
    def __init__(self, vocab_path="data/word2ind.json"):
        self.tokenizer = RegexpTokenizer(r"[а-яА-Я]+|[^\w\s]|\d+")

        # Load Vocabulary
        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.word2ind = json.load(f)
        else:
            # Fallback for testing/export without file
            print(f"Warning: {vocab_path} not found. Using dummy vocab.")
            self.word2ind = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}

        # Define special IDs safely
        self.pad_id = self.word2ind.get("[PAD]", 0)
        self.unk_id = self.word2ind.get("[UNK]", 1)
        self.cls_id = self.word2ind.get("[CLS]", 2)
        self.sep_id = self.word2ind.get("[SEP]", 3)

    def tokenize(self, sentence):
        return self.tokenizer.tokenize(sentence)

    def encode(
        self, text: str, max_length: int = 128, pad_to_max_length: bool = True
    ) -> np.ndarray:
        tokens = self.tokenize(text)

        ids = [self.cls_id]
        # Truncate to fit [CLS] ... [SEP]
        for t in tokens[: max_length - 2]:
            ids.append(self.word2ind.get(t, self.unk_id))
        ids.append(self.sep_id)

        if pad_to_max_length:
            padding = [self.pad_id] * (max_length - len(ids))
            ids.extend(padding)

        return np.array(ids, dtype=np.int64)

    def encode_batch(self, texts, max_length=128):
        batch = [self.encode(t, max_length) for t in texts]
        return np.stack(batch, axis=0)
