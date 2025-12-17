import onnxruntime as ort
from tokenizer import Tokenizer
import json

class ONNXNERModel:
    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(onnx_path)
        self.tokenizer = Tokenizer()

        with open('idx2tag.json', 'r', encoding='utf-8') as f:
            loaded_dict = json.load(f)
            self.idx2tag = {int(k): v for k, v in loaded_dict.items()}

    def predict(self, text: str):
        input_ids = self.tokenizer.encode_batch([text])

        logits = self.session.run(
            None,
            {"input_ids": input_ids}
        )[0]

        preds = logits.argmax(axis=-1)[0]
        tokens = self.tokenizer.tokenize(text)

        return [
            (token, self.idx2tag[p])
            for token, p in zip(tokens, preds)
        ]