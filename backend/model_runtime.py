import onnxruntime as ort
import numpy as np
import json
from tokenizer import Tokenizer


class ONNXNERModel:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

        self.tokenizer = Tokenizer()

        with open("data/idx2tag.json", "r", encoding="utf-8") as f:
            loaded_dict = json.load(f)
            self.idx2tag = {int(k): v for k, v in loaded_dict.items()}

        self.metadata = {}
        meta_map = self.session.get_modelmeta().custom_metadata_map
        for key, value in meta_map.items():
            self.metadata[key] = value

    def predict(self, text: str):
        input_ids = self.tokenizer.encode(text, max_length=128)
        input_ids = input_ids[None, ...]

        onnx_inputs = {self.session.get_inputs()[0].name: input_ids}
        onnx_outputs = self.session.run(None, onnx_inputs)

        logits = onnx_outputs[0]
        pred_ids = np.argmax(logits, axis=-1).flatten()

        raw_tokens = self.tokenizer.tokenize(text)
        result = []

        for idx_in_pred, token in enumerate(raw_tokens):
            if idx_in_pred < len(pred_ids):
                tag_id = pred_ids[idx_in_pred]
                result.append({"token": token, "tag": self.idx2tag.get(tag_id, "O")})

        return result
