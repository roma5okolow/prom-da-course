import torch
import onnx
from datetime import datetime
import subprocess
from model import NERClassifier

def get_git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"
    
class NEROnnxModel(torch.nn.Module):
    def __init__(self, encoder, mlp):
        super().__init__()
        self.encoder = encoder
        self.mlp = mlp

    def forward(self, input_ids):
        d, _, _ = self.encoder(input_ids)
        d = d.transpose(0, 1)
        return self.mlp(d)

# загружаем исходную модель
full_model = NERClassifier()
full_model.load_state_dict(torch.load("model_state_dict.pt"))
full_model.eval()

# собираем onnx-модель
onnx_model = NEROnnxModel(
    full_model.encoder,
    full_model.MLP
)
onnx_model.eval()

# dummy input: [batch, seq_len]
dummy_input = torch.randint(0, 100, (1, 16), dtype=torch.long)

onnx_path = "ner_model.onnx"

torch.onnx.export(
    onnx_model,
    dummy_input,
    onnx_path,
    input_names=["input_ids"],
    output_names=["logits"],
    opset_version=16,
    dynamic_axes={
        # Define BOTH batch (0) and seq (1) as dynamic for inputs and outputs
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "logits":    {0: "batch_size", 1: "seq_len"}
    }
)
onnx.checker.check_model('ner_model.onnx')

# metadata
onnx_model_proto = onnx.load(onnx_path)

metadata = {
    "git_commit": get_git_hash(),
    "export_date": datetime.utcnow().isoformat(),
    "experiment_name": "baseline_ner_lstm_no_tokenizer"
}

for k, v in metadata.items():
    meta = onnx_model_proto.metadata_props.add()
    meta.key = k
    meta.value = v

onnx.save(onnx_model_proto, onnx_path)
onnx.checker.check_model('ner_model.onnx')
print("ONNX export completed")
