import torch
from .config import ModelConfig
import os
import importlib.util
import sys

def _import_model_from_path(model_path: str, model_class_name: str) -> torch.nn.Module:
    """
    Dynamically imports a model class from a given file path.
    """
    spec = importlib.util.spec_from_file_location(model_class_name, model_path)
    if spec is None:
        raise ImportError(f"Could not load spec for module at {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model_class = getattr(module, model_class_name)
    return model_class

def convert_model(config: dict):
    """
    Converts a PyTorch model to TorchScript and then to ONNX based on the provided config.
    """
    model_config = ModelConfig(**config['model'])
    export_config = config['export']

    # Dynamically import and instantiate the model
    model_class = _import_model_from_path(model_config.path, model_config.class_name)
    model = model_class()

    if model_config.checkpoint:
        model.load_state_dict(torch.load(model_config.checkpoint))
    
    model.eval()

    dummy_input = torch.randn(model_config.input_shape)

    # Trace the model
    traced_model = torch.jit.trace(model, dummy_input)
    os.makedirs('out', exist_ok=True)
    traced_path = "out/model.pt"
    traced_model.save(traced_path)
    print(f"Traced model saved to {traced_path}")

    # Export to ONNX
    onnx_path = "out/model.onnx"
    torch.onnx.export(
        traced_model,
        dummy_input,
        onnx_path,
        opset_version=export_config.get('opset_version', 14),
        do_constant_folding=export_config.get('do_constant_folding', True),
        input_names=['input'],
        output_names=['output']
    )
    print(f"ONNX model saved to {onnx_path}")
