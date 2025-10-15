import pytest
import os
import torch
from oxidizedvision.convert import convert_model
from oxidizedvision.validate import validate_models

@pytest.fixture
def setup_test_model():
    os.makedirs("test_model", exist_ok=True)
    with open("test_model/model.py", "w") as f:
        f.write("""
import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1)
    def forward(self, x):
        return self.conv(x)
""")
    
    config = {
        "model": {
            "path": "test_model/model.py",
            "class_name": "MyModel",
            "input_shape": [1, 3, 32, 32]
        },
        "export": {
            "opset_version": 14,
            "do_constant_folding": True
        }
    }
    return config

def test_conversion_and_validation(setup_test_model):
    config = setup_test_model
    convert_model(config)

    assert os.path.exists("out/model.pt")
    assert os.path.exists("out/model.onnx")

    validate_models("out/model.pt", "out/model.onnx")

    # Clean up
    os.remove("out/model.pt")
    os.remove("out/model.onnx")
    os.remove("test_model/model.py")
    os.rmdir("test_model")
