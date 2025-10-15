from pydantic import BaseModel
from typing import List, Optional

class ModelConfig(BaseModel):
    path: str
    class_name: str
    input_shape: List[int]
    checkpoint: Optional[str] = None

class RunnerConfig(BaseModel):
    name: str
    optimize: bool
    use_cuda: Optional[bool] = None

class ExportConfig(BaseModel):
    opset_version: int
    do_constant_folding: bool

class ValidateConfig(BaseModel):
    num_tests: int
    tolerance: float

class Config(BaseModel):
    model: ModelConfig
    export: ExportConfig
    runners: List[RunnerConfig]
    validate: ValidateConfig
