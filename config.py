import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass
from enum import Enum

class ColumnType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    BOOLEAN = "boolean"
    ARRAY = "array"
    DATETIME = "datetime"

@dataclass
class ColumnConfig:
    name: str
    type: ColumnType
    embedding_dim: int = 32
    vocab_size: Optional[int] = None
    max_length: Optional[int] = None
    normalization: str = "standard"
    
@dataclass
class ModelConfig:
    columns: List[ColumnConfig]
    hidden_dims: List[int] = None
    output_dim: int = 1
    dropout: float = 0.3
    activation: str = "relu"
    output_type: str = "regression"
    num_classes: Optional[int] = None
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]