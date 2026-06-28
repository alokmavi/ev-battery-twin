import torch
from src.model import RULPredictorCNN

def test_cnn_forward_pass_shape():
    model = RULPredictorCNN(sequence_length=50, feature_dimensions=3)
    mock_tensor = torch.randn(4, 50, 3)
    
    output = model(mock_tensor)
    
    assert output.shape == (4, 1), "Output shape must match (batch_size, 1)"
    assert not torch.isnan(output).any(), "Model output contains NaN values"
