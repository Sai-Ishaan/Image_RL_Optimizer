import pytest
import numpy as np
from lightweight_rl.transforms import AtariTransformPipeline
from lightweight_rl.env_client import NatureCNN, WeightMappingParser
import torch

@pytest.fixture
def pipeline():
    ### Initialize preprocessing pipeline with default params
    return AtariTransformPipeline(stack_size=4, target_dim=(84,84))

@pytest.fixture
def model():
    ## Init Raw NatureCNN model with 6 default atari actions
    return NatureCNN(in_channels=4, num_actions=6).eval() ##Eval mode for parity actions

def test_rgb_frame_pipeline_flow(pipeline,model):
    ###Case 1: Standard Raw Atari emulator RGB Frame
    ## Verifies downsampling, grayscale conversion, stackinh, and model inference pass cleanly
    dummy_rgb_frame = np.random.randint(0,256, size=(210,160,3), dtype=np.uint8)

    ## Reset phase (all 4 channels should be populated with the identical frame)
    state_tensor = pipeline.reset(dummy_rgb_frame)
    assert state_tensor.shape == (1,4,84,84), f"Expected shape (1,4,84,84), got {state_tensor.shape}"
    assert state_tensor.dtype == torch.float32, "Tensor must be cast to Float32"
    assert state_tensor.max() <= 1.0 and state_tensor.min() >= 0.0, "Normalisation failed, values should be in [0,1]"

    ## Fwd step phase
    next_frame = np.random.randint(0, 256, size=(210, 160, 3), dtype=np.uint8)
    updated_tensor = pipeline.process_step(next_frame)

    ##mOdel inference verification
    with torch.no_grad():
        model_output = model(updated_tensor) ## q_values

    assert model_output.shape == (1,6), f"Expected q-value/model output shape (1,6), got {model_output.shape}."
    print(f"\nRGB Inferece Success!!. Output Q-Values shape: {model_output.shape}")

def test_flattened_array_fallback(pipeline):
    ###Case 2: The loopback server sends a flattened JSON Payload array
    ##This function verifies that the transformation pipeline has successfully re-constructed the array internally
    ##210X160X3 = 100800 flat elements
    flat_frame = np.random.randint(0,256,size=(100800,), dtype=np.uint8)
    try:
        state_tensor = pipeline.reset(flat_frame)
        assert state_tensor.shape == (1,4,84,84)
        print("Success! Flattebned array reshaped and parsed.")
    except ValueError as e:
        pytest.fail(f"Pipeline crahsed on flattened array: {e}")

def test_malformed_input_handling(pipeline):
    ###Case 3: Malformed input handling. Edge case check
    ##This function verifies the pipeline raises an intentional error if unexpected unwanted dimensions are passed in.
    malformed_frame = np.random.randint(0,256, size=(100,100), dtype=np.uint8)
    with pytest.raises(ValueError) as excinfo:
        pipeline.reset(malformed_frame)

    assert "Malformed input dimensions" in str(excinfo.value)
    print("Success! Pipeline correctly raised and rejected malformed dimensions.")
    