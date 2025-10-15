import torch
import onnxruntime as ort
import numpy as np

def validate_models(torchscript_path: str, onnx_path: str, num_tests: int = 8, tolerance: float = 1e-4):
    """
    Validates the outputs of a TorchScript model and an ONNX model.
    """
    # Load TorchScript model
    ts_model = torch.jit.load(torchscript_path)
    ts_model.eval()

    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name

    for i in range(num_tests):
        # Generate random input
        input_shape = ort_session.get_inputs()[0].shape
        # handle dynamic axes
        input_shape = [1 if isinstance(dim, str) else dim for dim in input_shape]
        dummy_input = torch.randn(input_shape)

        # TorchScript inference
        ts_output = ts_model(dummy_input)
        ts_output_np = ts_output.detach().numpy()

        # ONNX inference
        ort_output = ort_session.run(None, {input_name: dummy_input.numpy()})[0]

        # Compare outputs
        np.testing.assert_allclose(ts_output_np, ort_output, rtol=tolerance, atol=tolerance)
        print(f"Test {i+1}/{num_tests} passed.")

    print("All validation tests passed.")
