import torch
import onnx
from onnxsim import simplify
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input TorchScript model path')
    parser.add_argument('--output', required=True, help='Output ONNX model path')
    parser.add_argument('--input-shape', required=True, help='Input shape, e.g., "1,3,256,256"')
    args = parser.parse_args()

    shape = tuple(map(int, args.input_shape.split(',')))

    model = torch.jit.load(args.input)
    dummy = torch.randn(shape)

    # Save ONNX
    torch.onnx.export(model, dummy, args.output, opset_version=14, do_constant_folding=True, input_names=['x'], output_names=['y'])

    # simplify
    model_onnx = onnx.load(args.output)
    model_simp, check = simplify(model_onnx)
    assert check
    onnx.save(model_simp, args.output)
    print(f"Model saved to {args.output}")

if __name__ == '__main__':
    main()
