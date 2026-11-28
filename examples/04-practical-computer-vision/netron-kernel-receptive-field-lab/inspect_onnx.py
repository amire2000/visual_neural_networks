"""Inspect ONNX Conv weights, shapes, and estimated MACs."""

from math import prod
from pathlib import Path

import onnx
from onnx import helper, shape_inference

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
MODEL_NAMES = ("baseline", "optimized")


def get_shape_map(graph):
    """Map graph values and initializers to their known dimensions."""
    shapes = {}
    for value in (
        list(graph.input) + list(graph.value_info) + list(graph.output)
    ):
        dimensions = value.type.tensor_type.shape.dim
        shapes[value.name] = [
            dimension.dim_value if dimension.HasField("dim_value") else "?"
            for dimension in dimensions
        ]
    for initializer in graph.initializer:
        shapes[initializer.name] = list(initializer.dims)
    return shapes


def inspect_model(model_path):
    """Print each convolution's input/output, weight, group, and MAC information."""
    graph = shape_inference.infer_shapes(onnx.load(str(model_path))).graph
    shapes = get_shape_map(graph)
    weights = {initializer.name: initializer for initializer in graph.initializer}
    total_macs = 0
    print(f"\n{model_path.name}")
    for node in graph.node:
        if node.op_type != "Conv":
            continue
        attributes = {
            attribute.name: helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        weight = weights.get(node.input[1])
        input_shape = shapes.get(node.input[0], "?")
        output_shape = shapes.get(node.output[0], "?")
        weight_shape = list(weight.dims) if weight is not None else "dynamic"
        macs = 0
        if weight is not None and isinstance(output_shape, list) and all(
            isinstance(dimension, int) and dimension > 0
            for dimension in output_shape[2:]
        ):
            macs = prod(output_shape[2:]) * prod(weight_shape)
            total_macs += macs
        print(
            f"Conv {input_shape} → {output_shape}, weights={weight_shape}, "
            f"groups={attributes.get('group', 1)}, MACs={macs:,}"
        )
    print(f"Conv-only MAC total: {total_macs:,}")


def main():
    """Inspect both baseline and optimized graphs after export."""
    for model_name in MODEL_NAMES:
        model_path = OUTPUT_DIR / f"{model_name}.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"Export both models first; missing {model_path}")
        inspect_model(model_path)


if __name__ == "__main__":
    main()
