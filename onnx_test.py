import onnx

model = onnx.load("fusion_custom_cnn.onnx")
for i, node in enumerate(model.graph.node):
    print(f"  [{i:3d}] op={node.op_type:<20s} name={node.name}")