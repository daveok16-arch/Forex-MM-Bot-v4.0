import numpy as np
try:
    import onnx
    from onnx import numpy_helper, TensorProto
    from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info
    
    # Create a simple ONNX model that takes input and outputs fixed values
    input_tensor = make_tensor_value_info('input', TensorProto.FLOAT, [1, 10])
    output_tensor = make_tensor_value_info('output', TensorProto.FLOAT, [1, 3])
    
    # Create weights as constants
    W = numpy_helper.from_array(np.array([[0.3, 0.3, 0.4]], dtype=np.float32).reshape(10, 3), name='W')
    B = numpy_helper.from_array(np.array([0.1, 0.1, 0.1], dtype=np.float32), name='B')
    
    nodes = [
        make_node('MatMul', ['input', 'W'], ['matmul']),
        make_node('Add', ['matmul', 'B'], ['output'])
    ]
    
    graph = make_graph(nodes, 'dummy_model', [input_tensor], [output_tensor], [W, B])
    model = make_model(graph, opset_imports=[onnx.helper.make_opsetid('', 13)])
    
    onnx.save(model, 'data/models/hypernetwork_v4_fixed.onnx')
    print("✅ Dummy ONNX model created")
except ImportError:
    print("⚠️ onnx not available, creating placeholder file")
    open('data/models/hypernetwork_v4_fixed.onnx', 'w').close()
