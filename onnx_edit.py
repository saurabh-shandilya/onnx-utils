# ------------------------------------------------
# ONNX Model Editor and Graph Extractor
# License under The MIT License
# Written by Saurabh Shandilya
# -----------------------------------------------

import onnx
from onnx import helper, checker
from onnx import TensorProto
import re
import argparse

def createGraphMemberMap(graph_member_list):
    member_map=dict();
    for n in graph_member_list:
        member_map[n.name]=n;
    return member_map


def split_io_list(io_list,new_names_all):
    #splits input/output list to identify removed, retained and totally new nodes    
    removed_names=[]
    retained_names=[]
    for n in io_list:
        if n.name not in new_names_all:                
            removed_names.append(n.name)              
        if n.name in new_names_all:                
            retained_names.append(n.name)                      
    new_names=list(set(new_names_all)-set(retained_names)) 
    return [removed_names,retained_names,new_names]
          
def traceDependentNodes(graph,name,node_input_names,node_map, initializer_map):
    # recurisvely traces all dependent nodes for a given output nodes in a graph    
    for n in graph.node:
        for noutput in n.output:       
            if (noutput == name) and (n.name not in node_input_names):
                # give node "name" is node n's output, so add node "n" to node_input_names list 
                node_input_names.append(n.name)
                if n.name in node_map.keys():
                    for ninput in node_map[n.name].input:
                        # trace input node's inputs 
                        node_input_names = traceDependentNodes(graph,ninput,node_input_names,node_map, initializer_map)                                        
    # don't forget the initializers they can be terminal inputs on a path.                    
    if name in initializer_map.keys():
        node_input_names.append(name)                    
    return node_input_names     
    
def onnx_edit(input_model, output_model, new_input_node_names, input_shape_map, new_output_node_names, output_shape_map, verify):
    """ edits and modifies an onnx model to extract a subgraph based on input/output node names and shapes.
    Arguments: 
        input_model: path of input onnx model
        output_model: path of output onnx model    
        new_input_node_names: list of input node names including list of original input nodes if they are to be retained.
            If the list is empty original input nodes are assumed. 
        input_shape_map: dictionary/map of input node names to corresponding shapes. Shapes are needed for model checker to pass.
        new_output_node_names: list of output node names, including list of original output nodes if they are to be retained
            If the list if empty original output nodes are assumed.
        output_shape_map: dictionary/map of output node names to corresponding shape. Shapes are needed for model checker to pass.
        verify: set to true if input and output models need to be verified.
    """
    # LOAD MODEL AND PREP MAPS
    model = onnx.load(input_model)
    graph = model.graph
    if(verify):
        print("input model Errors: ", onnx.checker.check_model(model))
    
    #Generate a name for all node if they have none.
    nodeIdx = 0;
    for n in graph.node:
        if n.name == '':
            n.name = str(n.op_type) + str(nodeIdx)
            nodeIdx += 1
    
    node_map = createGraphMemberMap(graph.node)
    input_map = createGraphMemberMap(graph.input)
    output_map = createGraphMemberMap(graph.output)
    initializer_map = createGraphMemberMap(graph.initializer)
       
    if not new_input_node_names:
        new_input_node_names = list(input_map)
    if not new_output_node_names:
        new_output_node_names = list(output_map)
       
    # MODIFY INPUTS
    # Break the graph based on the new input node names
    [removed_names,retained_names,new_names]=split_io_list(graph.input,new_input_node_names)
    for name in removed_names:
        if name in input_map.keys():
            graph.input.remove(input_map[name])                              
    for name in new_names:
        # If a new input name corresponds to an existing node, it implies that original node in the graph needs to be replaced with an input node
        # Exactly here the graph is broken
        if name in node_map.keys():
            graph.node.remove(node_map[name])
        # Remove node where there output would match new input to avoid duplicate definitions
        nodesToRemoveToAvoidDuplicateEntries = []
        for n in graph.node:
            for noutput in n.output:       
                if (noutput == name):
                    nodesToRemoveToAvoidDuplicateEntries.append(n)
        for n in nodesToRemoveToAvoidDuplicateEntries:
            graph.node.remove(n)
        if(name in input_shape_map.keys()):
            new_nv = helper.make_tensor_value_info(name, TensorProto.FLOAT, input_shape_map[name])
        else:
            new_nv = helper.make_tensor_value_info(name, TensorProto.FLOAT, None)    
        graph.input.extend([new_nv])
    node_map = createGraphMemberMap(graph.node)
    input_map = createGraphMemberMap(graph.input)    

    # MODIFY OUTPUTS
    # Break the graph based on the new output node names   
    [removed_names,retained_names,new_names]=split_io_list(graph.output,new_output_node_names)
    for name in removed_names:
        if name in output_map.keys():
            graph.output.remove(output_map[name])                              
    for name in new_names:
        if(name in output_shape_map.keys()):
            new_nv = helper.make_tensor_value_info(name, TensorProto.FLOAT, output_shape_map[name])
        else:
            new_nv = helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
        graph.output.extend([new_nv])
    output_map = createGraphMemberMap(graph.output)      

    # CLEANUP NODES
    # Trace all dependent nodes for the current set of output nodes defined & prepare a list of invalid nodes
    valid_node_names=[]
    for new_output_node_name in new_output_node_names:
        valid_node_names=traceDependentNodes(graph,new_output_node_name,valid_node_names,node_map, initializer_map)
        valid_node_names=list(set(valid_node_names))
    invalid_node_names = list( (set(node_map.keys()) | set(initializer_map.keys())) - set(valid_node_names))
    # Remove all the invalid nodes from the graph               
    for name in invalid_node_names:
        if name in node_map.keys():
            graph.node.remove(node_map[name])        
        if name in initializer_map.keys():
            graph.initializer.remove(initializer_map[name])
        if name in input_map.keys():
            graph.input.remove(input_map[name])    

    # SAVE MODEL
    if(verify):    
        print("output model Errors: ", onnx.checker.check_model(model))
    onnx.save(model, output_model)

def parse_nodename_and_shape(name):
    # parses node names and shapes from input argument string
    inputs = []
    shapes = {}
    # input takes in most cases the format name:0, where 0 is the output number, and shapes
    # are appended to the same e.g. name:0[1,28,28,3]
    name_pattern = r"(?:([\w\d/\-\._:]+)(\[[\-\d,]+\])?),?"
    
    splits = re.split(name_pattern, name)
    for i in range(1, len(splits), 3):        
        inputs.append(splits[i])
        if splits[i + 1] is not None:
            shapes[splits[i]] = [int(n) for n in splits[i + 1][1:-1].split(",")]
    if not shapes:
        shapes = None
    return inputs, shapes    
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input onnx model")
    parser.add_argument("output", help="output onnx model")
    parser.add_argument("--inputs", help="comma separated model input names appended with shapes, e.g. --inputs <nodename>[1,2,3],<nodename1>[1,2,3] ")
    parser.add_argument("--outputs", help="comma separated model output names appended with shapes, e.g. --outputs <nodename>[1,2,3],<nodename1>[1,2,3] ")    
    parser.add_argument('--skipverify', dest='skipverify', action='store_true',
                    help='skip verification of model. Useful if shapes are not known')
    args = parser.parse_args()
        
    if args.inputs:
        new_input_node_names, input_shape_map = parse_nodename_and_shape(args.inputs)
        #print(new_input_node_names)
        #print(input_shape_map)
    else: 
        new_input_node_names = []
        input_shape_map = {}
        
    if args.outputs:
        new_output_node_names, output_shape_map = parse_nodename_and_shape(args.outputs)
        #print(new_output_node_names)
        #print(output_shape_map)
    else:
        new_output_node_names = []
        output_shape_map = {}
        
    onnx_edit(args.input,args.output,new_input_node_names, input_shape_map, new_output_node_names, output_shape_map, not args.skipverify)
    
        
        