import onnx
from onnx import helper, checker
from onnx import TensorProto
import re
import argparse

#import json
#from google.protobuf.json_format import MessageToJson
#from google.protobuf.json_format import Parse
import collections

def printDict(d, key_string, val_string):
    print('{}\n{} {}\n{}'.format('*'*10,key_string,val_string, '-'*10))
    for key, val in d.items():
        print('{} {}'.format(key, val))
    print('{}'.format('*'*10))


# Matches a pair of brackets which are not part of comments or not part of strings delimited by ""
# Returns
def match_brackets(str):
    pairs = {'(': ')',
             '{': '}',
             '[': ']'}
    ignore_delims={'"':'"',
                  '#':'\n'}
    q = []
    # { end_pos: (stat_char, start_pos)}
    pos_pair = collections.OrderedDict()
    pos=0
    ignore_char=None
    for c in str:
        if c in ignore_delims.keys() and ignore_char==None:
            ignore_char=c
        elif ignore_char:
            if c == ignore_delims[ignore_char]:
                ignore_char=None
        elif c in pairs.keys():
            entry = (c,pos)
            q.append(entry)
        elif c in pairs.values():
            if not q:
                return (False, None)
            entry = q.pop()
            if c != pairs[entry[0]]:
                print(str[:pos])
                return (False, None)
            pos_pair[pos]=entry
        pos=pos+1
    return (not q, pos_pair)

def analyze_onnx(model_file):
    model = onnx.load(model_file)
    graph = model.graph
    # Generate a name for all node if they have none.
    nodeIdx = 0
    opDict = collections.OrderedDict()
    for n in graph.node:
        if n.op_type not in opDict.keys():
            opDict[n.op_type] = 1
        else:
            opDict[n.op_type] = opDict[n.op_type] + 1
        if n.op_type == 'Loop':
            loop_body = "#" + str(n.attribute[0])
            loop_name = n.name.replace("\\", "_")
            loop_name = loop_name.replace("/", '_')
            match, bracket_dict = match_brackets(loop_body)
            if match and bracket_dict:
                last_brace_pos = list(bracket_dict.keys())[-1]
                first_bracket, start_pos = bracket_dict[last_brace_pos]
                loop_body = loop_body[start_pos:last_brace_pos + 1]
                loop_body = "graph " + loop_body
                onnxtxt_file = loop_name + '.onnxtxt'
                onnx_file = loop_name + '.onnx'
                print("Writing body for loop onnx operator " + n.name + " to file " + onnx_file + " .\n")
                text_file = open(onnxtxt_file, "w")
                n = text_file.write(loop_body)
                text_file.close()
                import os
                os.system('protoc onnx.proto --encode=onnx.ModelProto < ' + onnxtxt_file + ' > ' + onnx_file)
                analyze_onnx(onnx_file)
    print(model_file)
    printDict(opDict, 'op', 'count')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a summary of operators in the input onnx file and also dumps'
                                                 ' each loop body as an onnx file and also creates summary for each '
                                                 ' such loop body.')
    parser.add_argument("input", help="input onnx model")
    args = parser.parse_args()
        
    analyze_onnx(args.input)

'''    
    text_file = open("log.txt", "r")
    #read whole file to a string
    data = text_file.read() 
    #close file
    text_file.close()
    convert_model = Parse(data, onnx.ModelProto())
    
    print(convert_model)
    
    s = MessageToJson(onnx_model)
'''
