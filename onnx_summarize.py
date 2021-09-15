import onnx
from onnx import helper, checker
from onnx import TensorProto
import re
import argparse

import json
from google.protobuf.json_format import MessageToJson
from google.protobuf.json_format import Parse
import collections

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input onnx model")
    args = parser.parse_args()
        
    # LOAD MODEL AND PREP MAPS
    model = onnx.load(args.input)
    graph = model.graph
    
    #Generate a name for all node if they have none.
    nodeIdx = 0
    opDict=collections.OrderedDict()
    for n in graph.node:
        if n.op_type not in opDict.keys():
            opDict[n.op_type]=1
        else:
            opDict[n.op_type]=opDict[n.op_type]+1
        if n.op_type == 'Loop':
            print(n.name)
            loop_body = "#"+str(n.attribute[0])
            loop_name = n.name.replace("\\","_")
            loop_name = loop_name.replace("/",'_')
            match, bracket_dict=match_brackets(loop_body)
            if match and bracket_dict:
                last_brace_pos=list(bracket_dict.keys())[-1]
                first_bracket, start_pos= bracket_dict[last_brace_pos]
                loop_body = loop_body[start_pos:last_brace_pos+1]
                loop_body = "graph "+loop_body
                onnxtxt_file = loop_name+'.onnxtxt'
                onnx_file = loop_name+'.onnx'
                print("Writing body for loop onnx operator "+ n.name +" to file " + onnx_file+" .")
                text_file = open(onnxtxt_file, "w")
                n = text_file.write(loop_body)
                text_file.close()
                import os
                os.system('protoc onnx.proto --encode=onnx.ModelProto < '+onnxtxt_file+' > '+onnx_file)
    print(opDict)


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
