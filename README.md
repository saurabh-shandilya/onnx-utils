# onnx-utils

Utility scripts for editing or modifying onnx models. The script edits and modifies an onnx model to extract a subgraph based on input/output node names and shapes.

usage: onnx_edit.py [-h] [--inputs INPUTS] [--outputs OUTPUTS] [--skipverify]
                    input output

positional arguments:

  input              input onnx model  
  output             output onnx model
  

optional arguments:

  -h, --help         show this help message and exit
  
  --inputs INPUTS    comma separated model input names appended with shapes,
                     e.g. --inputs <nodename>[1,2,3],<nodename1>[1,2,3]
  
  --outputs OUTPUTS  comma separated model output names appended with shapes,
                     e.g. --outputs <nodename>[1,2,3],<nodename1>[1,2,3]
 
 --skipverify       skip verification of model. Useful if shapes are not
                     known
