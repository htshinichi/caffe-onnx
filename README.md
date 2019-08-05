# caffe-onnx
This tool converts caffe model convert to onnx model  
only use for inference

## Introduction  
This is the second version of converting caffe model to onnx model. In this version, all the parameters will be transformed to tensor and tensor value info when reading `.caffemodel` file and each operator node is constructed directly into the type of NodeProto in onnx.


## Dependencies  
- protobuf  
- onnx==1.4.0    

( caffe environment is not required! )

## How to Use  
```
usage: convert2onnx.py [-h] [caffe_graph_path] [caffe_params_path] [onnx_name] [save_dir]

positional arguments:
  caffe_graph_path          caffe's prototxt file path
  caffe_params_path         caffe's caffemodel file path
  onnx_name                 onnx model name
  save_dir                  onnx model file saved path
```  


## Current Support Operator  
BatchNorm  
Convolution  
Deconvolution  
Concat  
Dropout  
InnerProduct(Reshape+Gemm)  
LRN  
Pooling  
Unpooling  
ReLU  
Softmax  
Eltwise  
Upsample  
Scale  


## Test Caffe Model  
- Resnet50  
- AlexNet  
- Agenet  
- Yolo V3  
- vgg16  


## Visualization  
netron is recommended: https://github.com/lutzroeder/netron  
[netron Browser](https://lutzroeder.github.io/netron/)



