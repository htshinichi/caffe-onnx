# caffe-onnx
caffe model convert to onnx model  

## 介绍 Introduction  
前面写过一版caffe2onnx的工具，不过由于结构不合理以及对参数数据有很多多余的操作，会导致如vgg16这种参数量较大的模型难以转换，因此重写了一版，在读取参数的同时就将其转为tensor和tensor value info，并且在每个算子节点构建的时候直接转为onnx中NodeProto的类型。
## 依赖库 Dependencies  
- protobuf  
- onnx    
仅进行模型转换，不需要配置caffe环境
## 使用 Use  
```
usage: convert2onnx.py [-h] [CNP] [CMP] [ON] [OSP]

positional arguments:
  CNP         caffe's prototxt file path
  CMP         caffe's caffemodel file path
  ON          onnx model name
  OSP         onnx model file saved path

```  
默认值：  
CNP default="./caffemodel/test/test.prototxt"  
CMP default="./caffemodel/test/test.caffemodel"  
ON default="test"  
OSP default="./onnxmodel/"  
## 现在支持的算子 Current Support Operator  
BatchNorm  
Convolution  
Concat  
Dropout  
InnerProduct(Reshape+Gemm)  
LRN  
Pooling  
ReLU  
Softmax  
Eltwise  
Upsample  
Scale  


## 测试转换用模型 Test Caffe Model  
- Resnet50  
- AlexNet  
- Agenet  
- Yolo V3  
- vgg16  


## 可视化 Visualization  
可以使用神器netron https://github.com/lutzroeder/netron  
[netron网页版](https://lutzroeder.github.io/netron/)



