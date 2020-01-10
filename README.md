# caffe-onnx
This tool converts caffe model convert to onnx model  
only use for inference

## Introduction  
This is the second version of converting caffe model to onnx model. In this version, all the parameters will be transformed to tensor and tensor value info when reading `.caffemodel` file and each operator node is constructed directly into the type of NodeProto in onnx.


## Dependencies  
- protobuf  
- onnx==1.4.0    

```bash
$ pip install -r requirements.txt
```

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

Take ResNet-50 as an example, you can follow the instructions.

1. Download resnet50 `.caffemodel` file from BaiduDisk and put `resnet-50-model.caffemodel` to `./caffemodel/resnet-50/`  
    Link：https://pan.baidu.com/s/10YB42muAd0vGiNTCetvLsA  
    Code：7az4 

2. Convert resnet50 caffe model to onnx model
    ```bash
    $ python convert2onnx.py \
              caffemodel/resnet-50/resnet-50-model.prototxt \
              caffemodel/resnet-50/resnet-50-model.caffemodel \
              resnet50 onnxmodel
    ```

3. Visualize onnx model by netron
    ```bash
    $ netron onnxmodel/resnet50.onnx --host 0.0.0.0 --port 8008
    ```

4. Run test scripts
    ```bash
    $ python onnxmodel/test_resnet.py \
              --input_shape 224 224 \
              --img_path onnxmodel/airplane.jpg \
              --onnx_path onnxmodel/resnet50.onnx

    # you will get result 404 which is the class id of airplane in IMAGENET.
    ```

5. If you have custom layers in caffe which makes your `caffe.proto` is different than the one in the origin caffe code. The things you should do before convertion is:  
    - First of all, compile your proto file with `protoc`
        ```bash
        # for example
        $ protoc /your/path/to/caffe_ssd.proto --python_out ./proto
        ```

    - Then specify the caffe proto file by replacing the line `from proto import caffe_upsample_pb2 as caffe_pb2` with your module.
      


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



