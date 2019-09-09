import numpy as np
import src.c2oObject as Node
##---------------------------------------------------Conv层-------------------------------------------------------##
#获取超参数
def getConvAttri(layer):
    ##膨胀系数dilations
    dilations = [1, 1]
    if layer.convolution_param.dilation != []:
        dilation = layer.convolution_param.dilation[0]
        dilations = [dilation, dilation]
    ##填充pads
    pads = [0, 0, 0, 0]  # 默认为0
    if layer.convolution_param.pad != []:  # 若存在pad,则根据pad赋值
        pads = np.array([layer.convolution_param.pad] * 4).flatten().tolist()
    elif layer.convolution_param.pad_h != 0 or layer.convolution_param.pad_w != 0:  # 若存在pad_w,pad_h则根据其赋值
        pads = [layer.convolution_param.pad_h, layer.convolution_param.pad_w, layer.convolution_param.pad_h,
                layer.convolution_param.pad_w]
    ##步长strides
    strides = [1, 1]  # 默认为1
    if layer.convolution_param.stride != []:
        strides = np.array([layer.convolution_param.stride] * 2).flatten().tolist()
    ##卷积核尺寸kernel_shape
    kernel_shape = np.array([layer.convolution_param.kernel_size] * 2).flatten().tolist()
    if layer.convolution_param.kernel_size == []:
        kernel_shape = [layer.convolution_param.kernel_h, layer.convolution_param.kernel_w]
    ##分组group
    group = layer.convolution_param.group


    # 超参数字典
    dict = {  # "auto_pad":"NOTSET",
        "dilations": dilations,
        "group": group,
        "kernel_shape": kernel_shape,
        "pads": pads,
        "strides": strides
    }
    return dict
#计算输出维度
def getConvOutShape(input_shape,layer,dict):
    dilations = dict["dilations"]
    kernel_shape = dict["kernel_shape"]
    pads = dict["pads"]
    strides = dict["strides"]
    ##卷积核数量kernel_num
    kernel_num = layer.convolution_param.num_output

    #计算输入维度output_shape
    h = (input_shape[0][2] - kernel_shape[0] + pads[0] + pads[2] - (kernel_shape[0]-1)*(dilations[0]-1))/strides[0] + 1 # 输出维度N= ((输入维度I - 卷积核维度K + 2 * 填充P - (卷积核维度-1)*(膨胀系数-1))/步长S) + 1
    #当h非整数 ,且未设置pad ,在遇到输出为非整数情况 ,向上取整 ,即在右边和下边补1
    if h > int(h) and layer.convolution_param.pad == []:
        output_shape_h = int(h) + 1
        pads = [0,0,1,1]
    else:
        output_shape_h = int(h)
    
    w = (input_shape[0][3] - kernel_shape[1] + pads[1] + pads[3] - (kernel_shape[1]-1)*(dilations[1]-1))/strides[1] + 1 # 输出维度N= ((输入维度I - 卷积核维度K + 2 * 填充P - (卷积核维度-1)*(膨胀系数-1))/步长S) + 1
    #当h非整数 ,且未设置pad ,在遇到输出为非整数情况 ,向上取整 ,即在右边和下边补1
    if w > int(w) and layer.convolution_param.pad == []:
        output_shape_w = int(w) + 1
        pads = [0,0,1,1]
    else:
        output_shape_w = int(w)

    output_shape = [[input_shape[0][0],kernel_num,output_shape_h,output_shape_w]]

    return output_shape
#构建节点
def createConv(layer, nodename, inname, outname, input_shape):
    dict = getConvAttri(layer)
    output_shape = getConvOutShape(input_shape, layer, dict)
    #构建node
    node = Node.c2oNode(layer, nodename, "Conv", inname, outname, input_shape, output_shape, dict)
    print(nodename, "节点构建完成")
    return node

