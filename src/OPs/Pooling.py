import numpy as np
import src.c2oObject as Node
##-----------------------------------------------------Pooling层--------------------------------------------------##
#获取超参数
def getPoolingAttri(layer):
    ##池化核尺寸
    kernel_shape = np.array([layer.pooling_param.kernel_size]*2).reshape(1,-1)[0].tolist()
    if layer.pooling_param.kernel_size == []:
        kernel_shape = [layer.pooling_param.kernel_h,layer.pooling_param.kernel_w]
    ##步长
    strides = [1, 1]#默认为1
    if layer.pooling_param.stride != []:
        strides = np.array([layer.pooling_param.stride]*2).reshape(1,-1)[0].tolist()
    ##填充
    pads = [0, 0, 0, 0]#默认为0
    # 这里与卷积时一样,有pad,就按其值设置
    if layer.pooling_param.pad != []:
        pads = np.array([layer.pooling_param.pad] * 4).reshape(1, -1)[0].tolist()
    elif layer.pooling_param.pad_h != 0 or layer.pooling_param.pad_w != 0:
        pads = [layer.pooling_param.pad_h,layer.pooling_param.pad_w,layer.pooling_param.pad_h,layer.pooling_param.pad_w]

    #超参数字典
    dict = {"kernel_shape":kernel_shape,
            "strides":strides,
            "pads":pads
            }
    return dict
#计算输出维度
def getPoolingOutShape(input_shape,layer,dict):
    kernel_shape = dict["kernel_shape"]
    pads = dict["pads"]
    strides = dict["strides"]

    #计算输出维度,与卷积一样,若为非整数则向上取整
    h = (input_shape[0][2] - kernel_shape[0] + 2 * pads[0])/strides[0] + 1
    if h > int(h):
        output_shape_h = int(h) + 1
        pads = [0,0,1,1]
    else:
        output_shape_h = int(h)
    output_shape = [[input_shape[0][0],input_shape[0][1],output_shape_h,output_shape_h]]

    return output_shape
#构建节点
def createPooling(layer,nodename,inname,outname,input_shape):
    dict = getPoolingAttri(layer)
    output_shape = getPoolingOutShape(input_shape,layer,dict)

    #判断是池化种类,最大池化、平均池化
    if layer.pooling_param.pool == 0:
        node = Node.c2oNode(layer, nodename, "MaxPool", inname, outname, input_shape, output_shape, dict=dict)
    elif layer.pooling_param.pool == 1:
        node = Node.c2oNode(layer, nodename, "AveragePool", inname, outname, input_shape, output_shape, dict=dict)
    #Layers[i].pooling_param.pool==2为随机池化
    print(nodename, "节点构建完成")

    return node