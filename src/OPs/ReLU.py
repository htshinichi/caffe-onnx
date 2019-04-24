import src.c2oObject as Node
##--------------------------------------------------relu层------------------------------------------------------------##
#获取超参数
def getReluAttri(layer):
    dict = {}
    if layer.relu_param.negative_slope != 0:
        dict = {"alpha":layer.relu_param.negative_slope}
    return dict
#计算输出维度
def getReluOutShape(input_shape):
    #获取output_shape
    output_shape = input_shape
    return output_shape
#构建节点
def createRelu(layer,nodename,inname,outname,input_shape):
    dict = getReluAttri(layer)
    output_shape = getReluOutShape(input_shape)

    if dict == {}:
        node = Node.c2oNode(layer, nodename, "Relu", inname, outname, input_shape, output_shape)
    else:
        node = Node.c2oNode(layer, nodename, "LeakyRelu", inname, outname, input_shape, output_shape, dict=dict)

    print(nodename, "节点构建完成")
    return node