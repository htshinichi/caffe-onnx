import src.c2oObject as Node
##----------------------------------------------------Dropout层-------------------------------------------------------##
#获取超参数
def getDropoutAttri(layer):
    ##drop 比率
    ratio = layer.dropout_param.dropout_ratio
    # 超参数字典
    dict = {"ratio":ratio}
    return dict
def getDropoutOutShape(input_shape):
    # 计算输出维度output_shape
    output_shape = input_shape  # 与输入维度一样
    return output_shape
#构建节点
def createDropout(layer, nodename, inname, outname, input_shape):
    dict = getDropoutAttri(layer)
    output_shape = getDropoutOutShape(input_shape)
    # 构建node
    node = Node.c2oNode(layer, nodename, "Dropout", inname, outname, input_shape, output_shape, dict=dict)
    print(nodename, "节点构建完成")
    return node