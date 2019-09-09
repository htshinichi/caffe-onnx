import copy
import numpy as np
from onnx import helper

from . import OPs as op
from .c2oObject import *
from .op_layer_info import *


class Caffe2Onnx():
    def __init__(self,net,model,onnxname):
        #初始化一个c2oGraph对象
        self.onnxmodel = c2oGraph(onnxname)
        #网络和参数
        self.__NetLayer = self.__getNetLayer(net)
        self._ModelLayer = self.__getModelLayer(model)

        #模型的输入名和输入维度
        self.model_input_name = []
        self.model_input_shape = []

        #节点列表
        self.__n = 0
        self.NodeList = []

        #获取层列表
        LayerList = self.__addInputsTVIandGetLayerList(net)
        self.__getNodeList(LayerList)
        self.__addOutputsTVIandValueInfo()

    #获取网络层
    def __getNetLayer(self,net):
        if len(net.layer)==0 and len(net.layers)!=0:
            return net.layers
        elif len(net.layer)!=0 and len(net.layers)==0:
            return net.layer
        else:
            print("prototxt layer error")
            return -1

    #获取参数层
    def __getModelLayer(self,model):
        if len(model.layer) == 0 and len(model.layers) != 0:
            return model.layers
        elif len(model.layer) != 0 and len(model.layers) == 0:
            return model.layer
        else:
            print("caffemodel layer error")
            return -1

    #将模型输入信息添加到Inputs中并获取后续层列表
    def __addInputsTVIandGetLayerList(self,net):
        #如果第一个layer的类型为Input,且没有net.input存在
        if net.input == [] and self.__NetLayer[0].type == "Input":
            layer_list = []
            #考虑到整个网络会有多输入情况
            for lay in self.__NetLayer:
                if lay.type == "Input":
                    in_tvi = helper.make_tensor_value_info(lay.name+"_input", TensorProto.FLOAT, lay.input_param.shape[0].dim)
                    self.model_input_name.append(lay.name+"_input")
                    self.model_input_shape.append(lay.input_param.shape[0].dim)
                    self.onnxmodel.addInputsTVI(in_tvi)
                    print("添加模型输入信息")
                else:
                    layer_list.append(lay)
            return layer_list

        #如果存在net.input
        elif net.input !=[]:
            in_tvi = helper.make_tensor_value_info("input", TensorProto.FLOAT, net.input_dim)
            self.model_input_name.append("input")
            self.model_input_shape.append(net.input_dim)
            self.onnxmodel.addInputsTVI(in_tvi)
            print("添加模型输入信息")
            return self.__NetLayer

        #以上情况都不是,则该caffe模型没有输入,存在问题
        else:
            raise ValueError("the caffe model has no input")


    # 得到layer的参数shape
    def __getParamsShapeandData(self, layer):
        ParamShape = []
        ParamData = []
        #根据这个layer名找出对应的caffemodel中的参数
        for model_layer in self._ModelLayer:
            if layer.name == model_layer.name:
                Params = copy.deepcopy(model_layer.blobs)
                ParamShape = [p.shape.dim for p in Params]
                ParamData = [p.data for p in Params]
                if layer.type == "BatchNorm" or layer.type == "BN": 
                    if len(ParamShape) == 3:  
                        # 如果是bn层，则不用最后一层的滑动系数
                        ParamShape = ParamShape[:-1]
                        ParamData = ParamData[:-1]
                    elif len(ParamShape) == 2 and len(ParamShape[0]) != 1:
                        ParamShape = [[ParamShape[0][1]], [ParamShape[1][1]]]
                        ParamData = ParamData
        return ParamShape, ParamData



    #将参数添加到Inputs中,并生成tensor存储数据
    def __addInputsTVIfromParams(self,layer,ParamName,ParamType):
        #print(layer.type)
        ParamShape = []
        ParamData = []
        #根据这个layer名找出对应的caffemodel中的参数
        for model_layer in self._ModelLayer:
            if layer.name == model_layer.name:
                Params = copy.deepcopy(model_layer.blobs)
                ParamShape = [p.shape.dim for p in Params]
                ParamData = [p.data for p in Params]
                if layer.type == "BatchNorm" or layer.type == "BN":
                    if len(ParamShape) == 3:
                        # 如果是bn层，params为[mean, var, s]，则需要把mean和var除以滑动系数s
                        ParamShape = ParamShape[:-1]
                        ParamData = [[q/(Params[-1].data[0]) for q in p.data] if i==0 else [q/(Params[-1].data[0] + 1e-5) for q in p.data] for i,p in enumerate(Params[:-1])]  # with s
                    elif len(ParamShape) == 2 and len(ParamShape[0]) == 4:
                        ParamShape = [[ParamShape[0][1]], [ParamShape[1][1]]]
                        ParamData = [[q/1. for q in p.data] if i==0 else [q/(1. + 1e-5) for q in p.data] for i,p in enumerate(Params)]

                    # comment it for tvm because tvm use broadcast at prelu layer
                elif layer.type == "PReLU":
                    ParamShape = [[ParamShape[0][0], 1, 1]]
                break
        
        #判断是否有Param
        if ParamShape != []:
            ParamName = ParamName[0:len(ParamShape)]
            ParamType = ParamType[0:len(ParamShape)]
            for i in range(len(ParamShape)):
                #print(ParamName[i])
                ParamName[i] = layer.name+ParamName[i]
                p_tvi = helper.make_tensor_value_info(ParamName[i], ParamType[i], ParamShape[i])
                p_t = helper.make_tensor(ParamName[i],ParamType[i],ParamShape[i],ParamData[i])
                self.onnxmodel.addInputsTVI(p_tvi)
                self.onnxmodel.addInitTensor(p_t)
                print("添加参数" + ParamName[i] + "输入信息和tensor数据")
        if layer.type == "BatchNorm" or layer.type == "BN" or layer.type == "Scale":
            return ParamName, ParamShape
        return ParamName

    #手动将参数添加到输入信息中,并生成tensor存储数据
    def __addInputsTVIfromMannul(self,layer,ParamName,ParamType,ParamShape,ParamData):
        Param_Name = copy.deepcopy(ParamName)
        for i in range(len(ParamShape)):
            Param_Name[i] = layer.name + ParamName[i]
            p_tvi = helper.make_tensor_value_info(Param_Name[i], ParamType[i], ParamShape[i])
            p_t = helper.make_tensor(Param_Name[i], ParamType[i], ParamShape[i], ParamData[i])
            self.onnxmodel.addInputsTVI(p_tvi)
            self.onnxmodel.addInitTensor(p_t)
            print("添加参数"+Param_Name[i]+"输入信息和tensor数据")
        return Param_Name


    #获取上一层的输出名(即当前层的输入)
    def __getLastLayerOutNameAndShape(self,layer):
        outname = []
        outshape = []

        # 如果结点列表为空，或者当前层的bottom在input_name中，那么上一层输入一定是 Input
        if self.NodeList == []:
            outname += self.model_input_name
            outshape += self.model_input_shape

        else:
            for i in range(len(layer.bottom)):
                for j in range(len(self.model_input_name)):
                    if layer.bottom[i] + '_input' == self.model_input_name[j]:
                        outname.append(self.model_input_name[j])
                        outshape.append(self.model_input_shape[j])

                # 因为prototxt中存在top和bottom同名的情况，但是layer.bottom只能对应一个node，所以对每个layer.bottom，找到最末的那个同名节点作为上一层节点
                name = None
                shape = None
                for node in self.NodeList:
                    for j in range(len(node.top) if node.node.op_type != "MaxPool" else 1):   # comment if statement for original maxpool and maxunpool
                        if layer.bottom[i] == node.top[j]:
                            name = node.outputs_name[j]
                            shape = node.outputs_shape[j]
                
                if name:
                    outname.append(name)
                    outshape.append(shape)

        try:
            assert outname, "Failed at layer %s, layer's bottom not detected ..."%(layer.name)
        except:
            print("Failed at layer %s, layer's bottom not detected ..."%(layer.name))

        return outname, outshape

    #获取当前层的输出名，即layername+"_Y"
    def __getCurrentLayerOutName(self,layer):
        # return [layer.name+"_Y"]
        # 考虑有多个输出的情况
        if layer.top == layer.bottom and len(layer.top) == 1:
            return [layer.name+"_Y"]
        
        return [out+"_Y" for out in layer.top]



    def __getNodeList(self,Layers):
        for i in range(len(Layers)):
            # Convolution
            if Layers[i].type == "Convolution" or Layers[i].type == Layer_CONVOLUTION:
                #1.获取节点输入名、输入维度、输出名、节点名
                inname, input_shape = self.__getLastLayerOutNameAndShape(Layers[i])
                outname = self.__getCurrentLayerOutName(Layers[i])
                nodename = Layers[i].name

                #2.生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表
                conv_pname = self.__addInputsTVIfromParams(Layers[i],op_pname["Conv"],op_ptype["Conv"])
                inname.extend(conv_pname)

                #3.构建conv_node
                conv_node = op.createConv(Layers[i],nodename,inname,outname,input_shape)

                #4.添加节点到节点列表
                self.NodeList.append(conv_node)
                self.__n += 1

            #BatchNorm+Scale
            elif Layers[i].type == "BatchNorm" or Layers[i].type == "BN":
                #1.获取节点输入名、输入维度、输出名、节点名
                inname, input_shape = self.__getLastLayerOutNameAndShape(Layers[i])#获取输入名列表和输入形状
                outname = self.__getCurrentLayerOutName(Layers[i]) #获取输出名列表
                nodename = Layers[i].name

                #2.生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表
                if i < len(Layers) - 1 and Layers[i+1].type == "Scale":
                    scale_pname, scale_pshape = self.__addInputsTVIfromParams(Layers[i + 1], op_pname["Scale"], op_ptype["Scale"])
                    bn_pname, bn_pshape = self.__addInputsTVIfromParams(Layers[i], op_pname["BatchNorm"], op_ptype["BatchNorm"])
                    assert bn_pshape == scale_pshape, "BatchNorm and Scale params should share the same shape"
                    inname.extend(scale_pname)
                    inname.extend(bn_pname)

                else:
                    bn_pshape, _ = self.__getParamsShapeandData(Layers[i])
                    custom_params = [np.ones(shape=bn_pshape[0], dtype=np.float), 0.001 + np.zeros(shape=bn_pshape[1], dtype=np.float)]
                    scale_pname = self.__addInputsTVIfromMannul(Layers[i], op_pname["Scale"], op_ptype["Scale"], bn_pshape, custom_params)
                    bn_pname, bn_pshape = self.__addInputsTVIfromParams(Layers[i], op_pname["BatchNorm"], op_ptype["BatchNorm"])
                    inname.extend(scale_pname)
                    inname.extend(bn_pname)


                #3.构建bn_node
                bn_node = op.createBN(Layers[i], nodename, inname, outname, input_shape)

                #4.添加节点到节点列表
                self.NodeList.append(bn_node)
                self.__n += 1

            #Pooling
            elif Layers[i].type == "Pooling" or Layers[i].type == Layer_POOLING:
                #1.获取节点输入名、输入维度、输出名、节点名
                inname,input_shape = self.__getLastLayerOutNameAndShape(Layers[i])#获取输入名列表和输入形状
                outname = self.__getCurrentLayerOutName(Layers[i])#获取输出名列表
                nodename = Layers[i].name

                #2.构建pool_node
                pool_node = op.createPooling(Layers[i], nodename, inname, outname, input_shape)

                #3.添加节点到节点列表
                self.NodeList.append(pool_node)
                self.__n += 1


            # MaxUnPool
            elif Layers[i].type == "MaxUnpool":
                #1.获取节点输入名、输入维度、输出名、节点名
                inname, input_shape = self.__getLastLayerOutNameAndShape(Layers[i])#获取输入名列表和输入形状
                outname = self.__getCurrentLayerOutName(Layers[i])#获取输出名列表
                nodename = Layers[i].name

                #2.构建unpool_node
                unpool_node = op.createUnPooling(Layers[i], nodename, inname, outname, input_shape)

                #3.添加节点到节点列表
                self.NodeList.append(unpool_node)
                self.__n += 1


            #Eltwise
            elif Layers[i].type == "Eltwise" or Layers[i].type == Layer_ELTWISE:
                #1.获取节点输入名、输入维度、输出名、节点名
                inname,input_shape = self.__getLastLayerOutNameAndShape(Layers[i])#获取输入名列表和输入形状
                outname = self.__getCurrentLayerOutName(Layers[i])#获取输出名列表
                nodename = Layers[i].name

                #2.构建eltwise_node
                eltwise_node = op.createEltwise(Layers[i], nodename, inname, outname, input_shape)

                #3.添加节点到节点列表
                self.NodeList.append(eltwise_node)
                self.__n += 1

            #Softmax
            elif Layers[i].type == "Softmax" or Layers[i].type == Layer_SOFTMAX:
                #1.获取节点输入名、输入维度、输出名、节点名
                inname,input_shape = self.__getLastLayerOutNameAndShape(Layers[i])#获取输入名列表和输入形状
                outname = self.__getCurrentLayerOutName(Layers[i])#获取输出名列表
                nodename = Layers[i].name

                #2.构建softmax_node
                softmax_node = op.createSoftmax(Layers[i],nodename, inname, outname, input_shape)

                #3.添加节点到节点列表
                self.NodeList.append(softmax_node)
                self.__n += 1

            #Relu
            elif Layers[i].type == "ReLU" or Layers[i].type == Layer_RELU:
                #1.获取节点输入名、输入维度、输出名、节点名
                inname,input_shape = self.__getLastLayerOutNameAndShape(Layers[i])#获取输入名列表和输入形状
                outname = self.__getCurrentLayerOutName(Layers[i])#获取输出名列表
                nodename = Layers[i].name

                #2.构建relu_node
                relu_node = op.createRelu(Layers[i], nodename, inname, outname, input_shape)

                #3.添加节点到节点列表
                self.NodeList.append(relu_node)
                self.__n += 1

            #LRN
            elif Layers[i].type == "LRN" or Layers[i].type == Layer_LRN:
                #1.获取节点输入名、输入维度、输出名、节点名
                inname,input_shape = self.__getLastLayerOutNameAndShape(Layers[i])
                outname = self.__getCurrentLayerOutName(Layers[i])
                nodename = Layers[i].name

                #2.构建LRN_node
                LRN_node = op.createLRN(Layers[i],nodename, inname, outname, input_shape)

                #3.添加节点到节点列表
                self.NodeList.append(LRN_node)
                self.__n += 1

            #Dropout
            elif Layers[i].type == "Dropout" or Layers[i].type == Layer_DROPOUT:
                #1.获取节点输入名、输入维度、输出名、节点名
                inname,input_shape = self.__getLastLayerOutNameAndShape(Layers[i])
                outname = self.__getCurrentLayerOutName(Layers[i])
                nodename = Layers[i].name

                #2.构建Dropout_node
                Dropout_node = op.createDropout(Layers[i], nodename, inname, outname, input_shape)

                #3.添加节点到节点列表
                self.NodeList.append(Dropout_node)
                self.__n += 1


            #Upsample
            elif Layers[i].type == "Upsample" or Layers[i].type == Layer_UPSAMPLE:
                #1.获取节点输入名、输入维度、输出名、节点名
                inname, input_shape = self.__getLastLayerOutNameAndShape(Layers[i])
                outname = self.__getCurrentLayerOutName(Layers[i])
                nodename = Layers[i].name

                #2.生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表
                paramshape = [[4, 1]]
                paramdata = [[1.0, 1.0, Layers[i].upsample_param.scale, Layers[i].upsample_param.scale]]
                pname = self.__addInputsTVIfromMannul(Layers[i],op_pname["Upsample"],op_ptype["Upsample"],paramshape,paramdata)
                inname.extend(pname)

                #3.构建Upsample_node
                Upsample_node = op.createUpsample(Layers[i], nodename, inname, outname, input_shape)

                #4.添加节点到节点列表
                self.NodeList.append(Upsample_node)
                self.__n += 1

            #Concat
            elif Layers[i].type == "Concat" or Layers[i].type == Layer_CONCAT:
                #1.获取节点输入名、输入维度、输出名、节点名
                inname,input_shape = self.__getLastLayerOutNameAndShape(Layers[i])
                outname = self.__getCurrentLayerOutName(Layers[i])
                nodename = Layers[i].name

                #2.构建Concat_node
                Concat_node = op.createConcat(Layers[i], nodename, inname, outname, input_shape)

                #3.添加节点到节点列表
                self.NodeList.append(Concat_node)
                self.__n += 1

            #PRelu
            elif Layers[i].type == "PReLU":
                #1.获取节点输入名、输入维度、输出名、节点名
                inname,input_shape = self.__getLastLayerOutNameAndShape(Layers[i])
                outname = self.__getCurrentLayerOutName(Layers[i])
                nodename = Layers[i].name

                #2.生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表
                pname = self.__addInputsTVIfromParams(Layers[i], op_pname["PRelu"], op_ptype["PRelu"])
                inname.extend(pname)


                #3.构建PRelu_node
                PRelu_node = op.createPRelu(Layers[i], nodename, inname, outname, input_shape)

                #4.添加节点到节点列表
                self.NodeList.append(PRelu_node)
                self.__n += 1


            # InnerProduct
            # 由于onnx中没有全连接层，因此需要拆分，拆分有两种方法(Reshape+Gemm,Reshape+MatMul+Add)
            elif Layers[i].type == "InnerProduct" or Layers[i].type == Layer_INNER_PRODUCT:
                ####一、reshape
                reshape_layer = copy.deepcopy(Layers[i])  #深拷贝
                #1.获取节点输入名、输入维度、输出名、节点名
                reshape_inname, reshape_input_shape = self.__getLastLayerOutNameAndShape(reshape_layer)  #获取reshape的输入名列表和输入形状
                reshape_outname = [reshape_layer.name + "_Reshape_Y"]
                reshape_nodename = reshape_layer.name+"_Reshape"

                #2.生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表
                paramshape = [[2]]
                paramdata = op.getReshapeOutShape(Layers[i],reshape_input_shape)
                reshape_pname = self.__addInputsTVIfromMannul(reshape_layer,op_pname["Reshape"],op_ptype["Reshape"],paramshape,paramdata)
                reshape_inname.extend(reshape_pname)

                #3.构建reshape_node
                reshape_node = op.createReshape(reshape_layer,reshape_nodename, reshape_inname, reshape_outname, reshape_input_shape)

                #4.添加节点到节点列表
                self.NodeList.append(reshape_node)
                self.__n += 1


                ####二、Gemm
                gemm_layer = copy.deepcopy(Layers[i])#深拷贝
                #1.获取节点输入名、输入维度、输出名、节点名
                gemm_inname = reshape_outname
                gemm_input_shape = self.NodeList[self.__n-1].outputs_shape
                gemm_outname = [gemm_layer.name+"_Gemm_Y"]
                gemm_nodename = gemm_layer.name+"_Gemm"


                #2.生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表
                gemm_pname = self.__addInputsTVIfromParams(gemm_layer,op_pname["InnerProduct"],op_ptype["InnerProduct"])  # 获取输入参数，对于add来说blobs[1]里存放的是bias不需要,所以直接获取blobs[0]
                gemm_inname.extend(gemm_pname)


                #3.构建gemm_node
                matmul_node = op.createGemm(gemm_layer, gemm_nodename, gemm_inname, gemm_outname, gemm_input_shape, gemm_layer.inner_product_param.num_output)

                #4.添加节点到节点列表
                self.NodeList.append(matmul_node)
                self.__n += 1



            # Deconvolution
            elif Layers[i].type == "Deconvolution":
                #1.获取节点输入名、输入维度、输出名、节点名
                inname, input_shape = self.__getLastLayerOutNameAndShape(Layers[i])
                outname = self.__getCurrentLayerOutName(Layers[i])
                nodename = Layers[i].name

                #2.生成节点参数tensor value info,并获取节点参数名,将参数名加入节点输入名列表
                conv_pname = self.__addInputsTVIfromParams(Layers[i], op_pname["ConvTranspose"], op_ptype["ConvTranspose"])
                inname.extend(conv_pname)

                #3.构建conv_node
                conv_node = op.createConvTranspose(Layers[i], nodename, inname, outname, input_shape)
                if self.debug:
                    self.__print_debug_info(nodename, inname, outname, input_shape, conv_node.outputs_shape)

                #4.添加节点到节点列表
                self.NodeList.append(conv_node)
                self.__n += 1



    #判断当前节点是否是输出节点
    def judgeoutput(self,current_node,nodelist):
        for outname in current_node.outputs_name:
            for node in nodelist:
                if outname in node.inputs_name:
                    return False
        return True

    #添加模型输出信息和中间节点信息
    def __addOutputsTVIandValueInfo(self):
        for i in range(len(self.NodeList)):
            if self.judgeoutput(self.NodeList[i],self.NodeList):#构建输出节点信息
                lastnode = self.NodeList[i]
                for j in range(len(lastnode.outputs_shape)):
                    output_tvi = helper.make_tensor_value_info(lastnode.outputs_name[j], TensorProto.FLOAT,lastnode.outputs_shape[j])
                    self.onnxmodel.addOutputsTVI(output_tvi)
            else:#构建中间节点信息
                innernode = self.NodeList[i]
                for k in range(len(innernode.outputs_shape)):
                    hid_out_tvi = helper.make_tensor_value_info(innernode.outputs_name[k], TensorProto.FLOAT,innernode.outputs_shape[k])
                    self.onnxmodel.addValueInfoTVI(hid_out_tvi)
        print("添加模型输出信息和模型中间输出信息")

    #创建模型
    def createOnnxModel(self):
        node_def = [Node.node for Node in self.NodeList]
        graph_def = helper.make_graph(
            node_def,
            self.onnxmodel.name,
            self.onnxmodel.in_tvi,
            self.onnxmodel.out_tvi,
            self.onnxmodel.init_t,
            value_info=self.onnxmodel.hidden_out_tvi
        )
        model_def = helper.make_model(graph_def, producer_name='htshinichi')
        print("2.onnx模型转换完成")
        return model_def

