import os
import argparse
from src.load_save_model import loadcaffemodel, saveonnxmodel
from src.caffe2onnx import Caffe2Onnx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('caffe_graph_path',help="caffe's prototxt file path",nargs='?',default="./caffemodel/test/test.prototxt")
    parser.add_argument('caffe_params_path',help="caffe's caffemodel file path",nargs='?',default="./caffemodel/test/test.caffemodel")
    parser.add_argument('onnx_name',help="onnx model name",nargs='?',default="test")
    parser.add_argument('save_dir',help="onnx model file saved path",nargs='?',default="./onnxmodel/")
    args = parser.parse_args()
    return args


def main(args):
    caffe_graph_path = args.caffe_graph_path
    caffe_params_path = args.caffe_params_path
    onnx_name = args.onnx_name
    save_dir = args.save_dir
    save_path = os.path.join(save_dir, onnx_name+'.onnx')
    os.makedirs(save_dir, exist_ok=True)

    graph, params = loadcaffemodel(caffe_graph_path,caffe_params_path)
    # import ipdb; ipdb.set_trace()
    c2o = Caffe2Onnx(graph, params, onnx_name)
    onnxmodel = c2o.createOnnxModel()
    saveonnxmodel(onnxmodel, save_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
