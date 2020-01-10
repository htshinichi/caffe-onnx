import os
import importlib


def import_caffe_pb2(caffe_proto_name):
    caffe_pb2 = importlib.import_module("proto.%s_pb2"%caffe_proto_name)
    return caffe_pb2
