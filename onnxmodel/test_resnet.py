import onnxruntime
import numpy as np
import PIL.Image as Image
import argparse


def process_image(img_path,input_shape):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(input_shape)
    image = np.array(img, dtype=np.float32)
    image = image.transpose((2,0,1))[np.newaxis, ...]
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_shape', help="caffe's caffemodel file path", nargs='+', default=(224,224))
    parser.add_argument('--img_path', help="test image path", type=str, default="./onnxmodel/airplane.jpg")
    parser.add_argument('--onnx_path', help="onnx model file path", type=str, default="./onnxmodel/resnet50.onnx")
    args = parser.parse_args()


    input_shape = [int(x) for x in args.input_shape] #模型输入尺寸
    img_path = args.img_path
    onnx_path = args.onnx_path
    print("image path:",img_path)
    print("onnx model path:",onnx_path)

    data_input = process_image(img_path,input_shape)
    session = onnxruntime.InferenceSession(onnx_path)
    inname = [input.name for input in session.get_inputs()]
    outname = [output.name for output in session.get_outputs()]

    print("inputs name:",inname,"|| outputs name:",outname)
    data_output = session.run(outname, {inname[0]: data_input})

    output = data_output[0]
    print("Label predict: ", output.argmax())




if __name__ == '__main__':
    main()
