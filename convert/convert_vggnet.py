# Python script to convert the VGGNet weights to Metal CNN format.
#
# The VGGNet Caffe model stores the weights for each layer in this shape:
#    (outputChannels, inputChannels, kernelHeight, kernelWidth)
#
# The Metal API expects weights in the following shape:
#    (outputChannels, kernelHeight, kernelWidth, inputChannels)
#
# This script reads the VGGNet .caffemodel file, transposes the weight arrays,
# and writes out a big file called parameters.data that just contains the raw
# weights and bias values as 32-bit floats.
#
# Requirements:
# - numpy
# - google.protobuf
#
# Usage:
# - first download the prototxt file from:
#   https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/0067c9b32f60362c74f4c445a080beed06b07eb3/VGG_ILSVRC_16_layers_deploy.prototxt
# - also download the caffemodel file from:
#   http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
# - then run the script:
#   python3 convert_vggnet.py VGG_ILSVRC_16_layers_deploy.prototxt VGG_ILSVRC_16_layers.caffemodel ./output-folder
#
# The code is based on https://github.com/woodrush/neural-art-tf, which in turn
# is based on https://github.com/ethereon/caffe-tensorflow. Licensed under the
# MIT license: https://github.com/ethereon/caffe-tensorflow/blob/master/LICENSE.md
#
# Changes I made:
# - converted the source code to Python 3.5
# - used "protoc caffe.proto --python_out=." to convert the latest caffe.proto
#   file to Python 3 format
# - don't save as pickled numpy file but as raw bytes
# - got rid of the caffe stuff because I don't have it installed anyway
# - generates Swift code that should be placed in VGGNetData.swift

import os
import sys
import numpy as np

class CaffeDataReader(object):
    def __init__(self, def_path, data_path):
        self.def_path = def_path
        self.data_path = data_path
        self.load_using_pb()

    def load_using_pb(self):
        import caffe_pb2
        data = caffe_pb2.NetParameter()
        print("Loading the caffemodel. This takes a couple of minutes.")
        data.MergeFromString(open(self.data_path, 'rb').read())
        print("Done reading")
        pair = lambda layer: (layer.name, self.transform_data(layer))
        layers = data.layers or data.layer
        self.parameters = [pair(layer) for layer in layers if layer.blobs]
        print("Done transforming")

    def transform_data(self, layer):
        print("Transforming layer %s" % layer.name)
        transformed = []
        for idx, blob in enumerate(layer.blobs):
            c_o  = blob.num
            c_i  = blob.channels
            h    = blob.height
            w    = blob.width
            print("  %d: %d x %d x %d x %d" % (idx, c_o, c_i, h, w))

            arr = np.array(blob.data, dtype=np.float32)
            #print(arr.shape)

            # The fc6 layer is the first fully-connected layer. It has shape
            # (1, 1, 4096, 25088). We reshape it so that it gets transposed
            # correctly in the convert() function.
            if layer.name == "fc6" and idx == 0:
                data = arr.reshape(4096, 512, 7, 7)
            elif layer.name == "fc7" and idx == 0:
                data = arr.reshape(4096, 4096, 1, 1)
            elif layer.name == "fc8" and idx == 0:
                data = arr.reshape(1000, 4096, 1, 1)
            else:
                data = arr.reshape(c_o, c_i, h, w)

            transformed.append(data)
            #print(data.shape)

        print()
        return tuple(transformed)

    def dump(self, dst_path):
        params = []
        def convert(data):
            if data.ndim == 4:
                # (c_o, c_i, h, w) -> (c_o, h, w, c_i)
                data = data.transpose((0, 2, 3, 1))
            else:
                print("Unsupported layer:", data.shape)
            return data

        offset = 0
        s = ""
        all = np.array([], dtype=np.float32)
        for key, data_pair in self.parameters:
            print(key)
            ext = ["w", "b"]
            for i, data in enumerate(map(convert, data_pair)):
                s += ("  var %s_%s: UnsafeMutablePointer<Float> { return ptr + %d }\n" % (key, ext[i], offset))
                print("  ", data.shape)
                offset += data.size
                all = np.append(all, data.ravel())

                # Save the individual files.
                #g = open(dst_path + "/" + key + "-" + ext[i] + ".data", "wb")
                #data.tofile(g)
                #g.close()

        assert(all.shape[0] == 138357544)

        f = open(dst_path + "/parameters.data", "wb")
        all.tofile(f)
        f.close()

        print("\nCopy this code into VGGNetData.swift:")
        print(s)
        print("Done!")

def main():
    args = sys.argv[1:]
    if len(args) != 3:
        print("usage: %s path.prototxt path.caffemodel output-folder" % os.path.basename(__file__))
        exit(-1)
    def_path, data_path, dst_path = args
    CaffeDataReader(def_path, data_path).dump(dst_path)

if __name__ == '__main__':
    main()
