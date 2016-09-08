import MetalPerformanceShaders
import QuartzCore

/* Helper functions for creating the layers. */

private func makeConv(device: MTLDevice,
                      inDepth: Int,
                      outDepth: Int,
                      weights: UnsafePointer<Float>,
                      bias: UnsafePointer<Float>) -> MPSCNNConvolution {

  // After performing the convolution, the layer applies the ReLU activation
  // function. (Note: we could create a single ReLU instance and reuse it for
  // all layers.)
  let relu = MPSCNNNeuronReLU(device: device, a: 0)

  // All VGGNet conv layers use a 3x3 kernel with stride 1.
  let desc = MPSCNNConvolutionDescriptor(kernelWidth: 3,
                                         kernelHeight: 3,
                                         inputFeatureChannels: inDepth,
                                         outputFeatureChannels: outDepth,
                                         neuronFilter: relu)
  desc.strideInPixelsX = 1
  desc.strideInPixelsY = 1

  let conv = MPSCNNConvolution(device: device,
                               convolutionDescriptor: desc,
                               kernelWeights: weights,
                               biasTerms: bias,
                               flags: MPSCNNConvolutionFlags.none)

  // To preserve the width and height between conv layers, VGGNet assumes one
  // pixel of padding around the edges. Metal apparently has no problem reading
  // outside the source image, so we don't have to do anything special here.
  conv.edgeMode = .zero

  return conv
}

private func makePool(device: MTLDevice) -> MPSCNNPoolingMax {
  // All pooling layers in VGGNet are max pool, 2x2, stride 2. This chops the
  // width and height of the data volume in half but leaves the depth the same.
  return MPSCNNPoolingMax(device: device,
                          kernelWidth: 2,
                          kernelHeight: 2,
                          strideInPixelsX: 2,
                          strideInPixelsY: 2)
}

private func makeFC(device: MTLDevice,
                    inExtent: Int,
                    inDepth: Int,
                    fanOut: Int,
                    weights: UnsafePointer<Float>,
                    bias: UnsafePointer<Float>,
                    withRelu: Bool = true) -> MPSCNNFullyConnected {

  // The last fully-connected layer does not have a ReLU activation.
  // (Instead it uses a softmax but that is not an MPSCNNNeuron subclass
  // so we cannot apply that as a filter to the layer.)
  let filter: MPSCNNNeuron? = withRelu ? MPSCNNNeuronReLU(device: device, a: 0) : nil

  // A fully-connected layer is a special version of a convolutional layer
  // where the kernel size is equal to the width/height of the input volume.
  // The output volume is 1x1xfanOut.
  let desc = MPSCNNConvolutionDescriptor(kernelWidth: inExtent,
                                         kernelHeight: inExtent,
                                         inputFeatureChannels: inDepth,
                                         outputFeatureChannels: fanOut,
                                         neuronFilter: filter)

  let fc = MPSCNNFullyConnected(device: device,
                                convolutionDescriptor: desc,
                                kernelWeights: weights,
                                biasTerms: bias,
                                flags: MPSCNNConvolutionFlags.none)
  return fc
}

/*
  Implements the VGGNet neural network.
  
  Details can be found at http://www.robots.ox.ac.uk/~vgg/research/very_deep/
  and in the paper:
  
    Very Deep Convolutional Networks for Large-Scale Image Recognition
    K. Simonyan, A. Zisserman
    http://arxiv.org/pdf/1409.1556

  This is configuration D from the paper:
  
    - Input image is 224 x 224 pixels x 3 color channels (RGB).
    - All convolution kernels are 3x3.
    - All convolution layers use 1-element zero-padding to preserve the width 
      and height of the input volume.
    - Convolution is followed by a ReLU.
    - All pooling layers are max-pool, size 2, stride 2. These chop the input
      width and height in half but preserve the depth.
    - The fully-connected layers use a ReLU activation, except for the last one
      which applies the softmax function to produce a probability distribution.
    - Layers fc6 and fc7 have dropout regularization applied (ratio 0.5) during
      training, but we simply ignore this during inference.
*/
public class VGGNet {
  let device: MTLDevice
  let commandQueue: MTLCommandQueue

  // The custom compute kernels for preprocessing the input images.
  let pipelineRGB: MTLComputePipelineState
  let pipelineBGR: MTLComputePipelineState

  let outputImage: MPSImage

  // The neural network expects a 224x224 pixel image. We use a lanczos filter
  // to scale the input image down to these dimensions.
  let lanczos: MPSImageLanczosScale

  // After the last layer (fc8), we take the "softmax" of each output neuron.
  // This converts the last layer into a 1000-element vector of probabilities,
  // where each element in this vector corresponds to an ImageNet class label.
  let softmax: MPSCNNSoftMax

  /* The layers in the network: */

  let conv1_1: MPSCNNConvolution  // 224x224x3  input, 64 kernels (3x3x3x64  = 1728  weights + 64 bias)
  let conv1_2: MPSCNNConvolution  // 224x224x64 input, 64 kernels (3x3x64x64 = 36864 weights + 64 bias)
  let pool1  : MPSCNNPoolingMax   // 224x224x64 input -> 112x112x64 output

  let conv2_1: MPSCNNConvolution  // 112x112x64  input, 128 kernels (3x3x64x128  = 73728  weights + 128 bias)
  let conv2_2: MPSCNNConvolution  // 112x112x128 input, 128 kernels (3x3x128x128 = 147456 weights + 128 bias)
  let pool2  : MPSCNNPoolingMax   // 112x112x128 input -> 56x56x128 output

  let conv3_1: MPSCNNConvolution  // 56x56x128 input, 256 kernels (3x3x128x256 = 294912 weights + 256 bias)
  let conv3_2: MPSCNNConvolution  // 56x56x256 input, 256 kernels (3x3x256x256 = 589824 weights + 256 bias)
  let conv3_3: MPSCNNConvolution  // 56x56x256 input, 256 kernels (3x3x256x256 = 589824 weights + 256 bias)
  let pool3  : MPSCNNPoolingMax   // 56x56x256 input -> 28x28x256 output

  let conv4_1: MPSCNNConvolution  // 28x28x256 input, 512 kernels (3x3x256x512 = 1179648 weights + 512 bias)
  let conv4_2: MPSCNNConvolution  // 28x28x512 input, 512 kernels (3x3x512x512 = 2359296 weights + 512 bias)
  let conv4_3: MPSCNNConvolution  // 28x28x512 input, 512 kernels (3x3x512x512 = 2359296 weights + 512 bias)
  let pool4  : MPSCNNPoolingMax   // 28x28x512 input -> 14x14x512 output

  let conv5_1: MPSCNNConvolution  // 14x14x512 input, 512 kernels (3x3x512x512 = 2359296 weights + 512 bias)
  let conv5_2: MPSCNNConvolution  // 14x14x512 input, 512 kernels (3x3x512x512 = 2359296 weights + 512 bias)
  let conv5_3: MPSCNNConvolution  // 14x14x512 input, 512 kernels (3x3x512x512 = 2359296 weights + 512 bias)
  let pool5  : MPSCNNPoolingMax   // 14x14x512 input -> 7x7x512 output

  let fc6: MPSCNNFullyConnected   // 4096 neurons (7x7x512x4096  = 102760448 weights + 4096 bias)
  let fc7: MPSCNNFullyConnected   // 4096 neurons (1x1x4096x4096 = 16777216  weights + 4096 bias)
  let fc8: MPSCNNFullyConnected   // 1000 neurons (1x1x4096x1000 = 4096000   weights + 1000 bias)

  // Total parameters: 138.357.544. (The vast majority of those are in fc6!)
  // We store the weights and bias values as 32-bit floats, making the total
  // filesize 528 MB. Metal copies those values but converts them to 16-bit
  // floats so at runtime the parameters only take about 132 MB memory space.

  /* These MPSImage descriptors tell the network about the sizes of the data
     volumes that flow between the layers. */

  let input_id  = MPSImageDescriptor(channelFormat: .float16, width: 224, height: 224, featureChannels: 3)
  let conv1_id  = MPSImageDescriptor(channelFormat: .float16, width: 224, height: 224, featureChannels: 64)
  let pool1_id  = MPSImageDescriptor(channelFormat: .float16, width: 112, height: 112, featureChannels: 64)
  let conv2_id  = MPSImageDescriptor(channelFormat: .float16, width: 112, height: 112, featureChannels: 128)
  let pool2_id  = MPSImageDescriptor(channelFormat: .float16, width:  56, height:  56, featureChannels: 128)
  let conv3_id  = MPSImageDescriptor(channelFormat: .float16, width:  56, height:  56, featureChannels: 256)
  let pool3_id  = MPSImageDescriptor(channelFormat: .float16, width:  28, height:  28, featureChannels: 256)
  let conv4_id  = MPSImageDescriptor(channelFormat: .float16, width:  28, height:  28, featureChannels: 512)
  let pool4_id  = MPSImageDescriptor(channelFormat: .float16, width:  14, height:  14, featureChannels: 512)
  let conv5_id  = MPSImageDescriptor(channelFormat: .float16, width:  14, height:  14, featureChannels: 512)
  let pool5_id  = MPSImageDescriptor(channelFormat: .float16, width:   7, height:   7, featureChannels: 512)
  let fc_id     = MPSImageDescriptor(channelFormat: .float16, width:   1, height:   1, featureChannels: 4096)
  let output_id = MPSImageDescriptor(channelFormat: .float16, width:   1, height:   1, featureChannels: 1000)

  let labels = VGGNetLabels()

  public init(device: MTLDevice) {
    print("Setting up neural network...")
    let startTime = CACurrentMediaTime()

    self.device = device
    commandQueue = device.makeCommandQueue()

    outputImage = MPSImage(device: device, imageDescriptor: output_id)

    // Before we pass an image into the network, we need to adjust its RGB
    // values. This is done with a custom compute kernel. Here we load that
    // kernel (from Shaders.metal) and set up the compute pipeline.
    do {
      let library = device.newDefaultLibrary()!
      let adjust_mean_rgb = library.makeFunction(name: "adjust_mean_rgb")
      pipelineRGB = try device.makeComputePipelineState(function: adjust_mean_rgb!)

      let adjust_mean_bgr = library.makeFunction(name: "adjust_mean_bgr")
      pipelineBGR = try device.makeComputePipelineState(function: adjust_mean_bgr!)
    } catch {
      fatalError("Error initializing compute pipeline")
    }

    // Uncomment this to test the network with all zero weights.
    //let blob = VGGNetData()

    guard let path = Bundle.main.path(forResource: "parameters", ofType: "data"),
          let blob = VGGNetData(path: path) else {
      fatalError("Error loading network parameters")
    }

    lanczos = MPSImageLanczosScale(device: device)

    conv1_1 = makeConv(device: device, inDepth:   3, outDepth:  64, weights: blob.conv1_1_w, bias: blob.conv1_1_b)
    conv1_2 = makeConv(device: device, inDepth:  64, outDepth:  64, weights: blob.conv1_2_w, bias: blob.conv1_2_b)
    pool1   = makePool(device: device)

    conv2_1 = makeConv(device: device, inDepth:  64, outDepth: 128, weights: blob.conv2_1_w, bias: blob.conv2_1_b)
    conv2_2 = makeConv(device: device, inDepth: 128, outDepth: 128, weights: blob.conv2_2_w, bias: blob.conv2_2_b)
    pool2   = makePool(device: device)

    conv3_1 = makeConv(device: device, inDepth: 128, outDepth: 256, weights: blob.conv3_1_w, bias: blob.conv3_1_b)
    conv3_2 = makeConv(device: device, inDepth: 256, outDepth: 256, weights: blob.conv3_2_w, bias: blob.conv3_2_b)
    conv3_3 = makeConv(device: device, inDepth: 256, outDepth: 256, weights: blob.conv3_3_w, bias: blob.conv3_3_b)
    pool3   = makePool(device: device)

    conv4_1 = makeConv(device: device, inDepth: 256, outDepth: 512, weights: blob.conv4_1_w, bias: blob.conv4_1_b)
    conv4_2 = makeConv(device: device, inDepth: 512, outDepth: 512, weights: blob.conv4_2_w, bias: blob.conv4_2_b)
    conv4_3 = makeConv(device: device, inDepth: 512, outDepth: 512, weights: blob.conv4_3_w, bias: blob.conv4_3_b)
    pool4   = makePool(device: device)

    conv5_1 = makeConv(device: device, inDepth: 512, outDepth: 512, weights: blob.conv5_1_w, bias: blob.conv5_1_b)
    conv5_2 = makeConv(device: device, inDepth: 512, outDepth: 512, weights: blob.conv5_2_w, bias: blob.conv5_2_b)
    conv5_3 = makeConv(device: device, inDepth: 512, outDepth: 512, weights: blob.conv5_3_w, bias: blob.conv5_3_b)
    pool5   = makePool(device: device)

    fc6 = makeFC(device: device, inExtent: 7, inDepth:  512, fanOut: 4096, weights: blob.fc6_w, bias: blob.fc6_b)
    fc7 = makeFC(device: device, inExtent: 1, inDepth: 4096, fanOut: 4096, weights: blob.fc7_w, bias: blob.fc7_b)
    fc8 = makeFC(device: device, inExtent: 1, inDepth: 4096, fanOut: 1000, weights: blob.fc8_w, bias: blob.fc8_b, withRelu: false)

    softmax = MPSCNNSoftMax(device: device)

    let endTime = CACurrentMediaTime()
    print("Elapsed time: \(endTime - startTime) sec")
  }

  /* Performs the inference step. This takes the input image, converts it into
     the format the network expects, then feeds it into the network. The result
     is a 1000-element vector of probabilities. Returns the 5 ImageNet classes
     with the highest predicted probability values. */
  public func predict(image inputImage: MPSImage, bgr: Bool) -> [Prediction] {
    let startTime = CACurrentMediaTime()

    autoreleasepool{
      let commandBuffer = commandQueue.makeCommandBuffer()

      // This lets us squeeze some extra speed out of Metal.
      MPSTemporaryImage.prefetchStorage(with: commandBuffer, imageDescriptorList: [
        input_id, conv1_id, pool1_id, conv2_id, pool2_id, conv3_id, pool3_id,
        conv4_id, pool4_id, conv5_id, pool5_id, fc_id, output_id ])

      // Scale the input image to 224x224 pixels.
      let img1 = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: input_id)
      lanczos.encode(commandBuffer: commandBuffer, sourceTexture: inputImage.texture, destinationTexture: img1.texture)

      let img2 = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: input_id)

      // Adjust the RGB values of each pixel to be in the range -128...127
      // by subtracting the "mean pixel". If the input texture is RGB, this 
      // also swaps the R and B values because the model expects BGR pixels. 
      // As far as I can tell there is no MPS shader that can do these things,
      // so we use a custom compute kernel.
      let encoder = commandBuffer.makeComputeCommandEncoder()
      encoder.setComputePipelineState(bgr ? pipelineBGR : pipelineRGB)
      encoder.setTexture(img1.texture, at: 0)
      encoder.setTexture(img2.texture, at: 1)
      let threadsPerGroups = MTLSizeMake(8, 8, 1)
      let threadGroups = MTLSizeMake(img2.texture.width / threadsPerGroups.width,
                                     img2.texture.height / threadsPerGroups.height, 1)
      encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroups)
      encoder.endEncoding()
      img1.readCount -= 1    // see MPSTemporaryImage docs why this is needed

      // Now we take the output from our custom shader and pass it through the
      // layers of the neural network. For each layer we use a new "temporary"
      // MPSImage to hold the results.

      let conv1_1_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv1_id)
      conv1_1.encode(commandBuffer: commandBuffer, sourceImage: img2, destinationImage: conv1_1_img)

      let conv1_2_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv1_id)
      conv1_2.encode(commandBuffer: commandBuffer, sourceImage: conv1_1_img, destinationImage: conv1_2_img)

      let pool1_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: pool1_id)
      pool1.encode(commandBuffer: commandBuffer, sourceImage: conv1_2_img, destinationImage: pool1_img)

      let conv2_1_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv2_id)
      conv2_1.encode(commandBuffer: commandBuffer, sourceImage: pool1_img, destinationImage: conv2_1_img)

      let conv2_2_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv2_id)
      conv2_2.encode(commandBuffer: commandBuffer, sourceImage: conv2_1_img, destinationImage: conv2_2_img)

      let pool2_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: pool2_id)
      pool2.encode(commandBuffer: commandBuffer, sourceImage: conv2_2_img, destinationImage: pool2_img)

      let conv3_1_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv3_id)
      conv3_1.encode(commandBuffer: commandBuffer, sourceImage: pool2_img, destinationImage: conv3_1_img)

      let conv3_2_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv3_id)
      conv3_2.encode(commandBuffer: commandBuffer, sourceImage: conv3_1_img, destinationImage: conv3_2_img)

      let conv3_3_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv3_id)
      conv3_3.encode(commandBuffer: commandBuffer, sourceImage: conv3_2_img, destinationImage: conv3_3_img)

      let pool3_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: pool3_id)
      pool3.encode(commandBuffer: commandBuffer, sourceImage: conv3_3_img, destinationImage: pool3_img)

      let conv4_1_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv4_id)
      conv4_1.encode(commandBuffer: commandBuffer, sourceImage: pool3_img, destinationImage: conv4_1_img)

      let conv4_2_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv4_id)
      conv4_2.encode(commandBuffer: commandBuffer, sourceImage: conv4_1_img, destinationImage: conv4_2_img)

      let conv4_3_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv4_id)
      conv4_3.encode(commandBuffer: commandBuffer, sourceImage: conv4_2_img, destinationImage: conv4_3_img)

      let pool4_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: pool4_id)
      pool4.encode(commandBuffer: commandBuffer, sourceImage: conv4_3_img, destinationImage: pool4_img)

      let conv5_1_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv5_id)
      conv5_1.encode(commandBuffer: commandBuffer, sourceImage: pool4_img, destinationImage: conv5_1_img)

      let conv5_2_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv5_id)
      conv5_2.encode(commandBuffer: commandBuffer, sourceImage: conv5_1_img, destinationImage: conv5_2_img)

      let conv5_3_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv5_id)
      conv5_3.encode(commandBuffer: commandBuffer, sourceImage: conv5_2_img, destinationImage: conv5_3_img)

      let pool5_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: pool5_id)
      pool5.encode(commandBuffer: commandBuffer, sourceImage: conv5_3_img, destinationImage: pool5_img)

      let fc6_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: fc_id)
      fc6.encode(commandBuffer: commandBuffer, sourceImage: pool5_img, destinationImage: fc6_img)

      let fc7_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: fc_id)
      fc7.encode(commandBuffer: commandBuffer, sourceImage: fc6_img, destinationImage: fc7_img)

      let fc8_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: output_id)
      fc8.encode(commandBuffer: commandBuffer, sourceImage: fc7_img, destinationImage: fc8_img)

      // Finally, apply the softmax function to the output of the last layer.
      // The output image is not an MPSTemporaryImage but a regular MSPImage.
      softmax.encode(commandBuffer: commandBuffer, sourceImage: fc8_img, destinationImage: outputImage)

      // Tell the GPU to start and wait until it's done.
      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
    }

    // Convert the texture from outputImage into something we can use from
    // Swift and then find the ImageNet classes with the highest probability.
    let result = self.labels.top5Labels(prediction: self.outputImage.toFloatArray())

    let endTime = CACurrentMediaTime()
    print("Elapsed time: \(endTime - startTime) sec")

    return result
  }
}
