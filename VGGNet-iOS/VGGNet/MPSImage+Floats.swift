import Accelerate
import MetalPerformanceShaders

extension MPSImage {
  /* We receive the predicted output as an MPSImage. For VGGNet, this is a 1x1
     image with 1000 channels, each of which contains one float16 value.

     However, this is not a nice 1000-element array in memory. Because Metal is
     a graphics API, MPSImage stores the data in MTLTexture objects. Each pixel
     from the texture stores 4 channels: R contains the first channel, G is the
     second channel, B is the third, A is the first.

     This function converts these float16s from the Metal texture to a regular
     Swift array of 1000 Float values so we can use it from Swift. */
  public func toFloatArray() -> [Float] {
    assert(self.pixelFormat == .rgba16Float)

    let count = self.width * self.height * self.featureChannels
    var outputFloat16 = [UInt16](repeating: 0, count: count)

    let region = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                           size: MTLSize(width: self.width, height: self.height, depth: 1))

    // Copy the texture data into the outputFloat16 array.
    let numSlices = (self.featureChannels + 3)/4
    for i in 0..<numSlices {
      self.texture.getBytes(&(outputFloat16[self.width * self.height * 4 * i]),
                            bytesPerRow: self.width * 4 * MemoryLayout<UInt16>.size,
                            bytesPerImage: 0,
                            from: region,
                            mipmapLevel: 0,
                            slice: i)
    }

    // Use vImage to convert the float16 values to regular Swift Floats.
    var outputFloat32 = [Float](repeating: 0, count: count)
    var bufferFloat16 = vImage_Buffer(data: &outputFloat16, height: 1, width: UInt(count), rowBytes: count * 2)
    var bufferFloat32 = vImage_Buffer(data: &outputFloat32, height: 1, width: UInt(count), rowBytes: count * 4)

    if vImageConvert_Planar16FtoPlanarF(&bufferFloat16, &bufferFloat32, 0) != kvImageNoError {
      print("Error converting float16 to float32")
    }

    return outputFloat32
  }
}
