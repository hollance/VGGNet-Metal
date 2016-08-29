#include <metal_stdlib>

using namespace metal;

/*
  The input texture has four 16-bit floats per pixel, in the range 0.0...1.0.
  This shader function converts those floats to the range -128...127. 
  
  The values we subtract from the R/G/B components are the mean R/G/B values
  across the set of images that the neural network was trained on.
  
  The alpha component of outColor is not important, since our MPSImages only
  use the first 3 feature channels.

  NOTE: We flip RGB textures to BGR (inColor.x and inColor.z get swapped),
  since the tool that was used to train the network, Caffe, uses images with 
  BGR pixels. Therefore outColor.x is always B and outColor.y is always R.
*/

kernel void adjust_mean_rgb(texture2d<float, access::read> inTexture [[texture(0)]],
                            texture2d<float, access::write> outTexture [[texture(1)]],
                            uint2 gid [[thread_position_in_grid]]) {
  float4 inColor = inTexture.read(gid);
  float4 outColor = float4(inColor.z*255.0 - 103.939, inColor.y*255.0 - 116.779, inColor.x*255.0 - 123.68, 0.0);
  outTexture.write(outColor, gid);
}

kernel void adjust_mean_bgr(texture2d<float, access::read> inTexture [[texture(0)]],
                            texture2d<float, access::write> outTexture [[texture(1)]],
                            uint2 gid [[thread_position_in_grid]]) {
  float4 inColor = inTexture.read(gid);
  float4 outColor = float4(inColor.x*255.0 - 103.939, inColor.y*255.0 - 116.779, inColor.z*255.0 - 123.68, 0.0);
  outTexture.write(outColor, gid);
}
