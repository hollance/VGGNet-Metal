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

kernel void adjust_mean_rgb(texture2d<half, access::read> inTexture [[texture(0)]],
                            texture2d<half, access::write> outTexture [[texture(1)]],
                            uint2 gid [[thread_position_in_grid]]) {
  half4 inColor = inTexture.read(gid);
  half4 outColor = half4(inColor.z*255.0h - 103.939h, inColor.y*255.0h - 116.779h, inColor.x*255.0h - 123.68h, 0.0h);
  outTexture.write(outColor, gid);
}

kernel void adjust_mean_bgr(texture2d<half, access::read> inTexture [[texture(0)]],
                            texture2d<half, access::write> outTexture [[texture(1)]],
                            uint2 gid [[thread_position_in_grid]]) {
  half4 inColor = inTexture.read(gid);
  half4 outColor = half4(inColor.x*255.0h - 103.939h, inColor.y*255.0h - 116.779h, inColor.z*255.0h - 123.68h, 0.0h);
  outTexture.write(outColor, gid);
}
