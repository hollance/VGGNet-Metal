import UIKit
import AVFoundation
import CoreVideo
import Metal

public protocol VideoCaptureDelegate: class {
  func didCapture(texture: MTLTexture?, previewImage: UIImage?)
}

public class VideoCapture: NSObject, AVCapturePhotoCaptureDelegate {

  public var previewLayer: AVCaptureVideoPreviewLayer?
  public weak var delegate: VideoCaptureDelegate?

  var device: MTLDevice!
  var captureSession: AVCaptureSession!
	var photoOutput: AVCapturePhotoOutput!
  var textureCache: CVMetalTextureCache?

  public init(device: MTLDevice) {
    self.device = device
    super.init()
    setUp()
  }

  private func setUp() {
    guard CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache) == kCVReturnSuccess else {
      print("Error: Could not create a texture cache")
      return
    }

    captureSession = AVCaptureSession()
    captureSession.beginConfiguration()
    captureSession.sessionPreset = AVCaptureSessionPresetMedium

    guard let videoDevice = AVCaptureDevice.defaultDevice(withMediaType: AVMediaTypeVideo) else {
      print("Error: no video devices available")
      return
    }

    guard let videoInput = try? AVCaptureDeviceInput(device: videoDevice) else {
      print("Error: could not create AVCaptureDeviceInput")
      return
    }

    if captureSession.canAddInput(videoInput) {
      captureSession.addInput(videoInput)
    }

    if let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession) {
      previewLayer.videoGravity = AVLayerVideoGravityResizeAspect
      previewLayer.connection?.videoOrientation = .landscapeRight
      self.previewLayer = previewLayer
    }

    photoOutput = AVCapturePhotoOutput()
    if captureSession.canAddOutput(photoOutput) {
      captureSession.addOutput(photoOutput)
    }

    captureSession.commitConfiguration()
  }

  public func start() {
    captureSession.startRunning()
  }

  /* Captures a single frame of the camera input. */
  public func captureFrame() {
    let settings = AVCapturePhotoSettings(format: [
      kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: kCVPixelFormatType_32BGRA)
    ])

    settings.previewPhotoFormat = [
      kCVPixelBufferPixelFormatTypeKey as String: settings.availablePreviewPhotoPixelFormatTypes[0],
      kCVPixelBufferWidthKey as String: 480,
      kCVPixelBufferHeightKey as String: 360,
    ]

    photoOutput?.capturePhoto(with: settings, delegate: self)
  }

  public func capture(_ captureOutput: AVCapturePhotoOutput,
                      didFinishProcessingPhotoSampleBuffer photoSampleBuffer: CMSampleBuffer?,
                      previewPhotoSampleBuffer: CMSampleBuffer?,
                      resolvedSettings: AVCaptureResolvedPhotoSettings,
                      bracketSettings: AVCaptureBracketedStillImageSettings?,
                      error: Error?) {

    var imageTexture: MTLTexture?
    var previewImage: UIImage?

    // Convert the photo to a Metal texture.
    if error == nil, let textureCache = textureCache,
       let sampleBuffer = photoSampleBuffer,
       let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {

      let width = CVPixelBufferGetWidth(imageBuffer)
      let height = CVPixelBufferGetHeight(imageBuffer)

      var texture: CVMetalTexture?
      CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCache,
          imageBuffer, nil, .bgra8Unorm, width, height, 0, &texture)

      if let texture = texture {
        imageTexture = CVMetalTextureGetTexture(texture)
      }
    }

    // Convert the preview to a UIImage and show it on the screen.
    if error == nil, let sampleBuffer = previewPhotoSampleBuffer,
       let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {

      let width = CVPixelBufferGetWidth(imageBuffer)
      let height = CVPixelBufferGetHeight(imageBuffer)
      let rect = CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height))

      let ciImage = CIImage(cvPixelBuffer: imageBuffer)
      let ciContext = CIContext(options: nil)
      if let cgImage = ciContext.createCGImage(ciImage, from: rect) {
        previewImage = UIImage(cgImage: cgImage)
      }
    }

    delegate?.didCapture(texture: imageTexture, previewImage: previewImage)
  }
}
