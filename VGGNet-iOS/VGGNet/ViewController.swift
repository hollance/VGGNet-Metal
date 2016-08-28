import UIKit
import MetalKit
import MetalPerformanceShaders

class ViewController: UIViewController {

  @IBOutlet weak var spinner: UIActivityIndicatorView!
  @IBOutlet weak var spinnerPanel: UIView!

  private var textureLoader: MTKTextureLoader!
  private var nn: VGGNet!

  override func viewDidLoad() {
    super.viewDidLoad()

    spinnerPanel.isHidden = false
    spinner.startAnimating()

    createNeuralNetwork {
      self.spinner.stopAnimating()
      self.spinnerPanel.isHidden = true

      // Run the test image through the network. It should find a cat. ;)
      self.predict(imageNamed: "sophie.jpg") { print($0) }
    }
  }

  private func createNeuralNetwork(completion: @escaping () -> Void) {
    if let device = MTLCreateSystemDefaultDevice() {

      // Make sure the current device supports MetalPerformanceShaders.
      guard MPSSupportsMTLDevice(device) else {
        print("Error: Metal Performance Shaders not supported on this device")
        return
      }

      // We use MetalKit's texture loader to create MPSImage objects.
      textureLoader = MTKTextureLoader(device: device)

      // Because it takes a few seconds to load VGGNet's parameters, perform
      // the construction of the neural network in a background thread.
      DispatchQueue.global().async {
        self.nn = VGGNet(device: device)

        DispatchQueue.main.async(execute: completion)
      }
    }
  }

  private func predict(imageNamed filename: String, completion: @escaping ([String]) -> Void) {
    print("Predicting \(filename)")

    // It takes between 0.25-0.3 seconds to perform a forward pass of the net.
    // VGGNet.predict() blocks until the GPU is ready, so to prevent the app's
    // UI from being blocked we call that method from a background thread.
    DispatchQueue.global().async {
      if let texture = self.loadTexture(named: filename) {
        let inputImage = self.image(from: texture)
        let prediction = self.nn.predict(image: inputImage)

        DispatchQueue.main.async { completion(prediction) }
      }
    }
  }

  private func image(from texture: MTLTexture) -> MPSImage {
    // We set featureChannels to 3 because the neural network is only trained
    // on RGB data (the first 3 channels), not alpha (the 4th channel).
    return MPSImage(texture: texture, featureChannels: 3)
  }

  private func loadTexture(named filename: String) -> MTLTexture? {
    if let url = Bundle.main.url(forResource: filename, withExtension: "") {
      return loadTexture(url: url)
    } else {
      print("Error: could not find image \(filename)")
      return nil
    }
  }

  private func loadTexture(url: URL) -> MTLTexture? {
    do {
      // Note: the SRGB option should be set to false, otherwise the image
      // appears way too dark, since it wasn't actually saved as SRGB.
      return try textureLoader.newTexture(withContentsOf: url, options: [
        MTKTextureLoaderOptionSRGB : NSNumber(value: false)
      ])
    } catch {
      print("Error: could not load texture \(error)")
      return nil
    }
  }

  override func didReceiveMemoryWarning() {
    super.didReceiveMemoryWarning()
    print(#function)
  }

  override var preferredStatusBarStyle: UIStatusBarStyle {
    return .lightContent
  }
}
