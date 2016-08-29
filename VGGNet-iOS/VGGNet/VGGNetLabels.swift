import Foundation

public typealias Prediction = (label: String, probability: Float)

/*
  The list of ImageNet label names, loaded from synset_words.txt.
*/
public class VGGNetLabels {
  private var labels = [String](repeating: "", count: 1000)

  public init() {
    if let path = Bundle.main.path(forResource: "synset_words", ofType: "txt") {
      for (i, line) in lines(filename: path).enumerated() {
        if i < 1000 {
          // Strip off the WordNet ID (the first 10 characters).
          labels[i] = line.substring(from: line.index(line.startIndex, offsetBy: 10))
        }
      }
    }
  }

  private func lines(filename: String) -> [String] {
    do {
      let text = try String(contentsOfFile: filename, encoding: .ascii)
      let lines = text.components(separatedBy: NSCharacterSet.newlines)
      return lines
    } catch {
      fatalError("Could not load file: \(filename)")
    }
  }

  public subscript(i: Int) -> String {
    return labels[i]
  }

  /* Returns the labels for the top 5 guesses. */
  public func top5Labels(prediction: [Float]) -> [Prediction] {
    precondition(prediction.count == 1000)

    //print(prediction)

    // Combine the predicted probabilities and their array indices into a new 
    // list, then sort it from greatest probability to smallest. Finally, take
    // the top 5 items and convert them into strings.

    typealias tuple = (idx: Int, prob: Float)

    return zip(0...1000, prediction)
           .sorted(by: { (a: tuple, b: tuple) -> Bool in a.prob > b.prob })
           .prefix(through: 4)
           .map({ (x: tuple) -> Prediction in (labels[x.idx], x.prob) })
  }
}
