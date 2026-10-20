// vu-vision: read the text in images with macOS Vision. One JSON line per image:
// {"path": "...", "lines": [{"text": "...", "conf": 0.98, "box": [x, y, w, h]}]}
// Boxes are fractions of the image with the origin at the top left.
import AppKit
import Foundation
import Vision

func esc(_ s: String) -> String {
    let data = try! JSONSerialization.data(withJSONObject: [s], options: [])
    let str = String(data: data, encoding: .utf8)!
    return String(str.dropFirst().dropLast())
}

for path in CommandLine.arguments.dropFirst() {
    guard let image = NSImage(contentsOfFile: path),
          let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        print("{\"path\": \(esc(path)), \"lines\": [], \"error\": \"unreadable\"}")
        continue
    }
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    if #available(macOS 13.0, *) { request.automaticallyDetectsLanguage = true }
    let handler = VNImageRequestHandler(cgImage: cg, options: [:])
    try? handler.perform([request])
    var lines: [String] = []
    for obs in request.results ?? [] {
        guard let top = obs.topCandidates(1).first else { continue }
        let b = obs.boundingBox
        let box = String(format: "[%.3f, %.3f, %.3f, %.3f]", b.minX, 1 - b.maxY, b.width, b.height)
        lines.append("{\"text\": \(esc(top.string)), \"conf\": \(String(format: "%.2f", top.confidence)), \"box\": \(box)}")
    }
    print("{\"path\": \(esc(path)), \"lines\": [\(lines.joined(separator: ", "))]}")
}
