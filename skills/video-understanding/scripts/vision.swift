// vu-vision: read images with macOS Vision. One JSON line per image.
//
//   vu-vision a.jpg b.jpg            text:  {"path", "lines": [{"text", "conf", "box"}]}
//   vu-vision --faces a.jpg ...      faces: {"path", "faces": [{"box", "conf", "yaw", "roll", "pitch"}]}
//   vu-vision --pose a.jpg ...       body:  {"path", "people": [{"joints": {name: [x, y, conf]}}],
//                                            "hands": [{"chirality", "joints": {name: [x, y, conf]}}]}
//   vu-vision --person a.jpg ...     the person's mask, written next to the image as <name>.mask.png
//                                    {"path", "mask": "...png", "coverage": 0..1}
//
// Boxes and points are fractions of the image with the origin at the top left. Angles are degrees.
import AppKit
import Foundation
import Vision

func esc(_ s: String) -> String {
    let data = try! JSONSerialization.data(withJSONObject: [s], options: [])
    let str = String(data: data, encoding: .utf8)!
    return String(str.dropFirst().dropLast())
}

func f3(_ v: Double) -> String { String(format: "%.3f", v) }

func boxJSON(_ b: CGRect) -> String {
    "[\(f3(b.minX)), \(f3(1 - b.maxY)), \(f3(b.width)), \(f3(b.height))]"
}

func deg(_ n: NSNumber?) -> String {
    guard let n = n else { return "null" }
    return String(format: "%.1f", n.doubleValue * 180 / Double.pi)
}

func pointsJSON(_ pts: [VNRecognizedPointKey: VNRecognizedPoint]) -> String {
    var out: [String] = []
    for (k, p) in pts where p.confidence > 0.1 {
        out.append("\(esc(k.rawValue)): [\(f3(p.location.x)), \(f3(1 - p.location.y)), \(String(format: "%.2f", p.confidence))]")
    }
    return "{" + out.joined(separator: ", ") + "}"
}

var args = Array(CommandLine.arguments.dropFirst())
var mode = "text"
if let first = args.first, first.hasPrefix("--") {
    mode = String(first.dropFirst(2))
    args.removeFirst()
}

for path in args {
    guard let image = NSImage(contentsOfFile: path),
          let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        print("{\"path\": \(esc(path)), \"error\": \"unreadable\"}")
        continue
    }
    let handler = VNImageRequestHandler(cgImage: cg, options: [:])
    switch mode {
    case "faces":
        let request = VNDetectFaceRectanglesRequest()
        try? handler.perform([request])
        var faces: [String] = []
        for f in request.results ?? [] {
            faces.append("{\"box\": \(boxJSON(f.boundingBox)), \"conf\": \(String(format: "%.2f", f.confidence)), "
                + "\"yaw\": \(deg(f.yaw)), \"roll\": \(deg(f.roll)), \"pitch\": \(deg(f.pitch))}")
        }
        print("{\"path\": \(esc(path)), \"faces\": [\(faces.joined(separator: ", "))]}")
    case "pose":
        let body = VNDetectHumanBodyPoseRequest()
        let hands = VNDetectHumanHandPoseRequest()
        hands.maximumHandCount = 2
        try? handler.perform([body, hands])
        var people: [String] = []
        for o in body.results ?? [] {
            if let pts = try? o.recognizedPoints(.all) {
                var named: [VNRecognizedPointKey: VNRecognizedPoint] = [:]
                for (k, v) in pts { named[VNRecognizedPointKey(rawValue: k.rawValue.rawValue)] = v }
                people.append("{\"joints\": \(pointsJSON(named))}")
            }
        }
        var hs: [String] = []
        for o in hands.results ?? [] {
            if let pts = try? o.recognizedPoints(.all) {
                var named: [VNRecognizedPointKey: VNRecognizedPoint] = [:]
                for (k, v) in pts { named[VNRecognizedPointKey(rawValue: k.rawValue.rawValue)] = v }
                let side = o.chirality == .left ? "left" : o.chirality == .right ? "right" : "unknown"
                hs.append("{\"chirality\": \"\(side)\", \"joints\": \(pointsJSON(named))}")
            }
        }
        print("{\"path\": \(esc(path)), \"people\": [\(people.joined(separator: ", "))], \"hands\": [\(hs.joined(separator: ", "))]}")
    case "person":
        let request = VNGeneratePersonSegmentationRequest()
        request.qualityLevel = .balanced
        request.outputPixelFormat = kCVPixelFormatType_OneComponent8
        try? handler.perform([request])
        guard let buf = request.results?.first?.pixelBuffer else {
            print("{\"path\": \(esc(path)), \"mask\": null, \"coverage\": 0}")
            continue
        }
        let ci = CIImage(cvPixelBuffer: buf)
        let rep = NSBitmapImageRep(ciImage: ci)
        let out = (path as NSString).deletingPathExtension + ".mask.png"
        var on = 0, total = 0
        CVPixelBufferLockBaseAddress(buf, .readOnly)
        if let base = CVPixelBufferGetBaseAddress(buf) {
            let w = CVPixelBufferGetWidth(buf), h = CVPixelBufferGetHeight(buf), row = CVPixelBufferGetBytesPerRow(buf)
            let p = base.assumingMemoryBound(to: UInt8.self)
            for y in stride(from: 0, to: h, by: 2) { for x in stride(from: 0, to: w, by: 2) {
                total += 1; if p[y * row + x] > 127 { on += 1 }
            } }
        }
        CVPixelBufferUnlockBaseAddress(buf, .readOnly)
        try? rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: out))
        print("{\"path\": \(esc(path)), \"mask\": \(esc(out)), \"coverage\": \(f3(total > 0 ? Double(on) / Double(total) : 0))}")
    default:
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true
        if #available(macOS 13.0, *) { request.automaticallyDetectsLanguage = true }
        try? handler.perform([request])
        var lines: [String] = []
        for obs in request.results ?? [] {
            guard let top = obs.topCandidates(1).first else { continue }
            lines.append("{\"text\": \(esc(top.string)), \"conf\": \(String(format: "%.2f", top.confidence)), \"box\": \(boxJSON(obs.boundingBox))}")
        }
        print("{\"path\": \(esc(path)), \"lines\": [\(lines.joined(separator: ", "))]}")
    }
}
