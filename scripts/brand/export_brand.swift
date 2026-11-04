// Native SVG exports keep every raster pose tied to the editable vector master.
import AppKit

let brand = URL(fileURLWithPath: CommandLine.arguments[1])
let source = brand.appendingPathComponent("source")
let exports = brand.appendingPathComponent("exports")
let poses = URL(fileURLWithPath: CommandLine.arguments[2])
let avatarOnly = CommandLine.arguments.dropFirst(3).contains("--avatar-only")
func render(_ width: Int, _ height: Int, _ output: URL, _ draw: () -> Void) throws {
    let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height, bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmap)
    NSGraphicsContext.current!.imageInterpolation = .high
    draw()
    NSGraphicsContext.current!.flushGraphics()
    NSGraphicsContext.restoreGraphicsState()
    try bitmap.representation(using: .png, properties: [:])!.write(to: output)
}
func image(_ url: URL) -> NSImage { NSImage(contentsOf: url)! }
let master = image(source.appendingPathComponent("weft-mascot-rig.svg"))
let wordmark = source.appendingPathComponent("weft-wordmark.png")
precondition(FileManager.default.fileExists(atPath: wordmark.path), "Missing source wordmark")
if !avatarOnly {
    try render(1280,1280,exports.appendingPathComponent("weft-mascot.png")) {
        master.draw(in:NSRect(x:0,y:0,width:1280,height:1280))
    }
    try render(1280,1280,exports.appendingPathComponent("weft-mascot-sprites.png")) {
        for (i,name) in ["idle","blink","wave_a","wave_b"].enumerated() {
            image(poses.appendingPathComponent(name+".svg")).draw(in:NSRect(x:(i%2)*640,y:(1-i/2)*640,width:640,height:640))
        }
    }
    try render(1254,1254,exports.appendingPathComponent("weft-logo.png")) {
        image(wordmark).draw(in:NSRect(x:0,y:0,width:1254,height:1254))
        master.draw(in:NSRect(x:260,y:434,width:700,height:700))
    }
}
try render(1024,1024,exports.appendingPathComponent("weft-publisher-avatar.png")) {
    image(poses.appendingPathComponent("publisher-avatar.svg")).draw(in:NSRect(x:0,y:0,width:1024,height:1024))
}
try render(1024,1024,exports.appendingPathComponent("weft-publisher-developer.png")) {
    image(poses.appendingPathComponent("publisher-developer.svg")).draw(in:NSRect(x:0,y:0,width:1024,height:1024))
}
print(avatarOnly ? "Exported Weft publisher avatar" : "Exported mascot, sprites, logo, and publisher avatar")
