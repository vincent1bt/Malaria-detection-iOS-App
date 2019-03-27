import UIKit
import CoreImage
import Accelerate

// MARK: Structures That hold results
/**
 Stores one formatted inference.
 */
struct Inference {
    let className: String
    let confidence: Double
}

/**
 This class handles all data preprocessing and makes calls to run inference on a given frame through the TfliteWrapper. It then formats the inferences obtained and returns the top N results for a successful inference.
 */
class ModelDataHandler: NSObject {
    
    // MARK: Paremeters on which model was trained
    let wantedInputChannels = 3
    let wantedInputWidth = 96
    let wantedInputHeight = 96
    let stdDeviation: Float = 127.0
    let mean: Float = 1.0
    
    // MARK: Constants
    let threadCountLimit: Int32 = 10
    
    // MARK: Instance Variables
    var labels: [String] = []
    var threadCount: Int32 = 1
    private let resultCount = 1
    private let threshold = 0.5
    
    // MARK: TfliteWrapper
    private let tfLiteWrapper: TfliteWrapper
    
    // MARK: Initializer
    /**
     This is a failable initializer for ModelDataHandler. It successfully initializes an object of the class if the model file and labels file is found, labels can be loaded and the interpreter of TensorflowLite can be initialized successfully.
     */
    init?(modelFileName: String, labelsFileName: String, labelsFileExtension: String) {
        
        // Initializes TFliteWrapper and based on the setup result of interpreter, initializes the object of this class
        self.tfLiteWrapper = TfliteWrapper(modelFileName: modelFileName)
        guard self.tfLiteWrapper.setUpModelAndInterpreter() else {
            return nil
        }
        
        super.init()
        
        // Opens and loads the classes listed in labels file
        loadLabels(fromFileName: labelsFileName, fileExtension: labelsFileExtension)
    }
    
    // MARK: Methods for data preprocessing and post processing.
    /** Performs image preprocessing, calls the tfliteWrapper methods to feed the pixel buffer into the input tensor and  run inference
     on the pixel buffer.
     */
    func runModel(onFrame pixelBuffer: CVPixelBuffer) -> Inference? {
        
        let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
            sourcePixelFormat == kCVPixelFormatType_32BGRA || sourcePixelFormat == kCVPixelFormatType_32RGBA)
        
        
        let imageChannels = 4
        assert(imageChannels >= wantedInputChannels)
        
        // Crops the image to the biggest square in the center and scales it down to model dimensions.
        guard let thumbnailPixelBuffer = centerThumbnail(from: pixelBuffer, size: CGSize(width: wantedInputWidth, height: wantedInputHeight)) else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(thumbnailPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        
        guard let sourceStartAddrss = CVPixelBufferGetBaseAddress(thumbnailPixelBuffer) else {
            return nil
        }
        
        // Obtains the input tensor to feed the pixel buffer into
        guard  let tensorInputBaseAddress = tfLiteWrapper.inputTensor(at: 0) else {
            return nil
        }
        
        let inputImageBaseAddress = sourceStartAddrss.assumingMemoryBound(to: UInt8.self)
        
        for y in 0...wantedInputHeight - 1 {
            let tensorInputRow = tensorInputBaseAddress.advanced(by: (y * wantedInputWidth * wantedInputChannels))
            let inputImageRow = inputImageBaseAddress.advanced(by: y * wantedInputWidth * imageChannels)
            
            for x in 0...wantedInputWidth - 1 {
                
                let out_pixel = tensorInputRow.advanced(by: x * wantedInputChannels)
                let in_pixel = inputImageRow.advanced(by: x * imageChannels)
                
                var b = 2
                for c in 0...(wantedInputChannels) - 1 {
                    
                    // Pixel values are between 0-255. Model requires the values to be between -1 and 1.
                    // We are also reversing the order of pixels since the source pixel format is BGRA, but the model requires RGB format.
                    out_pixel[c] = Float32((Double(in_pixel[b]) / Double(stdDeviation)) - Double(mean))
                    b = b - 1
                }
            }
        }
        
        // Runs inference
        guard tfLiteWrapper.invokeInterpreter() else {
            return nil
        }
        
        // Gets the output tensor at index 0. This is a vector that holds the confidence values of the classes detected.
        guard let outputTensor = tfLiteWrapper.outputTensor(at: 0) else {
            return nil
        }
        
        CVPixelBufferUnlockBaseAddress(thumbnailPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        
        let result = getClass(prediction: outputTensor, threshold: threshold)

        return result
    }
    
    /**
     Returns thumbnail by cropping pixel buffer to biggest square and scaling the cropped image to model dimensions.
     */
    private func centerThumbnail(from pixelBuffer: CVPixelBuffer, size: CGSize ) -> CVPixelBuffer? {
        
        let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
        let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
        let pixelBufferType = CVPixelBufferGetPixelFormatType(pixelBuffer)
        
        assert(pixelBufferType == kCVPixelFormatType_32BGRA)
        
        let inputImageRowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let imageChannels = 4
        
        let thumbnailSize = min(imageWidth, imageHeight)
        CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        
        var originX = 0
        var originY = 0
        
        if imageWidth > imageHeight {
            originX = (imageWidth - imageHeight) / 2
        }
        else {
            originY = (imageHeight - imageWidth) / 2
        }
        
        // Finds the biggest square in the pixel buffer and advances rows based on it.
        guard let inputBaseAddress = CVPixelBufferGetBaseAddress(pixelBuffer)?.advanced(by: originY * inputImageRowBytes + originX * imageChannels) else {
            return nil
        }
        
        // Gets vImage Buffer from input image
        var inputVImageBuffer = vImage_Buffer(data: inputBaseAddress, height: UInt(thumbnailSize), width: UInt(thumbnailSize), rowBytes: inputImageRowBytes)
        
        let thumbnailRowBytes = Int(size.width) * imageChannels
        guard  let thumbnailBytes = malloc(Int(size.height) * thumbnailRowBytes) else {
            return nil
        }
        
        // Allocates a vImage buffer for thumbnail image.
        var thumbnailVImageBuffer = vImage_Buffer(data: thumbnailBytes, height: UInt(size.height), width: UInt(size.width), rowBytes: thumbnailRowBytes)
        
        // Performs the scale operation on input image buffer and stores it in thumbnail image buffer.
        let scaleError = vImageScale_ARGB8888(&inputVImageBuffer, &thumbnailVImageBuffer, nil, vImage_Flags(0))
        
        CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        
        guard scaleError == kvImageNoError else {
            return nil
        }
        
        let releaseCallBack: CVPixelBufferReleaseBytesCallback = {mutablePointer, pointer in
            
            if let pointer = pointer {
                free(UnsafeMutableRawPointer(mutating: pointer))
            }
        }
        
        var thumbnailPixelBuffer: CVPixelBuffer?
        
        // Converts the thumbnail vImage buffer to CVPixelBuffer
        let conversionStatus = CVPixelBufferCreateWithBytes(nil, Int(size.width), Int(size.height), pixelBufferType, thumbnailBytes, thumbnailRowBytes, releaseCallBack, nil, nil, &thumbnailPixelBuffer)
        
        guard conversionStatus == kCVReturnSuccess else {
            
            free(thumbnailBytes)
            return nil
        }
        
        return thumbnailPixelBuffer
    }
    
    func getClass(prediction: UnsafeMutablePointer<Float>, threshold: Double) -> Inference {
        let value = Double(prediction[0])
        
        if value > threshold {
            let result = Inference(className: "Parasitized", confidence: value)
            return result
        }
        
        let result = Inference(className: "Uninfected", confidence: value)
        return result
    }
    
    // MARK: Thread Update Methods
    /**
     Sets the number of threads on the interpreter through the TFliteWrapper
     */
    func set(numberOfThreads threadCount: Int32) {
        
        tfLiteWrapper.setNumberOfThreads(threadCount)
        self.threadCount = threadCount
    }
    
    /**
     Loads the labels from the labels file and stores it in an instance variable
     */
    func loadLabels(fromFileName fileName: String, fileExtension: String) {
        
        guard let fileURL = Bundle.main.url(forResource: fileName, withExtension: fileExtension) else {
            fatalError("Labels file not found in bundle. Please add a labels file with name \(fileName).\(fileExtension) and try again")
        }
        do {
            let contents = try String(contentsOf: fileURL, encoding: .utf8)
            self.labels = contents.components(separatedBy: ",")
            self.labels.removeAll { (label) -> Bool in
                return label == ""
            }
        }
        catch {
            fatalError("Labels file named \(fileName).\(fileExtension) cannot be read. Please add a valid labels file and try again.")
        }
        
    }
}
