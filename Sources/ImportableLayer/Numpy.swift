import Python
import TensorFlow
let np = Python.import("numpy")

public extension ImportableLayer {

    mutating func unsafeImport(fromNumpyArchive file: String, map: [String: (String, [Int]?)]) {
        let data = np.load(file)
        var parameters = [String: Tensor<Float>]()
        for label in data.files {
            if let label = String(label) {
                parameters[label] = Tensor<Float>(numpy: data[label])
            }
        }
        unsafeImport(parameters: parameters, map: map)
    }
}