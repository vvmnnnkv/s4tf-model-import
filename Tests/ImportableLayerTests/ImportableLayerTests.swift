import XCTest
@testable import ImportableLayer
import TensorFlow

final class ImportableLayerTests: XCTestCase {
    func testImport() {
        /// test model
        struct TestModel: Layer, ImportableLayer {
            typealias Input = Tensor<Float>
            typealias Output = Tensor<Float>
            var conv: Conv2D<Float>
            var something: Tensor<Float>
            init() {
                conv = Conv2D<Float>(filterShape: (3,3,1,16))
                something = Tensor<Float>(zeros: [1])
            }
            @differentiable
            func callAsFunction(_ input: Self.Input) -> Self.Output {
                return conv(input)
            }
        }

        let params = [
            "someparam": Tensor<Float>(ones: [1]),
            "conv.weight": Tensor<Float>(randomUniform: [16,1,3,3]),
            "conv.bias": Tensor<Float>(randomUniform: [1]),
        ]

        let map = [
            "conv.filter": ("conv.weight", [3,2,1,0]),
            "something": ("someparam", nil)
            // conv.bias has same property name and layout
        ]

        var model = TestModel()
        model.unsafeImport(parameters: params, map: map)
        XCTAssertEqual(model.conv.filter, params["conv.weight"]!.transposed(withPermutations: [3,2,1,0]))
        XCTAssertEqual(model.conv.bias, params["conv.bias"]!)
        XCTAssertEqual(model.something, params["someparam"]!)
    }

    static var allTests = [
        ("testImport", testImport),
    ]
}
