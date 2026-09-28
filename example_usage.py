from client import Int8SymmetricQuantizer

def run_example():
    print("=== GenPark INT8 Symmetric Quantizer Example ===")
    test_weights = [-3.5, -2.1, -0.5, 0.0, 0.45, 1.25, 2.8, 3.49]
    res = Int8SymmetricQuantizer.quantize(test_weights)
    print("Scale Factor:", res["scale"])
    print("Quantized INT8:", res["sample_quantized"])
    print("Reconstruction MSE:", res["mse_loss"])
    print("Signal-to-Noise Ratio (dB):", res["snr_db"])

if __name__ == "__main__":
    run_example()
