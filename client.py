import math
from typing import List, Dict, Any

class Int8SymmetricQuantizer:
    @staticmethod
    def quantize(tensor: List[float]) -> Dict[str, Any]:
        if not tensor:
            return {"error": "Tensor cannot be empty"}
        max_abs = max(abs(x) for x in tensor)
        scale = max_abs / 127.0 if max_abs > 0 else 1.0
        
        quantized = []
        for x in tensor:
            q = round(x / scale)
            q = max(-128, min(127, q))
            quantized.append(int(q))
            
        dequantized = [q * scale for q in quantized]
        mse = sum((o - d) ** 2 for o, d in zip(tensor, dequantized)) / len(tensor)
        signal_power = sum(x**2 for x in tensor) / len(tensor)
        snr = 10 * math.log10(signal_power / (mse + 1e-12)) if mse > 0 else 999.0
        
        return {
            "scale": round(scale, 6),
            "zero_point": 0,
            "quantized_length": len(quantized),
            "sample_quantized": quantized[:8],
            "sample_dequantized": [round(d, 4) for d in dequantized[:8]],
            "mse_loss": round(mse, 6),
            "snr_db": round(snr, 2),
            "compression_ratio": "4.0x (FP32 -> INT8)"
        }

    @staticmethod
    def benchmark_quantization() -> Dict[str, Any]:
        sample_weights = [0.15, -0.42, 1.88, -2.95, 3.1415, 0.0, -0.05, 0.98, -1.22, 2.45]
        return Int8SymmetricQuantizer.quantize(sample_weights)
