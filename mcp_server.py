import sys, json
from client import Int8SymmetricQuantizer

def handle_jsonrpc(line):
    try:
        req = json.loads(line)
        req_id = req.get("id")
        method = req.get("method")
        
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "genpark-int8-symmetric-per-tensor-quantizer-skill", "version": "1.0.0"},
                    "capabilities": {"tools": {}}
                }
            }
        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [
                        {
                            "name": "quantize_tensor",
                            "description": "Quantize float32 tensor into symmetric int8 representation with MSE and SNR telemetry.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"tensor": {"type": "array", "items": {"type": "number"}}},
                                "required": ["tensor"]
                            }
                        },
                        {
                            "name": "benchmark_quantization",
                            "description": "Run standard INT8 quantization benchmark suite.",
                            "inputSchema": {"type": "object", "properties": {}}
                        }
                    ]
                }
            }
        elif method == "tools/call":
            params = req.get("params", {})
            tool = params.get("name")
            args = params.get("arguments", {})
            if tool == "quantize_tensor":
                res = Int8SymmetricQuantizer.quantize(args.get("tensor", []))
            elif tool == "benchmark_quantization":
                res = Int8SymmetricQuantizer.benchmark_quantization()
            else:
                return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}
            return {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}}
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}
    except Exception as e:
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": str(e)}}

def main():
    for line in sys.stdin:
        if line.strip():
            print(json.dumps(handle_jsonrpc(line.strip())), flush=True)

if __name__ == "__main__":
    main()
