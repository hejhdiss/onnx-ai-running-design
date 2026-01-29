#!/usr/bin/env python3
#licensed under Apache 2.0.


import argparse
import configparser
import os
import sys
import socket
import gc
import onnxruntime as ort
import numpy as np

CONFIG_DIR = os.path.expanduser("~/.config/chat")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.ini")
SOCKET_PATH = os.path.join(CONFIG_DIR, "chat.sock")

def ensure_config_dir():
    os.makedirs(CONFIG_DIR, exist_ok=True)

def load_config():
    cfg = configparser.ConfigParser()
    if os.path.exists(CONFIG_FILE):
        cfg.read(CONFIG_FILE)
    return cfg

def save_config(args):
    ensure_config_dir()
    if not args.model or not os.path.exists(args.model):
        sys.exit(f"ERROR: model path does not exist: {args.model}")

    cfg = configparser.ConfigParser()
    cfg["chat"] = {
        "model": os.path.abspath(args.model),
        "threads": str(args.threads),
        "device": "cpu" if args.cpu else "auto"
    }
    with open(CONFIG_FILE, "w") as f:
        cfg.write(f)
    print(f"Config written.")
    sys.exit(0)

def get_providers(cfg):
    force_cpu = cfg["chat"]["device"] == "cpu"
    providers = ["CPUExecutionProvider"]
    if not force_cpu:
        providers.insert(0, "CUDAExecutionProvider")
    return providers

def create_warmup_session(cfg):
    import tempfile
    providers = get_providers(cfg)
    try:
        import onnx
        from onnx import helper, TensorProto
        node = helper.make_node("Relu", ["X"], ["Y"])
        graph = helper.make_graph([node], "warmup", [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1])], [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])])
        model = helper.make_model(graph)
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            onnx.save(model, tmp.name)
            tmp_path = tmp.name
        sess = ort.InferenceSession(tmp_path, providers=providers)
        sess.run(None, {"X": np.array([[1.0]], dtype=np.float32)})
        os.remove(tmp_path)
        return sess
    except Exception:
        return None

def infer_with_dynamic_load(cfg, prompt):
    model_path = cfg["chat"]["model"]
    threads = int(cfg["chat"]["threads"])
    providers = get_providers(cfg)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads

    print(f"Loading {os.path.basename(model_path)}...")
    session = ort.InferenceSession(model_path, opts, providers=providers)
    
    response = f"Processed: {prompt}"
    
    del session
    if "torch" in sys.modules:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    gc.collect()
    print("Model unloaded.")
    return response

def run_service():
    cfg = load_config()
    if not cfg.has_section("chat"):
        sys.exit("ERROR: run --config first")

    warmup = create_warmup_session(cfg)
    print("Service running. GPU warm.")

    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(1)

    try:
        while True:
            conn, _ = server.accept()
            try:
                data = conn.recv(4096).decode('utf-8')
                if data:
                    res = infer_with_dynamic_load(cfg, data)
                    conn.sendall(res.encode('utf-8'))
            except Exception as e:
                conn.sendall(str(e).encode('utf-8'))
            finally:
                conn.close()
    except KeyboardInterrupt:
        pass
    finally:
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)

def send_to_service(prompt):
    if not os.path.exists(SOCKET_PATH):
        return None
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(SOCKET_PATH)
        client.sendall(prompt.encode('utf-8'))
        response = client.recv(4096).decode('utf-8')
        client.close()
        return response
    except:
        return None

def main():
    parser = argparse.ArgumentParser(
        description="AI Chat CLI: A minimal tool that can run as a background service to keep hardware warm and load models on-demand."
    )
    parser.add_argument(
        "prompt", 
        nargs="?", 
        help="The input text prompt to process. If the service is running, it communicates with it; otherwise, it performs a local cold-start."
    )
    parser.add_argument(
        "--config", 
        action="store_true", 
        help="Save the current command-line parameters (model path, threads, device) to the persistent config file."
    )
    parser.add_argument(
        "--serve", 
        action="store_true", 
        help="Start the background service. This initializes the GPU/hardware and listens for incoming prompts via a Unix socket."
    )
    parser.add_argument(
        "-m", "--model", 
        metavar="PATH",
        help="Path to the ONNX model file. Required when using --config."
    )
    parser.add_argument(
        "--threads", 
        type=int, 
        default=4, 
        help="Number of threads for ONNX Runtime (default: 4)."
    )
    parser.add_argument(
        "--cpu", 
        action="store_true", 
        help="Force the usage of the CPU Execution Provider, disabling CUDA/GPU acceleration."
    )
    args = parser.parse_args()

    if args.config:
        save_config(args)
    if args.serve:
        run_service()
        return
    if not args.prompt:
        parser.print_help()
        sys.exit(1)

    response = send_to_service(args.prompt)
    if response:
        print(response)
    else:
        cfg = load_config()
        if not cfg.has_section("chat"):
            sys.exit("ERROR: No configuration found. Run with --config -m [path] first.")
        print(infer_with_dynamic_load(cfg, args.prompt))

if __name__ == "__main__":
    main()
