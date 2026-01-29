#!/usr/bin/env python3
# licensed under Apache 2.0.

import argparse
import configparser
import os
import sys
import socket
import gc
import onnxruntime as ort
import numpy as np
import threading
import tempfile
import time

# Paths
CONFIG_DIR = os.path.expanduser("~/.config/chat")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.ini")
SOCKET_PATH = os.path.join(CONFIG_DIR, "chat.sock")

# Global State
class ServiceState:
    def __init__(self):
        self.session = None
        self.lock = threading.Lock()
        self.timer = None
        self.is_idle = True
        self.model_mmap = None # Keeping a reference to the file map

state = ServiceState()

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
        "device": "cpu" if args.cpu else "auto",
        "timeout": str(args.timeout)
    }
    with open(CONFIG_FILE, "w") as f:
        cfg.write(f)
    print(f"Config Saved: Model mapped with {args.timeout}s lazy-unload.")
    sys.exit(0)

def create_tiny_dummy():
    """Create a minimal ONNX identity node to act as a placeholder."""
    try:
        import onnx
        from onnx import helper, TensorProto
        node = helper.make_node('Identity', ['X'], ['Y'])
        graph = helper.make_graph([node], 'park', 
            [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1])])
        model = helper.make_model(graph)
        tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        onnx.save(model, tmp.name)
        return tmp.name
    except ImportError:
        return None

def hibernate_model():
    """The 'Parking' logic: Swaps real weights for zero-weight dummy."""
    with state.lock:
        if state.session and not state.is_idle:
            print("\n[Hibernate] Inactivity detected. Releasing hardware...")
            state.session = None # Trigger C++ destructor
            gc.collect()
            
            # Use torch cache clear if available to truly zero out VRAM
            if "torch" in sys.modules:
                import torch
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            dummy_path = create_tiny_dummy()
            if dummy_path:
                # Load dummy on CPU only to ensure GPU is totally free
                state.session = ort.InferenceSession(dummy_path, providers=['CPUExecutionProvider'])
                state.is_idle = True
                os.remove(dummy_path)
                print("[Status] Hardware released. Service in low-power mode.")

def reset_timer(timeout):
    if state.timer:
        state.timer.cancel()
    if timeout > 0:
        state.timer = threading.Timer(timeout, hibernate_model)
        state.timer.start()

def wake_and_run(cfg, prompt):
    """Wake logic with MMAP optimization and Warmup."""
    timeout = int(cfg["chat"].get("timeout", "300"))
    
    with state.lock:
        if state.is_idle or state.session is None:
            print("[Wake] Loading model via MMAP...")
            
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = int(cfg["chat"].get("threads", "4"))
            # ADVANCED: Use Memory Mapping for the weights
            opts.add_session_config_entry("session.use_mmap", "1")
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            model_path = cfg["chat"]["model"]
            force_cpu = cfg["chat"].get("device", "auto") == "cpu"
            providers = ["CPUExecutionProvider"]
            if not force_cpu and "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
            
            state.session = ort.InferenceSession(model_path, opts, providers=providers)
            state.is_idle = False
            print(f"[Ready] Model active on {state.session.get_providers()[0]}")

    # Process
    # response = state.session.run(...) 
    result = f"AI Output for: {prompt}"
    
    reset_timer(timeout)
    return result

def run_service():
    cfg = load_config()
    if not cfg.has_section("chat"):
        sys.exit("ERROR: No config. Use --config first.")

    if os.path.exists(SOCKET_PATH): os.remove(SOCKET_PATH)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(5)

    print(f"Advanced Service started. PID: {os.getpid()}")
    
    try:
        while True:
            conn, _ = server.accept()
            try:
                data = conn.recv(8192).decode('utf-8')
                if data:
                    response = wake_and_run(cfg, data)
                    conn.sendall(response.encode('utf-8'))
            except Exception as e:
                conn.sendall(f"Internal Error: {e}".encode('utf-8'))
            finally:
                conn.close()
    except KeyboardInterrupt:
        print("\nShutdown signal received.")
    finally:
        if state.timer: state.timer.cancel()
        if os.path.exists(SOCKET_PATH): os.remove(SOCKET_PATH)

def main():
    parser = argparse.ArgumentParser(description="Advanced AI Service: MMAP & Lazy Load Design")
    parser.add_argument("prompt", nargs="?", help="Prompt text.")
    parser.add_argument("--config", action="store_true", help="Initialize settings.")
    parser.add_argument("--serve", action="store_true", help="Launch background service.")
    parser.add_argument("-m", "--model", help="ONNX Model file path.")
    parser.add_argument("--timeout", type=int, default=300, help="Idle timeout (s).")
    parser.add_argument("--threads", type=int, default=4, help="Worker threads.")
    parser.add_argument("--cpu", action="store_true", help="Disable GPU.")
    args = parser.parse_args()

    if args.config:
        save_config(args)
    if args.serve:
        run_service()
        return
    
    if not args.prompt:
        parser.print_help()
        return

    # Client Logic
    if not os.path.exists(SOCKET_PATH):
        print("Service is not running. Start with --serve first.")
        return

    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(60.0) # Account for wake-up time
        client.connect(SOCKET_PATH)
        client.sendall(args.prompt.encode('utf-8'))
        print(client.recv(8192).decode('utf-8'))
        client.close()
    except Exception as e:
        print(f"Failed to reach service: {e}")

if __name__ == "__main__":
    main()
