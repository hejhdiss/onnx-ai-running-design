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

CONFIG_DIR = os.path.expanduser("~/.config/chat")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.ini")
SOCKET_PATH = os.path.join(CONFIG_DIR, "chat.sock")

# Global state
active_session = None
session_lock = threading.Lock()
unload_timer = None
is_dummy_active = False

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
    print(f"Config updated. Timeout: {args.timeout}s.")
    sys.exit(0)

def create_dummy_model():
    """Generates a minimal 1KB ONNX model in a temp file to flush VRAM."""
    import onnx
    from onnx import helper, TensorProto
    
    node = helper.make_node('Identity', ['X'], ['Y'])
    graph = helper.make_graph([node], 'dummy', 
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1])])
    model = helper.make_model(graph)
    
    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    onnx.save(model, tmp.name)
    return tmp.name

def get_providers(cfg):
    force_cpu = cfg["chat"].get("device", "auto") == "cpu"
    providers = ["CPUExecutionProvider"]
    if not force_cpu:
        # Priority for CUDA
        providers.insert(0, "CUDAExecutionProvider")
    return providers

def swap_to_dummy():
    """Unloads real model and loads a tiny dummy to force hardware release."""
    global active_session, is_dummy_active
    with session_lock:
        if active_session is not None and not is_dummy_active:
            print("\n[Idle] Swapping to dummy model to free hardware resources...")
            # Release real session
            del active_session
            
            # Force GC and CUDA cache clear
            gc.collect()
            if "torch" in sys.modules:
                import torch
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            
            # Load minimal dummy
            dummy_path = create_dummy_model()
            try:
                active_session = ort.InferenceSession(dummy_path, providers=['CPUExecutionProvider'])
                is_dummy_active = True
                print("[Status] Hardware memory cleared. Dummy active.")
            finally:
                if os.path.exists(dummy_path): os.remove(dummy_path)

def schedule_swap(timeout_seconds):
    global unload_timer
    if unload_timer:
        unload_timer.cancel()
    if timeout_seconds > 0:
        unload_timer = threading.Timer(timeout_seconds, swap_to_dummy)
        unload_timer.start()

def load_real_model(cfg):
    global active_session, is_dummy_active
    with session_lock:
        if active_session is not None and not is_dummy_active:
            return active_session
        
        if is_dummy_active:
            print("[Wake] Evicting dummy...")
            del active_session
            is_dummy_active = False

        model_path = cfg["chat"]["model"]
        threads = int(cfg["chat"].get("threads", "4"))
        providers = get_providers(cfg)
        
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        
        print(f"[Load] Initializing {os.path.basename(model_path)}...")
        active_session = ort.InferenceSession(model_path, opts, providers=providers)
        return active_session

def process_request(cfg, prompt):
    timeout = int(cfg["chat"].get("timeout", "300"))
    session = load_real_model(cfg)
    
    # Placeholder for actual inference logic
    response = f"AI Response to '{prompt}'"
    
    schedule_swap(timeout)
    return response

def run_service():
    cfg = load_config()
    if not cfg.has_section("chat"):
        sys.exit("ERROR: Run --config first")

    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(5)

    print(f"Service active. Listening on {SOCKET_PATH}")
    print(f"Timeout set to {cfg['chat'].get('timeout', '300')}s")

    try:
        while True:
            conn, _ = server.accept()
            try:
                data = conn.recv(4096).decode('utf-8')
                if data:
                    res = process_request(cfg, data)
                    conn.sendall(res.encode('utf-8'))
            except Exception as e:
                conn.sendall(f"Inference Error: {str(e)}".encode('utf-8'))
            finally:
                conn.close()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if unload_timer: unload_timer.cancel()
        if os.path.exists(SOCKET_PATH): os.remove(SOCKET_PATH)

def send_to_service(prompt):
    if not os.path.exists(SOCKET_PATH):
        return None
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(120.0) # Longer timeout to allow for re-loading
        client.connect(SOCKET_PATH)
        client.sendall(prompt.encode('utf-8'))
        response = client.recv(4096).decode('utf-8')
        client.close()
        return response
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Optimized AI Chat CLI with Dummy Swap.")
    parser.add_argument("prompt", nargs="?", help="Text prompt.")
    parser.add_argument("--config", action="store_true", help="Save config.")
    parser.add_argument("--serve", action="store_true", help="Start service.")
    parser.add_argument("-m", "--model", help="Path to ONNX model.")
    parser.add_argument("--threads", type=int, default=4, help="Threads.")
    parser.add_argument("--timeout", type=int, default=300, help="Idle timeout (seconds).")
    parser.add_argument("--cpu", action="store_true", help="Force CPU.")
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
        print("Service not running. Run with --serve in another terminal.")

if __name__ == "__main__":
    main()
