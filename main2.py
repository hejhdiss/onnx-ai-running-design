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
import time

CONFIG_DIR = os.path.expanduser("~/.config/chat")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.ini")
SOCKET_PATH = os.path.join(CONFIG_DIR, "chat.sock")

# Global state for the service
active_session = None
last_activity_time = 0
session_lock = threading.Lock()
unload_timer = None

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
    print(f"Config written with {args.timeout}s timeout.")
    sys.exit(0)

def get_providers(cfg):
    force_cpu = cfg["chat"].get("device", "auto") == "cpu"
    providers = ["CPUExecutionProvider"]
    if not force_cpu:
        providers.insert(0, "CUDAExecutionProvider")
    return providers

def unload_model():
    global active_session
    with session_lock:
        if active_session is not None:
            print("\nIdle timeout reached. Unloading model...")
            del active_session
            active_session = None
            if "torch" in sys.modules:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            gc.collect()
            print("Hardware memory cleared.")

def schedule_unload(timeout_seconds):
    global unload_timer
    if unload_timer:
        unload_timer.cancel()
    
    if timeout_seconds > 0:
        unload_timer = threading.Timer(timeout_seconds, unload_model)
        unload_timer.start()

def get_or_create_session(cfg):
    global active_session
    with session_lock:
        if active_session is not None:
            return active_session
        
        model_path = cfg["chat"]["model"]
        threads = int(cfg["chat"].get("threads", "4"))
        providers = get_providers(cfg)
        
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads

        print(f"Loading {os.path.basename(model_path)} into memory...")
        active_session = ort.InferenceSession(model_path, opts, providers=providers)
        return active_session

def process_request(cfg, prompt):
    timeout = int(cfg["chat"].get("timeout", "300"))
    
    # Get session (loads if not present)
    session = get_or_create_session(cfg)
    
    # Dummy processing logic
    response = f"Processed: {prompt}"
    
    # Schedule the unload timer
    schedule_unload(timeout)
    
    return response

def run_service():
    cfg = load_config()
    if not cfg.has_section("chat"):
        sys.exit("ERROR: run --config first")

    print(f"Service running. Socket: {SOCKET_PATH}")
    print(f"Auto-unload timeout: {cfg['chat'].get('timeout', '300')}s")

    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(5)

    try:
        while True:
            conn, _ = server.accept()
            try:
                data = conn.recv(4096).decode('utf-8')
                if data:
                    res = process_request(cfg, data)
                    conn.sendall(res.encode('utf-8'))
            except Exception as e:
                error_msg = f"Inference Error: {str(e)}"
                conn.sendall(error_msg.encode('utf-8'))
            finally:
                conn.close()
    except KeyboardInterrupt:
        print("\nShutting down service...")
    finally:
        if unload_timer:
            unload_timer.cancel()
        unload_model()
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)

def send_to_service(prompt):
    if not os.path.exists(SOCKET_PATH):
        return None
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(60.0) # Wait up to 60s for loading/inference
        client.connect(SOCKET_PATH)
        client.sendall(prompt.encode('utf-8'))
        response = client.recv(4096).decode('utf-8')
        client.close()
        return response
    except Exception as e:
        return None

def main():
    parser = argparse.ArgumentParser(
        description="AI Chat CLI: Keep models in memory for a custom duration."
    )
    parser.add_argument(
        "prompt", 
        nargs="?", 
        help="Text prompt. If service is running, it stays loaded for --timeout seconds."
    )
    parser.add_argument(
        "--config", 
        action="store_true", 
        help="Save configuration."
    )
    parser.add_argument(
        "--serve", 
        action="store_true", 
        help="Start the background service."
    )
    parser.add_argument(
        "-m", "--model", 
        metavar="PATH",
        help="Path to the ONNX model file."
    )
    parser.add_argument(
        "--threads", 
        type=int, 
        default=4, 
        help="ONNX threads (default: 4)."
    )
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=300, 
        help="Seconds to keep model loaded after last prompt (default: 300)."
    )
    parser.add_argument(
        "--cpu", 
        action="store_true", 
        help="Force CPU mode."
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

    # Try communicating with service
    response = send_to_service(args.prompt)
    if response:
        print(response)
    else:
        cfg = load_config()
        if not cfg.has_section("chat"):
            sys.exit("ERROR: No configuration found. Run with --config -m [path] first.")
        providers = get_providers(cfg)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(cfg["chat"].get("threads", "4"))
        
        print("Service not detected. Performing local cold-start...")
        sess = ort.InferenceSession(cfg["chat"]["model"], opts, providers=providers)
        print(f"Processed: {args.prompt}")
        del sess
        gc.collect()

if __name__ == "__main__":
    main()
