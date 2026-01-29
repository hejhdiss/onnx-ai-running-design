#!/usr/bin/env python3
#Licensed under Apache 2.0.

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

def create_session(cfg):
    model_path = cfg["chat"]["model"]
    threads = int(cfg["chat"]["threads"])
    providers = get_providers(cfg)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    return ort.InferenceSession(model_path, opts, providers=providers)

def run_inference(session, prompt):
    inputs = session.get_inputs()
    input_name = inputs[0].name
    input_type = inputs[0].type
    input_shape = inputs[0].shape
    
    if "tensor(float)" in input_type:
        dtype = np.float32
    elif "tensor(int64)" in input_type:
        dtype = np.int64
    else:
        dtype = np.float32

    processed_shape = [1 if (isinstance(dim, str) or dim is None) else dim for dim in input_shape]
    
    try:
        raw_vals = [float(x) for x in prompt.split() if x.replace('.','',1).replace('-','',1).isdigit()]
        if not raw_vals:
            input_data = np.zeros(processed_shape, dtype=dtype)
        else:
            input_data = np.resize(np.array(raw_vals, dtype=dtype), processed_shape)
    except:
        input_data = np.zeros(processed_shape, dtype=dtype)

    outputs = session.run(None, {input_name: input_data})
    return str(outputs[0])

def run_service():
    cfg = load_config()
    if not cfg.has_section("chat"):
        sys.exit("ERROR: run --config first")
    
    print(f"Loading model: {cfg['chat']['model']}")
    session = create_session(cfg)
    print("Service running. Model persistent in memory.")

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
                    res = run_inference(session, data)
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
    parser = argparse.ArgumentParser(description="AI Chat CLI")
    parser.add_argument("prompt", nargs="?")
    parser.add_argument("--config", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("-m", "--model", metavar="PATH")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
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
            sys.exit("ERROR: No configuration found.")
        session = create_session(cfg)
        print(run_inference(session, args.prompt))

if __name__ == "__main__":
    main()
