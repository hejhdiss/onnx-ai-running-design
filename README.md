# ONNX AI Running Design - Basic Orchestration Patterns

> **⚠️ IMPORTANT NOTE:** This repository focuses **exclusively on model loading and orchestration patterns** - how to load/unload AI models in memory efficiently. This is just **one side of the system**. It does NOT discuss:
> - Which models to use
> - Model selection logic
> - Model inference implementation (placeholder only)
> - Model training or fine-tuning
> - Input/output processing
> - Multi-model coordination
> - Application integration
> - User interfaces
> - Or any other aspects of a complete AI system
> 
> This is **basic loading system design only** - the foundation for memory management in NeuroShellOS.

This repository contains **minimal, foundational orchestration designs** for running AI models (ONNX format) in NeuroShellOS. These designs were created with AI assistance (Gemini) to establish basic patterns that can be expanded based on specific use cases.

## 🎯 Purpose

These are **basic blueprints** demonstrating different modes of AI model orchestration. They represent the core architectural patterns needed for NeuroShellOS, where AI/ML models need to run efficiently alongside user workflows without interfering with other processes.

**Note:** This is NOT a production implementation - it's a foundational design that will evolve as NeuroShellOS develops. The actual orchestration layer in a full NeuroShellOS distribution could span thousands of lines with additional features like:
- Multi-model coordination
- Privacy protection mechanisms
- Resource isolation
- Hallucination detection
- Cross-process awareness
- Advanced memory management

## 📋 Design Modes

This repository contains **5 different orchestration modes**, each optimized for different use cases:

### Mode 1: Dynamic Load/Unload (`main__1_.py`)
- **Best for:** Gaming distros, resource-constrained systems
- **Behavior:** Loads model on-demand, processes request, immediately unloads
- **Memory:** Minimal footprint - model only in memory during inference
- **Startup:** Cold start on every request (slower)
- **Use case:** When you need maximum RAM/VRAM for other applications (e.g., games)

```bash
# Configure
python main__1_.py --config -m /path/to/model.onnx

# Run service (keeps hardware warm but not model)
python main__1_.py --serve

# Query
python main__1_.py "your prompt here"
```

### Mode 2: Persistent in Memory (`main1.py`)
- **Best for:** High-frequency usage, development work
- **Behavior:** Loads model once, keeps it in memory permanently
- **Memory:** Highest footprint - model always loaded
- **Startup:** Fast responses after initial load
- **Use case:** When you're actively using AI features throughout the day

```bash
# Configure
python main1.py --config -m /path/to/model.onnx

# Run service (model stays loaded)
python main1.py --serve

# Query (instant response)
python main1.py "your prompt here"
```

### Mode 3: Timeout-based Unload (`main2.py`)
- **Best for:** Intermittent usage, balanced approach
- **Behavior:** Keeps model loaded for N seconds after last use
- **Memory:** Medium footprint - smart memory management
- **Startup:** Fast during active period, cold after timeout
- **Use case:** Working sessions with breaks (default: 5 minutes)

```bash
# Configure with custom timeout
python main2.py --config -m /path/to/model.onnx --timeout 300

# Run service
python main2.py --serve

# Query (stays loaded for 5 minutes after last use)
python main2.py "your prompt here"
```

### Mode 4: Dummy Model Swap (`main3.py`)
- **Best for:** VRAM-sensitive workflows (video editing, 3D rendering)
- **Behavior:** Swaps to tiny dummy model when idle to free VRAM
- **Memory:** Smart VRAM management - frees GPU memory while staying ready
- **Startup:** Medium - reloads from dummy state
- **Use case:** Creative work where you need VRAM but also want AI available

```bash
# Configure
python main3.py --config -m /path/to/model.onnx --timeout 180

# Run service (swaps to dummy when idle)
python main3.py --serve

# Query
python main3.py "your prompt here"
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install onnxruntime numpy onnx
```

### Basic Usage
1. **Configure** (one time):
   ```bash
   python main2.py --config -m /path/to/your/model.onnx --threads 4
   ```

2. **Start service** (in background):
   ```bash
   python main2.py --serve &
   ```

3. **Use it**:
   ```bash
   python main2.py "What is the weather like?"
   ```

### Force CPU Mode
```bash
python main2.py --config -m model.onnx --cpu
```

## 🏗️ Architecture Notes

### Current Limitations (By Design)
- ✅ Single model at a time
- ✅ No inter-process awareness
- ✅ Basic Unix socket communication
- ✅ Simple ONNX inference (placeholder logic)
- ✅ Minimal error handling
- ✅ **ONNX-only** - Real NeuroShellOS needs multiple runtimes
- ✅ **CUDA/CPU only** - Real NeuroShellOS needs multi-hardware support

**Important Note on Hardware Support:**  
These basic designs only support **CUDA (NVIDIA) and CPU** execution. However, NeuroShellOS must support diverse hardware:
- **NVIDIA GPUs** (CUDA, TensorRT)
- **AMD GPUs** (ROCm, HIP)
- **Intel GPUs** (OneAPI, Level Zero)
- **Apple Silicon** (Metal, ANE)
- **ARM NPUs** (Qualcomm, MediaTek)
- **Intel NPUs** (VPU, GNA)
- **Google TPUs** (Edge TPU, Cloud TPU)
- **Specialized AI accelerators** (Groq, Cerebras, etc.)

**Since NeuroShellOS is Linux-based**, the OS will intelligently detect available hardware at runtime and choose appropriate drivers, runtimes, and execution providers. For example:
- Desktop with NVIDIA GPU → CUDA + TensorRT
- Laptop with Intel integrated GPU → OneAPI + OpenVINO
- ARM SBC with NPU → ARM Compute Library + appropriate NPU drivers
- Server with AMD GPU → ROCm + MIOpen
- No accelerator detected → Optimized CPU inference

The orchestration layer must query hardware capabilities and automatically select the best execution path without user configuration.

**Important Note on Runtime Support:**  
NeuroShellOS will support **multiple AI runtimes** beyond ONNX, including:
- **ONNX Runtime** (cross-platform compatibility)
- **PyTorch** (research and flexibility)
- **TensorFlow/TFLite** (mobile and edge deployment)
- **llama.cpp** (efficient CPU inference for LLMs)
- **vLLM** (optimized LLM serving)
- **TensorRT** (NVIDIA GPU optimization)
- **OpenVINO** (Intel hardware acceleration)

Each runtime has different strengths (speed, memory, hardware support, model compatibility). A production NeuroShellOS orchestration layer needs settings and management for all these runtimes, with automatic selection based on:
- Available hardware
- Model format
- Performance requirements
- Memory constraints
- User preferences

This basic design demonstrates the orchestration *pattern* - the actual implementation would be significantly more complex to handle multiple runtimes and hardware platforms simultaneously.

### Future NeuroShellOS Needs
- ⏳ Multi-model coordination
- ⏳ Process priority management
- ⏳ Security sandboxing
- ⏳ Privacy-preserving inference
- ⏳ Advanced caching strategies
- ⏳ Resource quotas per user/application
- ⏳ Distributed model orchestration

## 🎮 Distribution-Specific Optimizations

Different NeuroShellOS distributions will **deeply optimize and favor specific modes** based on their target use case, but **all modes remain available** for users who want different behavior. The prioritized modes receive more development focus, testing, and performance tuning for that distribution.

| Distribution | Deeply Optimized Mode | Reason |
|--------------|----------------------|--------|
| **Gaming Edition** | Mode 1 (Dynamic) | Maximum RAM/VRAM for games |
| **Developer Edition** | Mode 2 (Persistent) | Fast iteration, always-on AI tools |
| **Creator Edition** | Mode 4 (Dummy Swap) | VRAM for rendering, AI on-demand |
| **Standard Edition** | Mode 3 (Timeout) | Balanced for general use |
| **Minimal Edition** | Mode 1 (Dynamic) | Resource conservation |

**What "deeply optimized" means:**  
The favored mode for each distro gets extensive optimization - not just set as default. For example:

**Example: Gaming Distro Focus on Mode 1**  
Gaming Edition deeply optimizes Mode 1 (Dynamic Load/Unload) with:
- Automatic detection of game launches to immediately free all AI resources
- Sub-second cold-start times through aggressive preloading optimizations
- Priority scheduling that always favors gaming processes over AI processes
- Integration with game launchers (Steam, Lutris) to pre-emptively unload models
- Custom kernel patches for faster VRAM release
- Profiled model loading paths for common AI tasks gamers use

**Users can still choose any mode** - if a gamer wants Mode 2 (persistent) for streaming with AI overlays, they can configure it. The distro just makes its target mode work *extremely well* for that use case.

## 📝 Technical Details

### Communication
- Unix domain sockets (`~/.config/chat/chat.sock`)
- 4KB message buffer (expandable)
- 60-120s timeout for loading operations

**Security Note:**  
Currently, the Unix socket has **no permission restrictions**. This means:
- ✅ **Single-user systems**: Generally safe
- ⚠️ **Multi-user systems**: Any user on the system could connect to your AI service and send prompts

This basic design doesn't implement socket permissions (chmod/chown) or authentication. A production system must add:
- Proper file permissions (0600 for socket)
- User/group restrictions
- Authentication tokens
- Request validation

**Distribution-Specific Security Levels:**  
Different NeuroShellOS distributions will implement different security levels:

- **Standard Edition**: May have **relaxed restrictions in local user mode** - assumes single trusted user, focuses on convenience over strict isolation. May implement only pieces of the full security model.

- **Security-focused Distro**: Implements **all high-level security features** described in this README - full encryption, sandboxing, strict permissions, even for local single-user scenarios. This is where the most paranoid-level complexity lives.

- **Gaming/Minimal Editions**: Likely minimal security overhead to maximize performance.

**Standard Edition Philosophy:**  
Standard Edition is designed as "**all-in-one for normal users**" but carved down from the full security model. It includes:
- Basic permission checks
- Essential privacy protections
- Simple authentication
- **But NOT** the paranoid-level security of dedicated security distributions

Think of it as: Security Distro has the full fortress, Standard Edition has reasonable locks on doors, Gaming Edition has unlocked doors for speed.

**Important: ALL distributions still need thousands of lines of code**  
Even Gaming Edition or Minimal Edition needs thousands of lines - not just the Security-focused distro. Why? Because **each individual part is simple, but the overall system is complex**:

- Hardware detection: simple logic, but 50+ hardware types
- Runtime selection: simple per-runtime, but 7+ runtimes
- Memory management: simple concepts, but complex interactions
- Error handling: simple per-error, but hundreds of error paths
- Logging: simple writes, but complex filtering/rotation
- Configuration: simple parsing, but deep validation trees

**Example: Gaming Edition complexity**  
Even with minimal security, Gaming Edition needs thousands of lines for:
- Game launcher integrations (Steam, Lutris, Heroic, Bottles, etc.)
- Per-game profile management
- VRAM monitoring and preemptive unloading
- Anti-cheat compatibility (avoid AI processes triggering bans)
- Overlay integration (Discord, OBS, etc.)
- Performance profiling and auto-tuning
- Fallback chains for different GPU states

Each piece is simple. The combination is complex. **This is why every distro needs substantial code, just focused on different priorities.**

**The Security-Performance Challenge:**  
Even though this design doesn't address security, NeuroShellOS faces a critical challenge: **improving security often drops performance**. For example:
- Encryption adds latency and CPU overhead
- Permission checks slow down every request
- Sandboxing requires context switches
- Authentication adds round-trips

**Balancing Security + Performance requires:**
1. **Replacing IPC mechanisms**: Unix sockets may not be sufficient. Need to evaluate:
   - Shared memory (faster but needs careful synchronization)
   - io_uring (modern Linux async I/O)
   - eBPF-based communication (kernel-level filtering)
   - Custom ring buffers for zero-copy transfers

2. **Creating new communication systems** for specific areas:
   - Real-time inference paths (microsecond latency)
   - Batch processing paths (high throughput)
   - Control plane (can tolerate latency for security)

3. **Environment-aware automation**: The system must intelligently choose:
   - Trusted local process? → Skip encryption, use shared memory
   - Remote/untrusted request? → Full encryption + sandboxing
   - High-priority inference? → Fast path with minimal checks
   - Background task? → Full security validation

4. **Timing-aware security**: Security mechanisms that adapt based on:
   - Current system load
   - Inference urgency
   - Trust level of requester
   - Privacy sensitivity of data

**The goal:** Automated security decisions based on context - not one-size-fits-all. Maximum performance when safe, maximum security when needed, intelligent balance in between.

**Privacy Note:**  
This design has **no privacy protections** for model interactions. In NeuroShellOS, privacy is critical:
- User prompts could contain sensitive data
- Model responses might leak private information
- Conversation history needs protection
- Multi-user systems need isolation between users' AI sessions

Real implementations need careful privacy engineering:
- Encrypted communication channels
- Sandboxed model execution
- No logging of sensitive data
- Memory scrubbing after inference
- User data isolation

**Privacy and security are separate concerns** - both need deep attention in production NeuroShellOS.

### Configuration
- Stored in `~/.config/chat/config.ini`
- Per-user settings
- Thread count, device preference, timeout values

### Execution Providers
**Currently supported:**
1. **CUDA** (NVIDIA GPUs)
2. **CPU** (fallback or forced)

**NeuroShellOS will need to support:**
- AMD ROCm
- Intel OneAPI/OpenVINO
- Apple Metal/ANE
- ARM NPUs
- Various AI accelerators

## 🔮 About NeuroShellOS

NeuroShellOS is an **open blueprint** for AI-integrated operating systems. The actual orchestration layer in a production NeuroShellOS could be **tens of thousands of lines of code per distribution** just for this loading/orchestration section alone, handling:

- Privacy: No AI hallucination leaking into user workflows, strict data isolation
- Security: Sandboxed model execution, permission controls
- Performance: Smart scheduling across CPU/GPU/NPU
- Compatibility: Multiple model formats beyond ONNX
- Intelligence: Self-optimizing based on usage patterns

**Why tens of thousands of lines for just loading/orchestration?**  
When privacy becomes strict, complexity explodes:
- Every memory allocation needs tracking for scrubbing
- Inter-process communication requires encryption
- Model outputs need validation before reaching users
- Conversation history needs secure storage with access controls
- Multi-user isolation requires careful state management
- Audit logging without storing sensitive content
- Graceful handling of privacy violations

Privacy strictness transforms simple loading logic into complex privacy-preserving systems.

These designs are **foundational patterns** - starting points for the community to build upon.

## 🤝 Contributing

This is a **foundational design repository** for future NeuroShellOS development. The design and philosophy here are intentionally forward-looking.

**How to contribute to NeuroShellOS:**

Since NeuroShellOS is an **open blueprint** with concepts distributed across multiple articles and repositories, you can contribute by:

- **Building actual components** based on NeuroShellOS concepts
- **Implementing parts of the blueprint** in your own repositories
- **Writing articles** explaining specific NeuroShellOS subsystems
- **Creating proof-of-concepts** for different orchestration approaches
- **Documenting real-world testing** of these patterns
- **Sharing experience reports** from attempting to build parts of the vision
- **Proposing new blueprint sections** for areas not yet covered

The NeuroShellOS blueprint is distributed - not centralized. Contributions happen everywhere people build pieces of the vision.

**Your Contributions = Your Ownership**

Everything you build, implement, or create is **yours**:
- ✅ You own your code and implementations
- ✅ You can use any license you want for your code
- ✅ You can use the NeuroShellOS name and blueprint concepts freely
- ✅ You can commercialize, fork, modify, or do anything with your work
- ✅ No permission needed, no approval required

**Blueprint Licensing:**  
The NeuroShellOS blueprint itself is under **CC BY-SA 4.0** (Creative Commons Attribution-ShareAlike 4.0). This means:
- The *concepts and designs* are freely shareable with attribution
- Derivative blueprints must also be CC BY-SA 4.0
- **BUT your code implementations can use ANY license** - Apache 2.0, MIT, GPL, proprietary, whatever you choose

**Fully Open Source Supportive:**  
This project is built on the philosophy that:
- Ideas should be free and shared
- Implementations belong to their creators
- The community benefits when everyone can build without restrictions
- Open source thrives when people have ownership and freedom

Build what you want. Own what you build. Share what you learn.

## ⚖️ License

**Blueprint:** CC BY-SA 4.0 (Creative Commons Attribution-ShareAlike 4.0 International)  
**Code in this repository:** Apache 2.0

The NeuroShellOS blueprint concepts, designs, and documentation are licensed under CC BY-SA 4.0, allowing free sharing and adaptation with attribution. The actual code implementations in this repository use Apache 2.0.

**Your implementations based on this blueprint can use any license you choose.**

## 📖 Development Context

These orchestration patterns were developed as foundational blueprints for the NeuroShellOS project. The designs focus on establishing core architectural patterns that can be expanded based on specific use cases and hardware configurations. While ONNX Runtime is used here as an example inference engine, the patterns are designed to be runtime-agnostic and can be adapted for any AI/ML execution framework.

The simplicity of these designs is intentional - they demonstrate the essential orchestration logic without the complexity of production features like security sandboxing, multi-model coordination, or advanced resource management. This makes them ideal as learning materials and starting points for more sophisticated implementations.

As the NeuroShellOS blueprint evolves, these patterns will serve as references for understanding how different orchestration strategies (persistent vs. dynamic loading, timeout-based management, resource swapping) affect system performance and user experience across different workflows.

---

**Status:** Foundational design phase - not for production use  
