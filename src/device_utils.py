import torch


def detect_devices() -> tuple[str, torch.device, str]:
    """Detect the best available compute devices.

    Returns:
        yolo_device  — string passed to ultralytics (supports 'cuda', 'mps', 'cpu')
        lstm_device  — torch.device for PyTorch LSTM (supports more backends)
        description  — human-readable summary shown in the UI
    """
    # NVIDIA CUDA
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return "cuda", torch.device("cuda"), f"NVIDIA CUDA ({name})"

    # Apple Silicon MPS
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.device("mps"), "Apple MPS (Metal)"

    # Intel GPU — requires: pip install intel-extension-for-pytorch
    try:
        import intel_extension_for_pytorch  # noqa: F401
        if torch.xpu.is_available():
            return "cpu", torch.device("xpu"), "Intel XPU (LSTM) / CPU (YOLO)"
    except (ImportError, AttributeError):
        pass

    # AMD / Intel on Windows — requires: pip install torch-directml
    try:
        import torch_directml
        dml = torch_directml.device()
        gpu_name = torch_directml.device_name(0)
        return "cpu", dml, f"DirectML · {gpu_name} (LSTM) / CPU (YOLO)"
    except (ImportError, Exception):
        pass

    return "cpu", torch.device("cpu"), "CPU"
