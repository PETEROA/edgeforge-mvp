from fastapi import APIRouter
from app.models.schemas import DeviceProfile, DeviceProfileList

router = APIRouter(prefix="/v1/devices", tags=["Device Profiles"])

# Pre-built device profiles
DEVICE_PROFILES = {
    "android-low": DeviceProfile(
        id="android-low",
        name="Android Low-End",
        cpu="ARM Cortex-A53, 4-core",
        ram_mb=2048,
        gpu=None,
        target_runtime="tflite",
        max_model_size_mb=50,
        description="Budget Android phones (1-2 GB RAM). Target: TFLite / ONNX Runtime Mobile.",
    ),
    "android-mid": DeviceProfile(
        id="android-mid",
        name="Android Mid-Range",
        cpu="Snapdragon 680, 8-core",
        ram_mb=6144,
        gpu="Adreno 610",
        target_runtime="tflite",
        max_model_size_mb=200,
        description="Mid-range Android (4-6 GB RAM). Target: TFLite / NNAPI.",
    ),
    "rpi4": DeviceProfile(
        id="rpi4",
        name="Raspberry Pi 4",
        cpu="BCM2711, 4-core ARM Cortex-A72",
        ram_mb=4096,
        gpu=None,
        target_runtime="onnx",
        max_model_size_mb=500,
        description="Raspberry Pi 4 (2-8 GB RAM). Target: ONNX Runtime / TFLite.",
    ),
    "rpi5": DeviceProfile(
        id="rpi5",
        name="Raspberry Pi 5",
        cpu="BCM2712, 4-core ARM Cortex-A76",
        ram_mb=8192,
        gpu=None,
        target_runtime="onnx",
        max_model_size_mb=1000,
        description="Raspberry Pi 5 (4-8 GB RAM). Target: ONNX Runtime.",
    ),
    "jetson-nano": DeviceProfile(
        id="jetson-nano",
        name="NVIDIA Jetson Nano",
        cpu="ARM Cortex-A57, 4-core",
        ram_mb=4096,
        gpu="128-core Maxwell",
        target_runtime="tensorrt",
        max_model_size_mb=500,
        description="NVIDIA Jetson Nano (4 GB). Target: TensorRT.",
    ),
    "jetson-orin": DeviceProfile(
        id="jetson-orin",
        name="NVIDIA Jetson Orin Nano",
        cpu="ARM Cortex-A78AE, 6-core",
        ram_mb=8192,
        gpu="1024-core Ampere",
        target_runtime="tensorrt",
        max_model_size_mb=2000,
        description="Jetson Orin Nano (8 GB). Target: TensorRT.",
    ),
    "edge-server": DeviceProfile(
        id="edge-server",
        name="Generic Edge Server",
        cpu="x86_64, 4-8 core",
        ram_mb=16384,
        gpu="Optional T4/L4",
        target_runtime="onnx",
        max_model_size_mb=5000,
        description="Edge server with optional GPU. Target: ONNX Runtime / TensorRT.",
    ),
    "browser-wasm": DeviceProfile(
        id="browser-wasm",
        name="Browser (WASM)",
        cpu="Variable",
        ram_mb=2048,
        gpu=None,
        target_runtime="onnx-web",
        max_model_size_mb=50,
        description="In-browser inference via WebAssembly. Target: ONNX Runtime Web.",
    ),
    "ios-coreml": DeviceProfile(
        id="ios-coreml",
        name="iOS (CoreML)",
        cpu="Apple A14+",
        ram_mb=4096,
        gpu="Neural Engine",
        target_runtime="coreml",
        max_model_size_mb=500,
        description="iOS devices (A14+). Target: Core ML.",
    ),
    "tinyml": DeviceProfile(
        id="tinyml",
        name="Microcontroller (TinyML)",
        cpu="ARM Cortex-M4/M7",
        ram_mb=1,
        gpu=None,
        target_runtime="tflite-micro",
        max_model_size_mb=1,
        description="Microcontrollers (256 KB-1 MB RAM). Target: TFLite Micro.",
    ),
}


@router.get("/", response_model=DeviceProfileList)
def list_device_profiles():
    """List all available device profiles."""
    return DeviceProfileList(profiles=list(DEVICE_PROFILES.values()))


@router.get("/{profile_id}", response_model=DeviceProfile)
def get_device_profile(profile_id: str):
    """Get details of a specific device profile."""
    profile = DEVICE_PROFILES.get(profile_id)
    if not profile:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Device profile '{profile_id}' not found")
    return profile
