import tensorflow as tf
import psutil


def check_system_resources():
    """Check and print system resources (CPU cores and memory)"""
    # CPU Information
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)

    print("\nCPU Resources:")
    print(f"  Physical CPU cores: {physical_cores}")
    print(f"  Logical CPU cores: {logical_cores}")

    # Memory Information
    memory = psutil.virtual_memory()
    print("\nMemory Resources:")
    print(f"  Total Memory: {memory.total / (1024**3):.2f} GB")
    print(f"  Available Memory: {memory.available / (1024**3):.2f} GB")

    # TensorFlow Threading Information
    print("\nTensorFlow Threading:")
    print(
        f"  Inter-op parallelism threads: {tf.config.threading.get_inter_op_parallelism_threads()}"
    )
    print(
        f"  Intra-op parallelism threads: {tf.config.threading.get_intra_op_parallelism_threads()}"
    )

    return logical_cores


def check_gpu():
    """Check GPU availability and configuration"""
    print("TensorFlow version:", tf.__version__)

    # List physical devices
    print("\nPhysical devices:")
    physical_devices = tf.config.list_physical_devices()
    for device in physical_devices:
        print(f"  {device.device_type}: {device.name}")

    # Check specifically for GPUs
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"\nFound {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  {gpu.name}")

        # Configure memory growth
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("\nGPU memory growth enabled")
        except RuntimeError as e:
            print(f"\nError configuring GPU: {e}")
    else:
        print("\nNo GPUs found. Running on CPU")

    return bool(gpus)
