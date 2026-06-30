# GPU Setup

Verify:

nvidia-smi

Expected:

RTX 4060

Install:

tensorflow[and-cuda]

Verify:

python -c "
import tensorflow as tf
print(
tf.config.list_physical_devices(
'GPU'
)
)
"

Expected:

GPU detected

Enable:

memory growth

Enable:

mixed precision

Never disable GPU.

Avoid CPU fallback.

Monitor:

watch -n 1 nvidia-smi