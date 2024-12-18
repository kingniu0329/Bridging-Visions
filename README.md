# Visions

Our dataset can be accessed at: [Dataset link](https://drive.google.com/file/d/1jH3nr4chumkU99D2l-rzHL70g2vDnQsn/view?usp=sharing).

### Environment

To run the training process, you'll need the following environment setup. You can easily create the necessary environment using the provided `environment.yml` file.

1. **Python**: Version 3.8 or higher.
2. **Dependencies**: The required libraries and versions are listed in the `environment.yml` file provided in this repository.

#### Steps to Set Up the Environment:

1. **Create the Conda Environment**:
   To create a new `conda` environment from the `environment.yml` file, run the following command:

   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the Environment**:
   Once the environment is created, activate it using:

   ```bash
   conda activate visions
   ```
3. **Verify Installation**:
   After activating the environment, you can check if the libraries were successfully installed by running:

   ```bash
   conda list
   ```

### Train

The training command is as follows:

```bash
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=8 train.py \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=100 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="checkpoint/train" \
  --grad_scale 0.001 \
  --checkpointing_steps 100
```

#### Parameters breakdown:
- `--multi_gpu`: Enable multi-GPU training.
- `--mixed_precision=fp16`: Use mixed precision training to speed up the training and reduce memory usage.
- `--num_processes=8`: The number of processes to launch, typically corresponding to the number of GPUs.
- `--use_ema`: Use Exponential Moving Average (EMA) of model weights to improve stability.
- `--resolution=512`: Set the image resolution to 512x512.
- `--center_crop --random_flip`: Apply data augmentation techniques (center crop and random horizontal flip).
- `--train_batch_size=2`: The batch size used for each GPU.
- `--gradient_accumulation_steps=4`: Gradient accumulation steps to simulate a larger batch size while keeping memory usage manageable.
- `--gradient_checkpointing`: Reduce memory usage by storing intermediate activations during the forward pass.
- `--max_train_steps=100`: The total number of training steps.
- `--learning_rate=1e-05`: Learning rate used for optimization.
- `--max_grad_norm=1`: Maximum gradient norm to prevent exploding gradients.
- `--lr_scheduler="constant"`: Use a constant learning rate (no decay).
- `--lr_warmup_steps=0`: Number of steps for learning rate warmup. Set to 0 for no warmup.
- `--output_dir="checkpoint/train"`: Directory to save model checkpoints.
- `--grad_scale 0.001`: Scaling factor for gradients during mixed-precision training.
- `--checkpointing_steps 100`: Save a checkpoint every 100 steps.

### Notes:
- The `accelerate` tool is used for efficient multi-GPU training, and the training command provided uses mixed-precision and gradient checkpointing for memory efficiency.
- Adjust the `train_batch_size`, `max_train_steps`, and other hyperparameters as needed based on your hardware and dataset.
- You can monitor the training progress via the output directory where checkpoints are saved. Modify the `--output_dir` parameter to store checkpoints in a different location if needed.

This setup should allow you to start training your model effectively on multiple GPUs.
