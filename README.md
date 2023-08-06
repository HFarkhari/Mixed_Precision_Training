# Deep Learning with Mixed Precision Training

![Deep Learning with Mixed Precision Training](https://github.com/HFarkhari/Mixed_Precision_Training)

Welcome to the "Deep Learning with Mixed Precision Training" repository! Here, we provide comprehensive tutorials and code examples for training deep neural networks using mixed precision techniques in both PyTorch and TensorFlow.

## Contents

The repository is organized into the following sections:

### PyTorch_TensorFlow

The `PyTorch_TensorFlow` directory contains a set of base tutorials and code examples based on the "Mixed-Precision using Tensor Cores Series". These tutorials walk you through the fundamentals of mixed precision training in PyTorch and TensorFlow. They serve as an essential starting point for understanding the techniques used in subsequent sections.

### BERT_MLM_NSP_fp32_fp16

In the `BERT_MLM_NSP_fp32_fp16` directory, you will find various methods for training BERT (Bidirectional Encoder Representations from Transformers) models. This section covers training BERT models in both Masked Language Model (MLM) and Next Sentence Prediction (NSP) modes. Additionally, we demonstrate the combined training of MLM and NSP using a combination of techniques mentioned earlier. These techniques include mixed precision training with float32 (fp32) and float16 (fp16) precisions.

## Additional Resources

For more in-depth information on training BERT from scratch, you can find helpful tutorials from [James Briggs](https://youtube.com/playlist?list=PLIUOU7oqGTLgQ7tCdDT0ARlRoh1127NSO&feature=shared).


## Introduction

Deep learning models have achieved remarkable success in various fields. However, training these models can be computationally expensive and time-consuming. Mixed precision training offers an efficient solution by leveraging different numerical precisions for specific computations.

## Repository Focus

In this repository, we focus on the following aspects:

1. Training Deep Networks: We demonstrate how to train deep neural networks using mixed precision techniques in both PyTorch and TensorFlow.

2. Nvidia Helper Functions: We provide examples of leveraging Nvidia helper functions, as introduced in the [Mixed-Precision using Tensor Cores Series](https://youtube.com/playlist?list=PL5B692fm6--vi9vC5EDBFsfTBnrvVbl40&feature=shared), to further optimize performance on Nvidia GPUs with Tensor Cores.

3. Automatic Mixed Precision in PyTorch: We show how to use PyTorch's `torch.cuda.amp` package for automatic mixed precision training.

4. PyTorch 2.x Compile Method: We explore the compile method in PyTorch 2.x for mixed precision training.

5. Sophia Optimizer Comparison: We compare the performance in terms of training time while using the recent [Sophia optimizer](https://github.com/Liuhong99/Sophia).

6. Training BERT Model: We go beyond simple examples and train/fine-tune BERT models from scratch in MLM (Masked Language Model) and NSP (Next Sentence Prediction) modes. We compare GPU VRAM and training time requirements for each method.

7. Tensor Float 32 (TF32) Capability: We apply the methods on custom-designed deep networks to exploit the Tensor Float 32 capability in Nvidia Ampere GPUs.

9. Tips and Tricks: For each method, we share tips and tricks to achieve optimal results.



## Getting Started

To explore mixed precision training and train your deep neural networks more efficiently, follow these steps:

1. Clone this repository to your local machine using `git clone https://github.com/your_username/your_repository.git`.

2. Navigate to the repository folder and explore the `PyTorch_TensorFlow` directory for foundational tutorials.

3. Dive into the `BERT_MLM_NSP_fp32_fp16` directory to discover advanced methods for training BERT models in MLM and NSP modes using mixed precision techniques.

4. Run the provided scripts and notebooks in each directory to experiment with different precision settings and observe the training performance.

## Contribution Guidelines

We welcome contributions from the community to expand the repository with more examples and techniques. If you encounter any issues, have innovative ideas, or wish to optimize existing code, feel free to open an issue or submit a pull request. Together, let's accelerate the world of deep learning with mixed precision training!

## License

This repository is licensed under the [MIT License](LICENSE).

Let's unleash the power of mixed precision training and push the boundaries of deep learning! Happy coding!
