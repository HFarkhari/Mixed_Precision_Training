This repository contains code for training BERT (Bidirectional Encoder Representations from Transformers) in MLM (Masked Language Modeling) and NSP (Next Sentence Prediction) tasks, as presented by James Briggs in his tutorial at [here](https://youtube.com/playlist?list=PLIUOU7oqGTLgQ7tCdDT0ARlRoh1127NSO).

In this forked version, we have adapted the training process to explore different scenarios, including float point 32, half precision (fp16), and brain float 16, to assess the performance and efficiency of each technique. Furthermore, we integrated the compile method and helper functions provided by NVidia Developer in their tutorial, available [here](https://youtube.com/playlist?list=PL5B692fm6--vi9vC5EDBFsfTBnrvVbl40), to optimize the training process further.

Additionally, this repository integrates the powerful SophiaG optimizer into our codebase, allowing us to benefit from its advanced features. The Sophia.py script from [here](https://github.com/Liuhong99/Sophia) is utilized for seamless integration and evaluation of the optimizer's performance in our projects. SophiaG is known for its efficiency and optimization capabilities, making it an excellent addition to our development workflow.

By comparing the different training techniques and incorporating SophiaG, we aim to achieve superior model performance while minimizing the training time and resource requirements.

We extend our appreciation to James Briggs for his original tutorial, NVidia Developer for their valuable toturials in mixed precision training, and the SophiaG developers for their remarkable optimizer. Together, these contributions empower us to push the boundaries of BERT training and natural language processing in mixed precision training and speed up training process.
