# Image_Captioning_model
This repository implements an end-to-end image captioning system using a Transformer-based Vision–Language architecture. The project focuses on building a custom self-tuned ViT-GPT2 model that generates meaningful natural-language descriptions for images with limited training data.



Training Strategy
Pretrained model: nlpconnect/vit-gpt2-image-captioning
Encoder frozen; decoder fine-tuned for 8 epochs
Optimizer: AdamW
Loss function: Cross-Entropy Loss
Dataset: Mini-COCO 2014
Training performed on GPU with mixed-precision support



Performance Evaluation
The fine-tuned model was evaluated using standard Natural Language Generation metrics:
METEOR: 0.3420 (semantic similarity)
ROUGE-L: 0.3403 (structural similarity)
CIDEr: Not computed due to evaluation constraints
These results indicate that the model effectively captures visual semantics and produces grammatically coherent captions despite limited training data.



Deployment
The final model is deployed using Gradio and hosted on Hugging Face Spaces, enabling real-time image caption generation via a simple web interface.
🔗 Live Demo:
https://huggingface.co/spaces/Rushiparhad/image-caption-demo
