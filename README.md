
This app is primarily written by AI, intended to train AI models using AI-generated datasets. These models will be used by AI-based apps :)

Use it at your own risk, it won't break your PC, but I can guarantee nothing. I made it for myself, and I happily use it for my next cross-platform project.

The rest of the readme is written by AI...

==================

# 🤖 ONNX Model Trainer

A comprehensive GUI application for training, exporting, and testing transformer language models with full ONNX optimization support. Train models locally and export them for optimized deployment across different platforms.

## ✨ Key Features

### 🎯 Complete Training Pipeline
- **Model Training** - Train GPT-2, DialoGPT, GPT-Neo, OPT, Phi, and other transformer models
- **ONNX Export** - Export trained models to ONNX format with quantization options
- **Smart Presets** - Pre-configured training presets for different use cases
- **Device Optimization** - Automatic GPU/CPU detection with memory management

### 🧪 Interactive Model Testing
- **Real-time Testing** - Test your models immediately after training
- **Multiple Communication Modes** - Chat, Q&A, code completion, creative writing
- **Terminal Interface** - Interactive chat with conversation history
- **Generation Controls** - Fine-tune temperature, top-p, repetition penalty, and more

### 💻 Advanced Interface
- **Tabbed Design** - Separate training and testing environments
- **Intelligent Validation** - Real-time configuration validation and suggestions
- **Comprehensive Logging** - Detailed training logs with progress tracking
- **Model Information** - In-depth model specifications and recommendations

## 🚀 Quick Start

### Launch the Application
```bash
python3 trainer.py
```

The application will automatically perform a system check and enable features based on available dependencies.

## 📋 System Requirements

### Core Requirements
- **Python 3.8 or higher** (3.9+ recommended)
- **4GB RAM minimum** (8GB+ recommended for larger models)
- **2GB free disk space** (more for larger models and datasets)

### Required Python Packages
Install with: `pip install -r requirements.txt`

```
transformers>=4.21.0
torch>=1.12.0
datasets>=2.0.0
onnx>=1.12.0
onnxruntime>=1.12.0
optimum[onnxruntime]>=1.9.0
numpy>=1.21.0
```

### Optional (but recommended)
- **CUDA-capable GPU** - For faster training and inference
- **16GB+ RAM** - For training larger models (GPT-2 Large/XL, etc.)

## 🔧 Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd onnx-model-trainer
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the trainer**
   ```bash
   python3 trainer.py
   ```

## 📖 User Manual

### Main Interface Overview

The ONNX Model Trainer features a tabbed interface with two main sections:

#### **Training & Export Tab**
- **Left Panel**: Configuration and settings
- **Right Panel**: Real-time training logs and progress
- **Bottom**: Training control buttons

#### **Model Testing Tab**
- **Left Panel**: Testing parameters and model selection
- **Right Panel**: Interactive terminal for model communication
- **Bottom**: Generation control buttons

### Training Workflow

#### 1. Model Configuration
1. **Select Base Model**: Choose from 15+ pre-configured models:
   - **GPT-2 variants**: distilgpt2, gpt2, gpt2-medium, gpt2-large, gpt2-xl
   - **DialoGPT models**: Small, Medium, Large conversational models
   - **GPT-Neo**: 125M and 1.3B parameter models
   - **OPT models**: Meta's Open Pre-trained Transformers
   - **Microsoft Phi**: Compact, efficient models for code and reasoning

2. **Dataset Selection**: Load your training dataset (JSON format)
3. **Output Directory**: Choose where to save trained models
4. **Actions**: Select what to perform:
   - ✅ **Train**: Fine-tune the model on your dataset
   - ✅ **Export**: Convert to ONNX format
   - ✅ **Quantize**: Create compressed model versions

#### 2. Training Parameters
Choose from **6 training presets** or customize manually:

- **Quick Test**: 1 epoch, minimal settings for testing
- **Conservative**: Safe settings with low learning rate
- **Standard**: Balanced configuration (recommended)
- **High Quality**: Longer training with cosine scheduling
- **Fast**: Quick training with larger batches
- **Fine-tuning**: Gentle settings for pre-trained models

**Manual Parameters**:
- Epochs, batch size, learning rate
- Max sequence length, save/warmup steps
- Learning rate scheduler, gradient clipping
- Weight decay, advanced optimization settings

#### 3. Hardware Configuration
- **Automatic Device Detection**: GPU/CPU with memory information
- **Training Device Selection**: Choose optimal device for training
- **Memory Management**: Built-in optimization for resource constraints
- **Performance Tips**: Real-time memory usage and recommendations

#### 4. Start Training
1. Click **"Start Training"** to begin
2. Monitor progress in real-time logs
3. View detailed metrics, loss curves, and step information
4. Models are automatically saved with checkpoints

### Model Testing Workflow

#### 1. Model Selection
- **Browse Models**: Select any ONNX model file
- **Quick Select**: Choose from recently trained models
- **Auto-refresh**: Automatically detect new models in output directory

#### 2. Communication Modes
Choose from **6 specialized modes**:

- **Chat Conversation**: Interactive dialogue with memory
- **Text Generation**: Single-shot text completion
- **Question Answering**: Factual Q&A responses
- **Code Completion**: Programming assistance
- **Creative Writing**: Story and creative content generation
- **Batch Processing**: Process multiple prompts at once

#### 3. Generation Parameters
Fine-tune model behavior:
- **Temperature**: Creativity vs consistency (0.1-2.0)
- **Top-p**: Nucleus sampling for quality control
- **Top-k**: Vocabulary filtering
- **Repetition Penalty**: Avoid repetitive outputs
- **Length Controls**: Min/max response length

#### 4. Interactive Testing
- **Terminal Interface**: Chat naturally with your model
- **Conversation Memory**: Maintains context across exchanges
- **Special Commands**: 
  - `clear` - Reset conversation
  - `status` - Show conversation info
  - `help` - Display available commands

## 📊 Output Structure

After training, your output directory contains:

```
output/
├── [model-name]/
│   ├── pytorch/                 # Original PyTorch model
│   │   ├── config.json         # Model configuration
│   │   ├── pytorch_model.bin   # Model weights
│   │   ├── tokenizer.json      # Tokenizer files
│   │   └── training_args.json  # Training parameters
│   ├── onnx/                   # ONNX export
│   │   ├── model.onnx         # ONNX model file
│   │   ├── model_quantized.onnx # Quantized version
│   │   └── export_info.json   # Export metadata
│   ├── checkpoints/           # Training checkpoints
│   │   ├── checkpoint-500/
│   │   ├── checkpoint-1000/
│   │   └── ...
│   └── logs/                  # Training logs
│       ├── training_log.txt
│       └── progress_history.json
```

## 🎯 Model Specifications

### Supported Models Overview

| Model | Parameters | PyTorch Size | ONNX Size | Quantized | Context | Best For |
|-------|------------|--------------|-----------|-----------|---------|----------|
| DistilGPT-2 | 82M | 331 MB | 248 MB | 83 MB | 1024 | Fast training, testing |
| GPT-2 | 124M | 498 MB | 374 MB | 125 MB | 1024 | General purpose |
| GPT-2 Medium | 355M | 1.42 GB | 1.07 GB | 356 MB | 1024 | Balanced quality/speed |
| DialoGPT Small | 117M | 468 MB | 351 MB | 117 MB | 1024 | Conversations |
| GPT-Neo 125M | 125M | 502 MB | 376 MB | 126 MB | 2048 | Open source alternative |
| OPT 125M | 125M | 501 MB | 376 MB | 125 MB | 2048 | Meta's optimization |
| Phi 1 | 1.3B | 2.84 GB | 2.13 GB | 713 MB | 2048 | Code generation |
| Phi 1.5 | 1.3B | 5.20 GB | 3.90 GB | 1.30 GB | 2048 | Technical content |

### Recommended Settings by Model Size

**Small Models (≤200M parameters)**:
- Batch size: 4-16
- Learning rate: 1e-4 to 5e-5
- Memory required: 2-4GB

**Medium Models (200M-500M)**:
- Batch size: 2-8
- Learning rate: 3e-5 to 2e-5
- Memory required: 4-8GB

**Large Models (500M+)**:
- Batch size: 1-4
- Learning rate: 1e-5 to 5e-6
- Memory required: 8-16GB+

## 🔧 Dataset Format

### Standard Conversation Format
```json
[
    {
        "conversations": [
            {"from": "human", "value": "What is machine learning?"},
            {"from": "assistant", "value": "Machine learning is a subset of artificial intelligence..."}
        ]
    },
    {
        "conversations": [
            {"from": "human", "value": "Can you explain neural networks?"},
            {"from": "assistant", "value": "Neural networks are computational models inspired by the brain..."}
        ]
    }
]
```

### Alternative Formats (Auto-detected)
```json
[
    {"text": "Human: Hello\nAssistant: Hi there! How can I help you today?"},
    {"text": "Human: What's the weather like?\nAssistant: I don't have access to current weather data..."}
]
```

## � Troubleshooting

### Installation Issues
```bash
# If torch installation fails
pip install torch --index-url https://download.pytorch.org/whl/cpu

# If ONNX dependencies fail
pip install onnx onnxruntime --no-deps
pip install protobuf==3.20.3

# If GUI doesn't appear
sudo apt-get install python3-tk  # Ubuntu/Debian
```

### Training Problems
- **Out of Memory**: Reduce batch size or use gradient checkpointing
- **Slow Training**: Check GPU utilization, enable mixed precision
- **Poor Results**: Increase training epochs, adjust learning rate
- **Dataset Errors**: Validate JSON format, check encoding (UTF-8)

### Model Testing Issues
- **Generation Fails**: Verify ONNX model file integrity
- **Slow Inference**: Check device selection (CPU vs GPU)
- **Poor Quality**: Adjust temperature and generation parameters
- **Context Lost**: Ensure conversation mode is selected

### Performance Optimization
- **GPU Training**: Install CUDA-compatible PyTorch
- **Memory Usage**: Enable gradient checkpointing and CPU offload
- **Storage**: Use SSD for faster model loading
- **Network**: Pre-download models to avoid training interruptions

## 💡 Tips & Best Practices

### Training Tips
- **Start Small**: Use DistilGPT-2 for initial experiments
- **Validate Early**: Test with 1 epoch before full training
- **Monitor Memory**: Watch GPU/CPU usage during training
- **Save Frequently**: Use smaller save_steps for important runs

### Dataset Preparation
- **Quality over Quantity**: Clean, relevant data beats large noisy datasets
- **Balanced Conversations**: Mix question types and response lengths
- **Encoding**: Ensure UTF-8 encoding for special characters
- **Size Guidelines**: 1000+ conversations for fine-tuning

### Testing & Evaluation
- **Multiple Modes**: Test your model in different communication modes
- **Parameter Tuning**: Experiment with temperature and sampling settings
- **Conversation Flow**: Test multi-turn dialogues for coherence
- **Edge Cases**: Try unusual inputs to check model robustness

### Deployment Considerations
- **ONNX Export**: Always export successful models to ONNX
- **Quantization**: Use quantized models for production deployment
- **Model Size**: Consider file size vs quality trade-offs
- **Platform Testing**: Verify ONNX models work on target deployment platforms

---

## 🎉 Ready to Start?

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Launch the trainer**: `python3 trainer.py`
3. **Select a model**: Start with DistilGPT-2 for quick testing
4. **Load your dataset**: Use the included sample or your own data
5. **Start training**: Choose "Standard" preset for balanced results
6. **Test your model**: Switch to Testing tab and chat with your trained model

**Happy training!** 🚀
