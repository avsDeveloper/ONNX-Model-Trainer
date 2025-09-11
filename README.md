This app is primarily written by AI, intended to train AI models using AI-generated datasets. These models will be used by AI-based apps :)

Use it at your own risk, it won't break your PC, but I can guarantee nothing. I made it for myself, and I happily use it for my next cross-platform project.

The rest of the readme is written by AI...

==================

# 🤖 ONNX Model Trainer v0.8

A comprehensive GUI application for training, exporting, and testing transformer language models with full ONNX optimization support. Features timestamped session management, enhanced system monitoring, and interactive model testing capabilities.

<img width="1398" height="934" alt="Screenshot from 2025-07-29 14-31-54" src="https://github.com/user-attachments/assets/175568db-a46a-4bc9-9d3f-cb0ce307b911" /> <img width="1400" height="932" alt="Screenshot from 2025-07-29 14-35-16" src="https://github.com/user-attachments/assets/bcf19185-d31d-48e7-a98a-59bd168d2700" />



## ✨ Key Features

### 🎯 Complete Training Pipeline
- **Model Training** - Train GPT-2, DialoGPT, GPT-Neo, OPT, Phi, and other transformer models
- **ONNX Export** - Export trained models to ONNX format with quantization options
- **Smart Presets** - 6 pre-configured training presets for different use cases
- **Device Optimization** - Automatic GPU/CPU detection with memory management
- **Timestamped Sessions** - Automatic session directories with `YYYY_MM_DD_HH_MM_SS_ModelName_Actions` format

### 🧪 Interactive Model Testing
- **6 Communication Modes** - Chat, text generation, Q&A, code completion, creative writing, batch processing
- **Terminal Interface** - Interactive chat with conversation history and special commands
- **Real-time Testing** - Test your models immediately after training
- **Smart Context Management** - Automatic conversation context optimization
- **Generation Controls** - Fine-tune temperature, top-p, repetition penalty, and more

### 💻 Advanced Interface
- **Enhanced System Check** - 6-phase background system validation with detailed logging
- **Tabbed Design** - Separate training and testing environments
- **Intelligent Validation** - Real-time configuration validation and suggestions
- **Comprehensive Logging** - Detailed training logs with progress tracking
- **Smart Stop Functionality** - Confirmation dialogs with immediate process termination
- **Model Information** - In-depth model specifications and recommendations

### 🔧 System Reliability
- **Background System Check** - Non-blocking 6-phase system validation
- **Immediate Window Closing** - App closes instantly when requested
- **Enhanced Error Handling** - Robust error recovery and user feedback
- **Session Preservation** - No more overwritten output directories

## 🚀 Quick Start

### Launch the Application
```bash
python3 trainer.py
```

The application will automatically perform a comprehensive 6-phase system check:
1. **Phase 1**: Python Environment Validation
2. **Phase 2**: Basic Dependencies Check
3. **Phase 3**: ML Dependencies Assessment (Optional)
4. **Phase 4**: Hardware Capabilities Detection
5. **Phase 5**: System Resources Evaluation
6. **Phase 6**: Model Loading Framework Test

## 📋 System Requirements

### Core Requirements
- **Python 3.7 or higher** (3.9+ recommended)
- **4GB RAM minimum** (8GB+ recommended for larger models)
- **5GB free disk space** (more for larger models and datasets)

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
psutil>=5.8.0
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

## 📦 Project Portability

The ONNX Model Trainer is designed to be **fully portable** and can be easily moved between directories, devices, and systems:

### 🔄 Relative Path Architecture
- **Automatic Path Conversion**: All file dialogs automatically convert absolute paths to relative paths when files are within the project directory
- **Relative Defaults**: Default paths use relative notation (`./output`, `./dataset.json`)
- **Cross-Platform Compatibility**: Works seamlessly across Linux, Windows, and macOS
- **No Hardcoded Paths**: No absolute paths are stored in configuration files

### 📁 Safe Directory Structure
```
onnx-model-trainer/
├── trainer.py           # Main application
├── dataset.json         # Sample training data
├── README.md           # Documentation
└── output/             # Generated models (relative paths)
    └── ModelName/
        ├── 1_trained/   # PyTorch models
        ├── 2_converted/ # ONNX models  
        └── 3_quantized/ # Optimized models
```

### 🚀 Easy Migration
1. **Copy entire folder** to new location/device
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Run**: `python3 trainer.py` - everything works immediately!

*No configuration changes, no path fixes, no reinstallation required.*

## 📖 User Manual

### Main Interface Overview

The ONNX Model Trainer features a tabbed interface with two main sections:

#### **Training & Export Tab**
- **Left Panel**: Model selection, dataset configuration, training parameters
- **Right Panel**: Real-time training logs with 6-phase system check progress
- **Bottom**: Enhanced control buttons with confirmation dialogs

#### **Model Testing Tab**
- **Left Panel**: Testing parameters, model selection, and hardware information
- **Right Panel**: Interactive terminal interface with 6 communication modes
- **Bottom**: Generation control buttons with smart stopping

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

#### 4. Enhanced Start Training
1. Click **"Start Training"** to begin
2. **Enhanced Stop**: Confirmation dialog with operation-specific messages
3. **Immediate Termination**: Stop button now actually terminates processes
4. Monitor progress in real-time logs with detailed phase information
5. **Timestamped Output**: Models saved to unique session directories

### Session Management

#### Timestamped Directory Structure
Every training/conversion session creates a unique directory:
```
output/
├── 2025_01_29_14_30_45_distilgpt2_Training_Export_Quantization/
│   ├── 1_trained/              # Training output
│   ├── 2_converted/            # ONNX export
│   └── 3_quantized/            # Quantized models
├── 2025_01_29_15_15_20_gpt2_Training/
│   └── 1_trained/              # Training-only session
└── 2025_01_29_16_00_10_phi-1_Export_Quantization/
    ├── 2_converted/            # Export-only session
    └── 3_quantized/
```

**Benefits**:
- ✅ No more overwritten training results
- ✅ Clear session identification with timestamp and actions
- ✅ Easy tracking of different experiments
- ✅ Preserved training history

### Model Testing Workflow

#### 1. Model Selection
- **Browse Models**: Select any ONNX model file
- **Quick Select**: Choose from timestamped session outputs
- **Auto-refresh**: Automatically detect new models from recent sessions

#### 2. Communication Modes
Choose from **6 specialized modes** with optimized parameters:

- **Chat Conversation**: Interactive dialogue with memory management
  - Terminal interface with `chat>` prompt
  - Special commands: `clear`, `status`, `help`
  - Automatic context optimization
  - Conversation history preservation

- **Text Generation**: Single-shot text completion
  - Terminal interface with `text>` prompt
  - Direct prompt processing

- **Question Answering**: Factual Q&A responses
  - Terminal interface with `ask>` prompt
  - Optimized for accurate, focused answers

- **Code Completion**: Programming assistance
  - Terminal interface with `code>` prompt
  - Optimized for code generation

- **Creative Writing**: Story and creative content generation
  - Terminal interface with `story>` prompt
  - Enhanced creativity parameters

- **Batch Processing**: Process multiple prompts at once
  - Terminal interface with `batch>` prompt
  - Enter multiple prompts, type `EXECUTE` to process

#### 3. Terminal Interface Features
**Interactive Chat Experience**:
- Type directly in the terminal window
- Press Enter to send messages
- Real-time response generation with thinking indicators
- Smart response cleaning and formatting

**Special Commands** (available in all modes):
- `clear` or `reset` - Clear conversation history
- `status` or `info` - Show conversation statistics
- `help` or `?` - Display available commands

**Smart Context Management**:
- Automatic conversation length monitoring
- Dynamic context reset when approaching limits
- Intelligent conversation pruning
- Memory optimization for long chats

#### 4. Generation Parameters
Fine-tune model behavior with mode-optimized defaults:
- **Temperature**: Creativity vs consistency (0.1-2.0)
- **Top-p**: Nucleus sampling for quality control
- **Top-k**: Vocabulary filtering
- **Repetition Penalty**: Avoid repetitive outputs
- **Length Controls**: Min/max response length

### System Check Process

#### 6-Phase Background Validation
The enhanced system check runs independently in the background:

**Phase 1: Python Environment**
- Python version compatibility check
- System architecture validation

**Phase 2: Basic Dependencies**
- Critical module availability (tkinter, threading, json, pathlib, os, sys, time)
- Import verification for each dependency

**Phase 3: ML Dependencies (Optional)**
- PyTorch, Transformers, Datasets availability
- Graceful degradation to basic mode if missing

**Phase 4: Hardware Capabilities**
- CUDA GPU detection and VRAM measurement
- Multi-GPU system support
- CPU fallback confirmation

**Phase 5: System Resources**
- Disk space evaluation (5GB+ recommended)
- System memory assessment
- Performance optimization suggestions

**Phase 6: Model Loading Test**
- Tokenizer framework verification
- Model loading capability confirmation

**Features**:
- ✅ Non-blocking background execution
- ✅ Detailed logging for each phase
- ✅ Cancellation support between phases
- ✅ Graceful error handling and fallbacks
- ✅ Clear progress indication

## 📊 Output Structure

After training, your timestamped session directory contains:

```
2025_01_29_14_30_45_gpt2_Training_Export_Quantization/
├── 1_trained/                  # Training output
│   ├── config.json            # Model configuration
│   ├── pytorch_model.bin      # Model weights
│   ├── tokenizer.json         # Tokenizer files
│   ├── training_args.json     # Training parameters
│   └── trainer_state.json     # Training state
├── 2_converted/               # ONNX export
│   ├── model.onnx            # ONNX model file
│   ├── config.json           # Configuration
│   └── tokenizer.json        # Tokenizer
├── 3_quantized/              # Quantized version
│   ├── model_quantized.onnx  # Quantized ONNX model
│   ├── config.json           # Configuration
│   └── tokenizer.json        # Tokenizer
└── logs/                     # Session logs
    └── training_log.txt      # Complete training log
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
    {"text": "Human: Hello
Assistant: Hi there! How can I help you today?"},
    {"text": "Human: What's the weather like?
Assistant: I don't have access to current weather data..."}
]
```

## 🛠 Troubleshooting

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

### System Check Issues
- **Phase 1 Failure**: Update Python to 3.7+
- **Phase 2 Failure**: Install missing basic dependencies
- **Phase 3 Warning**: Install ML dependencies for full functionality
- **Phase 4 Warning**: GPU not detected, CPU mode available
- **Phase 5 Warning**: Insufficient disk space or memory
- **Phase 6 Failure**: Reinstall transformers library

### Training Problems
- **Out of Memory**: Reduce batch size or use gradient checkpointing
- **Slow Training**: Check GPU utilization, enable mixed precision
- **Poor Results**: Increase training epochs, adjust learning rate
- **Dataset Errors**: Validate JSON format, check encoding (UTF-8)
- **Process Won't Stop**: Use the enhanced Stop button with confirmation

### Model Testing Issues
- **Generation Fails**: Verify ONNX model file integrity, check device selection
- **Slow Inference**: Check device selection (CPU vs GPU)
- **Poor Quality**: Adjust temperature and generation parameters
- **Context Lost**: Use conversation mode, check context management
- **Terminal Not Responding**: Use special commands like `clear` or `help`

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
- **Use Timestamped Sessions**: Each run creates unique directories
- **Enhanced Stopping**: Use confirmation dialogs to properly terminate

### Dataset Preparation
- **Quality over Quantity**: Clean, relevant data beats large noisy datasets
- **Balanced Conversations**: Mix question types and response lengths
- **Encoding**: Ensure UTF-8 encoding for special characters
- **Size Guidelines**: 1000+ conversations for fine-tuning

### Testing & Evaluation
- **Multiple Modes**: Test your model in different communication modes
- **Parameter Tuning**: Experiment with temperature and sampling settings
- **Conversation Flow**: Test multi-turn dialogues for coherence
- **Terminal Commands**: Use `clear`, `status`, and `help` for better control
- **Context Management**: Monitor conversation length in chat mode

### System Management
- **System Check**: Let the 6-phase check complete for optimal setup
- **Resource Monitoring**: Check Phase 5 results for system capabilities
- **Session Organization**: Use timestamped directories for experiment tracking
- **Immediate Closing**: App closes instantly when requested

### Deployment Considerations
- **ONNX Export**: Always export successful models to ONNX
- **Quantization**: Use quantized models for production deployment
- **Model Size**: Consider file size vs quality trade-offs
- **Platform Testing**: Verify ONNX models work on target deployment platforms

---

## 🎉 Ready to Start?

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Launch the trainer**: `python3 trainer.py`
3. **Watch system check**: Observe the 6-phase validation process
4. **Select a model**: Start with DistilGPT-2 for quick testing
5. **Load your dataset**: Use the included sample or your own data
6. **Start training**: Choose "Standard" preset for balanced results
7. **Test your model**: Switch to Testing tab and chat with your trained model
8. **Use terminal interface**: Try different communication modes and special commands

**Happy training!** 🚀

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
    {"text": "Human: Hello
Assistant: Hi there! How can I help you today?"},
    {"text": "Human: What's the weather like?
Assistant: I don't have access to current weather data..."}
]
```

## 🛠 Troubleshooting

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
