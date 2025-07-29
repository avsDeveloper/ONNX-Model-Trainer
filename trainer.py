#!/usr/bin/env python3
"""
ONNX Model Trainer - Streamlined GUI Application with Integrated System Checks
Training and exporting language models with comprehensive system validation
"""

# Standard library imports
import json
import logging
import os
import random
import shutil
import signal
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

# GUI imports
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from tkinter import messagebox as msgbox  # Alternative alias for msgbox usage

# Optional third-party imports (imported conditionally)
try:
    import psutil
except ImportError:
    psutil = None

try:
    import accelerate
except ImportError:
    accelerate = None

# ML dependencies - imported conditionally to handle missing packages gracefully
torch = None
transformers = None
datasets = None
onnxruntime = None
onnx = None
numpy = None
optimum = None

# Import aliases for commonly used classes (will be set when ML deps are available)
AutoTokenizer = None
AutoModelForCausalLM = None
AutoConfig = None
Trainer = None
TrainingArguments = None
DataCollatorForLanguageModeling = None
TrainerCallback = None
Dataset = None
file_utils = None
ORTModelForCausalLM = None
quantize_dynamic = None
QuantType = None

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ML dependency status
ML_DEPENDENCIES_AVAILABLE = None
ML_IMPORT_ERROR = None

def check_ml_dependencies():
    """Check and import ML dependencies when needed"""
    global ML_DEPENDENCIES_AVAILABLE, ML_IMPORT_ERROR
    global torch, transformers, datasets, onnxruntime, onnx, numpy, optimum
    global AutoTokenizer, AutoModelForCausalLM, AutoConfig, Trainer, TrainingArguments
    global DataCollatorForLanguageModeling, TrainerCallback, Dataset, file_utils
    global ORTModelForCausalLM, quantize_dynamic, QuantType
    
    if ML_DEPENDENCIES_AVAILABLE is None:
        try:
            # Import core ML libraries
            import torch as _torch
            import transformers as _transformers
            import datasets as _datasets
            import onnxruntime as _onnxruntime
            import numpy as _numpy
            
            # Assign to global variables
            torch = _torch
            transformers = _transformers
            datasets = _datasets
            onnxruntime = _onnxruntime
            numpy = _numpy
            
            # Import commonly used classes
            from transformers import (
                AutoTokenizer as _AutoTokenizer,
                AutoModelForCausalLM as _AutoModelForCausalLM,
                AutoConfig as _AutoConfig,
                Trainer as _Trainer,
                TrainingArguments as _TrainingArguments,
                DataCollatorForLanguageModeling as _DataCollatorForLanguageModeling,
                TrainerCallback as _TrainerCallback,
                file_utils as _file_utils
            )
            from datasets import Dataset as _Dataset
            
            # Assign to global aliases
            AutoTokenizer = _AutoTokenizer
            AutoModelForCausalLM = _AutoModelForCausalLM
            AutoConfig = _AutoConfig
            Trainer = _Trainer
            TrainingArguments = _TrainingArguments
            DataCollatorForLanguageModeling = _DataCollatorForLanguageModeling
            TrainerCallback = _TrainerCallback
            Dataset = _Dataset
            file_utils = _file_utils
            
            # Try optional dependencies
            try:
                import onnx as _onnx
                onnx = _onnx
            except ImportError:
                onnx = None
                
            try:
                import optimum as _optimum
                optimum = _optimum
                # Import ONNX Runtime specific modules
                from optimum.onnxruntime import ORTModelForCausalLM as _ORTModelForCausalLM
                ORTModelForCausalLM = _ORTModelForCausalLM
            except ImportError:
                optimum = None
                ORTModelForCausalLM = None
                
            # Import quantization tools
            try:
                from onnxruntime.quantization import quantize_dynamic as _quantize_dynamic, QuantType as _QuantType
                quantize_dynamic = _quantize_dynamic
                QuantType = _QuantType
            except ImportError:
                quantize_dynamic = None
                QuantType = None
            
            ML_DEPENDENCIES_AVAILABLE = True
            ML_IMPORT_ERROR = None
            
        except ImportError as e:
            ML_DEPENDENCIES_AVAILABLE = False
            ML_IMPORT_ERROR = str(e)
            
    return ML_DEPENDENCIES_AVAILABLE


class ModelTrainer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ONNX Model Trainer v0.8")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 900)
        
        # System status
        self.system_ready = False
        self.dependencies_checked = False
        self.system_check_running = False
        self.system_check_cancelled = False
        self.system_check_thread = None
        
        # Add proper window closing protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Initialize variables
        self.training_thread = None
        self.is_training = False
        self.dataset_path = tk.StringVar()
        self.output_path = tk.StringVar(value="./output")
        self.model_name = tk.StringVar(value="distilgpt2")  # Default to distilgpt2
        
        # Training parameters - Enhanced with additional options
        self.epochs = tk.IntVar(value=3)
        self.batch_size = tk.IntVar(value=4)
        self.learning_rate = tk.StringVar(value="5e-5")  # Changed to StringVar for dropdown
        self.max_length = tk.IntVar(value=512)
        self.save_steps = tk.IntVar(value=500)
        self.warmup_steps = tk.IntVar(value=500)
        self.weight_decay = tk.DoubleVar(value=0.01)
        
        # Additional advanced training parameters
        self.lr_scheduler_type = tk.StringVar(value="linear")  # Learning rate scheduler
        self.max_grad_norm = tk.DoubleVar(value=1.0)  # Gradient clipping
        
        # Training preset selection
        self.training_preset = tk.StringVar(value="Standard Training")
        
        # Memory management options
        self.use_cpu_offload = tk.BooleanVar(value=True)
        self.gradient_checkpointing = tk.BooleanVar(value=True)
        self.low_cpu_mem_usage = tk.BooleanVar(value=True)
        
        # ONNX settings (for display purposes)
        self.opset_version = tk.IntVar(value=11)
        
        # Training mode options
        self.enable_training = tk.BooleanVar(value=True)  # Training is enabled by default
        
        # Action options
        self.action_train = tk.BooleanVar(value=True)  # Train checkbox
        self.action_export = tk.BooleanVar(value=True)  # Export checkbox - enabled by default  
        self.action_quantize = tk.BooleanVar(value=True)  # Quantize checkbox - enabled by default
        
        # Model testing variables
        self.model_paths = {}  # Store mapping of display names to paths
        self.generation_thread = None
        self.communication_mode = tk.StringVar(value="chat_conversation")  # Communication mode for testing
        
        # Device selection for training, export, and quantization
        self.user_selected_device = None  # Will be set by device detection
        
        # Model information database
        self.model_info_db = {
            "distilgpt2": {
                "name": "DistilGPT-2",
                "description": "Distilled version of GPT-2, faster and smaller",
                "parameters": "82M",
                "size_pytorch": "331 MB",
                "size_onnx": "248 MB",
                "size_quantized": "83 MB",
                "architecture": "Transformer decoder",
                "context_length": 1024,
                "vocabulary": "50,257 tokens",
                "recommended_lr": ["1e-4", "5e-5", "2e-5"],
                "recommended_batch": [4, 8, 16]
            },
            "gpt2": {
                "name": "GPT-2 (Base)",
                "description": "Original GPT-2 base model",
                "parameters": "124M",
                "size_pytorch": "498 MB",
                "size_onnx": "374 MB",
                "size_quantized": "125 MB",
                "architecture": "Transformer decoder",
                "context_length": 1024,
                "vocabulary": "50,257 tokens",
                "recommended_lr": ["5e-5", "3e-5", "2e-5"],
                "recommended_batch": [2, 4, 8]
            },
            "gpt2-medium": {
                "name": "GPT-2 Medium",
                "description": "Medium-sized GPT-2 model",
                "parameters": "355M",
                "size_pytorch": "1.42 GB",
                "size_onnx": "1.07 GB",
                "size_quantized": "356 MB",
                "architecture": "Transformer decoder",
                "context_length": 1024,
                "vocabulary": "50,257 tokens",
                "recommended_lr": ["3e-5", "2e-5", "1e-5"],
                "recommended_batch": [1, 2, 4]
            },
            "gpt2-large": {
                "name": "GPT-2 Large",
                "description": "Large GPT-2 model",
                "parameters": "774M",
                "size_pytorch": "3.09 GB",
                "size_onnx": "2.32 GB",
                "size_quantized": "774 MB",
                "architecture": "Transformer decoder",
                "context_length": 1024,
                "vocabulary": "50,257 tokens",
                "recommended_lr": ["2e-5", "1e-5", "5e-6"],
                "recommended_batch": [1, 2]
            },
            "gpt2-xl": {
                "name": "GPT-2 XL",
                "description": "Extra large GPT-2 model",
                "parameters": "1.5B",
                "size_pytorch": "6.17 GB",
                "size_onnx": "4.63 GB",
                "size_quantized": "1.55 GB",
                "architecture": "Transformer decoder",
                "context_length": 1024,
                "vocabulary": "50,257 tokens",
                "recommended_lr": ["1e-5", "5e-6", "2e-6"],
                "recommended_batch": [1]
            },
            "microsoft/DialoGPT-small": {
                "name": "DialoGPT Small",
                "description": "Conversational AI model, small version",
                "parameters": "117M",
                "size_pytorch": "468 MB",
                "size_onnx": "351 MB",
                "size_quantized": "117 MB",
                "architecture": "GPT-2 based conversational",
                "context_length": 1024,
                "vocabulary": "50,257 tokens",
                "recommended_lr": ["5e-5", "3e-5", "1e-5"],
                "recommended_batch": [4, 8, 16]
            },
            "microsoft/DialoGPT-medium": {
                "name": "DialoGPT Medium",
                "description": "Conversational AI model, medium version",
                "parameters": "355M",
                "size_pytorch": "1.42 GB",
                "size_onnx": "1.07 GB",
                "size_quantized": "356 MB",
                "architecture": "GPT-2 based conversational",
                "context_length": 1024,
                "vocabulary": "50,257 tokens",
                "recommended_lr": ["3e-5", "2e-5", "1e-5"],
                "recommended_batch": [2, 4, 8]
            },
            "microsoft/DialoGPT-large": {
                "name": "DialoGPT Large",
                "description": "Conversational AI model, large version",
                "parameters": "774M",
                "size_pytorch": "3.09 GB",
                "size_onnx": "2.32 GB",
                "size_quantized": "774 MB",
                "architecture": "GPT-2 based conversational",
                "context_length": 1024,
                "vocabulary": "50,257 tokens",
                "recommended_lr": ["2e-5", "1e-5", "5e-6"],
                "recommended_batch": [1, 2, 4]
            },
            "EleutherAI/gpt-neo-125M": {
                "name": "GPT-Neo 125M",
                "description": "Open source GPT-like model",
                "parameters": "125M",
                "size_pytorch": "502 MB",
                "size_onnx": "376 MB",
                "size_quantized": "126 MB",
                "architecture": "GPT-Neo transformer",
                "context_length": 2048,
                "vocabulary": "50,257 tokens",
                "recommended_lr": ["5e-5", "3e-5", "1e-5"],
                "recommended_batch": [4, 8, 16]
            },
            "EleutherAI/gpt-neo-1.3B": {
                "name": "GPT-Neo 1.3B",
                "description": "Large open source GPT-like model",
                "parameters": "1.3B",
                "size_pytorch": "5.19 GB",
                "size_onnx": "3.89 GB",
                "size_quantized": "1.30 GB",
                "architecture": "GPT-Neo transformer",
                "context_length": 2048,
                "vocabulary": "50,257 tokens",
                "recommended_lr": ["1e-5", "5e-6", "2e-6"],
                "recommended_batch": [1, 2]
            },
            "facebook/opt-125m": {
                "name": "OPT 125M",
                "description": "Open Pre-trained Transformer by Meta",
                "parameters": "125M",
                "size_pytorch": "501 MB",
                "size_onnx": "376 MB",
                "size_quantized": "125 MB",
                "architecture": "OPT transformer",
                "context_length": 2048,
                "vocabulary": "50,265 tokens",
                "recommended_lr": ["5e-5", "3e-5", "1e-5"],
                "recommended_batch": [4, 8, 16]
            },
            "facebook/opt-350m": {
                "name": "OPT 350M",
                "description": "Medium-sized OPT model by Meta",
                "parameters": "350M",
                "size_pytorch": "1.40 GB",
                "size_onnx": "1.05 GB",
                "size_quantized": "350 MB",
                "architecture": "OPT transformer",
                "context_length": 2048,
                "vocabulary": "50,265 tokens",
                "recommended_lr": ["3e-5", "2e-5", "1e-5"],
                "recommended_batch": [2, 4, 8]
            },
            "microsoft/phi-1": {
                "name": "Phi 1 (Code-Focused)",
                "description": "Compact language model by Microsoft specifically trained for code generation and programming tasks. Optimized for Python code completion, function generation, and technical documentation. Smaller and faster than Phi-1.5.",
                "parameters": "1.3B",
                "size_pytorch": "2.84 GB",
                "size_onnx": "2.13 GB",
                "size_quantized": "713 MB",
                "architecture": "Phi transformer",
                "context_length": 2048,
                "vocabulary": "51,200 tokens",
                "recommended_lr": ["1e-5", "5e-6", "2e-6"],
                "recommended_batch": [1, 2]
            },
            "microsoft/phi-1_5": {
                "name": "Phi 1.5 (Compact)",
                "description": "Compact and efficient language model by Microsoft trained primarily on code and technical content. Excels at programming tasks, technical explanations, and structured reasoning. May require specific prompting for natural conversation.",
                "parameters": "1.3B",
                "size_pytorch": "5.20 GB",
                "size_onnx": "3.90 GB",
                "size_quantized": "1.30 GB",
                "architecture": "Phi transformer",
                "context_length": 2048,
                "vocabulary": "51,200 tokens",
                "recommended_lr": ["1e-5", "5e-6", "2e-6"],
                "recommended_batch": [1, 2]
            }
        }
        
        # Common learning rates
        self.learning_rates = [
            "1e-3", "5e-4", "1e-4", "5e-5", "3e-5", "2e-5", "1e-5", "5e-6", "2e-6", "1e-6"
        ]
        
        # Training presets for different training scenarios
        self.training_presets = {
            "Quick Test": {
                "epochs": 1,
                "batch_size": 1,
                "learning_rate": "5e-5",
                "lr_scheduler_type": "constant",
                "weight_decay": 0.0,
                "warmup_steps": 50,
                "save_steps": 100,
                "max_grad_norm": 1.0,
                "description": "Minimal training for quick testing - 1 epoch, small batch"
            },
            "Conservative": {
                "epochs": 2,
                "batch_size": 2,
                "learning_rate": "3e-5",
                "lr_scheduler_type": "linear",
                "weight_decay": 0.01,
                "warmup_steps": 100,
                "save_steps": 250,
                "max_grad_norm": 1.0,
                "description": "Safe training settings with low learning rate - good for most models"
            },
            "Standard": {
                "epochs": 3,
                "batch_size": 4,
                "learning_rate": "5e-5",
                "lr_scheduler_type": "linear",
                "weight_decay": 0.01,
                "warmup_steps": 500,
                "save_steps": 500,
                "max_grad_norm": 1.0,
                "description": "Balanced training configuration - recommended for general use"
            },
            "High Quality": {
                "epochs": 5,
                "batch_size": 2,
                "learning_rate": "2e-5",
                "lr_scheduler_type": "cosine",
                "weight_decay": 0.05,
                "warmup_steps": 300,
                "save_steps": 200,
                "max_grad_norm": 0.5,
                "description": "Longer training with cosine schedule - best quality results"
            },
            "Fast": {
                "epochs": 2,
                "batch_size": 8,
                "learning_rate": "1e-4",
                "lr_scheduler_type": "linear",
                "weight_decay": 0.01,
                "warmup_steps": 200,
                "save_steps": 300,
                "max_grad_norm": 1.0,
                "description": "Quick training with larger batches - faster but may need good GPU"
            },
            "Fine-tuning": {
                "epochs": 3,
                "batch_size": 2,
                "learning_rate": "1e-5",
                "lr_scheduler_type": "cosine_with_restarts",
                "weight_decay": 0.02,
                "warmup_steps": 100,
                "save_steps": 150,
                "max_grad_norm": 0.8,
                "description": "Gentle fine-tuning with very low learning rate - for pre-trained models"
            }
        }
        
        self.setup_ui()
        self.start_system_check()
        
    def setup_ui(self):
        """Set up the main user interface with tabbed layout"""
        
        # Control buttons at the bottom with fixed height (create first to reserve space)
        self.setup_control_buttons()
        
        # Create main notebook (tabbed interface) that fills remaining space
        self.main_notebook = ttk.Notebook(self.root)
        self.main_notebook.pack(fill='both', expand=True, padx=10, pady=(10, 0))
        
        # Bind tab change event
        self.main_notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        
        # Tab 1: Training
        self.setup_training_tab()
        
        # Tab 2: Model Testing
        self.setup_testing_tab()
        
        # Initialize preset description
        self.on_preset_changed()
        
        # Sync action checkboxes with their corresponding variables
        self.sync_action_variables()
        
        # Initially disable all controls until system check passes
        self.disable_all_controls()
        
    def setup_training_tab(self):
        """Set up the training tab with settings on left and logs on right"""
        training_tab = ttk.Frame(self.main_notebook)
        self.main_notebook.add(training_tab, text="Training & Export")
        
        # Create left-right split in the training tab
        training_paned = ttk.PanedWindow(training_tab, orient='horizontal')
        training_paned.pack(fill='both', expand=True)
        
        # Left panel for training settings
        training_left_frame = ttk.Frame(training_paned)
        training_paned.add(training_left_frame, weight=1)
        
        # Right panel for training logs
        training_right_frame = ttk.Frame(training_paned)
        training_paned.add(training_right_frame, weight=2)
        
        # Set up training settings on the left
        self.setup_training_settings(training_left_frame)
        
        # Set up training logs on the right
        self.setup_training_logs(training_right_frame)
        
    def setup_testing_tab(self):
        """Set up the testing tab with settings on left and output on right"""
        testing_tab = ttk.Frame(self.main_notebook)
        self.main_notebook.add(testing_tab, text="Model Testing")
        
        # Create left-right split in the testing tab
        testing_paned = ttk.PanedWindow(testing_tab, orient='horizontal')
        testing_paned.pack(fill='both', expand=True)
        
        # Left panel for testing settings
        testing_left_frame = ttk.Frame(testing_paned)
        testing_paned.add(testing_left_frame, weight=1)
        
        # Right panel for testing output
        testing_right_frame = ttk.Frame(testing_paned)
        testing_paned.add(testing_right_frame, weight=2)
        
        # Set up testing settings on the left
        self.setup_testing_settings(testing_left_frame)
        
        # Set up testing output on the right
        self.setup_testing_output(testing_right_frame)
        
    def setup_training_settings(self, parent):
        """Set up the training settings panel"""
        
        # Model Configuration
        model_frame = ttk.LabelFrame(parent, text="Model Configuration")
        model_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(model_frame, text="Base Model:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_name, width=35, state='disabled')
        self.model_combo['values'] = list(self.model_info_db.keys())
        self.model_combo.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_changed)
        
        ttk.Label(model_frame, text="Output Directory:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        output_frame = ttk.Frame(model_frame)
        output_frame.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_path, width=25, state='disabled')
        self.output_entry.pack(side='left', fill='x', expand=True)
        self.output_button = ttk.Button(output_frame, text="Browse", command=self.browse_output, width=8, state='disabled')
        self.output_button.pack(side='right', padx=(5, 0))
        
        # Actions section
        ttk.Label(model_frame, text="Actions:").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        actions_frame = ttk.Frame(model_frame)
        actions_frame.grid(row=2, column=1, sticky='ew', padx=5, pady=5)
        
        self.train_checkbox = ttk.Checkbutton(actions_frame, text="Train", 
                                            variable=self.action_train, 
                                            command=self.on_train_action_changed)
        self.train_checkbox.pack(side='left', padx=(0, 10))
        
        self.export_checkbox = ttk.Checkbutton(actions_frame, text="Export", 
                                             variable=self.action_export,
                                             command=self.on_action_changed)
        self.export_checkbox.pack(side='left', padx=(0, 10))
        
        self.quantize_checkbox = ttk.Checkbutton(actions_frame, text="Quantize", 
                                               variable=self.action_quantize,
                                               command=self.on_action_changed)
        self.quantize_checkbox.pack(side='left')
        
        model_frame.columnconfigure(1, weight=1)
        
        # Training Parameters
        train_frame = ttk.LabelFrame(parent, text="Training Parameters")
        train_frame.pack(fill='x', pady=(0, 10))
        
        # Dataset File selection (moved here from model section)
        dataset_file_frame = ttk.Frame(train_frame)
        dataset_file_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(dataset_file_frame, text="Dataset File:").pack(side='left')
        dataset_entry_frame = ttk.Frame(dataset_file_frame)
        dataset_entry_frame.pack(side='left', fill='x', expand=True, padx=(5, 0))
        
        self.dataset_entry = ttk.Entry(dataset_entry_frame, textvariable=self.dataset_path, width=25, state='disabled')
        self.dataset_entry.pack(side='left', fill='x', expand=True)
        self.dataset_button = ttk.Button(dataset_entry_frame, text="Browse", command=self.browse_dataset, width=8, state='disabled')
        self.dataset_button.pack(side='right', padx=(5, 0))
        
        # Bind dataset path changes to validation
        self.dataset_path.trace('w', self.on_dataset_changed)
        
        # Training Preset selection (moved to top)
        preset_frame = ttk.Frame(train_frame)
        preset_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(preset_frame, text="Training Preset:").pack(side='left')
        self.preset_combo = ttk.Combobox(preset_frame, textvariable=self.training_preset, width=20, state='disabled')
        self.preset_combo['values'] = list(self.training_presets.keys())
        self.preset_combo.pack(side='left', padx=(5, 0), fill='x', expand=True)
        self.preset_combo.bind('<<ComboboxSelected>>', self.on_preset_changed)
        
        # Preset description
        preset_desc_frame = ttk.Frame(train_frame)
        preset_desc_frame.pack(fill='x', padx=5, pady=2)
        
        self.preset_description = ttk.Label(preset_desc_frame, text="Balanced training configuration - recommended for general use", font=('Arial', 8), foreground='gray')
        self.preset_description.pack(side='left')
        
        # Training parameters grid
        params_grid = ttk.Frame(train_frame)
        params_grid.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(params_grid, text="Epochs:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.epochs_spin = ttk.Spinbox(params_grid, from_=1, to=100, textvariable=self.epochs, width=10, state='disabled')
        self.epochs_spin.grid(row=0, column=1, sticky='w', padx=5, pady=2)
        self.epochs_spin.bind('<KeyRelease>', self.on_parameter_changed)
        self.epochs_spin.bind('<<Increment>>', self.on_parameter_changed)
        self.epochs_spin.bind('<<Decrement>>', self.on_parameter_changed)
        
        ttk.Label(params_grid, text="Batch Size:").grid(row=0, column=2, sticky='w', padx=5, pady=2)
        self.batch_spin = ttk.Spinbox(params_grid, from_=1, to=32, textvariable=self.batch_size, width=10, state='disabled')
        self.batch_spin.grid(row=0, column=3, sticky='w', padx=5, pady=2)
        self.batch_spin.bind('<KeyRelease>', self.on_parameter_changed)
        self.batch_spin.bind('<<Increment>>', self.on_parameter_changed)
        self.batch_spin.bind('<<Decrement>>', self.on_parameter_changed)
        
        ttk.Label(params_grid, text="Learning Rate:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.lr_combo = ttk.Combobox(params_grid, textvariable=self.learning_rate, width=12, state='disabled')
        self.lr_combo['values'] = self.learning_rates
        self.lr_combo.grid(row=1, column=1, sticky='w', padx=5, pady=2)
        self.lr_combo.bind('<<ComboboxSelected>>', self.on_parameter_changed)
        self.lr_combo.bind('<KeyRelease>', self.on_parameter_changed)
        
        ttk.Label(params_grid, text="Max Length:").grid(row=1, column=2, sticky='w', padx=5, pady=2)
        self.length_spin = ttk.Spinbox(params_grid, from_=128, to=2048, textvariable=self.max_length, width=10, state='disabled')
        self.length_spin.grid(row=1, column=3, sticky='w', padx=5, pady=2)
        self.length_spin.bind('<KeyRelease>', self.on_parameter_changed)
        self.length_spin.bind('<<Increment>>', self.on_parameter_changed)
        self.length_spin.bind('<<Decrement>>', self.on_parameter_changed)
        
        ttk.Label(params_grid, text="Save Steps:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.save_spin = ttk.Spinbox(params_grid, from_=100, to=5000, textvariable=self.save_steps, width=10, state='disabled')
        self.save_spin.grid(row=2, column=1, sticky='w', padx=5, pady=2)
        
        ttk.Label(params_grid, text="Warmup Steps:").grid(row=2, column=2, sticky='w', padx=5, pady=2)
        self.warmup_spin = ttk.Spinbox(params_grid, from_=0, to=2000, textvariable=self.warmup_steps, width=10, state='disabled')
        self.warmup_spin.grid(row=2, column=3, sticky='w', padx=5, pady=2)
        
        # Advanced Parameters row
        ttk.Label(params_grid, text="LR Scheduler:").grid(row=3, column=0, sticky='w', padx=5, pady=2)
        self.scheduler_combo = ttk.Combobox(params_grid, textvariable=self.lr_scheduler_type, width=15, state='disabled')
        self.scheduler_combo['values'] = ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
        self.scheduler_combo.grid(row=3, column=1, sticky='w', padx=5, pady=2)
        
        ttk.Label(params_grid, text="Max Grad Norm:").grid(row=3, column=2, sticky='w', padx=5, pady=2)
        self.grad_norm_spin = ttk.Spinbox(params_grid, from_=0.1, to=5.0, increment=0.1, textvariable=self.max_grad_norm, width=10, state='disabled')
        self.grad_norm_spin.grid(row=3, column=3, sticky='w', padx=5, pady=2)
        
        ttk.Label(params_grid, text="Weight Decay:").grid(row=4, column=0, sticky='w', padx=5, pady=2)
        self.weight_decay_spin = ttk.Spinbox(params_grid, from_=0.0, to=0.3, increment=0.01, textvariable=self.weight_decay, width=10, state='disabled')
        self.weight_decay_spin.grid(row=4, column=1, sticky='w', padx=5, pady=2)
        
        # Export Parameters
        export_frame = ttk.LabelFrame(parent, text="Export Parameters")
        export_frame.pack(fill='x', pady=(0, 10))
        
        onnx_settings = ttk.Frame(export_frame)
        onnx_settings.pack(fill='x', padx=5, pady=5)
        ttk.Label(onnx_settings, text="ONNX Opset:").pack(side='left')
        self.opset_spin = ttk.Spinbox(onnx_settings, from_=11, to=17, textvariable=self.opset_version, width=8, state='disabled')
        self.opset_spin.pack(side='left', padx=(5, 0))
        self.opset_spin.bind('<KeyRelease>', self.on_export_changed)
        self.opset_spin.bind('<<Increment>>', self.on_export_changed)
        self.opset_spin.bind('<<Decrement>>', self.on_export_changed)
        
        # Device Information (replaces Memory Management)
        device_frame = ttk.LabelFrame(parent, text="Hardware Information")
        device_frame.pack(fill='x', pady=(0, 10))
        
        # Device selection row
        device_select_frame = ttk.Frame(device_frame)
        device_select_frame.pack(fill='x', padx=5, pady=5)
        
        # Help button (question mark) on the left
        self.train_device_help_button = ttk.Button(device_select_frame, text="?", width=3, command=self.show_device_help)
        self.train_device_help_button.pack(side='left', padx=(0, 5))
        
        # Device selection dropdown
        ttk.Label(device_select_frame, text="Training Device:").pack(side='left')
        self.train_device_combo = ttk.Combobox(device_select_frame, width=20, state='readonly')
        self.train_device_combo.pack(side='left', padx=(5, 0), fill='x', expand=True)
        self.train_device_combo.bind('<<ComboboxSelected>>', self.on_train_device_changed)
        
        # Memory usage display
        train_memory_frame = ttk.Frame(device_frame)
        train_memory_frame.pack(fill='x', padx=5, pady=(0, 5))
        
        self.train_memory_usage_label = ttk.Label(train_memory_frame, text="Memory usage will be shown here", font=('Arial', 8), foreground='gray')
        self.train_memory_usage_label.pack(side='left')
        
        # Initialize training device options
        self.update_train_device_options()
        
        # Model Information
        info_frame = ttk.LabelFrame(parent, text="Model Information")
        info_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        self.model_info = scrolledtext.ScrolledText(info_frame, height=12, width=45, wrap=tk.WORD, state='disabled')
        self.model_info.pack(fill='both', expand=True, padx=5, pady=5)
        
    def setup_training_logs(self, parent):
        """Set up the training logs panel"""
        
        # Log text area
        logs_frame = ttk.LabelFrame(parent, text="Training Logs")
        logs_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(logs_frame, wrap=tk.WORD, font=('Consolas', 9))
        self.log_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Log controls
        log_controls = ttk.Frame(logs_frame)
        log_controls.pack(fill='x', padx=5, pady=5)
        
        self.clear_button = ttk.Button(log_controls, text="Clear Logs", command=self.clear_logs, state='disabled')
        self.clear_button.pack(side='left', padx=5)
        self.save_button = ttk.Button(log_controls, text="Save Logs", command=self.save_logs, state='disabled')
        self.save_button.pack(side='left', padx=5)
        self.test_download_button = ttk.Button(log_controls, text="Test Download", command=self.test_download_progress, state='disabled')
        self.test_download_button.pack(side='left', padx=5)
        
    def setup_testing_settings(self, parent):
        """Set up the testing settings panel"""
        
        # Model selection area
        model_select_frame = ttk.LabelFrame(parent, text="Select ONNX Model")
        model_select_frame.pack(fill='x', padx=5, pady=5)
        
        # Model path selection
        path_frame = ttk.Frame(model_select_frame)
        path_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(path_frame, text="Model Directory:").pack(side='left')
        self.model_path_var = tk.StringVar()
        self.model_path_entry = ttk.Entry(path_frame, textvariable=self.model_path_var, width=25, state='readonly')
        self.model_path_entry.pack(side='left', fill='x', expand=True, padx=(5, 0))
        
        self.browse_model_button = ttk.Button(path_frame, text="Browse", command=self.browse_onnx_model, width=8)
        self.browse_model_button.pack(side='right', padx=(5, 0))
        
        # Quick select from output folder
        quick_select_frame = ttk.Frame(model_select_frame)
        quick_select_frame.pack(fill='x', padx=5, pady=(0, 5))
        
        ttk.Label(quick_select_frame, text="Quick Select:").pack(side='left')
        self.refresh_models_button = ttk.Button(quick_select_frame, text="Refresh", command=self.refresh_model_list, width=8)
        self.refresh_models_button.pack(side='right')
        
        self.model_list_combo = ttk.Combobox(quick_select_frame, width=20, state='readonly')
        self.model_list_combo.pack(side='left', fill='x', expand=True, padx=(5, 5))
        self.model_list_combo.bind('<<ComboboxSelected>>', self.on_model_selected)

        # Generation parameters
        params_frame = ttk.LabelFrame(parent, text="Generation Parameters")
        params_frame.pack(fill='x', padx=5, pady=5)
        
        # Communication mode selection
        mode_frame = ttk.Frame(params_frame)
        mode_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(mode_frame, text="Communication Mode:").pack(side='left')
        self.mode_combo = ttk.Combobox(mode_frame, textvariable=self.communication_mode, width=20, state='readonly')
        self.mode_combo['values'] = [
            "generate_text",
            "chat_conversation", 
            "batch_processing",
            "creative_writing",
            "question_answering",
            "code_completion"
        ]
        self.mode_combo.pack(side='left', padx=(5, 0), fill='x', expand=True)
        self.mode_combo.bind('<<ComboboxSelected>>', self.on_mode_changed)
        
        # Mode descriptions
        mode_desc_frame = ttk.Frame(params_frame)
        mode_desc_frame.pack(fill='x', padx=5, pady=2)
        
        self.mode_description = ttk.Label(mode_desc_frame, text="Interactive chat with conversation history", font=('Arial', 8), foreground='gray')
        self.mode_description.pack(side='left')
        
        params_grid = ttk.Frame(params_frame)
        params_grid.pack(fill='x', padx=5, pady=5)
        
        # Row 1 - Max Length
        ttk.Label(params_grid, text="Max Length:").grid(row=0, column=0, sticky='w', padx=(0, 5), pady=2)
        self.test_max_length = tk.IntVar(value=100)
        ttk.Spinbox(params_grid, from_=10, to=500, textvariable=self.test_max_length, width=10).grid(row=0, column=1, sticky='w', pady=2)
        
        # Row 2 - Temperature
        ttk.Label(params_grid, text="Temperature:").grid(row=1, column=0, sticky='w', padx=(0, 5), pady=2)
        self.test_temperature = tk.DoubleVar(value=0.8)
        ttk.Spinbox(params_grid, from_=0.1, to=2.0, increment=0.1, textvariable=self.test_temperature, width=10).grid(row=1, column=1, sticky='w', pady=2)
        
        # Row 3 - Top-p
        ttk.Label(params_grid, text="Top-p:").grid(row=2, column=0, sticky='w', padx=(0, 5), pady=2)
        self.test_top_p = tk.DoubleVar(value=0.9)
        ttk.Spinbox(params_grid, from_=0.1, to=1.0, increment=0.1, textvariable=self.test_top_p, width=10).grid(row=2, column=1, sticky='w', pady=2)
        
        # Row 4 - Top-k
        ttk.Label(params_grid, text="Top-k:").grid(row=0, column=2, sticky='w', padx=(5, 5), pady=2)
        self.test_top_k = tk.IntVar(value=50)
        ttk.Spinbox(params_grid, from_=1, to=100, textvariable=self.test_top_k, width=10).grid(row=0, column=3, sticky='w', pady=2)
        
        # Row 5 - Repetition Penalty
        ttk.Label(params_grid, text="Repetition Penalty:").grid(row=1, column=2, sticky='w', padx=(5, 5), pady=2)
        self.test_repetition_penalty = tk.DoubleVar(value=1.1)
        ttk.Spinbox(params_grid, from_=1.0, to=2.0, increment=0.1, textvariable=self.test_repetition_penalty, width=10).grid(row=1, column=3, sticky='w', pady=2)
        
        # Row 6 - No Repeat N-gram Size
        ttk.Label(params_grid, text="No Repeat N-gram:").grid(row=2, column=2, sticky='w', padx=(5, 5), pady=2)
        self.test_no_repeat_ngram = tk.IntVar(value=2)
        ttk.Spinbox(params_grid, from_=0, to=5, textvariable=self.test_no_repeat_ngram, width=10).grid(row=2, column=3, sticky='w', pady=2)
        
        # Row 7 - Do Sample
        ttk.Label(params_grid, text="Sampling Method:").grid(row=3, column=0, sticky='w', padx=(0, 5), pady=2)
        self.test_do_sample = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_grid, text="Enable Sampling", variable=self.test_do_sample).grid(row=3, column=1, sticky='w', pady=2)
        
        # Row 8 - Early Stopping
        ttk.Label(params_grid, text="Early Stopping:").grid(row=3, column=2, sticky='w', padx=(5, 5), pady=2)
        self.test_early_stopping = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_grid, text="Enable", variable=self.test_early_stopping).grid(row=3, column=3, sticky='w', pady=2)
        
        # Row 9 - Min Length
        ttk.Label(params_grid, text="Min Length:").grid(row=4, column=0, sticky='w', padx=(0, 5), pady=2)
        self.test_min_length = tk.IntVar(value=10)
        ttk.Spinbox(params_grid, from_=1, to=100, textvariable=self.test_min_length, width=10).grid(row=4, column=1, sticky='w', pady=2)
        
        # Row 10 - Length Penalty
        ttk.Label(params_grid, text="Length Penalty:").grid(row=4, column=2, sticky='w', padx=(5, 5), pady=2)
        self.test_length_penalty = tk.DoubleVar(value=1.0)
        ttk.Spinbox(params_grid, from_=0.5, to=2.0, increment=0.1, textvariable=self.test_length_penalty, width=10).grid(row=4, column=3, sticky='w', pady=2)
        
        params_grid.columnconfigure(1, weight=1)
        params_grid.columnconfigure(3, weight=1)
        
        # Device information section
        device_info_frame = ttk.LabelFrame(parent, text="Hardware Information")
        device_info_frame.pack(fill='x', padx=5, pady=5)
        
        # Device selection row
        device_select_frame = ttk.Frame(device_info_frame)
        device_select_frame.pack(fill='x', padx=5, pady=5)
        
        # Help button (question mark) on the left
        self.device_help_button = ttk.Button(device_select_frame, text="?", width=3, command=self.show_device_help)
        self.device_help_button.pack(side='left', padx=(0, 5))
        
        # Device selection dropdown
        ttk.Label(device_select_frame, text="Inference Device:").pack(side='left')
        self.device_combo = ttk.Combobox(device_select_frame, width=20, state='readonly')
        self.device_combo.pack(side='left', padx=(5, 0), fill='x', expand=True)
        self.device_combo.bind('<<ComboboxSelected>>', self.on_device_changed)
        
        # Memory usage display
        memory_frame = ttk.Frame(device_info_frame)
        memory_frame.pack(fill='x', padx=5, pady=(0, 5))
        
        self.memory_usage_label = ttk.Label(memory_frame, text="Memory usage will be shown here", font=('Arial', 8), foreground='gray')
        self.memory_usage_label.pack(side='left')
        
        # Initialize device options
        self.update_device_options()
        
        # Technical logs section
        tech_log_frame = ttk.LabelFrame(parent, text="Technical Logs")
        tech_log_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.tech_log_widget = scrolledtext.ScrolledText(tech_log_frame, wrap=tk.WORD, font=('Consolas', 8), height=8)
        self.tech_log_widget.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Initialize model list
        self.refresh_model_list()
        
    def setup_testing_output(self, parent):
        """Set up the testing output panel"""
        
        # Output area
        output_frame = ttk.LabelFrame(parent, text="Terminal Interface")
        output_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.test_output = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, font=('Consolas', 9))
        self.test_output.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Bind Enter key for terminal-like interaction
        self.test_output.bind('<Return>', self.on_terminal_enter)
        self.test_output.bind('<KeyPress>', self.on_terminal_keypress)
        
        # Track current line for terminal behavior
        self.current_prompt_line = None
        self.prompt_start_pos = None
        
        # Initialize terminal interface with current mode
        # Use after_idle to ensure the widget is fully created
        self.root.after_idle(self.initialize_terminal_interface)
    
    def initialize_terminal_interface(self):
        """Initialize the terminal interface with the default mode"""
        if hasattr(self, 'test_output') and hasattr(self, 'communication_mode'):
            # Only initialize if the terminal is empty
            current_content = self.test_output.get('1.0', tk.END).strip()
            if not current_content:
                self.on_mode_changed(None)
        
    def setup_control_buttons(self):
        """Set up main control buttons at the bottom with minimal size"""
        # Create control frame with minimal height at the bottom
        control_frame = ttk.Frame(self.root, height=80)
        control_frame.pack(side='bottom', fill='x', padx=10, pady=5)
        control_frame.pack_propagate(False)  # Prevent frame from shrinking
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Performing system check...", font=('Arial', 10, 'bold'))
        self.status_label.pack(pady=(5, 3))
        
        # Buttons frame
        self.buttons_frame = ttk.Frame(control_frame)
        self.buttons_frame.pack(pady=3)
        
        # Training buttons
        self.training_buttons_frame = ttk.Frame(self.buttons_frame)
        
        self.train_button = ttk.Button(self.training_buttons_frame, text="Start Training", command=self.start_training, state='disabled')
        self.train_button.pack(side='left', padx=5)
        
        self.stop_button = ttk.Button(self.training_buttons_frame, text="Stop", command=self.stop_training, state='disabled')
        self.stop_button.pack(side='left', padx=5)
        
        # Testing buttons
        self.testing_buttons_frame = ttk.Frame(self.buttons_frame)
        
        self.generate_button = ttk.Button(self.testing_buttons_frame, text="Generate Text", command=self.generate_text, state='disabled')
        self.generate_button.pack(side='left', padx=5)
        
        self.stop_generation_button = ttk.Button(self.testing_buttons_frame, text="Stop Generation", command=self.stop_generation, state='disabled')
        self.stop_generation_button.pack(side='left', padx=5)
        
        # Show training buttons by default
        self.training_buttons_frame.pack()
        
        # Initialize the default mode (chat_conversation)
        self.on_mode_changed(None)
        
    def on_tab_changed(self, event):
        """Handle tab change to show/hide appropriate buttons"""
        tab_index = self.main_notebook.index(self.main_notebook.select())
        
        # Hide all button frames first
        self.training_buttons_frame.pack_forget()
        self.testing_buttons_frame.pack_forget()
        
        # Show appropriate buttons based on active tab
        if tab_index == 0:  # Training tab
            self.training_buttons_frame.pack()
            # Validate training configuration when switching to training tab
            if hasattr(self, 'system_ready') and self.system_ready:
                self.validate_training_configuration()
        elif tab_index == 1:  # Testing tab
            self.testing_buttons_frame.pack()
            # Clear any training validation errors when switching to testing tab
            if hasattr(self, 'system_ready') and self.system_ready:
                self.update_task_status("Ready - Select model and generate text")
            
            # Initialize terminal interface if it's empty
            if hasattr(self, 'test_output'):
                current_content = self.test_output.get('1.0', tk.END).strip()
                if not current_content:
                    # Terminal interface is empty, initialize it
                    self.on_mode_changed(None)
    
    def on_mode_changed(self, event):
        """Handle communication mode change"""
        mode = self.communication_mode.get()
        
        # Update mode description and button text
        mode_descriptions = {
            "generate_text": "Generate single text responses from prompts",
            "chat_conversation": "Interactive chat with conversation history",
            "batch_processing": "Process multiple prompts at once",
            "creative_writing": "Creative writing assistance and storytelling",
            "question_answering": "Answer questions with factual responses",
            "code_completion": "Complete and generate code snippets"
        }
        
        button_texts = {
            "generate_text": "Clear Output",
            "chat_conversation": "Clear Output", 
            "batch_processing": "Clear Output",
            "creative_writing": "Clear Output",
            "question_answering": "Clear Output",
            "code_completion": "Clear Output"
        }
        
        # Update description
        if hasattr(self, 'mode_description'):
            description = mode_descriptions.get(mode, "Unknown mode")
            self.mode_description.config(text=description)
        
        # Optimize parameters for the selected mode - using simple script approach
        if hasattr(self, 'test_temperature'):
            if mode == "chat_conversation":
                # Use parameters similar to the simple script for natural conversation
                self.test_temperature.set(0.8)
                self.test_top_p.set(0.95)
                self.test_top_k.set(50)
                self.test_repetition_penalty.set(1.2)
                self.test_no_repeat_ngram.set(0)  # Disable n-gram restriction like simple script
                self.test_max_length.set(128)  # Match simple script length
                self.test_do_sample.set(True)
            elif mode == "question_answering":
                # Optimize for accurate, focused answers
                self.test_temperature.set(0.3)
                self.test_top_p.set(0.8)
                self.test_top_k.set(30)
                self.test_repetition_penalty.set(1.1)
                self.test_no_repeat_ngram.set(2)
                self.test_max_length.set(100)
                self.test_do_sample.set(True)
            elif mode == "code_completion":
                # Optimize for code generation
                self.test_temperature.set(0.2)
                self.test_top_p.set(0.95)
                self.test_top_k.set(50)
                self.test_repetition_penalty.set(1.05)
                self.test_no_repeat_ngram.set(0)
                self.test_max_length.set(150)
                self.test_do_sample.set(True)
            elif mode == "creative_writing":
                # Optimize for creativity
                self.test_temperature.set(0.9)
                self.test_top_p.set(0.95)
                self.test_top_k.set(60)
                self.test_repetition_penalty.set(1.1)
                self.test_no_repeat_ngram.set(4)
                self.test_max_length.set(200)
                self.test_do_sample.set(True)
            else:
                # Default settings for generate_text and batch_processing
                self.test_temperature.set(0.8)
                self.test_top_p.set(0.9)
                self.test_top_k.set(50)
                self.test_repetition_penalty.set(1.1)
                self.test_no_repeat_ngram.set(2)
                self.test_max_length.set(100)
                self.test_do_sample.set(True)
        
        # Update button text
        if hasattr(self, 'generate_button'):
            button_text = button_texts.get(mode, "Generate")
            self.generate_button.config(text=button_text)
        
        # Update output area based on selected mode
        if hasattr(self, 'test_output'):
            self.test_output.delete('1.0', tk.END)
            
            if mode == "chat_conversation":
                # Clear conversation history when switching to chat mode
                if hasattr(self, 'conversation_history'):
                    self.conversation_history = []
                self.test_output.insert('1.0', "💬 Chat Mode - Terminal Interface\n")
                self.test_output.insert(tk.END, "Type your messages directly in this window and press Enter to send.\n")
                self.test_output.insert(tk.END, "💡 Commands: 'clear' to reset conversation, 'help' for more info\n\n")
                self.add_terminal_prompt("chat")
                        
            elif mode == "batch_processing":
                # Show instructions for batch mode
                self.test_output.insert('1.0', 
                    "📋 Batch Processing Mode - Terminal Interface\n"
                    "Enter multiple prompts (one per line) and press Enter after each.\n"
                    "Type 'EXECUTE' and press Enter to process all prompts.\n\n")
                self.add_terminal_prompt("batch")
                    
            elif mode == "creative_writing":
                self.test_output.insert('1.0', 
                    "✍️ Creative Writing Mode - Terminal Interface\n"
                    "Enter your creative writing prompts directly in this window.\n"
                    "Press Enter to generate creative content.\n\n")
                self.add_terminal_prompt("story")
                        
            elif mode == "question_answering":
                self.test_output.insert('1.0', 
                    "❓ Question Answering Mode - Terminal Interface\n"
                    "Type your questions directly in this window.\n"
                    "Press Enter to get answers.\n\n")
                self.add_terminal_prompt("ask")
                        
            elif mode == "code_completion":
                self.test_output.insert('1.0', 
                    "💻 Code Completion Mode - Terminal Interface\n"
                    "Type your code prompts directly in this window.\n"
                    "Press Enter to complete code.\n\n")
                self.add_terminal_prompt("code")
                        
            else:  # generate_text mode
                self.test_output.insert('1.0', 
                    "📝 Text Generation Mode - Terminal Interface\n"
                    "Type your prompts directly in this window.\n"
                    "Press Enter to generate text.\n\n")
                self.add_terminal_prompt("text")
        
    def add_terminal_prompt(self, mode_prefix):
        """Add a terminal-style prompt to the output area"""
        prompt_indicators = {
            "text": "text> ",
            "chat": "chat> ",
            "batch": "batch> ",
            "story": "story> ",
            "ask": "ask> ",
            "code": "code> "
        }
        
        prompt = prompt_indicators.get(mode_prefix, "$ ")
        self.test_output.insert(tk.END, prompt)
        self.test_output.mark_set("prompt_start", f"{tk.END}-{len(prompt)}c")
        self.test_output.mark_gravity("prompt_start", "left")
        self.test_output.see(tk.END)
        self.test_output.focus_set()
    
    def on_terminal_enter(self, event):
        """Handle Enter key press in terminal interface"""
        # Get current cursor position
        cursor_pos = self.test_output.index(tk.INSERT)
        
        # Get the current line
        line_start = f"{cursor_pos.split('.')[0]}.0"
        line_end = f"{cursor_pos.split('.')[0]}.end"
        current_line = self.test_output.get(line_start, line_end)
        
        # Extract user input (everything after the prompt indicator)
        user_input = ""
        for prefix in ["text> ", "chat> ", "batch> ", "story> ", "ask> ", "code> "]:
            if current_line.startswith(prefix):
                user_input = current_line[len(prefix):].strip()
                break
        
        if user_input:
            # Move to end and add newline
            self.test_output.mark_set(tk.INSERT, tk.END)
            self.test_output.insert(tk.END, "\n")
            
            # Process the input based on current mode
            mode = self.communication_mode.get()
            self.process_terminal_input(user_input, mode)
            
        return "break"  # Prevent default Enter behavior
    
    def on_terminal_keypress(self, event):
        """Handle key presses in terminal interface"""
        # Get current cursor position
        cursor_pos = self.test_output.index(tk.INSERT)
        
        # Don't allow editing before the current prompt
        try:
            prompt_start = self.test_output.index("prompt_start")
            if self.test_output.compare(cursor_pos, "<", prompt_start):
                self.test_output.mark_set(tk.INSERT, tk.END)
                return "break"
        except tk.TclError:
            # prompt_start mark doesn't exist, allow editing
            pass
    
    def process_terminal_input(self, user_input, mode):
        """Process user input from terminal interface"""
        if not user_input.strip():
            self.add_terminal_prompt(self.get_mode_prefix(mode))
            return
        
        # Special commands for conversation management
        if user_input.lower() in ['clear', 'reset', 'restart']:
            # Clear conversation history
            if hasattr(self, 'conversation_history'):
                old_length = len(self.conversation_history)
                self.conversation_history = []
                self.test_output.insert(tk.END, f"🔄 Conversation history cleared ({old_length} lines removed). Starting fresh!\n\n")
                self.tech_log(f"💬 Conversation history manually reset by user ({old_length} lines)")
            else:
                self.test_output.insert(tk.END, "🔄 Conversation history was already empty. Starting fresh!\n\n")
                self.tech_log("💬 Conversation reset requested but history was empty")
            self.add_terminal_prompt(self.get_mode_prefix(mode))
            return
        elif user_input.lower() in ['status', 'info']:
            # Show conversation status
            if hasattr(self, 'conversation_history') and self.conversation_history:
                full_prompt = "\n".join(self.conversation_history)
                status_text = f"""
📊 Conversation Status:
• Lines in history: {len(self.conversation_history)}
• Character count: {len(full_prompt):,}
• Model: {getattr(self, 'model_name', tk.StringVar()).get() or 'Unknown'}

💡 Commands: 'clear' to reset, 'help' for more info
"""
                self.test_output.insert(tk.END, status_text + "\n")
            else:
                self.test_output.insert(tk.END, "📊 Conversation Status: Empty (no history)\n\n")
            self.add_terminal_prompt(self.get_mode_prefix(mode))
            return
        elif user_input.lower() in ['help', '?']:
            help_text = """
💡 Available commands:
• 'clear' or 'reset' - Clear conversation history
• 'status' or 'info' - Show conversation statistics
• 'help' or '?' - Show this help
• Just type normally to chat with the AI

🔧 If AI stops responding:
• Try 'clear' to reset the conversation
• Use shorter messages
• Try rephrasing your question
"""
            self.test_output.insert(tk.END, help_text + "\n")
            self.add_terminal_prompt(self.get_mode_prefix(mode))
            return
        
        # Start generation based on the input
        if mode == "batch_processing" and user_input.upper() == "EXECUTE":
            # Special command for batch processing
            self.test_output.insert(tk.END, "🔄 Processing batch commands...\n\n")
            # For now, just show a message - full batch processing can be implemented later
            self.test_output.insert(tk.END, "📋 Batch processing functionality coming soon!\n")
            self.test_output.insert(tk.END, "💡 For now, you can process prompts one by one in other modes.\n\n")
            self.add_terminal_prompt(self.get_mode_prefix(mode))
        else:
            # Generate response for the input
            self._start_terminal_generation(user_input, mode)
    
    def get_mode_prefix(self, mode):
        """Get the terminal prefix for a given mode"""
        mode_prefixes = {
            "generate_text": "text",
            "chat_conversation": "chat",
            "batch_processing": "batch",
            "creative_writing": "story",
            "question_answering": "ask",
            "code_completion": "code"
        }
        return mode_prefixes.get(mode, "text")
    
    def _start_terminal_generation(self, prompt, mode):
        """Start generation from terminal input - using simple script approach"""
        model_path = self.model_path_var.get()
        if not model_path:
            self.test_output.insert(tk.END, "❌ Error: Please select an ONNX model first\n\n")
            self.add_terminal_prompt(self.get_mode_prefix(mode))
            return
        
        # Initialize conversation history if not exists
        if not hasattr(self, 'conversation_history'):
            self.conversation_history = []
        
        # Build conversation prompt like the simple script
        if mode == "chat_conversation":
            # Dynamic context management BEFORE adding new input - check if reset is needed
            if hasattr(self, 'conversation_history') and self.conversation_history:
                self._check_and_reset_context_before_input(prompt)
            
            # Add current user input to history
            self.conversation_history.append(f"User: {prompt}")
            self.conversation_history.append("AI:")
            
            # Build the full conversation prompt similar to simple script
            formatted_prompt = "\n".join(self.conversation_history)
            
        elif mode == "question_answering":
            # Direct Q&A format
            formatted_prompt = f"Q: {prompt}\nA:"
        elif mode == "code_completion":
            # Code mode - let it generate code naturally
            formatted_prompt = f"# {prompt}\n"
        else:
            # For other modes, use simple format
            formatted_prompt = f"User: {prompt}\nAI:"
        
        # Store the prompt for response cleaning
        self.current_prompt = formatted_prompt
        
        # Add thinking indicator with some variation
        thinking_messages = [
            "🤔 AI is thinking...",
            "💭 Processing your request...", 
            "⚡ Generating response...",
            "🧠 Analyzing and responding...",
            "✨ Crafting a response..."
        ]
        thinking_msg = random.choice(thinking_messages)
        self.test_output.insert(tk.END, f"{thinking_msg}\n")
        self.test_output.see(tk.END)
        self.test_output.update()  # Force immediate display
        
        # Disable generate button and enable stop button
        self.generate_button.config(state='disabled')
        self.stop_generation_button.config(state='normal')
        
        # Start generation in separate thread
        self.generation_thread = threading.Thread(
            target=self.run_terminal_generation,
            args=(model_path, formatted_prompt, mode),
            daemon=True
        )
        self.generation_thread.start()
    
    def run_terminal_generation(self, model_path, prompt, mode):
        """Run text generation from terminal input"""
        try:
            # Use the existing generation logic but update output differently
            self.run_generation(model_path, prompt, mode + "_terminal")
        except Exception as e:
            self.root.after(0, lambda: self.remove_thinking_indicator())
            self.root.after(0, lambda: self.test_output.insert(tk.END, f"❌ Error: {str(e)}\n\n"))
        finally:
            # Re-enable controls and add new prompt
            self.root.after(0, self.terminal_generation_finished)
    
    def terminal_generation_finished(self):
        """Clean up after terminal generation"""
        self.generate_button.config(state='normal')
        self.stop_generation_button.config(state='disabled')
        
        # Add new prompt for next input
        mode = self.communication_mode.get()
        self.test_output.insert(tk.END, "\n")
        self.add_terminal_prompt(self.get_mode_prefix(mode))
        
    def disable_all_controls(self):
        """Disable all UI controls during system check"""
        controls = [
            self.model_combo, self.dataset_entry, self.dataset_button,
            self.output_entry, self.output_button, self.epochs_spin,
            self.batch_spin, self.lr_combo, self.length_spin,
            self.save_spin, self.warmup_spin, self.model_info,
            self.train_button, self.clear_button, self.save_button,
            self.test_download_button, self.train_checkbox, self.export_checkbox
        ]
        
        for control in controls:
            control.config(state='disabled')
            
    def enable_all_controls(self):
        """Enable all UI controls after successful system check"""
        controls = [
            self.model_combo, self.dataset_entry, self.dataset_button,
            self.output_entry, self.output_button, self.epochs_spin,
            self.batch_spin, self.lr_combo, self.length_spin,
            self.save_spin, self.warmup_spin, self.model_info,
            self.clear_button, self.save_button, self.test_download_button,
            self.preset_combo, self.scheduler_combo, self.grad_norm_spin,
            self.weight_decay_spin, self.train_checkbox, self.export_checkbox,
            self.opset_spin, self.train_device_help_button
        ]
        
        for control in controls:
            try:
                control.config(state='normal')
            except:
                pass  # Skip controls that might not exist
        
        # Enable training/export buttons only if ML dependencies are available
        if ML_DEPENDENCIES_AVAILABLE:
            try:
                self.train_button.config(state='normal')
                self.generate_button.config(state='normal')
            except:
                pass
        else:
            # Disable ML-dependent features
            try:
                self.train_button.config(state='disabled')
                self.generate_button.config(state='disabled')
            except:
                pass
        
        # Initialize device options for both training and testing tabs
        if hasattr(self, 'update_train_device_options'):
            self.update_train_device_options()
        if hasattr(self, 'update_device_options'):
            self.update_device_options()
            
        # Update controls state based on training mode
        self.update_controls_state()
        
        # Update action-related controls state
        self.update_action_controls_state()
        
        # Update model info for default selection
        self.update_model_info()
        
    def update_controls_state(self):
        """Update control states based on training mode selection"""
        training_enabled = self.enable_training.get()
        
        # Dataset controls - always enabled when training is enabled
        dataset_state = 'normal' if training_enabled else 'disabled'
        self.dataset_entry.config(state=dataset_state)
        self.dataset_button.config(state=dataset_state)
        
        # Update train button text
        if training_enabled:
            self.train_button.config(text="Start Training")
        else:
            self.train_button.config(text="Convert & Export")
        
        # Always update training parameter states (they will be disabled if training is not enabled)
        self.update_training_controls_state()
    
    def update_training_controls_state(self):
        """Update training parameter controls based on dataset file selection"""
        if not self.system_ready:
            return  # Don't update during system initialization
            
        # Check if dataset is selected
        dataset_selected = bool(self.dataset_path.get().strip())
        training_enabled = self.enable_training.get()
        
        # Training parameters should only be enabled if:
        # 1. Training is enabled AND
        # 2. Dataset file is selected
        controls_state = 'normal' if (training_enabled and dataset_selected) else 'disabled'
        
    
        
        # Training parameter controls (only include controls that actually exist)
        training_controls = []
        
        # Add spinboxes and comboboxes
        control_names = ['epochs_spin', 'batch_spin', 'lr_combo', 'length_spin',
                        'save_spin', 'warmup_spin', 'preset_combo',
                        'scheduler_combo', 'grad_norm_spin', 'weight_decay_spin']
        
        for control_name in control_names:
            if hasattr(self, control_name):
                control = getattr(self, control_name)
                if control and hasattr(control, 'config'):
                    training_controls.append(control)
        
        # Add memory management checkboxes if they exist
        checkbox_names = ['cpu_offload_check', 'gradient_check', 'low_mem_check']
        for checkbox_name in checkbox_names:
            if hasattr(self, checkbox_name):
                control = getattr(self, checkbox_name)
                if control and hasattr(control, 'config'):
                    training_controls.append(control)
        
        # Update all training controls
        for control in training_controls:
            try:
                control.config(state=controls_state)
            except Exception as e:
                if hasattr(self, 'tech_log'):
                    self.tech_log(f"⚠️ Error updating control state: {e}")
        
        # Start Training button should only be enabled if dataset is selected (when training is enabled)
        if hasattr(self, 'train_button'):
            if training_enabled:
                train_button_state = 'normal' if dataset_selected else 'disabled'
                self.train_button.config(state=train_button_state)
            else:
                # For export mode, always enable the button
                self.train_button.config(state='normal')
    
            
    def start_system_check(self):
        """Start comprehensive system check in background - non-blocking and interruptible"""
        self.log_message("🔍 Starting comprehensive system check...")
        self.update_task_status("System check in progress...")
        self.system_check_running = True
        self.system_check_cancelled = False
        
        # Run system check in separate daemon thread (won't block app shutdown)
        self.system_check_thread = threading.Thread(target=self.run_system_check, daemon=True)
        self.system_check_thread.start()
        
        # Start a periodic check to see if system check completed or was cancelled
        self.check_system_status()
    
    def check_system_status(self):
        """Periodically check system check status - non-blocking"""
        if self.system_check_cancelled:
            # User closed window, stop checking
            return
            
        if not self.system_check_running:
            # System check completed
            return
            
        # Check again in 100ms - keeps UI responsive
        self.root.after(100, self.check_system_status)
        
    def run_system_check(self):
        """Run comprehensive system check - can be interrupted"""
        try:
            # Check if cancelled before starting
            if self.system_check_cancelled:
                return
                
            self.log_message("🐍 Checking Python environment...")
            python_version = sys.version_info
            
            if python_version >= (3, 7):
                self.log_message(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
            else:
                self.log_message(f"❌ Python 3.7+ required, found {python_version.major}.{python_version.minor}")
                self.system_check_failed("Python version too old")
                return
            
            # Check for cancellation
            if self.system_check_cancelled:
                return
                
            # Check basic dependencies first
            self.log_message("📦 Checking basic dependencies...")
            basic_modules = {
                'tkinter': 'Tkinter (GUI)',
                'threading': 'Threading',
                'json': 'JSON',
                'pathlib': 'Path utilities'
            }
            
            for module, name in basic_modules.items():
                try:
                    __import__(module)
                    self.log_message(f"✅ {name} available")
                except ImportError:
                    self.log_message(f"❌ {name} missing")
                    self.system_check_failed(f"Missing basic dependency: {name}")
                    return
                
            # Check if ML dependencies are available (non-blocking)
            if self.system_check_cancelled:
                return
                
            self.log_message("🤖 Checking ML dependencies (optional for GUI)...")
            if not check_ml_dependencies():
                self.log_message(f"⚠️ ML dependencies not available: {ML_IMPORT_ERROR}")
                self.log_message("💡 Some features will be disabled. Install with: pip install torch transformers datasets optimum onnxruntime")
                self.log_message("✅ GUI will start in basic mode")
            else:
                self.log_message("✅ All ML dependencies available")
                
                # Check for cancellation before GPU check
                if self.system_check_cancelled:
                    return
                
                # Additional ML checks only if dependencies are available
                self.log_message("🖥️ Checking compute capabilities...")
                try:
                    if torch and torch.cuda.is_available():
                        gpu_count = torch.cuda.device_count()
                        gpu_name = torch.cuda.get_device_name(0)
                        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        self.log_message(f"✅ CUDA GPU available: {gpu_name} ({gpu_count} device(s))")
                        self.log_message(f"   VRAM: {vram_gb:.1f} GB")
                    else:
                        self.log_message("⚠️ No CUDA GPU detected - will use CPU (slower)")
                except Exception as e:
                    self.log_message(f"⚠️ GPU check failed: {e}")
            
            # Check for cancellation before disk check
            if self.system_check_cancelled:
                return
            
            # Check disk space
            self.log_message("💾 Checking disk space...")
            try:
                output_path = self.output_path.get()
                if output_path and os.path.exists(os.path.dirname(output_path)):
                    stat = os.statvfs(os.path.dirname(output_path))
                    free_space_gb = (stat.f_frsize * stat.f_bavail) / (1024**3)
                    self.log_message(f"✅ Available disk space: {free_space_gb:.1f} GB")
                    
                    if free_space_gb < 1:
                        self.log_message("⚠️ Low disk space - recommend at least 1GB free")
                else:
                    self.log_message("✅ Disk space check skipped (output path not set)")
            except Exception as e:
                self.log_message(f"⚠️ Disk space check failed: {e}")
                
            # Test model loading only if ML dependencies are available and not cancelled
            if ML_DEPENDENCIES_AVAILABLE and not self.system_check_cancelled:
                self.log_message("🤖 Testing model loading capability...")
                try:
                    # Quick test - just try to import the tokenizer class (no download)
                    from transformers import AutoTokenizer
                    self.log_message("✅ Model loading capability verified")
                except Exception as e:
                    self.log_message(f"⚠️ Model loading test failed: {e}")
            
            # Final cancellation check
            if self.system_check_cancelled:
                return
            
            # All checks passed
            self.log_message("🎉 System check completed successfully!")
            if ML_DEPENDENCIES_AVAILABLE:
                self.log_message("✅ All components ready for training and testing")
            else:
                self.log_message("✅ GUI ready in basic mode (install ML dependencies for full features)")
            self.system_check_passed()
            
        except Exception as e:
            if not self.system_check_cancelled:
                self.log_message(f"❌ System check error: {str(e)}")
                self.system_check_failed(f"System check error: {str(e)}")
        finally:
            self.system_check_running = False
            
    def system_check_passed(self):
        """Handle successful system check"""
        self.system_ready = True
        self.dependencies_checked = True
        
        # Update UI in main thread
        self.root.after(0, self._enable_ui_after_check)
        
    def _enable_ui_after_check(self):
        """Enable UI after successful system check (main thread)"""
        self.update_task_status("Ready - Configure settings and start training")
        self.enable_all_controls()
        
        # Update hardware info now that dependencies are loaded
        self.update_hardware_info()
        
        # Update model info for default selection
        self.update_model_info()
        
        # Validate training configuration on startup
        self.validate_training_configuration()
        self.log_message("🚀 You can now configure training settings and start training!")
        
    def system_check_failed(self, reason):
        """Handle failed system check"""
        self.system_ready = False
        
        # Update UI in main thread
        self.root.after(0, lambda: self._handle_check_failure(reason))
        
    def _handle_check_failure(self, reason):
        """Handle check failure in main thread"""
        self.update_task_status(f"System check failed: {reason}")
        self.log_message(f"❌ Please resolve the issues above and restart the application")
    
    def on_closing(self):
        """Handle window close event - ensure app can always be closed"""
        try:
            # Cancel any running system check
            if self.system_check_running:
                self.system_check_cancelled = True
                self.log_message("🛑 Cancelling system check...")
            
            # Wait briefly for system check to respond to cancellation
            if self.system_check_running:
                self.root.after(500, self._force_close)  # Force close after 500ms
            else:
                self._force_close()
                
        except Exception:
            # If anything goes wrong, force close immediately
            self._force_close()
    
    def _force_close(self):
        """Force close the application"""
        try:
            # Stop any background threads
            if hasattr(self, 'system_check_thread') and self.system_check_thread:
                self.system_check_cancelled = True
            
            # Destroy the window
            self.root.quit()
            self.root.destroy()
        except Exception:
            # Last resort - exit the process
            import sys
            sys.exit(0)
        
    def on_model_changed(self, event=None):
        """Handle model selection change"""
        if self.system_ready:
            self.update_model_info()
            
            # Update memory usage displays in both tabs to reflect new model size
            if hasattr(self, 'update_train_memory_usage_display'):
                self.update_train_memory_usage_display()
            if hasattr(self, 'update_memory_usage_display'):
                self.update_memory_usage_display()
            
    def on_parameter_changed(self, event=None):
        """Handle parameter changes"""
        if self.system_ready:
            # Small delay to allow the widget to update
            self.root.after(100, self.update_model_info)
            
    def update_task_status(self, status_text):
        """Update the task status label with normal text"""
        if hasattr(self, 'status_label'):
            self.status_label.config(text=status_text, foreground='black')
            
        # Track current operation for better stop feedback
        if 'training' in status_text.lower():
            self.current_operation = 'training'
        elif 'export' in status_text.lower() or 'convert' in status_text.lower():
            self.current_operation = 'export'
        elif 'quantiz' in status_text.lower():
            self.current_operation = 'quantization'
        else:
            self.current_operation = 'operation'
            
    def update_task_status_error(self, error_text):
        """Update the task status label with error text in red color"""
        if hasattr(self, 'status_label'):
            self.status_label.config(text=error_text, foreground='red')
            
    def validate_training_configuration(self):
        """Validate training configuration and update status if there are issues"""
        if not self.system_ready:
            return True  # Don't validate during system check
        
        # Check if we're on the training tab
        try:
            current_tab = self.main_notebook.index(self.main_notebook.select())
            if current_tab != 0:  # Not on training tab (index 0)
                return True  # No validation needed when not on training tab
        except:
            # If there's any issue getting tab info, skip validation
            return True
            
        training_enabled = self.enable_training.get()
        if not training_enabled:
            # Clear any training validation errors when training is disabled
            self.update_task_status("Ready - Configure settings and export model")
            return True  # No validation needed if training is disabled
            
        # Check for missing required fields
        dataset_path = self.dataset_path.get().strip()
        output_path = self.output_path.get().strip()
        
        if not dataset_path:
            self.update_task_status_error("Dataset file not selected - Please browse and select a dataset file")
            return False
        elif not output_path:
            self.update_task_status_error("Output directory not selected - Please browse and select an output directory")
            return False
        else:
            # Configuration is valid, restore normal status
            self.update_task_status("Ready - Configure settings and start training")
            return True
            
    def set_training_progress(self, epoch, total_epochs, completion_percent, gpu_percent=0, cpu_percent=0):
        """Update training progress status"""
        if gpu_percent > 0:
            status = f"Training (GPU {gpu_percent}% / CPU {cpu_percent}%): Epoch {epoch}/{total_epochs} - {completion_percent}%"
        else:
            status = f"Training: Epoch {epoch}/{total_epochs} - {completion_percent}%"
        self.update_task_status(status)
        
    def set_export_progress(self, percent):
        """Update export progress status"""
        self.update_task_status(f"Export: {percent}%")
        
    def set_quantization_progress(self, percent):
        """Update quantization progress status"""
        self.update_task_status(f"Quantization: {percent}%")
            
    def on_export_changed(self, event=None):
        """Handle export option changes"""
        if self.system_ready:
            # Small delay to allow the widget to update
            self.root.after(100, self.update_model_info)
    
    def on_training_mode_changed(self, event=None):
        """Handle training mode selection change"""
        if self.system_ready:
            self.update_controls_state()
            self.update_model_info()
            # Validate configuration when training mode changes
            self.validate_training_configuration()
    
    def on_train_action_changed(self, event=None):
        """Handle train action checkbox change - sync with enable_training"""
        # Keep the enable_training variable in sync with action_train
        self.enable_training.set(self.action_train.get())
        # Call the original training mode changed handler
        self.on_training_mode_changed(event)
    
    def on_action_changed(self, event=None):
        """Handle export/quantize action checkbox changes"""
        if self.system_ready:
            self.update_action_controls_state()
            self.update_model_info()
    
    def update_action_controls_state(self):
        """Update action-related controls based on action selections"""
        # Enable/disable ONNX Opset spinner based on export action
        export_enabled = self.action_export.get()
        opset_state = 'normal' if export_enabled else 'disabled'
        if hasattr(self, 'opset_spin'):
            self.opset_spin.config(state=opset_state)
        
        # Enable/disable Quantize action based on export action
        quantize_state = 'normal' if export_enabled else 'disabled'
        if hasattr(self, 'quantize_checkbox'):
            self.quantize_checkbox.config(state=quantize_state)
            # If export is disabled, also disable quantize
            if not export_enabled:
                self.action_quantize.set(False)
    
    def sync_action_variables(self):
        """Sync action checkboxes with their corresponding variables"""
        # Make sure action_train is in sync with enable_training
        self.action_train.set(self.enable_training.get())
            
    def update_model_info(self):
        """Update model information based on current selections"""
        if not self.system_ready:
            return
            
        try:
            model_key = self.model_name.get()
            if model_key not in self.model_info_db:
                return
                
            model_data = self.model_info_db[model_key]
            
            # Calculate estimated sizes based on parameters
            current_epochs = self.epochs.get()
            current_batch = self.batch_size.get()
            current_lr = self.learning_rate.get()
            current_length = self.max_length.get()
            export_enabled = self.action_export.get()
            quantize_enabled = self.action_quantize.get()
            
            # Memory management settings
            cpu_offload = self.use_cpu_offload.get()
            gradient_checkpoint = self.gradient_checkpointing.get()
            low_mem = self.low_cpu_mem_usage.get()
            
            # Build comprehensive info
            ml_status = "Available" if ML_DEPENDENCIES_AVAILABLE else "Not Available"
            
            info = f"""📋 {model_data['name']}

🔢 Model Specifications:
• Parameters: {model_data['parameters']}
• Architecture: {model_data['architecture']}
• Context Length: {model_data['context_length']:,} tokens
• Vocabulary: {model_data['vocabulary']}

💾 Model Sizes:
• PyTorch Model: {model_data['size_pytorch']}
• ONNX Export: {model_data['size_onnx']}
• Quantized ONNX: {model_data['size_quantized']}

⚙️ Current Training Configuration:
• Epochs: {current_epochs}
• Batch Size: {current_batch}
• Learning Rate: {current_lr}
• Max Sequence Length: {current_length:,}

📊 Training Estimates:
• Memory Usage: ~{self.estimate_memory_usage(model_key, current_batch)} GB
• Training Time: ~{self.estimate_training_time(model_key, current_epochs)} min
• Output Size: ~{self.estimate_output_size(model_key, export_enabled, quantize_enabled)}

🧠 Memory Management:
• CPU Offloading: {'Enabled' if cpu_offload else 'Disabled'}
• Gradient Checkpointing: {'Enabled' if gradient_checkpoint else 'Disabled'}
• Low CPU Memory: {'Enabled' if low_mem else 'Disabled'}
• Strategy: {'Hybrid GPU+CPU' if cpu_offload else 'GPU Only'}
• Precision: {'FP32 (safer with CPU offload)' if cpu_offload else 'FP16/FP32 (auto)'}

💡 Recommendations:
• Suggested Learning Rates: {', '.join(model_data['recommended_lr'])}
• Recommended Batch Sizes: {', '.join(map(str, model_data['recommended_batch']))}

📝 Description:
{model_data['description']}

✅ Export Configuration:
• ONNX Export: {'Enabled' if export_enabled else 'Disabled'}
• Optimization: {'Enabled' if quantize_enabled else 'Disabled'}
• Opset Version: {self.opset_version.get()}

🔧 System Status:
• ML Dependencies: {ml_status}
• Training Features: {'Available' if ML_DEPENDENCIES_AVAILABLE else 'Disabled (install ML dependencies)'}
• GUI Mode: {'Full' if ML_DEPENDENCIES_AVAILABLE else 'Basic'}"""

            # Update the text widget
            self.model_info.config(state='normal')
            self.model_info.delete('1.0', tk.END)
            self.model_info.insert('1.0', info)
            self.model_info.config(state='disabled')
            
        except Exception as e:
            self.log_message(f"⚠️ Error updating model info: {e}")
            # Show basic info
            try:
                self.model_info.config(state='normal')
                self.model_info.delete('1.0', tk.END)
                self.model_info.insert('1.0', f"Model: {self.model_name.get()}\n\nML Dependencies: {'Available' if ML_DEPENDENCIES_AVAILABLE else 'Not Available'}\n\nInstall dependencies for full model information.")
                self.model_info.config(state='disabled')
            except:
                pass
            
    def estimate_memory_usage(self, model_key, batch_size):
        """Estimate memory usage during training"""
        try:
            # Rough estimates based on model parameters
            param_counts = {
                "distilgpt2": 82, "gpt2": 124, "gpt2-medium": 355,
                "gpt2-large": 774, "gpt2-xl": 1500,
                "microsoft/DialoGPT-small": 117,
                "microsoft/DialoGPT-medium": 355,
                "microsoft/DialoGPT-large": 774,
                "EleutherAI/gpt-neo-125M": 125,
                "EleutherAI/gpt-neo-1.3B": 1300,
                "facebook/opt-125m": 125,
                "facebook/opt-350m": 350,
                "microsoft/phi-1_5": 1300
            }
            
            params = param_counts.get(model_key, 124)  # Default to GPT-2 base
            
            # Check if CUDA is available (only import torch when needed)
            cuda_available = False
            if ML_DEPENDENCIES_AVAILABLE:
                try:
                    cuda_available = torch and torch.cuda.is_available()
                except (ImportError, AttributeError):
                    pass
            
            # Adjust for FP16 vs FP32 and CPU offloading
            if self.use_cpu_offload.get():
                # CPU offloading uses FP32, but splits across devices
                memory_gb = (params * 4 * batch_size) / 2000  # Roughly half the memory on GPU
                return f"{memory_gb:.1f} (split GPU+CPU)"
            else:
                # Pure GPU training can use FP16
                bytes_per_param = 2 if cuda_available else 4  # FP16 vs FP32
                memory_gb = (params * bytes_per_param * batch_size) / 1000
                precision = "FP16" if cuda_available else "FP32"
                return f"{memory_gb:.1f} ({precision})"
        except Exception as e:
            return f"Unknown (error: {str(e)[:20]}...)"
        
    def estimate_training_time(self, model_key, epochs):
        """Estimate training time"""
        # Very rough estimates (depends heavily on hardware)
        base_times = {
            "distilgpt2": 5, "gpt2": 8, "gpt2-medium": 20,
            "gpt2-large": 45, "gpt2-xl": 90,
            "microsoft/DialoGPT-small": 8,
            "microsoft/DialoGPT-medium": 20,
            "microsoft/DialoGPT-large": 45,
            "EleutherAI/gpt-neo-125M": 8,
            "EleutherAI/gpt-neo-1.3B": 75,
            "facebook/opt-125m": 8,
            "facebook/opt-350m": 18,
            "microsoft/phi-1_5": 70
        }
        
        base_time = base_times.get(model_key, 8)
        total_time = base_time * epochs
        
        if total_time > 60:
            return f"{total_time/60:.1f} hours"
        else:
            return f"{total_time}"
            
    def estimate_output_size(self, model_key, export_enabled, quantize_enabled):
        """Estimate total output size"""
        model_data = self.model_info_db.get(model_key, {})
        
        # Parse size strings (e.g., "331 MB" -> 331)
        pytorch_size = self.parse_size(model_data.get('size_pytorch', '500 MB'))
        
        total_size = pytorch_size  # Base model
        total_size += pytorch_size * 0.1  # Checkpoints and logs
        
        if export_enabled:
            onnx_size = self.parse_size(model_data.get('size_onnx', '400 MB'))
            total_size += onnx_size
            
            if quantize_enabled:
                quantized_size = self.parse_size(model_data.get('size_quantized', '150 MB'))
                total_size += quantized_size
                
        if total_size > 1000:
            return f"{total_size/1000:.1f} GB"
        else:
            return f"{total_size:.0f} MB"
            
    def parse_size(self, size_str):
        """Parse size string like '331 MB' to number in MB"""
        try:
            parts = size_str.split()
            if len(parts) == 2:
                value = float(parts[0])
                unit = parts[1].upper()
                if unit == 'GB':
                    return value * 1000
                elif unit == 'MB':
                    return value
            return 500  # Default fallback
        except:
            return 500
            
    def browse_dataset(self):
        """Browse for dataset file"""
        filename = filedialog.askopenfilename(
            title="Select Dataset File",
            filetypes=[
                ("JSON files", "*.json"),
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.dataset_path.set(filename)
            # Trigger validation after setting the path
            self.validate_training_configuration()
    
    def on_dataset_changed(self, *args):
        """Handle dataset file selection changes"""
        if self.system_ready:
            self.update_training_controls_state()
            self.validate_training_configuration()
            
    def browse_output(self):
        """Browse for output directory"""
        dirname = filedialog.askdirectory(title="Select Output Directory")
        if dirname:
            self.output_path.set(dirname)
            # Trigger validation after setting the path
            self.validate_training_configuration()
            
    def start_training(self):
        """Start the training or conversion process"""
        if not self.system_ready:
            messagebox.showerror("Error", "System check has not completed successfully")
            return
            
        # Check if training is enabled and validate dataset
        training_enabled = self.enable_training.get()
        if training_enabled and not self.dataset_path.get():
            messagebox.showerror("Error", "Please select a dataset file for training")
            return
            
        if not self.output_path.get():
            messagebox.showerror("Error", "Please specify an output directory")
            return
            
        self.is_training = True
        self.train_button.config(state='disabled')
        self.stop_button.config(state='normal')
        
        if training_enabled:
            self.update_task_status("Training: Initializing...")
        else:
            self.update_task_status("Export: Initializing...")
        
        # Start training/conversion in a separate thread
        self.training_thread = threading.Thread(target=self.run_training)
        self.training_thread.daemon = True
        self.training_thread.start()
        
    def run_training(self):
        """Run the actual training or conversion process"""
        try:
            training_enabled = self.enable_training.get()
            
            if training_enabled:
                self.log_message("🚀 Starting training process...")
                self.log_message(f"Model: {self.model_name.get()}")
                self.log_message(f"Dataset: {self.dataset_path.get()}")
            else:
                self.log_message("🚀 Starting model conversion process...")
                self.log_message(f"Model: {self.model_name.get()}")
                
            self.log_message(f"Output: {self.output_path.get()}")
            
            # Create timestamped output directory structure
            base_output_dir = Path(self.output_path.get())
            base_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create timestamped session directory
            timestamped_dir = self._create_timestamped_output_dir(base_output_dir)
            self.log_message(f"📁 Session directory: {timestamped_dir.name}")
            
            # Create subdirectories for each step
            train_dir = timestamped_dir / "1_trained"
            convert_dir = timestamped_dir / "2_converted"
            quantize_dir = timestamped_dir / "3_quantized"
            
            if training_enabled:
                # Step 1: Training
                train_dir.mkdir(exist_ok=True)
                self.log_message(f"📁 Created training output directory: {train_dir}")
                self.log_message("📚 Step 1: Training model...")
                self.run_actual_training(train_dir)
                source_dir = train_dir
            else:
                # For conversion mode, use the pretrained model directly
                self.log_message("� Using pretrained model for conversion...")
                source_dir = None  # Will use model name directly
            
            # Step 2: ONNX Export
            if self.action_export.get():
                convert_dir.mkdir(exist_ok=True)
                if training_enabled:
                    self.log_message("🔄 Step 2: Converting trained model to ONNX...")
                    self.set_export_progress(0)
                else:
                    self.log_message("🔄 Step 1: Converting pretrained model to ONNX...")
                    self.set_export_progress(0)
                self.run_onnx_export(source_dir, convert_dir)
                
                # Step 3: Quantization
                if self.action_quantize.get():
                    quantize_dir.mkdir(exist_ok=True)
                    if training_enabled:
                        self.log_message("⚙️ Step 3: Quantizing ONNX model...")
                    else:
                        self.log_message("⚙️ Step 2: Quantizing ONNX model...")
                    self.set_quantization_progress(0)
                    self.run_onnx_quantization(convert_dir, quantize_dir)
                    
            if self.is_training:
                if training_enabled:
                    self.log_message("🎉 Training pipeline completed successfully!")
                else:
                    self.log_message("🎉 Model conversion completed successfully!")
                self.log_message(f"📁 All outputs saved to: {timestamped_dir}")
            else:
                self.log_message("⏹️ Training stopped by user")
                
        except Exception as e:
            self.log_message(f"❌ Training error: {str(e)}")
            logger.exception("Training error")
        finally:
            self.training_finished()
            
    def _run_with_gpu_fallback(self, func, *args, operation_name="operation"):
        """
        Run a function with automatic GPU memory fallback.
        If CUDA out of memory error occurs, retry with CPU-only mode.
        """
        try:
            # Store original training device before attempting GPU operation
            self.store_original_training_device()
            
            # First attempt with auto device strategy
            return func(*args, "auto")
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for CUDA out of memory errors
            if ('cuda' in error_msg and 'out of memory' in error_msg) or \
               ('memory' in error_msg and 'allocate' in error_msg) or \
               ('runtimeerror' in error_msg and 'cuda' in error_msg):
                
                self.log_message(f"⚠️ GPU memory insufficient for {operation_name}")
                self.log_message("🔄 Automatically switching to CPU-only mode...")
                
                # Update training device display to show CPU fallback
                self.update_training_device_display("cpu_only")
                
                # Clear GPU cache if possible
                try:
                    if torch and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        self.log_message("🧹 GPU memory cache cleared")
                        
                        # Force garbage collection
                        import gc
                        gc.collect()
                        
                        # Additional cleanup - try to clear all GPU memory
                        try:
                            torch.cuda.ipc_collect()
                        except:
                            pass
                except:
                    pass
                
                # Set environment variable to hide CUDA devices before retry
                import os
                old_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                
                try:
                    # Retry with CPU-only mode
                    result = func(*args, "cpu_only")
                    self.log_message(f"✅ {operation_name.capitalize()} completed successfully using CPU")
                    return result
                except Exception as cpu_error:
                    self.log_message(f"❌ {operation_name.capitalize()} failed even with CPU-only mode: {str(cpu_error)}")
                    raise cpu_error
                finally:
                    # Restore original CUDA_VISIBLE_DEVICES
                    if old_cuda_visible is not None:
                        os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_visible
                    else:
                        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                    
                    # Restore original training device display after operation
                    self.root.after(1000, lambda: self.update_training_device_display("auto"))
            else:
                # Not a memory error, re-raise original exception
                raise e
        
    def run_actual_training(self, train_dir):
        """Run standard model training using HuggingFace transformers with GPU memory fallback"""
        return self._run_with_gpu_fallback(self._run_actual_training_impl, train_dir, operation_name="training")
    
    def _run_actual_training_impl(self, train_dir, device_strategy="auto"):
        """Implementation of actual training with device strategy parameter"""
        try:
            # Simple progress callback to track training progress
            class ProgressCallback(TrainerCallback):
                def __init__(self, trainer_ui):
                    self.trainer_ui = trainer_ui
                    self.total_epochs = trainer_ui.epochs.get()
                    self.current_epoch = 0
                    
                def on_epoch_begin(self, args, state, control, **kwargs):
                    self.current_epoch = state.epoch + 1
                    self.trainer_ui.root.after(0, lambda: self.trainer_ui.set_training_progress(
                        self.current_epoch, self.total_epochs, 0, 0, 0
                    ))
                    
                def on_step_end(self, args, state, control, **kwargs):
                    if state.max_steps > 0:
                        step_percent = int((state.global_step / state.max_steps) * 100)
                        self.trainer_ui.root.after(0, lambda: self.trainer_ui.set_training_progress(
                            self.current_epoch, self.total_epochs, step_percent, 0, 0
                        ))
                        
                def on_epoch_end(self, args, state, control, **kwargs):
                    self.trainer_ui.root.after(0, lambda: self.trainer_ui.set_training_progress(
                        self.current_epoch, self.total_epochs, 100, 0, 0
                    ))
            
            # Load model and tokenizer using standard HuggingFace procedures
            model_name = self.model_name.get()
            self.log_message(f"🔍 Loading model: {model_name}")
            
            # Load tokenizer with no caching
            self.log_message("📝 Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=None)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            self.log_message("✅ Tokenizer loaded")
                
            # Load model with device strategy consideration
            self.log_message("🔄 Loading model...")
            if device_strategy == "cpu_only":
                # Force CPU-only loading - ensure no GPU usage at all
                import os
                os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide CUDA devices
                
                # Disable CUDA backends
                import torch
                torch.backends.cudnn.enabled = False
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    cache_dir=None,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    low_cpu_mem_usage=True
                )
                
                # Force model to CPU and make sure it stays there
                model = model.to("cpu")
                
                self.log_message("✅ Model loaded (CPU-only mode)")
            else:
                # Auto device mapping (default behavior)
                model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=None)
                self.log_message("✅ Model loaded")
            
            # Load and format dataset
            self.log_message("📚 Loading training dataset...")
            train_dataset = self.load_and_format_dataset()
            self.log_message(f"✅ Dataset loaded: {len(train_dataset)} examples")
            
            # Tokenize dataset
            self.log_message("🔤 Tokenizing dataset...")
            def tokenize(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=self.max_length.get(),
                    padding=False,
                )
            
            tokenized_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )
            
            # Training arguments with device strategy consideration
            training_args_kwargs = {
                "output_dir": str(train_dir),
                "num_train_epochs": self.epochs.get(),
                "per_device_train_batch_size": self.batch_size.get(),
                "learning_rate": float(self.learning_rate.get()),
                "weight_decay": self.weight_decay.get(),
                "warmup_steps": self.warmup_steps.get(),
                "logging_steps": 50,
                "save_steps": self.save_steps.get(),
                "report_to": "none",
                "save_total_limit": 3,
            }
            
            # Adjust settings for CPU-only mode
            if device_strategy == "cpu_only":
                training_args_kwargs.update({
                    "fp16": False,
                    "bf16": False,
                    "dataloader_pin_memory": False,
                    "no_cuda": True,  # Force no CUDA usage
                })
                # Reduce batch size for CPU training if it's too large
                if self.batch_size.get() > 4:
                    training_args_kwargs["per_device_train_batch_size"] = min(4, self.batch_size.get())
                    self.log_message(f"🔧 Reduced batch size to {training_args_kwargs['per_device_train_batch_size']} for CPU training")
            
            training_args = TrainingArguments(**training_args_kwargs)
            
            # Create trainer
            progress_callback = ProgressCallback(self)
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
                callbacks=[progress_callback],
            )
            
            # Start training
            if device_strategy == "cpu_only":
                self.log_message("🚀 Starting model training (CPU-only mode)...")
            else:
                self.log_message("🚀 Starting model training...")
            self.log_message(f"📊 Training: {self.epochs.get()} epochs, batch size {training_args.per_device_train_batch_size}")
            trainer.train()
            self.log_message("✅ Training completed successfully")
            
            # Save model
            self.log_message("💾 Saving trained model...")
            model.save_pretrained(train_dir)
            tokenizer.save_pretrained(train_dir)
            self.log_message("✅ Model saved successfully")
            self.log_message(f"📁 Model saved to: {train_dir}")
            
        except Exception as e:
            self.log_message(f"❌ Training error: {str(e)}")
            raise
            
    def get_device_map(self):
        """Determine the optimal device mapping strategy based on user selection"""
        try:
            if not torch:
                return "CPU only"
        except ImportError:
            return "CPU only"
        
        # Get user selected device from the device selection UI
        selected_device = getattr(self, 'user_selected_device', None)
        if not selected_device:
            # Fallback: if no device selected, use CPU-only mode
            return "CPU only"
        
        # Parse user device selection
        if "CPU Only" in selected_device:
            return "CPU only"
        elif "CPU + GPU" in selected_device or "Hybrid" in selected_device:
            # Check if accelerate is available for advanced device mapping
            try:
                import accelerate
                return "Auto (GPU + CPU offloading)"
            except ImportError:
                return "Manual (GPU primary + CPU fallback)"
        elif "GPU" in selected_device:
            # Extract GPU index from device name like "GPU 0: RTX 4090 (24.0GB)"
            try:
                gpu_index = int(selected_device.split("GPU ")[1].split(":")[0])
                return f"GPU {gpu_index} only"
            except:
                return "GPU only"
        else:
            # Default fallback
            return "GPU only" if torch.cuda.is_available() else "CPU only"
            
    def log_device_placement(self, model):
        """Log where different parts of the model are placed"""
        try:
            device_counts = {}
            for name, param in model.named_parameters():
                device = str(param.device)
                device_counts[device] = device_counts.get(device, 0) + 1
                
            self.log_message("📍 Model device placement:")
            for device, count in device_counts.items():
                self.log_message(f"   {device}: {count} parameters")
                
        except Exception as e:
            self.log_message(f"⚠️ Could not log device placement: {e}")
            
    def is_large_model(self, model):
        """Check if model is considered large (>1B parameters)"""
        try:
            total_params = sum(p.numel() for p in model.parameters())
            param_gb = total_params * 4 / (1024**3)  # Assuming float32
            self.log_message(f"Model size: {total_params/1e6:.1f}M parameters ({param_gb:.2f} GB)")
            return total_params > 1e9  # 1B parameters
        except:
            return False
            
    def apply_manual_offloading(self, model):
        """Apply manual CPU offloading for large models"""
        try:
            self.log_message("🔄 Applying manual CPU offloading...")
            
            # Move embedding layers to GPU (most frequently accessed)
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
                model.transformer.wte = model.transformer.wte.cuda()
                self.log_message("   Embeddings -> GPU")
            
            # Move some transformer layers to CPU
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                layers = model.transformer.h
                gpu_layers = len(layers) // 2  # Keep half on GPU
                
                for i, layer in enumerate(layers):
                    if i < gpu_layers:
                        layer = layer.cuda()
                    else:
                        layer = layer.cpu()
                        
                self.log_message(f"   Layers 0-{gpu_layers-1} -> GPU")
                self.log_message(f"   Layers {gpu_layers}-{len(layers)-1} -> CPU")
            
            # Keep final layers on GPU for output generation
            if hasattr(model, 'lm_head'):
                model.lm_head = model.lm_head.cuda()
                self.log_message("   Output head -> GPU")
                
            return model
            
        except Exception as e:
            self.log_message(f"⚠️ Manual offloading failed: {e}")
            self.log_message("🔄 Falling back to standard GPU placement")
            return model.cuda()
            
    def log_memory_usage(self, stage):
        """Log current memory usage"""
        try:
            if torch and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                self.log_message(f"🧠 Memory usage ({stage}): {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        except (ImportError, AttributeError, Exception) as e:
            self.log_message(f"⚠️ Could not check memory usage: {e}")
    
    def create_onnx_session(self, model_path):
        """Create ONNX Runtime session with GPU support and CPU fallback"""
        # Use global onnxruntime and numpy imports
        ort = onnxruntime
        np = numpy
        
        # Try to get available providers
        available_providers = ort.get_available_providers()
        self.tech_log(f"📋 Available ONNX providers: {available_providers}")
        
        # Check user's device preference
        user_device = getattr(self, 'user_selected_device', None)
        if user_device:
            self.tech_log(f"👤 User selected device: {user_device}")
        
        # Define provider preference order based on user selection
        preferred_providers = []
        
        # If user explicitly selected CPU Only, skip GPU providers
        if user_device and "CPU Only" in user_device:
            self.tech_log("🖥️ User selected CPU Only - skipping GPU providers")
            preferred_providers.append('CPUExecutionProvider')
        else:
            # Check for GPU providers (only if user didn't select CPU Only)
            if 'CUDAExecutionProvider' in available_providers:
                try:
                    if torch and torch.cuda.is_available():
                        # If user selected a specific GPU, use that device
                        if user_device and user_device.startswith("GPU ") and ":" in user_device:
                            # Extract GPU index from selection like "GPU 0: GeForce RTX 3080 (10.0GB)"
                            try:
                                gpu_index = int(user_device.split(":")[0].replace("GPU ", "").strip())
                                cuda_device = gpu_index
                                self.tech_log(f"🎯 Using user-selected GPU {gpu_index}")
                            except (ValueError, IndexError):
                                cuda_device = torch.cuda.current_device()
                                self.tech_log(f"⚠️ Could not parse GPU selection, using default GPU {cuda_device}")
                        else:
                            cuda_device = torch.cuda.current_device()
                            
                        vram_gb = torch.cuda.get_device_properties(cuda_device).total_memory / (1024**3)
                        self.tech_log(f"🎮 CUDA available: GPU {cuda_device} with {vram_gb:.1f} GB VRAM")
                        
                        # Configure CUDA provider with memory optimization
                        cuda_provider_options = {
                            'device_id': cuda_device,
                            'arena_extend_strategy': 'kSameAsRequested',  # Conservative memory allocation
                            'gpu_mem_limit': int(vram_gb * 0.8 * 1024**3),  # Use 80% of VRAM
                            'cudnn_conv_algo_search': 'EXHAUSTIVE',
                            'do_copy_in_default_stream': True,
                        }
                        preferred_providers.append(('CUDAExecutionProvider', cuda_provider_options))
                        self.tech_log("✅ CUDA provider configured with memory optimization")
                    else:
                        self.tech_log("⚠️ CUDA provider available but no CUDA devices detected")
                except ImportError:
                    self.tech_log("⚠️ CUDA provider available but PyTorch not installed")
            
            if 'ROCMExecutionProvider' in available_providers:
                preferred_providers.append('ROCMExecutionProvider')
                self.tech_log("✅ ROCm provider available for AMD GPUs")
            
            if 'OpenVINOExecutionProvider' in available_providers:
                preferred_providers.append('OpenVINOExecutionProvider')
                self.tech_log("✅ OpenVINO provider available for Intel hardware")
            
            # Always add CPU as fallback
            preferred_providers.append('CPUExecutionProvider')
        
        # Create session with provider priority
        session = None
        used_provider = None
        
        for provider in preferred_providers:
            try:
                if isinstance(provider, tuple):
                    # Provider with options (like CUDA)
                    provider_name, provider_options = provider
                    self.tech_log(f"🔄 Trying provider: {provider_name} with options...")
                    session = ort.InferenceSession(model_path, providers=[(provider_name, provider_options)])
                    used_provider = provider_name
                else:
                    # Simple provider
                    self.tech_log(f"🔄 Trying provider: {provider}...")
                    session = ort.InferenceSession(model_path, providers=[provider])
                    used_provider = provider
                
                # Test the session with a dummy input to ensure it works
                input_info = session.get_inputs()
                
                # Create proper dummy inputs that match the model structure
                dummy_inputs = {}
                
                for inp in input_info:
                    if inp.name == 'input_ids':
                        dummy_inputs[inp.name] = np.array([[1, 2, 3]], dtype=np.int64)
                    elif inp.name == 'attention_mask':
                        dummy_inputs[inp.name] = np.array([[1, 1, 1]], dtype=np.int64)
                    elif inp.name == 'position_ids':
                        dummy_inputs[inp.name] = np.array([[0, 1, 2]], dtype=np.int64)
                    elif inp.name.startswith('past_key_values'):
                        # For past key values, create zero tensors with appropriate shape
                        # Typical shape is [batch_size, num_heads, seq_length, head_dim]
                        # For initial generation, past sequence length is 0
                        shape = inp.shape
                        concrete_shape = []
                        for dim in shape:
                            if isinstance(dim, str) or dim == -1:
                                if 'batch' in str(dim).lower() or dim == -1:
                                    concrete_shape.append(1)  # batch size
                                elif 'sequence' in str(dim).lower() or 'seq' in str(dim).lower():
                                    concrete_shape.append(0)  # past sequence length (0 for initial)
                                else:
                                    concrete_shape.append(32)  # default dimension
                            else:
                                concrete_shape.append(dim)
                        
                        # Create zero tensor for past key values
                        dummy_inputs[inp.name] = np.zeros(concrete_shape, dtype=np.float32)
                    else:
                        # For any other inputs, try to create appropriate dummy data
                        shape = inp.shape
                        concrete_shape = []
                        for dim in shape:
                            if isinstance(dim, str) or dim == -1:
                                concrete_shape.append(1)
                            else:
                                concrete_shape.append(dim)
                        
                        if 'int' in inp.type.lower():
                            dummy_inputs[inp.name] = np.ones(concrete_shape, dtype=np.int64)
                        else:
                            dummy_inputs[inp.name] = np.zeros(concrete_shape, dtype=np.float32)
                
                # Try a test inference with proper inputs
                try:
                    session.run(None, dummy_inputs)
                    self.tech_log(f"✅ Successfully created session with {used_provider}")
                    break
                except Exception as test_error:
                    # If the test fails but it's a complex model, still try to use this provider
                    # The actual inference code is more robust and handles complex inputs
                    test_error_msg = str(test_error)
                    if any(keyword in test_error_msg.lower() for keyword in ['required inputs', 'missing from input feed', 'past_key_values']):
                        self.tech_log(f"⚠️ Provider {used_provider} test failed (complex model structure), but will try during actual inference")
                        # Don't break here, still use this provider for actual inference
                        break
                    else:
                        # For other errors, actually fail and try next provider
                        raise test_error
                
            except Exception as e:
                error_msg = str(e)
                self.tech_log(f"❌ Provider {provider if isinstance(provider, str) else provider[0]} failed: {error_msg}")
                
                # Check for specific GPU memory issues
                if 'memory' in error_msg.lower() or 'out of memory' in error_msg.lower():
                    self.tech_log("💡 GPU memory insufficient, falling back to next provider...")
                elif 'cuda' in error_msg.lower() and 'initialize' in error_msg.lower():
                    self.tech_log("💡 CUDA initialization failed, falling back to next provider...")
                
                session = None
                continue
        
        if session is None:
            # Last resort: try basic CPU session
            self.tech_log("🆘 All providers failed, trying basic CPU session...")
            try:
                session = ort.InferenceSession(model_path)
                used_provider = "CPU (basic)"
                self.tech_log("✅ Basic CPU session created successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to create ONNX session with any provider: {e}")
        
        # Log final configuration
        actual_providers = session.get_providers()
        self.tech_log(f"🎯 Final session providers: {actual_providers}")
        self.tech_log(f"🚀 Using {used_provider} for model inference")
        
        # Update GUI to show actual provider being used
        if hasattr(self, 'memory_usage_label'):
            try:
                # Create display text showing actual provider
                if 'CUDA' in used_provider:
                    provider_display = f"🎮 Currently using GPU: {used_provider}"
                    status_color = 'darkgreen'
                elif 'CPU' in used_provider:
                    provider_display = f"🖥️ Currently using CPU: {used_provider}"
                    status_color = 'darkorange'
                else:
                    provider_display = f"⚡ Currently using: {used_provider}"
                    status_color = 'darkblue'
                
                # Update the memory usage label to also show current provider
                self.root.after(0, lambda: self._update_provider_status(provider_display, status_color))
            except Exception as e:
                self.tech_log(f"⚠️ Could not update GUI provider status: {e}")
        
        return session
    
    def _update_provider_status(self, provider_display, status_color):
        """Update GUI to show the currently active provider"""
        try:
            # Get current selections and status
            selected_device = self.device_combo.get() if hasattr(self, 'device_combo') else "Unknown"
            
            # Check if the actual provider differs from selected device
            actual_differs = False
            actual_provider_name = ""
            
            if 'CPU' in provider_display and 'GPU' in selected_device:
                actual_differs = True
                actual_provider_name = "CPU Only"
            elif 'GPU' in provider_display and 'CPU Only' in selected_device:
                actual_differs = True
                # Try to match to a specific GPU option if available
                if hasattr(self, 'device_combo'):
                    available_options = self.device_combo['values']
                    gpu_options = [opt for opt in available_options if opt.startswith("GPU")]
                    if gpu_options:
                        actual_provider_name = gpu_options[0]  # Use first available GPU option
                    else:
                        actual_provider_name = "GPU (Active)"
            
            if actual_differs:
                # Update the dropdown to show the actual active provider
                if actual_provider_name and hasattr(self, 'device_combo'):
                    # Temporarily change dropdown to show actual provider
                    self.device_combo.set(f"{actual_provider_name} (Auto-selected)")
                    
                # Create fallback status message
                status_text = f"⚠️ Fallback: {selected_device} → {actual_provider_name}"
                status_color = 'darkorange'
                
                # Log the fallback
                self.tech_log(f"⚠️ Device fallback: {selected_device} → {actual_provider_name}")
            else:
                # Providers match - show success status
                status_text = f"✅ Active: {selected_device}"
                status_color = 'darkgreen'
                self.tech_log(f"✅ Device match: Using {selected_device} as selected")
            
            # Now update the memory display with proper information
            self._update_memory_display_with_status(status_text, status_color)
            
        except Exception as e:
            self.tech_log(f"❌ Error updating provider status in GUI: {e}")
    
    def _update_memory_display_with_status(self, status_text, status_color):
        """Update memory display with current status and preserve model size info"""
        try:
            # Get current model path to calculate model size
            model_path = self.model_path_var.get() if hasattr(self, 'model_path_var') else None
            
            memory_info = []
            
            # Get current device (could be fallback device)
            current_device = self.device_combo.get()
            
            # Add memory information based on current active device
            if "GPU" in current_device and "CPU" not in current_device:
                # GPU mode - show VRAM info
                try:
                    if torch and torch.cuda.is_available():
                        gpu_id = 0  # Default to first GPU
                        if current_device.startswith("GPU ") and ":" in current_device:
                            try:
                                gpu_id = int(current_device.split()[1].rstrip(":"))
                            except:
                                pass
                        
                        vram_total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
                        vram_free = (torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_allocated(gpu_id)) / (1024**3)
                        memory_info.append(f"🎮 VRAM: {vram_free:.1f}GB free / {vram_total:.1f}GB total")
                except Exception:
                    memory_info.append("🎮 VRAM: Unable to get GPU memory info")
                    
            elif "CPU + GPU" in current_device or "Hybrid" in current_device:
                # Hybrid mode - show both RAM and VRAM
                try:
                    import psutil
                    ram_info = psutil.virtual_memory()
                    ram_free = ram_info.available / (1024**3)
                    ram_total = ram_info.total / (1024**3)
                    memory_info.append(f"💾 RAM: {ram_free:.1f}GB free / {ram_total:.1f}GB total")
                    
                    if torch and torch.cuda.is_available():
                        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        vram_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
                        memory_info.append(f"🎮 VRAM: {vram_free:.1f}GB free / {vram_total:.1f}GB total")
                except ImportError:
                    memory_info.append("💾 RAM: psutil not available for memory info")
                except Exception:
                    memory_info.append("💾 Memory info unavailable")
            else:
                # CPU only mode - show RAM info
                try:
                    import psutil
                    ram_info = psutil.virtual_memory()
                    ram_free = ram_info.available / (1024**3)
                    ram_total = ram_info.total / (1024**3)
                    memory_info.append(f"💾 RAM: {ram_free:.1f}GB free / {ram_total:.1f}GB total")
                except ImportError:
                    memory_info.append("💾 RAM: Install psutil for memory info")
                except Exception:
                    memory_info.append("💾 RAM: Unable to get memory info")
            
            # Add model size if model is selected
            if model_path and os.path.exists(model_path):
                try:
                    model_size = 0
                    for root, dirs, files in os.walk(model_path):
                        for file in files:
                            if file.endswith('.onnx'):
                                model_size += os.path.getsize(os.path.join(root, file))
                    
                    if model_size > 0:
                        model_size_gb = model_size / (1024**3)
                        if model_size_gb >= 1.0:
                            memory_info.append(f"📦 Model: {model_size_gb:.1f}GB")
                        else:
                            model_size_mb = model_size / (1024**2)
                            memory_info.append(f"📦 Model: {model_size_mb:.0f}MB")
                except Exception:
                    memory_info.append("📦 Model: Size unknown")
            
            # Build final display text
            if memory_info:
                full_display = f"{status_text} • {' • '.join(memory_info)}"
            else:
                full_display = status_text
            
            # Update the display
            self.memory_usage_label.config(text=full_display, foreground=status_color)
            
        except Exception as e:
            # Fallback display
            self.memory_usage_label.config(text=f"{status_text} • Error getting memory info", foreground='red')
            self.tech_log(f"❌ Error updating memory display: {e}")
    
    def update_device_options(self):
        """Update available device options based on hardware capabilities"""
        try:
            # Check if ML dependencies are loaded yet
            if not ML_DEPENDENCIES_AVAILABLE and onnxruntime is None:
                # Dependencies not loaded yet - show loading status
                self.device_combo['values'] = ["Checking devices..."]
                self.device_combo.set("Checking devices...")
                self.device_combo.config(state='disabled')
                self.memory_usage_label.config(text="⏳ Checking hardware capabilities...", foreground='gray')
                self.device_help_button.config(state='disabled')
                return
            
            # Get available providers and hardware info
            available_providers = onnxruntime.get_available_providers() if onnxruntime else []
            has_cuda = any('CUDA' in p for p in available_providers)
            has_gpu_hardware = torch and torch.cuda.is_available() if torch else False
            
            device_options = []
            
            if has_cuda and has_gpu_hardware:
                # GPU is available and working
                try:
                    gpu_count = torch.cuda.device_count()
                    for i in range(gpu_count):
                        gpu_name = torch.cuda.get_device_name(i)
                        vram_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                        device_options.append(f"GPU {i}: {gpu_name} ({vram_gb:.1f}GB)")
                    
                    # Add hybrid option
                    device_options.append("CPU + GPU (Hybrid)")
                except Exception as e:
                    device_options.append("GPU (Details unavailable)")
            
            # Always add CPU option
            device_options.append("CPU Only")
            
            # Update combo box
            self.device_combo['values'] = device_options
            
            # Set default selection and store it as user preference
            if device_options:
                if has_cuda and has_gpu_hardware:
                    default_device = device_options[0]  # Select first GPU by default
                    self.device_combo.set(default_device)
                    self.user_selected_device = default_device
                else:
                    self.device_combo.set("CPU Only")
                    self.user_selected_device = "CPU Only"
            
            self.device_combo.config(state='readonly')
            self.device_help_button.config(state='normal')
            
            # Update memory usage display
            self.update_memory_usage_display()
            
        except Exception as e:
            # Fallback for any errors
            self.device_combo['values'] = ["Error detecting devices"]
            self.device_combo.set("Error detecting devices")
            self.device_combo.config(state='disabled')
            self.memory_usage_label.config(text=f"❌ Device detection failed: {str(e)}", foreground='red')
            self.device_help_button.config(state='disabled')
    
    def update_memory_usage_display(self):
        """Update the memory usage display based on selected device and model"""
        try:
            selected_device = self.device_combo.get()
            
            if "Checking" in selected_device or "Error" in selected_device:
                return
            
            # Use the same display logic as the status update
            status_text = f"🔧 Selected: {selected_device}"
            status_color = 'black'
            self._update_memory_display_with_status(status_text, status_color)
                
        except Exception as e:
            self.memory_usage_label.config(text=f"Error updating memory info: {str(e)}", foreground='red')
    
    def on_device_changed(self, event=None):
        """Handle device selection change"""
        try:
            selected_device = self.device_combo.get()
            
            # Store the user's device preference
            self.user_selected_device = selected_device
            
            # Also update the training device combo to match
            if hasattr(self, 'train_device_combo') and self.train_device_combo['values']:
                try:
                    self.train_device_combo.set(selected_device)
                    # Update memory management variables
                    if "CPU Only" in selected_device:
                        self.use_cpu_offload.set(False)
                        self.gradient_checkpointing.set(True)
                    elif "CPU + GPU" in selected_device or "Hybrid" in selected_device:
                        self.use_cpu_offload.set(True)
                        self.gradient_checkpointing.set(True)
                    else:  # GPU only
                        self.use_cpu_offload.set(False)
                        self.gradient_checkpointing.set(True)
                except:
                    pass
            
            self.update_memory_usage_display()
            if hasattr(self, 'update_train_memory_usage_display'):
                self.update_train_memory_usage_display()
            
            # Log device change
            if hasattr(self, 'tech_log'):
                self.tech_log(f"🔄 Inference device changed to: {selected_device}")
                
        except Exception as e:
            if hasattr(self, 'tech_log'):
                self.tech_log(f"❌ Error changing device: {str(e)}")
    
    def show_device_help(self):
        """Show device selection help dialog"""
        try:
            # Create a custom dialog window
            help_window = tk.Toplevel(self.root)
            help_window.title("Device Selection Help")
            help_window.geometry("480x480")
            help_window.resizable(False, False)
            
            # Center the window
            help_window.transient(self.root)
            help_window.grab_set()
            
            # Position window in center of parent
            help_window.update_idletasks()
            x = (help_window.winfo_screenwidth() // 2) - (480 // 2)
            y = (help_window.winfo_screenheight() // 2) - (480 // 2)
            help_window.geometry(f"480x480+{x}+{y}")
            
            # Create main frame with padding
            main_frame = ttk.Frame(help_window, padding="15")
            main_frame.pack(fill='both', expand=True)
            
            # Title
            title_label = ttk.Label(main_frame, text="🖥️ Device Selection Guide", 
                                  font=('Segoe UI', 12, 'bold'))
            title_label.pack(anchor='w', pady=(0, 10))
            
            # Create scrollable text area
            text_frame = ttk.Frame(main_frame)
            text_frame.pack(fill='both', expand=True, pady=(0, 15))
            
            # Text widget with scrollbar
            text_widget = tk.Text(text_frame, wrap='word', font=('Segoe UI', 9),
                                bg='#f8f9fa', fg='#2c3e50', relief='flat',
                                padx=10, pady=10, height=20)
            scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)
            
            # Pack text widget and scrollbar
            text_widget.pack(side='left', fill='both', expand=True)
            scrollbar.pack(side='right', fill='y')
            
            # Help content with better formatting
            help_content = """GPU OPTIONS

🎮 GPU 0, GPU 1, etc.
   • Uses dedicated graphics memory (VRAM)
   • Fastest inference performance
   • Best for real-time chat and large models
   • Requires compatible NVIDIA/AMD GPU

💾 CPU + GPU (Hybrid)
   • Distributes processing between CPU and GPU
   • Uses both system RAM and VRAM
   • Good for models larger than available VRAM
   • Slower than pure GPU but handles bigger models

🖥️ CPU Only
   • Uses system RAM exclusively
   • Works on any computer
   • Slower inference but universal compatibility
   • Good for smaller models or testing

PERFORMANCE GUIDELINES

Small Models (< 2GB)
   ✓ Any device will work well
   ✓ CPU sufficient for testing

Medium Models (2-8GB)
   ✓ GPU recommended for speed
   ✓ Hybrid if VRAM limited

Large Models (> 8GB)
   ✓ Hybrid mode often required
   ✓ CPU fallback for very large models

MEMORY CONSIDERATIONS

• Check memory usage indicator below device selection
• GPU memory is typically 8-24GB on modern cards
• System RAM is usually 16-32GB on workstations
• Choose device based on your hardware limits

TIPS FOR BEST PERFORMANCE

1. Use GPU when available for speed
2. Monitor memory usage to avoid crashes
3. Try hybrid mode if getting memory errors
4. CPU mode works as reliable fallback
5. Smaller models = more device flexibility"""
            
            # Insert text and configure tags for styling
            text_widget.insert('1.0', help_content)
            
            # Configure text styling
            text_widget.tag_configure('heading', font=('Segoe UI', 10, 'bold'), 
                                    foreground='#2980b9', spacing1=10, spacing3=5)
            text_widget.tag_configure('subheading', font=('Segoe UI', 9, 'bold'), 
                                    foreground='#27ae60', spacing1=8, spacing3=3)
            text_widget.tag_configure('bullet', lmargin1=20, lmargin2=30)
            
            # Apply styling to headings
            content_lines = help_content.split('\n')
            line_num = 1
            for line in content_lines:
                if line and not line.startswith('   ') and not line.startswith('✓') and not line.startswith('•') and not line.isdigit():
                    if line.isupper():
                        text_widget.tag_add('heading', f'{line_num}.0', f'{line_num}.end')
                    elif line and not line.startswith(' '):
                        text_widget.tag_add('subheading', f'{line_num}.0', f'{line_num}.end')
                elif line.startswith('   ') or line.startswith('✓') or line.startswith('•'):
                    text_widget.tag_add('bullet', f'{line_num}.0', f'{line_num}.end')
                line_num += 1
            
            # Make text read-only
            text_widget.config(state='disabled')
            
            # Close button
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill='x', pady=(5, 0))
            
            close_button = ttk.Button(button_frame, text="Close", 
                                    command=help_window.destroy)
            close_button.pack(side='right')
            
            # Focus and keyboard handling
            help_window.focus_set()
            help_window.bind('<Escape>', lambda e: help_window.destroy())
            close_button.bind('<Return>', lambda e: help_window.destroy())
            
        except Exception as e:
            # Fallback to simple message box if custom dialog fails
            import tkinter.messagebox as msgbox
            fallback_text = """Device Selection Help

GPU: Fastest, uses graphics memory
CPU+GPU: Hybrid mode for large models  
CPU: Slowest but universal compatibility

Choose based on your hardware and model size."""
            msgbox.showinfo("Device Selection Help", fallback_text)
            print(f"Error showing device help: {e}")

    def update_train_device_options(self):
        """Update available device options for training tab based on hardware capabilities"""
        try:
            # Check if ML dependencies are loaded yet
            if not ML_DEPENDENCIES_AVAILABLE and onnxruntime is None:
                # Dependencies not loaded yet - show loading status
                self.train_device_combo['values'] = ["Checking devices..."]
                self.train_device_combo.set("Checking devices...")
                self.train_device_combo.config(state='disabled')
                self.train_memory_usage_label.config(text="⏳ Checking hardware capabilities...", foreground='gray')
                self.train_device_help_button.config(state='disabled')
                return
            
            # Get available providers and hardware info
            available_providers = onnxruntime.get_available_providers() if onnxruntime else []
            has_cuda = any('CUDA' in p for p in available_providers)
            has_gpu_hardware = torch and torch.cuda.is_available() if torch else False
            
            device_options = []
            
            if has_cuda and has_gpu_hardware:
                # GPU is available and working
                try:
                    gpu_count = torch.cuda.device_count()
                    for i in range(gpu_count):
                        gpu_name = torch.cuda.get_device_name(i)
                        vram_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                        device_options.append(f"GPU {i}: {gpu_name} ({vram_gb:.1f}GB)")
                    
                    # Add hybrid option
                    device_options.append("CPU + GPU (Hybrid)")
                except Exception as e:
                    device_options.append("GPU (Details unavailable)")
            
            # Always add CPU option
            device_options.append("CPU Only")
            
            # Update combo box
            self.train_device_combo['values'] = device_options
            
            # Set default selection and store it as user preference
            if device_options:
                if has_cuda and has_gpu_hardware:
                    default_device = device_options[0]  # Select first GPU by default
                    self.train_device_combo.set(default_device)
                    self.user_selected_device = default_device
                else:
                    self.train_device_combo.set("CPU Only")
                    self.user_selected_device = "CPU Only"
            
            self.train_device_combo.config(state='readonly')
            self.train_device_help_button.config(state='normal')
            
            # Update memory usage display for training
            self.update_train_memory_usage_display()
            
        except Exception as e:
            # Fallback for any errors
            self.train_device_combo['values'] = ["Error detecting devices"]
            self.train_device_combo.set("Error detecting devices")
            self.train_device_combo.config(state='disabled')
            self.train_memory_usage_label.config(text=f"❌ Device detection failed: {str(e)}", foreground='red')
            self.train_device_help_button.config(state='disabled')
    
    def update_training_device_display(self, device_strategy):
        """Update the training device display based on current device strategy"""
        try:
            if not hasattr(self, 'train_device_combo'):
                return
                
            current_device = self.train_device_combo.get()
            
            if device_strategy == "cpu_only":
                # Update to show CPU fallback
                fallback_text = "CPU Only (GPU Memory Fallback)"
                self.train_device_combo.set(fallback_text)
                
                # Update memory usage display for CPU mode
                try:
                    import psutil
                    ram_info = psutil.virtual_memory()
                    ram_free = ram_info.available / (1024**3)
                    ram_total = ram_info.total / (1024**3)
                    
                    current_model = self.model_name.get()
                    model_info_text = ""
                    if current_model and current_model in self.model_info_db:
                        model_info = self.model_info_db[current_model]
                        model_size_str = model_info.get('size_pytorch', 'Unknown')
                        if model_size_str != 'Unknown':
                            model_info_text = f" • 📦 Model: {model_size_str}"
                    
                    memory_text = f"💾 RAM: {ram_free:.1f}GB free / {ram_total:.1f}GB total{model_info_text} • 🔄 Automatic CPU fallback active"
                    self.train_memory_usage_label.config(text=memory_text, foreground='orange')
                    
                except ImportError:
                    self.train_memory_usage_label.config(text="💾 CPU Mode (GPU Memory Fallback) • Install psutil for memory info", foreground='orange')
                except Exception:
                    self.train_memory_usage_label.config(text="💾 CPU Mode (GPU Memory Fallback) • Memory info unavailable", foreground='orange')
                    
            elif device_strategy == "auto":
                # Restore original device selection
                if hasattr(self, 'original_training_device'):
                    self.train_device_combo.set(self.original_training_device)
                else:
                    # Restore to first GPU or CPU if no original stored
                    device_options = self.train_device_combo['values']
                    if device_options:
                        self.train_device_combo.set(device_options[0])
                
                # Update memory usage display normally
                self.update_train_memory_usage_display()
                
        except Exception as e:
            self.log_message(f"⚠️ Error updating training device display: {str(e)}")

    def store_original_training_device(self):
        """Store the original training device selection before fallback"""
        try:
            if hasattr(self, 'train_device_combo'):
                self.original_training_device = self.train_device_combo.get()
        except Exception:
            pass

    def update_train_memory_usage_display(self):
        """Update the memory usage display for training based on selected device and model"""
        try:
            selected_device = self.train_device_combo.get()
            current_model = self.model_name.get()
            
            if not selected_device or selected_device in ["Checking devices...", "Error detecting devices"]:
                return
            
            memory_info = []
            
            if "GPU" in selected_device and "CPU" not in selected_device:
                # Pure GPU mode
                try:
                    if torch and torch.cuda.is_available():
                        gpu_idx = int(selected_device.split()[1].rstrip(':'))
                        vram_total = torch.cuda.get_device_properties(gpu_idx).total_memory / (1024**3)
                        vram_free = (torch.cuda.get_device_properties(gpu_idx).total_memory - torch.cuda.memory_allocated(gpu_idx)) / (1024**3)
                        memory_info.append(f"🎮 VRAM: {vram_free:.1f}GB free / {vram_total:.1f}GB total")
                    else:
                        memory_info.append("⚠️ GPU selected but not available")
                except Exception:
                    memory_info.append("🎮 VRAM: Unable to get GPU memory info")
                    
            elif "CPU + GPU" in selected_device:
                # Hybrid mode
                try:
                    import psutil
                    ram_info = psutil.virtual_memory()
                    ram_free = ram_info.available / (1024**3)
                    ram_total = ram_info.total / (1024**3)
                    memory_info.append(f"💾 RAM: {ram_free:.1f}GB free / {ram_total:.1f}GB total")
                    
                    if torch and torch.cuda.is_available():
                        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        vram_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
                        memory_info.append(f"🎮 VRAM: {vram_free:.1f}GB free / {vram_total:.1f}GB total")
                except ImportError:
                    memory_info.append("💾 RAM: psutil not available for memory info")
                except Exception:
                    memory_info.append("💾 Memory info unavailable")
                    
            else:
                # CPU only mode
                try:
                    import psutil
                    ram_info = psutil.virtual_memory()
                    ram_free = ram_info.available / (1024**3)
                    ram_total = ram_info.total / (1024**3)
                    memory_info.append(f"💾 RAM: {ram_free:.1f}GB free / {ram_total:.1f}GB total")
                except ImportError:
                    memory_info.append("💾 RAM: Install psutil for memory info")
                except Exception:
                    memory_info.append("� RAM: Unable to get memory info")
            
            # Add model size estimate based on selected model
            if current_model and current_model in self.model_info_db:
                model_info = self.model_info_db[current_model]
                model_size_str = model_info.get('size_pytorch', 'Unknown')
                if model_size_str != 'Unknown':
                    memory_info.append(f"📦 Model: {model_size_str}")
            
            # Update display
            if memory_info:
                self.train_memory_usage_label.config(text=" • ".join(memory_info), foreground='black')
            else:
                self.train_memory_usage_label.config(text="Memory usage information unavailable", foreground='gray')
            
        except Exception as e:
            self.train_memory_usage_label.config(text=f"Error updating memory info: {str(e)}", foreground='red')
    
    def on_train_device_changed(self, event=None):
        """Handle training device selection change"""
        try:
            selected_device = self.train_device_combo.get()
            
            # Store the user's device preference (same as inference device)
            self.user_selected_device = selected_device
            
            # Update memory management variables based on device selection
            if "CPU Only" in selected_device:
                self.use_cpu_offload.set(False)  # No offloading needed for CPU-only
                self.gradient_checkpointing.set(True)  # Enable for memory efficiency
            elif "CPU + GPU" in selected_device or "Hybrid" in selected_device:
                self.use_cpu_offload.set(True)  # Enable hybrid mode
                self.gradient_checkpointing.set(True)  # Enable for memory efficiency
            else:  # GPU only
                self.use_cpu_offload.set(False)  # Pure GPU mode
                self.gradient_checkpointing.set(True)  # Still useful for large models
            
            # Also update the inference device combo to match
            if hasattr(self, 'device_combo') and self.device_combo['values']:
                try:
                    self.device_combo.set(selected_device)
                except:
                    pass  # If device not available in inference combo, skip
            
            self.update_train_memory_usage_display()
            
            # Log device change
            self.log_message(f"🔄 Training device changed to: {selected_device}")
                
        except Exception as e:
            self.log_message(f"❌ Error changing training device: {str(e)}")

    def update_hardware_info(self):
        """Update hardware information display in the Model Testing tab"""
        # This method now primarily updates the device options
        # Most hardware detection is handled by update_device_options()
        self.update_device_options()
    
    def test_download_progress(self):
        """Test download progress tracking - shows cache status and recommendations"""
        self.log_message("🧪 Testing download progress tracking...")
        self.log_message("💡 To see download progress, select a model that's not cached")
        
        # List of small models that are good for testing
        test_models = [
            "distilgpt2",
            "gpt2", 
            "microsoft/DialoGPT-small",
            "facebook/opt-125m"
        ]
        
        self.log_message(f"📝 Recommended test models: {', '.join(test_models)}")
        self.log_message("🔧 To force a download test:")
        self.log_message("   1. Select a model you haven't used before")
        self.log_message("   2. Or clear the HuggingFace cache: ~/.cache/huggingface/")
        
        # Check current cache status
        try:
            # Use global file_utils import
            cache_dir = file_utils.default_cache_path
            self.log_message(f"📁 Current cache directory: {cache_dir}")
            
            if os.path.exists(cache_dir):
                cached_models = []
                for item in os.listdir(cache_dir):
                    if item.startswith("models--"):
                        model_name = item.replace("models--", "").replace("--", "/")
                        cached_models.append(model_name)
                
                if cached_models:
                    self.log_message(f"💾 Currently cached models: {', '.join(cached_models[:5])}")
                    if len(cached_models) > 5:
                        self.log_message(f"   ... and {len(cached_models) - 5} more")
                else:
                    self.log_message("💾 No models currently cached")
            else:
                self.log_message("💾 Cache directory doesn't exist yet")
                
        except Exception as e:
            self.log_message(f"⚠️ Could not check cache status: {e}")
            
        self.log_message("✅ Download test information complete")
            
    def load_and_format_dataset(self):
        """Load and format the dataset from JSON file"""
        try:
            # Use global Dataset import
            with open(self.dataset_path.get(), "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Format data similar to the original script
            formatted_data = []
            for item in data:
                if "text" in item and isinstance(item["text"], str):
                    text = item["text"]
                    if "<|prompt|>" in text and "<|reply|>" in text:
                        try:
                            parts = text.partition("<|prompt|>")
                            prompt_reply_part = parts[2]
                            
                            prompt_parts = prompt_reply_part.partition("<|reply|>")
                            prompt = prompt_parts[0].strip()
                            reply = prompt_parts[2].strip()
                            
                            formatted_data.append({"text": f"user: {prompt}\nbot: {reply}"})
                        except IndexError:
                            self.log_message(f"Warning: Could not parse prompt/reply in item")
                            continue
                    else:
                        # If no special delimiters, use the text as-is
                        formatted_data.append({"text": text})
                else:
                    self.log_message(f"Warning: Item missing 'text' key or 'text' is not a string")
                    continue
            
            return Dataset.from_list(formatted_data)
            
        except Exception as e:
            self.log_message(f"❌ Dataset loading error: {str(e)}")
            raise
            
    def save_compatible_tokenizer(self, tokenizer, model_dir):
        """Save only essential tokenizer files for optimized deployment"""
        try:
            
            self.log_message("🔧 Saving essential tokenizer files for optimized deployment...")
            
            # Save tokenizer normally first to get all files
            tokenizer.save_pretrained(model_dir)
            
            # Load and fix tokenizer.json 
            tokenizer_json_path = Path(model_dir) / "tokenizer.json"
            
            if tokenizer_json_path.exists():
                with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
                    tokenizer_data = json.load(f)
                
                fixes_applied = []
                
                # Ensure merges are in string format (standard HuggingFace format)
                if 'model' in tokenizer_data and 'merges' in tokenizer_data['model']:
                    merges = tokenizer_data['model']['merges']
                    if merges and isinstance(merges[0], list):
                        # Convert from array format to string format
                        string_merges = []
                        for merge in merges:
                            if isinstance(merge, list) and len(merge) == 2:
                                string_merges.append(f"{merge[0]} {merge[1]}")
                            elif isinstance(merge, str):
                                string_merges.append(merge)
                        
                        tokenizer_data['model']['merges'] = string_merges
                        fixes_applied.append(f"Converted {len(string_merges)} merges to standard string format")
                
                # Ensure pre_tokenizer is in standard ByteLevel format (without pattern field)
                if 'pre_tokenizer' in tokenizer_data:
                    pre_tokenizer = tokenizer_data['pre_tokenizer']
                    # Remove pattern field if it exists (not standard for ByteLevel)
                    if 'pattern' in pre_tokenizer:
                        del pre_tokenizer['pattern']
                        fixes_applied.append("Removed non-standard pattern field from pre_tokenizer")
                
                # Save the corrected tokenizer.json in standard format
                with open(tokenizer_json_path, 'w', encoding='utf-8') as f:
                    json.dump(tokenizer_data, f, indent=2, ensure_ascii=False)
                
                if fixes_applied:
                    self.log_message("✅ Applied compatibility fixes:")
                    for fix in fixes_applied:
                        self.log_message(f"   - {fix}")
                else:
                    self.log_message("✅ Tokenizer already in standard HuggingFace format")
            
            # Remove unnecessary files, keep only essential ones for optimized deployment
            # Essential files for deployment: tokenizer.json, config.json, ONNX files, and external data files
            essential_files = {"tokenizer.json", "config.json"}
            model_dir_path = Path(model_dir)
            
            removed_files = []
            for file_path in model_dir_path.glob("*"):
                if file_path.is_file():
                    # Keep essential files, ONNX files, and external data files
                    if (file_path.name in essential_files or 
                        file_path.name.endswith('.onnx') or 
                        file_path.name.endswith('.onnx_data')):
                        continue  # Keep this file
                    
                    # Remove non-essential files
                    try:
                        file_path.unlink()  # Delete the file
                        removed_files.append(file_path.name)
                    except Exception as e:
                        self.log_message(f"⚠️ Could not remove {file_path.name}: {e}")
            
            if removed_files:
                self.log_message(f"🗑️ Removed unnecessary files: {', '.join(removed_files)}")
            
            # List final files
            final_files = [f.name for f in model_dir_path.glob("*") if f.is_file()]
            self.log_message(f"📁 Final deployment-ready files: {', '.join(sorted(final_files))}")
                
        except Exception as e:
            self.log_message(f"⚠️ Warning: Could not ensure tokenizer compatibility: {e}")
            self.log_message("   Falling back to standard save - tokenizer should still work")
            
    def run_onnx_export(self, source_dir, convert_dir):
        """Export trained model or pretrained model to ONNX format with GPU memory fallback"""
        return self._run_with_gpu_fallback(self._run_onnx_export_impl, source_dir, convert_dir, operation_name="ONNX export")
    
    def _run_onnx_export_impl(self, source_dir, convert_dir, device_strategy="auto"):
        """Export trained model or pretrained model to ONNX format"""
        try:
            
            if source_dir is None:
                # Export pretrained model directly
                model_name = self.model_name.get()
                if device_strategy == "cpu_only":
                    self.log_message(f"🔄 Converting pretrained model to ONNX (CPU-only): {model_name}")
                else:
                    self.log_message(f"🔄 Converting pretrained model to ONNX: {model_name}")
                
                # First, validate that we can load the model
                self.log_message("🔍 Validating model availability...")
                try:
                    # Test loading with minimal configuration
                    config = AutoConfig.from_pretrained(model_name)
                    self.log_message(f"✅ Model config loaded: {config.model_type}")
                    
                    # Test tokenizer loading
                    test_tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.log_message("✅ Tokenizer validation successful")
                    
                    # Quick model load test with device strategy consideration
                    if device_strategy == "cpu_only":
                        test_model = AutoModelForCausalLM.from_pretrained(
                            model_name, 
                            torch_dtype=torch.float32,
                            device_map="cpu",
                            low_cpu_mem_usage=True
                        )
                    else:
                        test_model = AutoModelForCausalLM.from_pretrained(
                            model_name, 
                            torch_dtype=torch.float32,
                            device_map="auto" if torch.cuda.is_available() else "cpu",
                            low_cpu_mem_usage=True
                        )
                    self.log_message("✅ Model validation successful")
                    del test_model, test_tokenizer  # Free memory immediately
                    
                except Exception as val_error:
                    self.log_message(f"❌ Model validation failed: {val_error}")
                    raise Exception(f"Cannot load model {model_name}: {val_error}")
                
                # Try ONNX export with device strategy consideration
                self.log_message("🔄 Starting ONNX export...")
                try:
                    # Force CPU export for CPU-only mode
                    export_kwargs = {
                        "export": True,
                        "use_cache": False  # Disable caching to avoid state conflicts
                    }
                    
                    if device_strategy == "cpu_only":
                        # Force export on CPU to avoid GPU memory issues
                        export_kwargs.update({
                            "torch_dtype": torch.float32,
                            "device_map": "cpu"
                        })
                        
                    ort_model = ORTModelForCausalLM.from_pretrained(
                        model_name, 
                        **export_kwargs
                    )
                    self.log_message("✅ ONNX export completed")
                except Exception as export_error:
                    self.log_message(f"❌ ONNX export failed: {export_error}")
                    # Try alternative approach without optimum
                    raise Exception(f"ONNX export failed for {model_name}: {export_error}")
                
            else:
                # Export trained model
                self.log_message("🔄 Converting trained model to ONNX...")
                ort_model = ORTModelForCausalLM.from_pretrained(
                    str(source_dir), 
                    export=True
                )
                self.log_message("✅ Trained model exported to ONNX")
            
            # Save the ONNX model
            self.log_message("� Saving ONNX model...")
            ort_model.save_pretrained(str(convert_dir))
            
            # Save tokenizer
            if source_dir is None:
                tokenizer = AutoTokenizer.from_pretrained(self.model_name.get())
            else:
                tokenizer = AutoTokenizer.from_pretrained(str(source_dir))
            tokenizer.save_pretrained(convert_dir)
            
            # Check file size
            onnx_file = convert_dir / "model.onnx"
            if onnx_file.exists():
                size_mb = onnx_file.stat().st_size / (1024*1024)
                self.log_message(f"✅ ONNX export completed successfully")
                self.log_message(f"📁 Output location: {convert_dir}")
                self.log_message(f"� Model size: {size_mb:.1f} MB")
                self.set_export_progress(100)
            else:
                raise Exception("ONNX export failed - model.onnx not found")
                
        except Exception as e:
            self.log_message(f"❌ ONNX export error: {str(e)}")
            raise
            
    def run_onnx_quantization(self, convert_dir, quantize_dir):
        """Quantize ONNX model with GPU memory fallback"""
        return self._run_with_gpu_fallback(self._run_onnx_quantization_impl, convert_dir, quantize_dir, operation_name="quantization")
    
    def _run_onnx_quantization_impl(self, convert_dir, quantize_dir, device_strategy="auto"):
        """Quantize ONNX model using standard procedures"""
        try:
            
            input_onnx = convert_dir / "model.onnx"
            output_onnx = quantize_dir / "model.onnx"
            
            if not input_onnx.exists():
                raise Exception(f"ONNX model not found at {input_onnx}")
            
            if device_strategy == "cpu_only":
                self.log_message("🔧 Starting quantization (CPU-only mode)...")
            else:
                self.log_message("🔧 Starting quantization...")
            quantize_dir.mkdir(parents=True, exist_ok=True)
            
            # Standard quantization (quantization is typically CPU-bound anyway)
            quantize_dynamic(
                model_input=str(input_onnx),
                model_output=str(output_onnx),
                weight_type=QuantType.QInt8
            )
            self.log_message("✅ Quantization completed successfully")
            
            # Copy tokenizer files
            tokenizer = AutoTokenizer.from_pretrained(str(convert_dir))
            tokenizer.save_pretrained(quantize_dir)
            
            # Copy config files
            for file in ["config.json"]:
                src = convert_dir / file
                if src.exists():
                    shutil.copy2(str(src), str(quantize_dir / file))
            
            # Report results
            original_size = input_onnx.stat().st_size / (1024*1024)
            quantized_size = output_onnx.stat().st_size / (1024*1024)
            
            self.log_message(f"� Quantization Results:")
            self.log_message(f"   • Original: {original_size:.1f} MB")
            self.log_message(f"   • Quantized: {quantized_size:.1f} MB")
            self.log_message(f"   • Reduction: {((original_size - quantized_size) / original_size * 100):.1f}%")
            
            self.set_quantization_progress(100)
            
        except Exception as e:
            self.log_message(f"❌ Quantization error: {str(e)}")
            raise
            
    def test_quantized_model(self, quantize_dir):
        """Test the quantized model with a simple prompt"""
        try:
            
            # Load quantized model
            model = ORTModelForCausalLM.from_pretrained(str(quantize_dir), provider="CPUExecutionProvider")
            tokenizer = AutoTokenizer.from_pretrained(str(quantize_dir))
            
            # Test with a simple prompt
            test_prompt = "Hello! How are you today?"
            self.log_message(f"🧪 Testing with prompt: '{test_prompt}'")
            
            # Encode the prompt
            inputs = tokenizer.encode(test_prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
            
            # Decode the response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = full_response[len(test_prompt):].strip()
            
            self.log_message(f"  → Generated: {generated_text[:100]}{'...' if len(generated_text) > 100 else ''}")
            self.log_message(f"✓ Quantized model test completed!")
            
        except Exception as e:
            self.log_message(f"⚠ Warning: Could not test quantized model: {e}")
            self.log_message("  Model files created, but manual testing recommended")
            
    def test_quantized_model(self, quantize_dir):
        """Test the quantized model with a simple prompt"""
        try:
            # Use global AutoTokenizer and torch imports
            
            # Load quantized model
            model = ORTModelForCausalLM.from_pretrained(str(quantize_dir), provider="CPUExecutionProvider")
            tokenizer = AutoTokenizer.from_pretrained(str(quantize_dir))
            
            # Test with multiple prompts suitable for Phi-1.5
            test_prompts = [
                "Hello! How are you today?",
                "What is the capital of France?",
                "Explain what artificial intelligence is:",
                "Write a simple Python function:"
            ]
            
            for i, test_prompt in enumerate(test_prompts):
                self.log_message(f"🧪 Test {i+1}: '{test_prompt}'")
                
                # Encode the prompt
                inputs = tokenizer.encode(test_prompt, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_length=inputs.shape[1] + 50,  # Generate 50 new tokens
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        do_sample=True,
                        top_k=40,
                        top_p=0.9,
                        temperature=0.7,
                        repetition_penalty=1.1,
                        eos_token_id=tokenizer.eos_token_id,
                        no_repeat_ngram_size=3,  # Prevent repetitive sequences
                    )
                
                # Decode the full response
                full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract just the generated part (everything after the prompt)
                generated_text = full_response[len(test_prompt):].strip()
                
                self.log_message(f"  → Generated: {generated_text[:100]}{'...' if len(generated_text) > 100 else ''}")
                
                # Stop after first successful generation to avoid spam
                if generated_text and len(generated_text) > 5:
                    break
            
            self.log_message(f"✓ Quantized model test completed!")
            
        except Exception as e:
            self.log_message(f"⚠ Warning: Could not test quantized model: {e}")
            self.log_message("  Model files created, but manual testing recommended")
            
    def stop_training(self):
        """Stop the current operation (training, ONNX conversion, or quantization)"""
        self.is_training = False
        
        # Determine what operation is being stopped based on current status
        current_status = getattr(self, 'current_operation', 'operation')
        
        if 'training' in current_status.lower():
            self.log_message("🛑 Training operation interrupted by user")
        elif 'export' in current_status.lower() or 'convert' in current_status.lower():
            self.log_message("🛑 ONNX conversion operation interrupted by user")
        elif 'quantiz' in current_status.lower():
            self.log_message("🛑 Quantization operation interrupted by user")
        else:
            self.log_message("🛑 Current operation interrupted by user")
        
        # Update UI immediately
        self.update_task_status("Stopped by user - Ready for next task")
        
    def training_finished(self):
        """Clean up after training is finished"""
        self.root.after(0, self._training_finished_ui)
        
    def _training_finished_ui(self):
        """Update UI after training is finished (must run in main thread)"""
        self.is_training = False
        self.train_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.update_task_status("Training completed - Ready for next task")
        
        # Restore original training device display
        self.update_training_device_display("auto")
    
    def refresh_model_list(self):
        """Refresh the list of available ONNX models"""
        try:
            output_dir = Path("./output")
            models = []
            
            if output_dir.exists():
                # Find all 3_quantized directories (now in timestamped folders)
                for quantized_dir in output_dir.glob("**/3_quantized"):
                    if (quantized_dir / "model.onnx").exists():
                        # Get timestamped parent directory name for display
                        parent_name = quantized_dir.parent.name
                        # Extract meaningful parts from the timestamped name
                        # Format: yyyy_mm_dd_hh_mm_ss_modelName_actions
                        parts = parent_name.split('_')
                        if len(parts) >= 6:
                            # Get date/time part
                            date_part = f"{parts[0]}-{parts[1]}-{parts[2]} {parts[3]}:{parts[4]}:{parts[5]}"
                            # Get model and action parts (everything after timestamp)
                            remaining_parts = '_'.join(parts[6:])
                            display_name = f"{date_part} - {remaining_parts}"
                        else:
                            # Fallback to simple parent name
                            display_name = f"{parent_name}"
                        
                        models.append((str(quantized_dir), display_name))
                
                # Sort by modification time (newest first)
                models.sort(key=lambda x: Path(x[0]).stat().st_mtime, reverse=True)
            
            # Update combobox
            self.model_list_combo['values'] = [display_name for _, display_name in models]
            self.model_paths = {display_name: path for path, display_name in models}
            
            if models:
                # Select the most recent model
                self.model_list_combo.set(models[0][1])
                self.model_path_var.set(models[0][0])
                self.generate_button.config(state='normal')
            else:
                self.model_list_combo.set("")
                self.model_path_var.set("")
                self.generate_button.config(state='disabled')
                
        except Exception as e:
            self.tech_log(f"❌ Error refreshing model list: {e}")
    
    def on_model_selected(self, event=None):
        """Handle model selection from dropdown"""
        selected = self.model_list_combo.get()
        if selected in self.model_paths:
            self.model_path_var.set(self.model_paths[selected])
            self.generate_button.config(state='normal')
            self.tech_log(f"📁 Selected model: {selected}")
            # Update memory usage display when model changes
            if hasattr(self, 'update_memory_usage_display'):
                self.update_memory_usage_display()
    
    def browse_onnx_model(self):
        """Browse for ONNX model directory"""
        from tkinter import filedialog
        
        dirname = filedialog.askdirectory(
            title="Select ONNX Model Directory (containing model.onnx)",
            initialdir="./output"
        )
        
        if dirname:
            model_path = Path(dirname)
            if (model_path / "model.onnx").exists():
                self.model_path_var.set(dirname)
                self.generate_button.config(state='normal')
                self.tech_log(f"📁 Selected model directory: {dirname}")
                # Update memory usage display when model changes
                if hasattr(self, 'update_memory_usage_display'):
                    self.update_memory_usage_display()
            else:
                messagebox.showerror("Error", "Selected directory does not contain model.onnx")
    
    def generate_text(self):
        """Clear output and reset to current mode"""
        # Clear the terminal and reset to current mode
        mode = self.communication_mode.get()
        self.on_mode_changed(None)  # This will reset the terminal for current mode
    
    def single_prompt_mode(self):
        """Single prompt generation mode"""
        prompt = self.test_prompt_var.get().strip()
        if not prompt:
            prompt = "Hello, I am"
            self.test_prompt_var.set(prompt)
        
        # Clear previous output
        self.test_output.delete('1.0', tk.END)
        self.test_output.insert(tk.END, "📝 Generated Text:\n")
        self.test_output.insert(tk.END, "=" * 30 + "\n\n")
        
        # Start generation
        self._start_generation(prompt, "generate_text")
    
    def interactive_chat_mode(self):
        """Interactive chat mode with conversation history"""
        prompt = self.test_prompt_var.get().strip()
        if not prompt:
            messagebox.showwarning("Warning", "Please enter a message to send")
            return
        
        # Initialize conversation history if not exists
        if not hasattr(self, 'conversation_history'):
            self.conversation_history = []
        
        # Display user message
        self.test_output.insert(tk.END, f"\n💬 You: {prompt}\n")
        self.test_output.insert(tk.END, "🤖 Assistant: ")
        self.test_output.see(tk.END)
        
        # For chat, use just the user's message with a conversational prompt
        chat_prompt = f"Human: {prompt}\nAssistant:"
        
        # Clear prompt for next input
        self.test_prompt_var.set("")
        
        # Add to conversation history for context tracking
        self.conversation_history.append(f"Human: {prompt}")
        
        # Dynamic context management - check and reset if needed BEFORE generation
        if hasattr(self, 'conversation_history') and self.conversation_history:
            self._check_and_reset_context_before_generation()
        
        # Start generation
        self._start_generation(chat_prompt, "chat_conversation")
    
    def _manage_conversation_context(self):
        """Dynamically manage conversation context with intelligent reset"""
        if not hasattr(self, 'conversation_history') or not self.conversation_history:
            return
        
        # Calculate context metrics
        full_context = "\n".join(self.conversation_history)
        context_length = len(full_context)
        history_count = len(self.conversation_history)
        
        # Estimate token count (rough approximation: 1 token ≈ 4 characters)
        estimated_tokens = context_length // 4
        
        # Dynamic thresholds based on context growth
        base_token_limit = 1000
        base_history_limit = 16
        base_char_limit = 4000
        
        # Progressive scaling - tighter limits as conversation grows
        if history_count > 20:
            token_limit = base_token_limit * 0.8  # 800 tokens
            char_limit = base_char_limit * 0.8    # 3200 chars
        elif history_count > 12:
            token_limit = base_token_limit * 0.9  # 900 tokens
            char_limit = base_char_limit * 0.9    # 3600 chars
        else:
            token_limit = base_token_limit         # 1000 tokens
            char_limit = base_char_limit          # 4000 chars
        
        # Check if reset is needed
        should_reset = (
            estimated_tokens > token_limit or
            history_count > base_history_limit or
            context_length > char_limit or
            self._detect_repetitive_patterns()
        )
        
        if should_reset:
            old_count = history_count
            old_tokens = estimated_tokens
            
            # Smart reset - keep the most recent meaningful exchanges
            if history_count > 8:
                # Keep last 6 entries (3 user-AI pairs)
                self.conversation_history = self.conversation_history[-6:]
            else:
                # Full reset for short conversations
                self.conversation_history = []
            
            # Notify user about the reset
            self.test_output.insert(tk.END, "\n🔄 chat> Context reset! (Previous conversation was getting too long)\n")
            self.test_output.see(tk.END)
            
            # Log the reset for debugging
            if hasattr(self, 'tech_log'):
                self.tech_log(f"🔄 Dynamic context reset: {old_count} entries ({old_tokens} tokens) → {len(self.conversation_history)} entries")
    
    def _detect_repetitive_patterns(self):
        """Detect if conversation is becoming repetitive"""
        if not hasattr(self, 'conversation_history') or len(self.conversation_history) < 6:
            return False
        
        # Check last few AI responses for similarity
        ai_responses = [line for line in self.conversation_history[-6:] if line.startswith("AI:")]
        
        if len(ai_responses) >= 2:
            # Simple repetition detection
            last_response = ai_responses[-1].lower()
            for prev_response in ai_responses[:-1]:
                # If responses are very similar, consider it repetitive
                if len(last_response) > 10 and last_response in prev_response.lower():
                    return True
        
        return False
    
    def _check_and_reset_context_before_input(self, new_prompt):
        """Check context size before adding new input and reset if needed"""
        if not hasattr(self, 'conversation_history') or not self.conversation_history:
            return
        
        # Calculate what the context would be with the new input
        current_context = "\n".join(self.conversation_history)
        new_entry = f"User: {new_prompt}\nAI:"
        projected_context = current_context + "\n" + new_entry
        
        projected_length = len(projected_context)
        projected_tokens = projected_length // 4
        history_count = len(self.conversation_history)
        
        # Very aggressive limits to prevent overloading
        MAX_TOKENS = 600      # Much lower limit
        MAX_HISTORY = 10      # Much fewer entries  
        MAX_CHARS = 2400      # Much smaller character limit
        
        # Check if we need to reset
        should_reset = (
            projected_tokens > MAX_TOKENS or
            history_count > MAX_HISTORY or
            projected_length > MAX_CHARS
        )
        
        if should_reset:
            old_count = len(self.conversation_history)
            # Aggressive reset - keep only last 4 entries (2 exchanges)
            if old_count > 4:
                self.conversation_history = self.conversation_history[-4:]
            else:
                self.conversation_history = []
            
            # Notify user
            self.test_output.insert(tk.END, "\n🔄 chat> Context reset! (Preventing model overload)\n")
            self.test_output.see(tk.END)
            
            # Log the reset
            if hasattr(self, 'tech_log'):
                self.tech_log(f"🔄 Aggressive context reset: {old_count} → {len(self.conversation_history)} entries")
                self.tech_log(f"   Projected: {projected_tokens} tokens, {projected_length} chars")
    
    def _check_and_reset_context_before_generation(self):
        """Check context before generation starts - for interactive chat"""
        if not hasattr(self, 'conversation_history') or not self.conversation_history:
            return
        
        # Current context analysis
        current_context = "\n".join(self.conversation_history)
        context_length = len(current_context)
        estimated_tokens = context_length // 4
        history_count = len(self.conversation_history)
        
        # Aggressive limits to prevent overloading  
        MAX_TOKENS = 500      # Very low limit
        MAX_HISTORY = 8       # Very few entries
        MAX_CHARS = 2000      # Small character limit
        
        should_reset = (
            estimated_tokens > MAX_TOKENS or
            history_count > MAX_HISTORY or
            context_length > MAX_CHARS
        )
        
        if should_reset:
            old_count = history_count
            # Keep only the most recent entry (the one we just added)
            if old_count > 2:
                self.conversation_history = self.conversation_history[-2:]  # Keep last user input
            else:
                self.conversation_history = self.conversation_history[-1:]  # Keep just the current input
                
            # Notify user
            self.test_output.insert(tk.END, "\n🔄 chat> Context reset! (Preventing overload)\n")
            self.test_output.see(tk.END)
            
            # Log the reset
            if hasattr(self, 'tech_log'):
                self.tech_log(f"🔄 Pre-generation reset: {old_count} → {len(self.conversation_history)} entries")
                self.tech_log(f"   Context was: {estimated_tokens} tokens, {context_length} chars")
    
    def _post_generation_context_check(self):
        """Check and manage context after generation completes"""
        if not hasattr(self, 'conversation_history') or not self.conversation_history:
            return
        
        # Get current context size
        full_context = "\n".join(self.conversation_history)
        context_length = len(full_context)
        estimated_tokens = context_length // 4
        
        # More aggressive post-generation limits since we just added a response
        if estimated_tokens > 1200 or len(self.conversation_history) > 18 or context_length > 4800:
            # Keep only the most recent conversation
            if len(self.conversation_history) > 8:
                self.conversation_history = self.conversation_history[-8:]
                if hasattr(self, 'tech_log'):
                    self.tech_log(f"🔄 Post-generation context trim: kept last 8 entries")
    
    def batch_processing_mode(self):
        """Batch processing mode for multiple prompts"""
        # Get prompts from the text area (each line is a prompt)
        all_text = self.test_output.get('1.0', tk.END).strip()
        prompts = [line.strip() for line in all_text.split('\n') if line.strip()]
        
        if not prompts:
            # If no prompts in output area, use the prompt entry
            prompt = self.test_prompt_var.get().strip()
            if not prompt:
                messagebox.showwarning("Warning", "Please enter prompts in the output area (one per line) or a single prompt in the input field")
                return
            prompts = [prompt]
        
        # Clear output and show batch info
        self.test_output.delete('1.0', tk.END)
        self.test_output.insert(tk.END, f"📋 Batch Processing Mode - {len(prompts)} prompts\n")
        self.test_output.insert(tk.END, "=" * 50 + "\n\n")
        
        # Disable generate button and enable stop button
        self.generate_button.config(state='disabled')
        self.stop_generation_button.config(state='normal')
        
        # Start batch generation in separate thread
        self.generation_thread = threading.Thread(
            target=self.run_batch_generation,
            args=(self.model_path_var.get(), prompts),
            daemon=True
        )
        self.generation_thread.start()
    
    def creative_writing_mode(self):
        """Creative writing mode with storytelling enhancements"""
        prompt = self.test_prompt_var.get().strip()
        if not prompt:
            messagebox.showwarning("Warning", "Please enter a creative writing prompt")
            return
        
        # Add creative writing context
        enhanced_prompt = f"Creative writing request: {prompt}\n\nPlease write creatively and engagingly:"
        
        # Clear previous output with creative header
        self.test_output.delete('1.0', tk.END)
        self.test_output.insert(tk.END, "✍️ Creative Writing Response:\n")
        self.test_output.insert(tk.END, "=" * 40 + "\n\n")
        
        # Start generation
        self._start_generation(enhanced_prompt, "creative_writing")
    
    def question_answering_mode(self):
        """Question answering mode with factual response focus"""
        prompt = self.test_prompt_var.get().strip()
        if not prompt:
            messagebox.showwarning("Warning", "Please enter a question")
            return
        
        # Add Q&A context
        enhanced_prompt = f"Question: {prompt}\n\nAnswer:"
        
        # Clear previous output with Q&A header
        self.test_output.delete('1.0', tk.END)
        self.test_output.insert(tk.END, f"❓ Question: {prompt}\n")
        self.test_output.insert(tk.END, "💡 Answer: ")
        
        # Start generation
        self._start_generation(enhanced_prompt, "question_answering")
    
    def code_completion_mode(self):
        """Code completion mode with programming context"""
        prompt = self.test_prompt_var.get().strip()
        if not prompt:
            messagebox.showwarning("Warning", "Please enter code to complete")
            return
        
        # Add code context
        enhanced_prompt = f"Complete this code:\n{prompt}"
        
        # Clear previous output with code header
        self.test_output.delete('1.0', tk.END)
        self.test_output.insert(tk.END, "💻 Code Input:\n")
        self.test_output.insert(tk.END, f"{prompt}\n\n")
        self.test_output.insert(tk.END, "🔧 Completion:\n")
        
        # Start generation
        self._start_generation(enhanced_prompt, "code_completion")
    
    def _start_generation(self, prompt, mode):
        """Helper method to start generation for different modes"""
        # Disable generate button and enable stop button
        self.generate_button.config(state='disabled')
        self.stop_generation_button.config(state='normal')
        
        # Start generation in separate thread
        self.generation_thread = threading.Thread(
            target=self.run_generation,
            args=(self.model_path_var.get(), prompt, mode),
            daemon=True
        )
        self.generation_thread.start()
    
    def run_generation(self, model_path, prompt, mode="single"):
        """Run text generation in background thread"""
        
        # Validate generation parameters
        try:
            max_length = max(1, min(512, self.test_max_length.get()))
            min_length = max(1, min(max_length - 1, self.test_min_length.get()))
            temperature = max(0.1, min(2.0, self.test_temperature.get()))
            top_p = max(0.1, min(1.0, self.test_top_p.get()))
            top_k = max(1, min(100, self.test_top_k.get()))
            repetition_penalty = max(1.0, min(2.0, self.test_repetition_penalty.get()))
            no_repeat_ngram_size = max(0, min(5, self.test_no_repeat_ngram.get()))
            length_penalty = max(0.5, min(2.0, self.test_length_penalty.get()))
            
            self.tech_log("🎛️ Generation Parameters:")
            self.tech_log(f"   Length: {min_length} to {max_length} tokens")
            self.tech_log(f"   Sampling: Temperature={temperature}, Top-p={top_p}, Top-k={top_k}")
            self.tech_log(f"   Quality: Rep.penalty={repetition_penalty}, No-repeat={no_repeat_ngram_size}-gram")
            self.tech_log(f"   Control: Do sample={self.test_do_sample.get()}, Early stop={self.test_early_stopping.get()}")
            
        except Exception as e:
            self.tech_log(f"❌ Parameter validation error: {e}")
            return
        try:
            self.tech_log(f"🚀 Starting text generation... Mode: {mode}")
            self.tech_log(f"📁 Model: {model_path}")
            self.tech_log(f"💬 Prompt: '{prompt}'")
            self.tech_log(f"⚙️ Max length: {self.test_max_length.get()}, Min length: {self.test_min_length.get()}")
            self.tech_log(f"🌡️ Temperature: {self.test_temperature.get()}, Top-p: {self.test_top_p.get()}, Top-k: {self.test_top_k.get()}")
            self.tech_log(f"🔄 Repetition penalty: {self.test_repetition_penalty.get()}, No-repeat n-gram: {self.test_no_repeat_ngram.get()}")
            self.tech_log(f"📊 Sampling: {self.test_do_sample.get()}, Early stopping: {self.test_early_stopping.get()}, Length penalty: {self.test_length_penalty.get()}")
            
            # Import required libraries
            # Use global AutoTokenizer import
            # Use global numpy and onnxruntime imports
            ort = onnxruntime
            np = numpy
            
            # Load tokenizer
            self.tech_log("📦 Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load ONNX model with GPU support and fallback
            self.tech_log("🔧 Loading ONNX model...")
            session = self.create_onnx_session(str(Path(model_path) / "model.onnx"))
            
            # Show device info to user
            providers = session.get_providers()
            device_info = "🖥️ CPU"
            if any('CUDA' in p for p in providers):
                device_info = "🎮 GPU (CUDA)"
            elif any('ROCm' in p.upper() for p in providers):
                device_info = "🎮 GPU (ROCm)"
            elif any('OpenVINO' in p for p in providers):
                device_info = "⚡ Intel OpenVINO"
            
            # Tokenize input
            input_ids = tokenizer.encode(prompt, return_tensors="np")
            self.tech_log(f"✅ Model loaded. Input tokens: {input_ids.shape[1]}")
            
            # Generate text
            generated_ids = input_ids.copy()
            max_length = self.test_max_length.get()
            min_length = self.test_min_length.get()
            temperature = self.test_temperature.get()
            top_p = self.test_top_p.get()
            top_k = self.test_top_k.get()
            repetition_penalty = self.test_repetition_penalty.get()
            no_repeat_ngram_size = self.test_no_repeat_ngram.get()
            do_sample = self.test_do_sample.get()
            early_stopping = self.test_early_stopping.get()
            length_penalty = self.test_length_penalty.get()
            
            # Track generated n-grams for repetition avoidance
            generated_ngrams = set()
            
            # Performance optimization: use simplified mode for small models
            use_fast_mode = max_length <= 200 and input_ids.shape[1] <= 100
            if use_fast_mode:
                self.tech_log("🚀 Using fast generation mode for small context")
            
            start_time = time.time()
            
            for step in range(max_length - input_ids.shape[1]):
                if not hasattr(self, 'generation_thread') or not self.generation_thread.is_alive():
                    break
                
                # Prepare inputs
                inputs = {"input_ids": generated_ids.astype(np.int64)}
                
                # Get input names from the model
                input_names = [inp.name for inp in session.get_inputs()]
                
                # Add attention mask if needed
                if "attention_mask" in input_names:
                    attention_mask = np.ones_like(generated_ids, dtype=np.int64)
                    inputs["attention_mask"] = attention_mask
                
                # Add position_ids if needed
                if "position_ids" in input_names:
                    position_ids = np.arange(generated_ids.shape[1], dtype=np.int64).reshape(1, -1)
                    inputs["position_ids"] = position_ids
                
                # Add past key values if needed (for cached models)
                for inp in session.get_inputs():
                    if inp.name.startswith("past_key_values"):
                        # For the first step, initialize with empty cache (zeros)
                        # For subsequent steps, we would need to maintain the cache
                        # For simplicity, we'll use zeros which may affect quality but will work
                        shape = inp.shape
                        # Replace dynamic dimensions with concrete values
                        concrete_shape = []
                        for dim in shape:
                            if isinstance(dim, str) or dim == -1:
                                # For sequence dimension, start with 0 for past
                                if "sequence" in str(dim).lower() or dim == -1:
                                    concrete_shape.append(0)
                                else:
                                    concrete_shape.append(1)  # Batch size
                            else:
                                concrete_shape.append(dim)
                        
                        # Create zero tensor for past key values
                        past_kv = np.zeros(concrete_shape, dtype=np.float32)
                        inputs[inp.name] = past_kv
                
                # Run inference
                try:
                    outputs = session.run(None, inputs)
                    logits = outputs[0]
                except Exception as e:
                    if use_fast_mode:
                        # Fast mode: minimal error handling for performance
                        self.tech_log(f"❌ Generation error: {str(e)}")
                        break
                    else:
                        # Regular mode: comprehensive error handling
                        error_msg = str(e)
                    self.tech_log(f"❌ Generation error: {error_msg}")
                    self.tech_log(f"   Expected inputs: {[inp.name for inp in session.get_inputs()]}")
                    self.tech_log(f"   Provided inputs: {list(inputs.keys())}")
                    self.tech_log(f"   Input shapes: {[inputs[k].shape for k in inputs.keys()]}")
                    
                    # Check for common issues and apply intelligent recovery
                    if "invalid" in error_msg.lower() or "shape" in error_msg.lower():
                        self.tech_log("💡 Shape mismatch detected - this may indicate context overflow")
                    elif "memory" in error_msg.lower():
                        self.tech_log("💡 Memory issue - context may be too long")
                    elif "range" in error_msg.lower() or "bounds" in error_msg.lower():
                        self.tech_log("💡 Index out of range - likely context length exceeded")
                    
                    # Intelligent context recovery based on sequence length
                    current_length = generated_ids.shape[1]
                    
                    if current_length > 512:  # Long context detected
                        self.tech_log(f"� Long context detected ({current_length} tokens). Applying intelligent truncation...")
                        
                        # Strategy: Keep the beginning (context) and recent tokens, remove middle
                        context_tokens = 200  # Keep first 200 tokens as context
                        recent_tokens = 200   # Keep last 200 tokens as recent conversation
                        
                        if current_length > context_tokens + recent_tokens:
                            # Extract beginning and end
                            beginning = generated_ids[:, :context_tokens]
                            end = generated_ids[:, -recent_tokens:]
                            
                            # Concatenate with a small gap indicator
                            # Use a neutral token (space or period) as separator
                            separator_token = np.array([[tokenizer.encode(" ")[0]]], dtype=np.int64)
                            generated_ids = np.concatenate([beginning, separator_token, end], axis=1)
                            
                            self.tech_log(f"✅ Context intelligently truncated: {beginning.shape[1]} + 1 + {end.shape[1]} = {generated_ids.shape[1]} tokens")
                        else:
                            # Simple truncation if intelligent method doesn't apply
                            keep_tokens = min(400, current_length // 2)
                            generated_ids = generated_ids[:, -keep_tokens:]
                            self.tech_log(f"✅ Simple truncation applied: keeping last {keep_tokens} tokens")
                        
                        # Update inputs with truncated context
                        inputs["input_ids"] = generated_ids.astype(np.int64)
                        if "attention_mask" in inputs:
                            inputs["attention_mask"] = np.ones_like(generated_ids, dtype=np.int64)
                        if "position_ids" in inputs:
                            inputs["position_ids"] = np.arange(generated_ids.shape[1], dtype=np.int64).reshape(1, -1)
                        
                        # Retry inference with recovered context
                        try:
                            outputs = session.run(None, inputs)
                            logits = outputs[0]
                            self.tech_log("✅ Generation recovered with intelligent context management")
                        except Exception as e2:
                            self.tech_log(f"❌ Intelligent recovery failed: {str(e2)}")
                            
                            # Last resort: very aggressive truncation
                            if generated_ids.shape[1] > 100:
                                generated_ids = generated_ids[:, -100:]  # Keep only last 100 tokens
                                inputs["input_ids"] = generated_ids.astype(np.int64)
                                if "attention_mask" in inputs:
                                    inputs["attention_mask"] = np.ones_like(generated_ids, dtype=np.int64)
                                if "position_ids" in inputs:
                                    inputs["position_ids"] = np.arange(generated_ids.shape[1], dtype=np.int64).reshape(1, -1)
                                
                                try:
                                    outputs = session.run(None, inputs)
                                    logits = outputs[0]
                                    self.tech_log("🆘 Emergency recovery successful with aggressive truncation")
                                except Exception as e3:
                                    self.tech_log(f"🆘 Emergency recovery failed: {str(e3)}")
                                    break
                            else:
                                break
                    elif current_length > 200:  # Medium context
                        self.tech_log(f"🔄 Medium context detected ({current_length} tokens). Applying standard truncation...")
                        # Standard truncation for medium contexts
                        generated_ids = generated_ids[:, -300:]  # Keep last 300 tokens
                        
                        # Update inputs
                        inputs["input_ids"] = generated_ids.astype(np.int64)
                        if "attention_mask" in inputs:
                            inputs["attention_mask"] = np.ones_like(generated_ids, dtype=np.int64)
                        if "position_ids" in inputs:
                            inputs["position_ids"] = np.arange(generated_ids.shape[1], dtype=np.int64).reshape(1, -1)
                        
                        try:
                            outputs = session.run(None, inputs)
                            logits = outputs[0]
                            self.tech_log("✅ Generation recovered with standard truncation")
                        except Exception as e2:
                            self.tech_log(f"❌ Standard recovery failed: {str(e2)}")
                            break
                    else:
                        # Short context but still failing - probably a different issue
                        self.tech_log("❌ Error not related to context length. Stopping generation.")
                        break
                
                # Apply temperature scaling
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    # Apply penalty to tokens already in the sequence
                    for token_id in generated_ids[0]:
                        if next_token_logits[token_id] < 0:
                            next_token_logits[token_id] = next_token_logits[token_id] * repetition_penalty
                        else:
                            next_token_logits[token_id] = next_token_logits[token_id] / repetition_penalty
                
                # Apply no-repeat n-gram filtering
                if no_repeat_ngram_size > 0 and generated_ids.shape[1] >= no_repeat_ngram_size:
                    # Get the last (n-1) tokens to form n-grams
                    last_ngram_tokens = generated_ids[0, -(no_repeat_ngram_size-1):].tolist()
                    
                    # For each possible next token, check if it would form a repeated n-gram
                    for next_token_candidate in range(len(next_token_logits)):
                        candidate_ngram = tuple(last_ngram_tokens + [next_token_candidate])
                        if candidate_ngram in generated_ngrams:
                            next_token_logits[next_token_candidate] = -float('inf')
                
                # Apply Top-k filtering
                if top_k > 0:
                    top_k_indices = np.argpartition(next_token_logits, -top_k)[-top_k:]
                    top_k_mask = np.zeros_like(next_token_logits, dtype=bool)
                    top_k_mask[top_k_indices] = True
                    next_token_logits[~top_k_mask] = -float('inf')
                
                # Apply Top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_indices = np.argsort(next_token_logits)[::-1]
                    sorted_logits = next_token_logits[sorted_indices]
                    cumulative_probs = np.cumsum(self.softmax(sorted_logits))
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                    sorted_indices_to_remove[0] = False
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token or use greedy selection
                if do_sample:
                    probabilities = self.softmax(next_token_logits)
                    # Ensure probabilities are valid
                    if np.any(np.isnan(probabilities)) or np.sum(probabilities) == 0:
                        # Fallback to greedy selection if sampling fails
                        next_token = np.argmax(next_token_logits)
                    else:
                        next_token = np.random.choice(len(probabilities), p=probabilities)
                else:
                    # Greedy selection
                    next_token = np.argmax(next_token_logits)
                
                # Add to sequence
                generated_ids = np.concatenate([generated_ids, [[next_token]]], axis=1)
                
                # Update n-gram tracking
                if no_repeat_ngram_size > 0 and generated_ids.shape[1] >= no_repeat_ngram_size:
                    new_ngram = tuple(generated_ids[0, -no_repeat_ngram_size:].tolist())
                    generated_ngrams.add(new_ngram)
                
                # Update output based on mode - optimize by updating less frequently
                current_length = generated_ids.shape[1] - input_ids.shape[1]
                # Update UI every 3 tokens or on final token to reduce overhead
                should_update_ui = (current_length % 3 == 0) or (current_length >= max_length)
                
                if should_update_ui:
                    # Decode once and reuse for different modes to avoid redundant tokenization
                    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    
                    if mode in ["generate_text", "single"]:
                        self.root.after(0, lambda text=full_text: self.update_generation_output(text))
                    elif mode in ["chat_conversation", "interactive"]:
                        # For chat, extract only the Assistant's response
                        if "Assistant:" in full_text:
                            response_start = full_text.find("Assistant:") + len("Assistant:")
                            assistant_response = full_text[response_start:].strip()
                        else:
                            # Fallback: show text after the original prompt
                            assistant_response = full_text[len(prompt):].strip() if len(full_text) > len(prompt) else full_text
                        
                        # Update the current response display
                        self.root.after(0, lambda text=assistant_response: self.update_chat_response(text))
                    elif mode.endswith("_terminal"):
                        # Handle terminal modes - only update at the end, not incrementally
                        # Don't update during generation to avoid spam
                        pass
                    elif mode in ["creative_writing", "question_answering", "code_completion"]:
                        # Show full response for specialized modes
                        # For these modes, show only the generated part after the prompt
                        generated_part = full_text[len(prompt):] if len(full_text) > len(prompt) else full_text
                        self.root.after(0, lambda text=generated_part: self.append_generation_output(text))
                
                # Check for end of sequence with min_length and early_stopping consideration
                
                if next_token == tokenizer.eos_token_id:
                    if current_length >= min_length:
                        # We've reached EOS and minimum length, can stop
                        if early_stopping:
                            break
                        # If early_stopping is False, continue generation (ignore EOS)
                    # If we haven't reached min_length, continue generation (ignore EOS)
                
                # Check if we've reached maximum length
                if current_length >= max_length:
                    break
            
            generation_time = time.time() - start_time
            tokens_generated = generated_ids.shape[1] - input_ids.shape[1]
            
            # Handle mode-specific completion
            if mode in ["chat_conversation", "interactive"]:
                # Add assistant response to conversation history
                full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                response = full_text[len(prompt):] if len(full_text) > len(prompt) else ""
                if hasattr(self, 'conversation_history'):
                    self.conversation_history.append(f"Assistant: {response}")
                    # Check context after adding response - manage if needed
                    self.root.after(0, lambda: self._post_generation_context_check())
                self.root.after(0, lambda: self.test_output.insert(tk.END, "\n\n"))
            elif mode.endswith("_terminal"):
                # Handle terminal modes - display final result
                full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                generated_part = full_text[len(prompt):] if len(full_text) > len(prompt) else ""
                self.root.after(0, lambda text=generated_part: self.update_terminal_output(text))
            elif mode in ["creative_writing", "question_answering", "code_completion"]:
                # Add completion markers for specialized modes
                self.root.after(0, lambda: self.test_output.insert(tk.END, "\n\n--- End of Response ---\n"))
            
            self.tech_log(f"✅ Generation complete!")
            self.tech_log(f"📊 Tokens generated: {tokens_generated}, Time: {generation_time:.2f}s, Speed: {tokens_generated/generation_time:.1f} tokens/sec")
            
        except Exception as e:
            self.tech_log(f"❌ Generation error: {str(e)}")
        finally:
            # Re-enable controls
            self.root.after(0, self.generation_finished)
    
    def softmax(self, x):
        """Compute softmax values"""
        # Use global numpy import
        np = numpy
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def run_batch_generation(self, model_path, prompts):
        """Run batch text generation for multiple prompts"""
        try:
            self.tech_log(f"🚀 Starting batch generation for {len(prompts)} prompts...")
            
            # Import required libraries
            # Use global AutoTokenizer import
            # Use global imports
            ort = onnxruntime
            np = numpy
            
            # Load tokenizer and model once
            self.tech_log("📦 Loading tokenizer and model...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            session = self.create_onnx_session(str(Path(model_path) / "model.onnx"))
            
            total_start_time = time.time()
            
            for i, prompt in enumerate(prompts, 1):
                if not hasattr(self, 'generation_thread') or not self.generation_thread.is_alive():
                    break
                
                self.root.after(0, lambda p=prompt, idx=i: self.test_output.insert(tk.END, f"📝 Prompt {idx}: {p}\n🤖 Response: "))
                
                # Generate for this prompt
                input_ids = tokenizer.encode(prompt, return_tensors="np")
                generated_ids = input_ids.copy()
                max_length = min(self.test_max_length.get(), input_ids.shape[1] + 100)  # Shorter for batch
                
                for step in range(max_length - input_ids.shape[1]):
                    if not hasattr(self, 'generation_thread') or not self.generation_thread.is_alive():
                        break
                    
                    # Same generation logic as single mode but optimized
                    inputs = {"input_ids": generated_ids.astype(np.int64)}
                    
                    # Add required inputs
                    input_names = [inp.name for inp in session.get_inputs()]
                    if "attention_mask" in input_names:
                        inputs["attention_mask"] = np.ones_like(generated_ids, dtype=np.int64)
                    if "position_ids" in input_names:
                        inputs["position_ids"] = np.arange(generated_ids.shape[1], dtype=np.int64).reshape(1, -1)
                    
                    # Add past key values
                    for inp in session.get_inputs():
                        if inp.name.startswith("past_key_values"):
                            shape = inp.shape
                            concrete_shape = []
                            for dim in shape:
                                if isinstance(dim, str) or dim == -1:
                                    concrete_shape.append(0 if "sequence" in str(dim).lower() else 1)
                                else:
                                    concrete_shape.append(dim)
                            inputs[inp.name] = np.zeros(concrete_shape, dtype=np.float32)
                    
                    try:
                        outputs = session.run(None, inputs)
                        logits = outputs[0]
                    except Exception as e:
                        self.tech_log(f"❌ Error in batch item {i}: {str(e)}")
                        break
                    
                    # Simplified sampling for batch mode
                    next_token_logits = logits[0, -1, :] / self.test_temperature.get()
                    probabilities = self.softmax(next_token_logits)
                    next_token = np.random.choice(len(probabilities), p=probabilities)
                    
                    generated_ids = np.concatenate([generated_ids, [[next_token]]], axis=1)
                    
                    if next_token == tokenizer.eos_token_id:
                        break
                
                # Get final response for this prompt
                full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                response = full_text[len(prompt):] if len(full_text) > len(prompt) else ""
                
                self.root.after(0, lambda r=response: self.test_output.insert(tk.END, f"{r}\n\n"))
                self.root.after(0, lambda: self.test_output.see(tk.END))
            
            total_time = time.time() - total_start_time
            self.tech_log(f"✅ Batch generation complete! Processed {len(prompts)} prompts in {total_time:.2f}s")
            
        except Exception as e:
            self.tech_log(f"❌ Error during batch generation: {str(e)}")
            self.tech_log(f"🔍 Traceback: {traceback.format_exc()}")
        finally:
            self.root.after(0, self.generation_finished)
    
    def update_generation_output(self, text):
        """Update the generation output in the UI"""
        self.test_output.delete('1.0', tk.END)
        self.test_output.insert('1.0', text)
        self.test_output.see(tk.END)
    
    def append_generation_output(self, text):
        """Append text to the generation output in the UI"""
        self.test_output.insert(tk.END, text)
        self.test_output.see(tk.END)
    
    def generation_finished(self):
        """Re-enable controls after generation"""
        self.generate_button.config(state='normal')
        self.stop_generation_button.config(state='disabled')
    
    def generation_complete(self):
        """Alias for generation_finished"""
        self.generation_finished()
    
    def update_chat_response(self, response):
        """Update the output for chat conversation mode with clean response"""
        # Clear the output area for clean display
        self.test_output.delete(1.0, tk.END)
        
        # Add conversation history
        for exchange in self.conversation_history:
            self.test_output.insert(tk.END, f"Human: {exchange['human']}\n\n")
            self.test_output.insert(tk.END, f"Assistant: {exchange['assistant']}\n\n")
        
        # Add the latest response
        if hasattr(self, 'current_prompt'):
            self.test_output.insert(tk.END, f"Human: {self.current_prompt}\n\n")
        self.test_output.insert(tk.END, f"Assistant: {response}\n\n")
        
        # Scroll to the end
        self.test_output.see(tk.END)
    
    def remove_thinking_indicator(self):
        """Remove the thinking indicator from the terminal output"""
        try:
            # Get all content
            content = self.test_output.get('1.0', tk.END)
            
            # Check if the last non-empty line contains any thinking indicator
            lines = content.split('\n')
            
            # List of possible thinking indicators
            thinking_indicators = [
                "🤔 AI is thinking...",
                "💭 Processing your request...", 
                "⚡ Generating response...",
                "🧠 Analyzing and responding...",
                "✨ Crafting a response..."
            ]
            
            # Find and remove any thinking indicator line
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i].strip()
                if line and any(indicator in line for indicator in thinking_indicators):
                    # Remove this line by deleting it from the text widget
                    line_start = f"{i+1}.0"
                    line_end = f"{i+2}.0"
                    self.test_output.delete(line_start, line_end)
                    break
                    
        except Exception as e:
            # If we can't remove it cleanly, just log the error
            self.tech_log(f"⚠️ Could not remove thinking indicator: {e}")
    
    def update_terminal_output(self, generated_text):
        """Update the terminal output with generated text - improved for distilgpt2"""
        try:
            # Remove thinking indicator if it exists
            self.remove_thinking_indicator()
            
            if not generated_text or not generated_text.strip():
                # Model overloaded - emergency context reset and retry
                if hasattr(self, 'conversation_history') and len(self.conversation_history) > 2:
                    old_count = len(self.conversation_history)
                    
                    # Extract the latest user prompt before reset
                    latest_user_prompt = None
                    for entry in reversed(self.conversation_history):
                        if entry.startswith("User: "):
                            latest_user_prompt = entry[6:]  # Remove "User: " prefix
                            break
                    
                    # Reset context
                    self.conversation_history = []  # Complete reset
                    self.test_output.insert(tk.END, "🚨 Model overloaded! Context completely reset.\n")
                    self.tech_log(f"🚨 Emergency reset: cleared {old_count} entries due to overload")
                    
                    # If we found the user's prompt, retry generation with clean context
                    if latest_user_prompt:
                        self.test_output.insert(tk.END, "🔄 Retrying with fresh context...\n")
                        self.tech_log(f"🔄 Retrying generation for: '{latest_user_prompt[:50]}...'")
                        
                        # Add thinking indicator for retry
                        self.test_output.insert(tk.END, "💭 Generating response with clean context...\n")
                        self.test_output.see(tk.END)
                        self.test_output.update()
                        
                        # Retry generation with just the latest prompt and clean context
                        model_path = self.model_path_var.get()
                        if model_path:
                            # Simple retry prompt without accumulated context
                            retry_prompt = f"User: {latest_user_prompt}\nAI:"
                            
                            # Store this as the new conversation history
                            self.conversation_history = [f"User: {latest_user_prompt}", "AI:"]
                            
                            # Start retry generation in separate thread
                            retry_thread = threading.Thread(
                                target=self._retry_generation_after_reset,
                                args=(model_path, retry_prompt, latest_user_prompt),
                                daemon=True
                            )
                            retry_thread.start()
                            return
                        else:
                            self.test_output.insert(tk.END, "❌ Cannot retry: No model selected\n\n")
                    else:
                        self.test_output.insert(tk.END, "❌ Cannot retry: Could not find user prompt\n\n")
                else:
                    self.test_output.insert(tk.END, "AI: [No response generated - model may be overloaded]\n\n")
                self.tech_log("⚠️ Empty generated text received")
                return
            
            # Get the original prompt if available
            prompt_text = getattr(self, 'current_prompt', '')
            
            # Remove the prompt from generated text to get only the response
            if prompt_text and generated_text.startswith(prompt_text):
                response = generated_text[len(prompt_text):].strip()
            else:
                response = generated_text.strip()
            
            self.tech_log(f"🔍 Raw response: '{response[:100]}...' (length: {len(response)})")
            
            # Clean the response with improved logic for distilgpt2
            if response:
                # First, split by lines and clean each line
                lines = [line.strip() for line in response.split('\n') if line.strip()]
                
                ai_response = ""
                
                # Strategy 1: Look for the first clean sentence
                for line in lines:
                    # Skip obvious artifacts
                    if (line and 
                        not line.startswith('#') and 
                        not line.startswith('```') and
                        not line.startswith('def ') and
                        not line.startswith('import ') and
                        not line.startswith('class ') and
                        not line.startswith('User:') and
                        not line.startswith('AI:') and
                        not line.startswith('Q:') and
                        not line.startswith('A:') and
                        len(line) > 3 and  # Must be meaningful length
                        len(line) < 500):  # Not too long to be garbage
                        
                        # Additional filters for distilgpt2 artifacts
                        if not any(artifact in line.lower() for artifact in [
                            'function', 'variable', 'script', 'console', 'error',
                            'undefined', 'null', 'typeof', 'document'
                        ]):
                            ai_response = line
                            break
                
                # Strategy 2: If no good line, try sentence-level extraction
                if not ai_response and response:
                    # Try to extract sentences
                    text = response.replace('\n', ' ').strip()
                    
                    # Split by sentence endings
                    sentences = []
                    for delimiter in ['. ', '! ', '? ']:
                        if delimiter in text:
                            sentences = text.split(delimiter)
                            break
                    
                    if not sentences:
                        sentences = [text]  # Treat whole thing as one sentence
                    
                    for sentence in sentences:
                        sentence = sentence.strip().rstrip('.!?')
                        if (sentence and 
                            len(sentence) > 10 and 
                            len(sentence) < 200 and
                            not any(keyword in sentence.lower() for keyword in 
                                   ['function', 'variable', 'script', 'console', 'def ', 'import '])):
                            ai_response = sentence
                            break
                
                # Strategy 3: If still nothing, use a very basic filter
                if not ai_response and response:
                    # Just take the first reasonable chunk
                    words = response.split()
                    if words and len(words) <= 50:  # Not too many words
                        clean_words = []
                        for word in words:
                            # Stop at first code-like token
                            if any(x in word for x in ['(', ')', '{', '}', '=', ';', 'function']):
                                break
                            clean_words.append(word)
                        
                        if clean_words and len(clean_words) >= 3:
                            ai_response = ' '.join(clean_words)
                
                # Display the response
                if ai_response:
                    # Clean up the response further
                    ai_response = ai_response.strip()
                    if not ai_response.endswith(('.', '!', '?')):
                        ai_response += '.'  # Add period if missing
                    
                    self.test_output.insert(tk.END, f"AI: {ai_response}\n\n")
                    self.tech_log(f"✅ Response extracted: '{ai_response}'")
                    
                    # Update conversation history for chat mode
                    if (hasattr(self, 'conversation_history') and 
                        self.communication_mode.get() == "chat_conversation"):
                        # Update the last "AI:" entry with the actual response
                        if self.conversation_history and self.conversation_history[-1] == "AI:":
                            self.conversation_history[-1] = f"AI: {ai_response}"
                else:
                    # Fallback response
                    fallback = "I understand your message. Could you try rephrasing it differently?"
                    self.test_output.insert(tk.END, f"AI: {fallback}\n\n")
                    self.tech_log("⚠️ Could not extract clean response, using fallback")
                    
                    # Update history with fallback
                    if (hasattr(self, 'conversation_history') and 
                        self.communication_mode.get() == "chat_conversation"):
                        if self.conversation_history and self.conversation_history[-1] == "AI:":
                            self.conversation_history[-1] = f"AI: {fallback}"
            else:
                # No response at all
                self.test_output.insert(tk.END, "AI: [Model generated empty response]\n\n")
                self.tech_log("❌ No response content after prompt removal")
                
        except Exception as e:
            self.test_output.insert(tk.END, f"AI: [Error processing response: {str(e)}]\n\n")
            self.tech_log(f"❌ Response processing error: {e}")
        
        # Dynamic context management after terminal response
        self._post_generation_context_check()
        
        self.test_output.see(tk.END)
    
    def _retry_generation_after_reset(self, model_path, retry_prompt, user_prompt):
        """Retry generation with clean context after emergency reset"""
        try:
            self.tech_log(f"🔄 Starting retry generation with clean context")
            
            # Import required libraries
            ort = onnxruntime
            np = numpy
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            session = self.create_onnx_session(str(Path(model_path) / "model.onnx"))
            
            # Tokenize the simple retry prompt
            input_ids = tokenizer.encode(retry_prompt, return_tensors="np")
            self.tech_log(f"🔧 Retry input tokens: {input_ids.shape[1]}")
            
            # Use simple generation parameters to avoid overload
            max_new_tokens = min(50, self.test_max_length.get())  # Much shorter for retry
            temperature = max(0.7, self.test_temperature.get())   # Slightly higher temperature
            
            # Simple generation loop with aggressive limits
            generated_ids = input_ids.copy()
            
            for step in range(max_new_tokens):
                # Prepare inputs
                inputs = {"input_ids": generated_ids.astype(np.int64)}
                
                # Add attention mask
                input_names = [inp.name for inp in session.get_inputs()]
                if "attention_mask" in input_names:
                    attention_mask = np.ones_like(generated_ids, dtype=np.int64)
                    inputs["attention_mask"] = attention_mask
                
                try:
                    # Run inference
                    outputs = session.run(None, inputs)
                    logits = outputs[0]
                    
                    # Simple sampling with temperature
                    next_token_logits = logits[0, -1, :] / temperature
                    
                    # Apply softmax
                    exp_logits = np.exp(next_token_logits - np.max(next_token_logits))
                    probabilities = exp_logits / np.sum(exp_logits)
                    
                    # Sample next token
                    next_token = np.random.choice(len(probabilities), p=probabilities)
                    
                    # Add to sequence
                    generated_ids = np.concatenate([generated_ids, [[next_token]]], axis=1)
                    
                    # Check for end of sequence
                    if next_token == tokenizer.eos_token_id:
                        break
                        
                except Exception as inference_error:
                    self.tech_log(f"⚠️ Inference error during retry: {inference_error}")
                    break
            
            # Decode the full response
            full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            response = full_text[len(retry_prompt):].strip() if len(full_text) > len(retry_prompt) else ""
            
            self.tech_log(f"🔧 Retry generated: '{response[:50]}...'")
            
            # Update UI in main thread with the retry result
            if response and len(response.strip()) > 2:
                # Clean the response
                clean_response = response.strip()
                if not clean_response.endswith(('.', '!', '?')):
                    clean_response += '.'
                
                self.root.after(0, lambda: self._display_retry_success(clean_response))
            else:
                # Even the retry failed, use a contextual fallback
                self.root.after(0, lambda: self._display_retry_fallback(user_prompt))
            
        except Exception as e:
            self.tech_log(f"❌ Retry generation failed: {str(e)}")
            # Update UI in main thread with fallback response
            self.root.after(0, lambda: self._display_retry_fallback(user_prompt))
    
    def _display_retry_success(self, clean_response):
        """Display successful retry result"""
        # Remove retry thinking indicators
        self.remove_thinking_indicator()
        
        self.test_output.insert(tk.END, f"AI: {clean_response}\n\n")
        self.tech_log(f"✅ Retry successful: '{clean_response}'")
        
        # Update conversation history with the successful response
        if hasattr(self, 'conversation_history') and self.conversation_history:
            if self.conversation_history[-1] == "AI:":
                self.conversation_history[-1] = f"AI: {clean_response}"
        
        self.test_output.see(tk.END)
        
        # Add new prompt for next input
        self.add_terminal_prompt(self.get_mode_prefix(self.communication_mode.get()))
    
    def _display_retry_fallback(self, user_prompt):
        """Display fallback response when retry also fails"""
        # Remove any retry thinking indicators
        self.remove_thinking_indicator()
        
        # Create a contextual fallback response based on the user's prompt
        prompt_lower = user_prompt.lower()
        
        if any(word in prompt_lower for word in ['hello', 'hi', 'hey', 'greet']):
            fallback_response = "Hello! I'm here to help you."
        elif any(word in prompt_lower for word in ['how', 'what', 'when', 'where', 'why', '?']):
            fallback_response = "That's an interesting question. Could you rephrase it more simply?"
        elif any(word in prompt_lower for word in ['help', 'assist', 'support']):
            fallback_response = "I'd be happy to help! Could you tell me more specifically what you need?"
        elif any(word in prompt_lower for word in ['thank', 'thanks']):
            fallback_response = "You're welcome! Is there anything else I can help you with?"
        else:
            fallback_response = "I understand. Could you try asking that in a different way?"
        
        self.test_output.insert(tk.END, f"AI: {fallback_response}\n\n")
        self.tech_log(f"🆘 Using contextual fallback: '{fallback_response}'")
        
        # Update conversation history with the fallback
        if hasattr(self, 'conversation_history') and self.conversation_history:
            if self.conversation_history[-1] == "AI:":
                self.conversation_history[-1] = f"AI: {fallback_response}"
        
        self.test_output.see(tk.END)
        
        # Add new prompt for next input
        self.add_terminal_prompt(self.get_mode_prefix(self.communication_mode.get()))
    
    def stop_generation(self):
        """Stop the current generation"""
        if hasattr(self, 'generation_thread') and self.generation_thread.is_alive():
            # Remove thinking indicator if generation is stopped
            self.remove_thinking_indicator()
            # Note: This is a simple way to stop - in a real implementation,
            # you might want to use a threading.Event for cleaner stopping
            self.generation_thread = None
            self.tech_log("⏹️ Generation stopped by user")
            self.generation_finished()
    
    def clear_test_output(self):
        """Clear the test output area"""
        self.test_output.delete('1.0', tk.END)
        self.tech_log("🗑️ Output cleared")
    
    def tech_log(self, message):
        """Add a message to the technical log window"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        # Update in main thread
        self.root.after(0, lambda: self._update_tech_log(log_entry))
    
    def _update_tech_log(self, log_entry):
        """Update technical log in main thread"""
        if hasattr(self, 'tech_log_widget'):
            self.tech_log_widget.insert(tk.END, log_entry)
            self.tech_log_widget.see(tk.END)
    
    def test_log(self, message):
        """Redirect test log to technical log window"""
        self.tech_log(message)
    
    def _create_timestamped_output_dir(self, base_output_dir):
        """Create a timestamped directory for the current training/conversion session"""
        from datetime import datetime
        import re
        
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        
        # Get short model name (clean up the model name for directory)
        model_name = self.model_name.get()
        # Extract just the model name part (after the last slash if it exists)
        short_model_name = model_name.split('/')[-1] if '/' in model_name else model_name
        # Clean up any special characters for directory name
        short_model_name = re.sub(r'[^\w\-_]', '_', short_model_name)
        # Limit length to keep directory names reasonable
        if len(short_model_name) > 20:
            short_model_name = short_model_name[:20]
        
        # Build action suffixes based on what operations are enabled
        action_parts = []
        if self.action_train.get():
            action_parts.append("Training")
        if self.action_export.get():
            action_parts.append("Export")
        if self.action_quantize.get():
            action_parts.append("Quantization")
        
        # Create directory name
        action_suffix = "_".join(action_parts) if action_parts else "NoAction"
        dir_name = f"{timestamp}_{short_model_name}_{action_suffix}"
        
        # Create the timestamped directory
        timestamped_dir = base_output_dir / dir_name
        timestamped_dir.mkdir(parents=True, exist_ok=True)
        
        return timestamped_dir
    
    def log_message(self, message):
        """Add a message to the log"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        # Update log in main thread
        self.root.after(0, lambda: self._update_log(log_entry))
    
    def log_download_progress(self, model_name, progress_percentage):
        """Log download progress with percentage"""
        # Only log every 10% to avoid spam
        if progress_percentage % 10 == 0 or progress_percentage >= 95:
            timestamp = time.strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] ⬇️  Downloading {model_name}: {progress_percentage}%\n"
            self.root.after(0, lambda: self._update_log(log_entry))
    
    def create_download_progress_callback(self, model_name):
        """Create a progress callback for model downloads"""
        def progress_callback(total_size, downloaded_size):
            if total_size > 0:
                progress = int((downloaded_size / total_size) * 100)
                # Track last logged percentage to avoid spam
                if not hasattr(self, '_last_progress'):
                    self._last_progress = {}
                
                last_progress = self._last_progress.get(model_name, -1)
                
                # Log every 10% increment or when reaching 100%
                if progress >= 100 or (progress // 10 > last_progress // 10):
                    self.log_download_progress(model_name, progress)
                    self._last_progress[model_name] = progress
                    
        return progress_callback
    
    def load_model_with_progress(self, model_name, model_class, **kwargs):
        """Load model with download progress tracking"""
        # Use global file_utils import
        
        # Check if model is already cached
        is_cached = False
        try:
            cache_dir = file_utils.default_cache_path if file_utils else None
            if cache_dir:
                cached_path = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
            
            if os.path.exists(cached_path):
                # Check if all necessary files are present
                refs_dir = os.path.join(cached_path, "refs")
                snapshots_dir = os.path.join(cached_path, "snapshots")
                
                if os.path.exists(refs_dir) and os.path.exists(snapshots_dir):
                    # Check if there are actual model files
                    for snapshot_dir in os.listdir(snapshots_dir):
                        snapshot_path = os.path.join(snapshots_dir, snapshot_dir)
                        if os.path.isdir(snapshot_path):
                            # Check for model files
                            model_files = [f for f in os.listdir(snapshot_path) 
                                         if f.endswith(('.bin', '.safetensors', '.json'))]
                            if len(model_files) > 2:  # At least config + model weights
                                is_cached = True
                                break
        except:
            pass  # If cache check fails, assume not cached
        
        if is_cached:
            self.log_message(f"📁 Using cached model: {model_name}")
            return model_class.from_pretrained(model_name, **kwargs)
        
        # Model not cached, will need to download
        self.log_message(f"⬇️  Model not cached, initiating download: {model_name}")
        
        # Reset progress tracking for this model
        if not hasattr(self, '_last_progress'):
            self._last_progress = {}
        self._last_progress[model_name] = -1
        
        # Start a background thread to monitor download progress
        download_completed = threading.Event()
        download_started = threading.Event()
        
        def monitor_download():
            """Monitor download progress by checking cache directory growth"""
            try:
                progress = 0
                start_time = time.time()
                initial_cache_size = 0
                target_cache_size = 0
                
                # Wait for download to actually start
                download_started.wait(5)
                
                # Try to estimate download size based on model type
                model_size_estimates = {
                    'gpt2': 500, 'distilgpt2': 250, 'gpt2-medium': 1500, 'gpt2-large': 3000,
                    'opt-125m': 500, 'opt-350m': 1400, 'opt-1.3b': 5000,
                    'phi-1': 2800, 'phi-1_5': 5200
                }
                
                # Estimate target size
                for key in model_size_estimates:
                    if key in model_name.lower():
                        target_cache_size = model_size_estimates[key] * 1024 * 1024  # Convert to bytes
                        break
                
                if target_cache_size == 0:
                    target_cache_size = 1000 * 1024 * 1024  # Default 1GB
                
                self.log_message(f"📥 Downloading {model_name} (estimated size: {target_cache_size // (1024*1024)} MB)")
                
                while not download_completed.is_set() and progress < 90:
                    try:
                        # Check current cache directory size
                        current_size = 0
                        cache_dir = file_utils.default_cache_path
                        model_cache_path = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
                        
                        if os.path.exists(model_cache_path):
                            for root, dirs, files in os.walk(model_cache_path):
                                for file in files:
                                    try:
                                        current_size += os.path.getsize(os.path.join(root, file))
                                    except:
                                        pass
                        
                        if initial_cache_size == 0:
                            initial_cache_size = current_size
                        
                        # Calculate progress
                        if target_cache_size > 0 and current_size > initial_cache_size:
                            downloaded = current_size - initial_cache_size
                            progress = min(int((downloaded / target_cache_size) * 100), 90)
                        else:
                            # Fallback: time-based progress
                            elapsed = time.time() - start_time
                            progress = min(int(elapsed * 2), 90)  # 2% per second, max 90%
                        
                        # Log every 10% increment
                        if progress >= 10 and progress // 10 != self._last_progress.get(model_name, -1) // 10:
                            self.log_download_progress(model_name, progress)
                            self._last_progress[model_name] = progress
                    
                    except Exception:
                        pass  # Ignore monitoring errors
                    
                    download_completed.wait(2)  # Check every 2 seconds
                    
            except Exception:
                pass  # Ignore any monitoring errors
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_download, daemon=True)
        monitor_thread.start()
        
        try:
            # Signal that download is starting
            download_started.set()
            
            # Attempt to load model (this will trigger download)
            model = model_class.from_pretrained(model_name, **kwargs)
            
            # Signal download completion
            download_completed.set()
            
            # Log final completion
            self.log_download_progress(model_name, 100)
            self.log_message(f"✅ Successfully downloaded and loaded: {model_name}")
            return model
            
        except Exception as e:
            download_completed.set()
            self.log_message(f"❌ Failed to download/load model {model_name}: {str(e)}")
            raise
        
    def _update_log(self, message):
        """Update log text widget (must run in main thread)"""
        self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)
        
    def clear_logs(self):
        """Clear the log text"""
        self.log_text.delete('1.0', tk.END)
        
    def save_logs(self):
        """Save logs to file"""
        filename = filedialog.asksaveasfilename(
            title="Save Training Logs",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.log_text.get('1.0', tk.END))
                self.log_message(f"📄 Logs saved to: {filename}")
            except Exception as e:
                self.log_message(f"❌ Error saving logs: {str(e)}")
                
    def clear_test_output(self):
        """Clear the test output area"""
        self.test_output.delete('1.0', tk.END)
    
    def browse_onnx_model(self):
        """Browse for ONNX model directory"""
        dirname = filedialog.askdirectory(title="Select ONNX Model Directory")
        if dirname:
            self.model_path_var.set(dirname)
    
    def refresh_model_list(self):
        """Refresh the list of available models from output directory"""
        try:
            output_dir = Path(self.output_path.get())
            models = []
            
            # Look for ONNX models in timestamped subdirectories
            for timestamped_dir in output_dir.glob("*"):
                if timestamped_dir.is_dir():
                    # Check for converted and quantized models in this timestamped directory
                    for subdir in ['2_converted', '3_quantized']:
                        model_dir = timestamped_dir / subdir
                        if model_dir.exists() and (model_dir / 'model.onnx').exists():
                            # Create readable display name
                            session_name = timestamped_dir.name
                            # Extract meaningful parts from the timestamped name
                            parts = session_name.split('_')
                            if len(parts) >= 6:
                                date_part = f"{parts[0]}-{parts[1]}-{parts[2]} {parts[3]}:{parts[4]}:{parts[5]}"
                                remaining_parts = '_'.join(parts[6:])
                                display_name = f"{date_part} - {remaining_parts} - {subdir}"
                            else:
                                display_name = f"{session_name} - {subdir}"
                            
                            models.append(display_name)
                            self.model_paths[display_name] = str(model_dir)
            
            # Update combobox
            self.model_list_combo['values'] = list(self.model_paths.keys()) if models else ["No models found"]
            
            if models:
                self.model_list_combo.set(list(self.model_paths.keys())[0])
                self.model_path_var.set(list(self.model_paths.values())[0])
                
        except Exception as e:
            self.tech_log(f"⚠️ Error refreshing model list: {e}")
    
    def test_log(self, message):
        """Log message to test output (fallback method)"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if hasattr(self, 'test_output'):
            self.test_output.insert(tk.END, f"[{timestamp}] {message}\n")
            self.test_output.see(tk.END)
        else:
            print(f"[{timestamp}] {message}")
    
    def on_preset_changed(self, event=None):
        """Apply selected training preset to all parameters"""
        try:
            preset_name = self.training_preset.get()
            if not preset_name or preset_name not in self.training_presets:
                return
                
            preset = self.training_presets[preset_name]
            
            # Apply preset values to all parameters
            self.epochs.set(preset.get('epochs', 3))
            self.batch_size.set(preset.get('batch_size', 4))
            self.learning_rate.set(preset.get('learning_rate', 5e-5))
            self.warmup_steps.set(preset.get('warmup_steps', 500))
            self.weight_decay.set(preset.get('weight_decay', 0.01))
            self.lr_scheduler_type.set(preset.get('lr_scheduler_type', 'linear'))
            self.max_grad_norm.set(preset.get('max_grad_norm', 1.0))
            
            # Update preset description
            description = preset.get('description', 'Training preset configuration')
            if hasattr(self, 'preset_description'):
                self.preset_description.config(text=description)
            
            self.log_message(f"✅ Applied training preset: {preset_name}")
            
        except Exception as e:
            self.log_message(f"⚠️ Error applying preset: {e}")
    
    def update_eval_steps_visibility(self, event=None):
        """Placeholder method for backward compatibility"""
        pass
                
    def run(self):
        """Start the application"""
        self.log_message("🚀 ONNX Model Trainer v0.8 started")
        self.log_message("⏳ Running system checks before enabling controls...")
        self.root.mainloop()

def main():
    """Main entry point"""
    try:
        app = ModelTrainer()
        app.run()
    except KeyboardInterrupt:
        print("\n👋 Application interrupted by user")
    except Exception as e:
        print(f"❌ Application error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()