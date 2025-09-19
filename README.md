# 🇹🇭 Thai Text-to-Speech Fine-tuning with MMS-TTS

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samiabat/thai-colab/blob/main/Thai_TTS_Finetune_MMS_Colab.ipynb)

A comprehensive Google Colab notebook for fine-tuning **Meta's MMS-TTS (Massively Multilingual Speech)** model specifically for Thai language text-to-speech synthesis.

## 🚀 Quick Start

**Click the "Open in Colab" badge above** to start training your own Thai TTS model immediately!

## 📋 Overview

This project provides a complete pipeline for:
- Fine-tuning Meta's pre-trained MMS-TTS Thai model on your custom dataset
- Processing and normalizing Thai text using PyThaiNLP
- Converting audio files to the optimal format (16kHz mono WAV)
- Training a high-quality Thai text-to-speech model
- Generating synthetic Thai speech

## ✨ Features

- **🎯 Robust Training Pipeline**: Complete end-to-end training with error handling
- **🔧 Audio Processing**: Automatic resampling and format conversion
- **📝 Thai Text Normalization**: Proper handling of Thai script using PyThaiNLP
- **⚡ GPU Accelerated**: Optimized for Google Colab's free GPU
- **📊 Progress Monitoring**: Real-time training metrics and checkpointing
- **🎵 Audio Generation**: Quick inference testing with sample outputs
- **💾 Disk-based Audio Loading**: Efficient CSV-based dataset loading that reads audio files directly from disk
- **🛡️ Error Resilient**: Graceful handling of corrupted or missing audio files

## 🔄 Recent Updates

- **✅ Fixed Audio Loading Issues**: Implemented robust disk-based audio loading instead of problematic HuggingFace dataset format
- **✅ CSV Dataset Support**: Training script now automatically detects and properly handles CSV files with audio paths
- **✅ Improved Error Handling**: Better error messages and graceful handling of missing or corrupted files
- **✅ Memory Efficiency**: Audio files are loaded on-demand rather than pre-loaded into memory

## 🛠️ Requirements

### For Google Colab (Recommended)
- Google account
- Google Drive for storing datasets and model checkpoints
- GPU runtime (T4/V100/A100)

### For Local Development
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ VRAM for training
- Required packages (automatically installed in notebook)

## 📁 Dataset Format

Your dataset should contain audio files and their corresponding text transcriptions. The training script supports three dataset formats with automatic detection.

### ✅ Option 1: CSV Format (Recommended for new datasets)
A CSV file with columns:
- `path`: **Full path** to audio file (WAV format preferred)
- `text`: Corresponding Thai text

**Key Advantages:**
- 🚀 **Better Performance**: Direct disk loading without HuggingFace dataset overhead
- 🛡️ **More Robust**: Graceful handling of missing/corrupted files
- 💾 **Memory Efficient**: Audio loaded on-demand during training
- 🔧 **Auto-Detection**: Training script automatically detects `.csv` files

Example:
```csv
path,text
/full/path/to/audio1.wav,สวัสดีครับ วันนี้อากาศดีมาก
/full/path/to/audio2.wav,ขอบคุณสำหรับความช่วยเหลือ
```

### Option 2: Folder Structure
```
your_dataset/
├── audio1.wav
├── audio1.txt
├── audio2.wav
├── audio2.txt
└── ...
```

Each `.wav` file should have a corresponding `.txt` file with the same basename containing the Thai transcription. The notebook will automatically convert this to CSV format.

### 🆕 Option 3: Saved HuggingFace Dataset (New!)
You can now use datasets saved with HuggingFace's `save_to_disk()` function:

```python
# Save your dataset
dataset.save_to_disk('/path/to/saved_dataset')

# Use in training config
"dataset_name": "/path/to/saved_dataset"
```

**Key Advantages:**
- 🔄 **Preprocessing Persistence**: Save preprocessed datasets for reuse
- ⚡ **Faster Loading**: No need to reprocess data each time
- 🔀 **Split Support**: Supports DatasetDict with train/eval splits
- 🔧 **Auto-Detection**: Training script automatically detects saved datasets

**Supported Formats:**
- Single dataset: `Dataset.from_list(data).save_to_disk(path)`
- Multiple splits: `DatasetDict({'train': train_ds, 'eval': eval_ds}).save_to_disk(path)`

## 🎵 Audio Guidelines

For best results, your audio data should:
- **Duration**: 1-12 seconds per clip
- **Format**: WAV, 16kHz, mono (automatically converted if needed)
- **Quality**: Clean speech with minimal background noise
- **Quantity**: 1-3 hours for voice cloning, 5-10+ hours for robust models
- **Speaker**: Single speaker for consistency

## 🔧 Configuration

### Key Parameters to Adjust

1. **Paths Configuration**:
   ```python
   DATA_ROOT = '/content/drive/MyDrive/your-dataset/'
   TRANSCRIPT_CSV = '/content/drive/MyDrive/your-dataset/metadata.csv'
   WORK_DIR = '/content/drive/MyDrive/thai_tts/work'
   ```

2. **Training Parameters**:
   ```python
   "per_device_train_batch_size": 8,  # Reduce if OOM
   "max_steps": 20000,               # Increase for longer training
   "learning_rate": 0.0001,          # Adjust based on dataset size
   ```

3. **Audio Filtering**:
   ```python
   MIN_DUR = 1.0    # Minimum clip duration (seconds)
   MAX_DUR = 12.0   # Maximum clip duration (seconds)
   ```

## 📊 Training Process

The notebook includes these main steps:

1. **Environment Setup**: Install dependencies and compile required extensions
2. **Data Loading**: Mount Google Drive and configure paths
3. **Preprocessing**: Audio conversion and Thai text normalization
4. **Dataset Creation**: Build HuggingFace dataset for training
5. **Model Preparation**: Convert MMS checkpoint for fine-tuning
6. **Training**: Fine-tune the model with your data
7. **Inference**: Test the trained model with sample text
8. **Export**: Save checkpoints to Google Drive

## 🎛️ Advanced Usage

### Memory Optimization
If you encounter out-of-memory errors:
- Reduce `per_device_train_batch_size` from 8 to 4 or 2
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Use shorter audio clips (< 8 seconds)

### Training Time Optimization
- Start with fewer `max_steps` (e.g., 5000) to validate the pipeline
- Use smaller datasets for initial testing
- Monitor loss curves to avoid overfitting

### Quality Improvements
- Use high-quality, noise-free audio recordings
- Ensure consistent pronunciation and speaking style
- Balance your dataset across different phonemes and words
- Consider data augmentation for small datasets

## 🚨 Common Issues & Solutions

### 1. ModuleNotFoundError: monotonic_align
**Solution**: The notebook now automatically compiles the required extension.

### 2. Audio Column Error
**Solution**: Fixed - the notebook now uses correct column names.

### 3. Out of Memory (OOM)
**Solutions**:
- Reduce batch size: `"per_device_train_batch_size": 4`
- Increase gradient accumulation: `"gradient_accumulation_steps": 4`
- Use shorter audio clips

### 4. Slow Training
**Solutions**:
- Ensure GPU runtime is enabled in Colab
- Use smaller datasets for testing
- Consider using higher-end GPU (A100)

## 📈 Training Tips

1. **Start Small**: Begin with a subset of your data to validate the pipeline
2. **Monitor Progress**: Check loss curves and sample outputs regularly
3. **Experiment**: Try different learning rates and batch sizes
4. **Quality over Quantity**: Clean, well-recorded data is more valuable than large, noisy datasets
5. **Backup**: Regularly save checkpoints to Google Drive

## 🎵 Sample Usage After Training

```python
from transformers import pipeline
import soundfile as sf

# Load your fine-tuned model
tts = pipeline("text-to-speech", model="/path/to/your/checkpoint")

# Generate speech
text = "สวัสดีครับ นี่คือเสียงสังเคราะห์ภาษาไทย"
output = tts(text)

# Save audio
sf.write("output.wav", output["audio"], output["sampling_rate"])
```

## 📝 License & Attribution

- **MMS-TTS**: Released under CC-BY-NC 4.0 license by Meta
- **This Notebook**: MIT License
- **Commercial Use**: For commercial applications, consider training from scratch or using a different base model

## 🤝 Contributing

Feel free to:
- Report issues or bugs
- Suggest improvements
- Share your trained models
- Contribute code enhancements

## 📚 References

- [Meta MMS Paper](https://research.facebook.com/publications/scaling-speech-technology-to-1000-languages/)
- [finetune-hf-vits Repository](https://github.com/ylacombe/finetune-hf-vits)
- [PyThaiNLP Documentation](https://pythainlp.github.io/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

## 🏷️ Tags

`thai-tts` `text-to-speech` `mms-tts` `google-colab` `machine-learning` `nlp` `pytorch` `huggingface` `thai-language` `speech-synthesis`

---

**Happy Training! 🎉**

If you find this project helpful, please ⭐ star the repository and share it with others interested in Thai TTS!
