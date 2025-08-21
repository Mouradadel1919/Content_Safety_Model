# Content Safety Model

## Overview
This is an end-to-end **Content Safety Model** designed to classify **text**, **images**, and **voices**.  
The system includes both a **hard classifier** and a **soft classifier**:

- **Hard Classifier**: LLaMA Guard  
- **Soft Classifier**: Fine-tuned DistilBERT on safety data, achieving **95% F1-Score**

The soft classifier acts as an additional safety layer to detect unsafe content behind LLaMA Guard.

---

## Features
- **Text Classification**:  
  - Preprocessing includes stop word removal, lemmatization, and synonym augmentation.  
  - DistilBERT fine-tuned for safety classification.  

- **Image Classification**:  
  - BLIP captioning LLM is used to generate captions for images.  
  - The generated captions are then classified for safety.  

- **Voice Classification**:  
  - Whisper model for speech-to-text transcription.  
  - Transcribed text is classified for safety.  

---

## Notebook
You can view the full notebook here:  
[Content Safety Notebook on Kaggle](https://www.kaggle.com/code/mouradadel333/toxic-content-classifiere586bd1c1f)

---

## Deployment
The model is deployed using **FastAPI**.  

### Installation
1. Clone the repository
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. ffmpeg should be installed if you need voice classification
3. Run app
   ```bash
   python app.py
   ```
