import re
import nltk
import torch
import numpy as np
import pandas as pd
import tensorflow as tf
import shutil
import os
from PIL import UnidentifiedImageError, Image

from fastapi import Body
from peft import PeftModel
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoProcessor,  BitsAndBytesConfig, AutoModelForCausalLM, AutoModelForSequenceClassification, BlipProcessor, BlipForConditionalGeneration, pipeline
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from fastapi import FastAPI
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import JSONResponse
import io

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# 1. Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove non-letters
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# 2. Lemmatization function
def lemmatize_text(text):
    tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens])

# 3. Full pipeline
def text_pipeline(text):
    cleaned = clean_text(text)
    lemmatized = lemmatize_text(cleaned)
    return lemmatized


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''Bert Model'''
tokenizer_path = "./saved_models/soft_model/soft_tokenizer"
base_path = "./saved_models/soft_model/base_model"
adapter_path = "./saved_models/soft_model"

def load_lora_model(base_model_name, tokenizer_path, adapter_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=9).to(device)
    lora_model = PeftModel.from_pretrained(base_model, adapter_path).to(device)

    return lora_model, tokenizer

bert_model, bert_tokenizer = load_lora_model(base_path, tokenizer_path, adapter_path, device)


'''Blip Model'''
blip_model_path = "./saved_models/blip/blip_model"
blip_processor_path = "./saved_models/blip/blip_processor"

blip_processor = BlipProcessor.from_pretrained(blip_processor_path)
blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_path)

'''LLaMA-Guard Model'''


llama_model_path = "./saved_models/llama/llama_model"
llama_processor_path = "./saved_models/llama/llama_tokenizer"


quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

llama_tokenizer = AutoTokenizer.from_pretrained(llama_processor_path, token=token)

llama_model = AutoModelForCausalLM.from_pretrained(
    llama_model_path,
    use_auth_token=token,
    quantization_config=quant_config,
    torch_dtype=torch.float16,  # avoids mixed precision issues
    device_map="auto"  # will automatically put parts on CPU/GPU
)


'''Whisper Model'''
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained("./saved_models/whisper_tiny_model")
whisper_processor = AutoProcessor.from_pretrained("./saved_models/whisper_tiny_model")

# Create the ASR pipeline from local files
pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=whisper_processor.tokenizer,
    feature_extractor=whisper_processor.feature_extractor,
    device=device  # change to "cpu" if no GPU
)

def SoftClassifier(bert_model, tokenizer, text, device=None):
    """
    Predicts the class label for the given text using a pretrained BERT model.
    """
    text = text_pipeline(text)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenized_sentence = tokenizer(
        text= text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    # Model prediction
    with torch.no_grad():
        outputs = bert_model(**tokenized_sentence)

    # Probabilities
    probs = torch.softmax(outputs.logits, dim=-1)

    # Prediction
    predicted_idx = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][predicted_idx].item()

    label_map = {0: "Safe", 1: "Unsafe"}

    predicted_label = label_map[predicted_idx] if label_map else predicted_idx

    return f"Category: {predicted_label} with Confidence: {confidence:.4f}"

def image_captioning(blip_model, processor, image):

    # Process the image
    inputs = processor(images=image, return_tensors="pt")
    outputs = blip_model.generate(**inputs)

    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption


def classify_text(text):
    pipe_classifier = pipeline("text-generation", model=llama_model, tokenizer=llama_tokenizer, device_map="auto")

    def get_prompt(input_text):
        return f"""<s>[INST] <<SYS>>
                    You are a text safety classifier. You will classify the following user input as either SAFE or UNSAFE.
                    Your output is 'Safe' or 'Unsafe' only.
                    <</SYS>>
                    

                    Text: {input_text}
                    Classification: [/INST]"""
    

    response = response = pipe_classifier(
                                        get_prompt(text),
                                        max_new_tokens=20,
                                        do_sample=False,      # deterministic output
                                        temperature=0.0,      # no randomness
                                        top_p=1.0
                                    )[0]['generated_text']

    hard = "None"
    soft = "None"
    if 'unsafe' in response:
        hard = f'Hard Classifier : {"unsafe"}'

    elif 'safe' in response:
        hard = f'Hard Classifier : {"safe"}'
        soft = f'Soft Classifier :  {SoftClassifier(bert_model, bert_tokenizer, text)}'

    else:
        hard = 'unknown'
        soft = 'unknown'
    return hard, soft


app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get('/', response_class=HTMLResponse)
@app.get('/home', response_class=HTMLResponse)

async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.get("/predict", response_class=HTMLResponse)
async def get_predict(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})




@app.post("/predict")
async def predict(
    request_type: str = Form(None),
    text_input: str = Form(None),
    image_input: UploadFile = File(None),
    voice_input: UploadFile = File(None)
):
    try:
        result_hard = None
        result_soft = None

        if text_input and text_input.strip():
            processed_text = text_pipeline(text_input)
            result_hard, result_soft = classify_text(processed_text)
            return JSONResponse({
            "hard": result_hard,
            "soft": result_soft,
        })

        if image_input is not None:
            try:
                image_bytes = await image_input.read()
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                caption = image_captioning(blip_model=blip_model, processor=blip_processor, image=image)
                result_hard, result_soft = classify_text(caption)

                return JSONResponse({
            "hard": result_hard,
            "soft": result_soft,
            "caption": caption
        })
            except UnidentifiedImageError:
                return JSONResponse({"error": "Invalid image file"}, status_code=400)
            
        # --- VOICE CLASSIFICATION ---
        if voice_input is not None:
            temp_path = "uploaded_audio.wav"
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(voice_input.file, buffer)

            # Transcribe audio
            transcription_result = pipe(temp_path)  # Your ASR pipeline
            transcription = transcription_result["text"]

            # Classify transcription like text
            processed_text = text_pipeline(transcription)
            result_hard, result_soft = classify_text(processed_text)

            # Clean up temporary file
            os.remove(temp_path)

            return JSONResponse({
            "hard": result_hard,
            "soft": result_soft,
            "transcription": transcription
            })

         
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        port=8000
    )