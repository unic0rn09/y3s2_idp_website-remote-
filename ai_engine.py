import os
import json
import re
import glob
import torch
import numpy as np
import soundfile as sf
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import pipeline as hf_pipeline
from peft import PeftModel
from openai import OpenAI
from dotenv import load_dotenv
import threading

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================
# CONFIGURATIONS
# ============================================
BASE_MODEL_ID = "mesolitica/malaysian-whisper-medium-v2"
ROJAK_ADAPTER_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "rojak_medium_lora_adapter")
KELANTAN_ADAPTER_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "kelantan_lora_adapter")
TARGET_SR = 16000
INSTANCE_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "instance")
os.makedirs(INSTANCE_FOLDER, exist_ok=True)

# ============================================
# FILE MANAGEMENT HELPERS
# ============================================
def _to_safe_visit_id(patient_id):
    return str(patient_id).replace(" ", "_").replace("/", "_")


def clear_old_audio(patient_id):
    safe_vid = _to_safe_visit_id(patient_id)
    # The wildcard * ensures it catches chunks, full.wav, and temp.wav
    pattern = os.path.join(INSTANCE_FOLDER, f"visit_{safe_vid}_*") 
    for file_path in glob.glob(pattern):
        try:
            os.remove(file_path)
            print(f"🧹 Cleaned up local storage: {file_path}")
        except OSError as e:
             print(f"⚠️ Could not delete {file_path}: {e}")
             
# def clear_old_audio(patient_id):
#     safe_vid = _to_safe_visit_id(patient_id)
#     pattern = os.path.join(INSTANCE_FOLDER, f"visit_{safe_vid}_*.wav")
#     for file_path in glob.glob(pattern):
#         try:
#             os.remove(file_path)
#             print(f"🧹 Cleaned up local storage: {file_path}")
#         except OSError:
#             pass

# ============================================
# 🚀 RTX 4060 GPU OPTIMIZATION: GLOBAL CACHING
# ============================================
_asr_processor = None
_asr_model = None
_asr_lock = threading.Lock()

# ============================================
# 1. ASR ENGINE (Whisper)
# ============================================
def get_asr(mode="normal"):
    global _asr_processor, _asr_model
    with _asr_lock:
        if _asr_model is None:
            print("🚀 Loading Whisper to RTX 4060 GPU...")
            _asr_processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            base_model = WhisperForConditionalGeneration.from_pretrained(
                BASE_MODEL_ID, 
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa" # Highly recommended optimization for RTX 4000 series
            )
            
            has_rojak = os.path.isdir(ROJAK_ADAPTER_DIR)
            has_kelantan = os.path.isdir(KELANTAN_ADAPTER_DIR)
            
            if has_rojak:
                _asr_model = PeftModel.from_pretrained(base_model, ROJAK_ADAPTER_DIR, adapter_name="normal")
                if has_kelantan:
                    _asr_model.load_adapter(KELANTAN_ADAPTER_DIR, adapter_name="kelantan")
            elif has_kelantan:
                _asr_model = PeftModel.from_pretrained(base_model, KELANTAN_ADAPTER_DIR, adapter_name="kelantan")
            else:
                _asr_model = base_model
                
            _asr_model = _asr_model.to(device)
            _asr_model.eval()

        if hasattr(_asr_model, "set_adapter"):
            adapter_to_set = "kelantan" if mode == "kelantan" else "normal"
            if adapter_to_set in getattr(_asr_model, "peft_config", {}):
                _asr_model.set_adapter(adapter_to_set)
                
    return _asr_processor, _asr_model

def transcribe_wav(audio_path, mode="normal"):
    processor, model = get_asr(mode)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    audio = _load_audio(audio_path)
    inputs = processor(audio, return_tensors="pt", sampling_rate=TARGET_SR)
    input_features = inputs.input_features.to(device, dtype=dtype)
    with torch.no_grad():
        generated_ids = model.generate(input_features, max_new_tokens=440)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()

def transcribe_with_timestamps(audio_path, mode="normal"):
    """Returns raw text chunks using HF Pipeline to prevent OOM."""
    processor, model = get_asr(mode)
    asr_pipeline = hf_pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16,
        device=0 if torch.cuda.is_available() else -1,
        return_timestamps=True,
        chunk_length_s=30 
    )
    result = asr_pipeline(audio_path, batch_size=4, generate_kwargs={"max_new_tokens": 440})
    
    # VRAM CLEANUP
    del asr_pipeline
    torch.cuda.empty_cache()
    
    return result.get("chunks", [])

def _load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1: 
        audio = np.mean(audio, axis=1).astype(np.float32)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    return audio

# ============================================
# 2. SEMANTIC DIARIZATION ENGINE (The "Who" Logic)
# ============================================
def generate_diarized_transcript(patient_id, mode="normal"):
    """Step 1: Gathers chunks, transcribes them, and performs Semantic Diarization."""
    safe_vid = _to_safe_visit_id(patient_id)
    
    # Finds ALL chunks mathematically, skipping none
    pattern = os.path.join(INSTANCE_FOLDER, f"visit_{safe_vid}_chunk*.wav")
    chunk_files = glob.glob(pattern)
    
    if not chunk_files:
        return "No audio found for this consultation."
        
    # Sort files by their exact number so the audio is in perfect chronological order
    chunk_files.sort(key=lambda x: int(re.search(r'chunk(\d+)\.wav', x).group(1)))
    
    print("🎙️ Step 1: Transcribing chunks individually to bypass timestamp hallucinations...")
    
    raw_text_pieces = []
    for chunk_path in chunk_files:
        try:
            # Transcribe each pre-sliced chunk
            text = transcribe_wav(chunk_path, mode=mode)
            if text:
                raw_text_pieces.append(text)
        except Exception as e:
            print(f"❌ Error transcribing {chunk_path}: {e}")
            
    # Glue the transcribed TEXT strings together and inject the Anchor
    raw_text = " ".join(raw_text_pieces).strip()
    raw_text = raw_text + "\n\n[END OF CONSULTATION]"
    
    print(f"\n--- RAW ASR TEXT ---\n{raw_text}\n--------------------\n")
    
    print("🧠 Step 2: Calling OpenAI API for Semantic Diarization...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are a strict medical transcriptionist. Format this raw, messy ASR transcript into a clean 'Doctor:' and 'Patient:' dialogue. Whenever theres a change in speaker, make it into a new line"
                        "CRITICAL RULES:\n"
                        "1. DIARIZATION: Accurately assign speakers. Pay attention to context.\n"
                        "2. MEDICAL & PHONETIC CORRECTIONS: The ASR makes severe phonetic mistakes with Malaysian accents. You MUST correct them based on clinical context "
                        "(e.g., changing 'very cold winds' to 'varicose veins', 'weakness insufficient' to 'venous insufficiency', 'kulali' to 'buku lali', 'lift a dent' to 'leave a dent', 'other bawah ubat' to 'ada bawa ubat', 'bulit' to 'kulit').\n"
                        "3. MANDATORY HTML TAGS: Every single time you correct an ASR misheard word, you ABSOLUTELY MUST wrap the corrected word in exact HTML tags. "
                        "Do not skip this HTML formatting. Example: <span class='text-red-600 font-bold'>varicose veins</span>.\n"
                        "4. THE ANCHOR RULE: The raw text ends with the phrase [END OF CONSULTATION]. You MUST process every single word of the transcript and you are forbidden from stopping until you output the phrase [END OF CONSULTATION] exactly as it appears.\n\n"
                        "--- EXAMPLE ---\n"
                        "Raw Input: Doctor family history of very cold winds Patient my mother had very cold winds Doctor that can worsen the weakness issue Patient yes Doctor possible weakness insufficient features [END OF CONSULTATION]\n"
                        "Expected Output:\n"
                        "Doctor: Family history of <span class='text-red-600 font-bold'>varicose veins</span>?\n"
                        "Patient: My mother had <span class='text-red-600 font-bold'>varicose veins</span>.\n"
                        "Doctor: That can worsen the <span class='text-red-600 font-bold'>venous</span> issue.\n"
                        "Patient: Yes.\n"
                        "Doctor: Possible <span class='text-red-600 font-bold'>venous insufficiency</span> features.\n"
                        "[END OF CONSULTATION]\n"
                        "--- END OF EXAMPLE ---\n\n"
                        "Now, process the following transcript exactly like the example above."
                    )
                },
                {
                    "role": "user", 
                    "content": "Raw Input:\n" + raw_text
                }
            ],
            temperature=0.0,
            max_tokens=4000
        )
        final_transcript = response.choices[0].message.content.strip()
        final_transcript = final_transcript.replace("[END OF CONSULTATION]", "").strip()
        
    except Exception as e:
        print(f"❌ OpenAI API Error: {e}")
        return f"OpenAI formatting failed. Raw text:\n\n{raw_text}"
        
    return final_transcript

#Translation task
def translate_rojak(numbered_text):
    """Translate Rojak transcript to formal English while keeping HTML tags."""
    print("🧠 Calling OpenAI API for Rojak Translation...")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical translator specializing in "
                        "Malaysian multilingual clinical conversations.\n\n"
                        "Translate the following transcript into formal English.\n"
                        "The transcript may contain:\n"
                        "- Bahasa Melayu\n"
                        "- English\n"
                        "- Mandarin (romanized or characters)\n"
                        "- Bahasa Kelantan dialect\n"
                        "- Mixed code-switching between any of the above\n\n"
                        "CRITICAL RULES:\n"
                        "1. Translate ALL non-English content to English.\n"
                        "2. Preserve medical terms exactly as stated.\n"
                        "3. MANDATORY HTML PRESERVATION: The input contains words wrapped in HTML tags (e.g., <span class='text-red-600 font-bold'>...</span>). You MUST preserve these exact HTML tags around the translated or original words. Do NOT remove or modify the HTML tags.\n"
                        "4. Maintain speaker labels (Doctor:/Patient:).\n"
                        "5. Do NOT interpret or add clinical meaning — translate literally."
                    ),
                },
                {"role": "user", "content": numbered_text},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"❌ Translation failed: {e}")
        return None


def process_clinical_tasks(labeled_text):
    """Step 2: Extracts the 9 specific boxes using a Few-Shot Medical Prompt."""
    print("⏳ GPT is structuring medical notes...")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini", # You can change this to "gpt-4o" for even higher clinical accuracy if needed
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system", 
                "content": (
                    "You are an expert clinical physician. Your task is to extract medical details from a doctor-patient transcript and format them into a highly professional, structured clinical note in English. "
                    "You must translate layman terms into accurate medical terminology (e.g., 'sweating' -> 'diaphoresis', 'heart attack' -> 'myocardial infarction', 'high blood pressure' -> 'Hypertension').\n\n"
                    "You MUST return a JSON object with EXACTLY these keys: 'cc', 'hopi', 'pmh', 'dh', 'fh', 'sh', 'allergies', 'ros'.\n\n"
                    "FORMATTING RULES:\n"
                    "- 'cc' (Chief Complaint): A single concise string (e.g., 'Chest pain × 1 hour').\n"
                    "- 'allergies': A single string (e.g., 'NKDA' or 'Penicillin').\n"
                    "- For 'hopi', 'pmh', 'dh', 'fh', 'sh', and 'ros': Return a single string containing a bulleted list. Each bullet must start with '* ' and be separated by a newline '\\n'.\n"
                    "- If a category is absolutely not mentioned in the text, output 'N/A' or 'None reported'.\n\n"
                    "--- EXAMPLE INPUT ---\n"
                    "Doctor: Hi encik... Before we start, can I confirm your name and age?\n"
                    "Patient: I’m Rahman, 58 years old.\n"
                    "Doctor: I understand encik came because of chest pain. Boleh encik cerita dulu from the beginning what happened?\n"
                    "Patient: Doctor, about one hour ago I suddenly felt pain in the middle of my chest. I was climbing stairs at work. It was quite sudden.\n"
                    "Doctor: Does the pain move anywhere? Menjalar ke tangan, rahang atau belakang?\n"
                    "Patient: Yes, to my left arm and jaw. Like heavy pressure sitting on my chest. Maybe 8 out of 10. Constant.\n"
                    "Doctor: Anything makes it better or worse?\n"
                    "Patient: Rest helps slightly but still painful. Yes, a bit shortness of breath and sweating a lot. Felt nauseous but didn’t vomit. A bit lightheaded.\n"
                    "Doctor: Before today, pernah ada sakit macam ni?\n"
                    "Patient: Mild chest discomfort before, but never this bad.\n"
                    "Doctor: Do you have diabetes, high blood pressure, or cholesterol problems?\n"
                    "Patient: I have diabetes and hypertension for 10 years. Doctor said cholesterol high before.\n"
                    "Doctor: What medications are you on?\n"
                    "Patient: Metformin, amlodipine and atorvastatin. Sometimes I miss them.\n"
                    "Doctor: Any allergies to medication? Any family history of heart disease?\n"
                    "Patient: No allergies. My older brother had heart attack at age 50.\n"
                    "Doctor: Do you smoke or drink alcohol? What work do you do?\n"
                    "Patient: Smoke one pack a day since my twenties. Alcohol occasionally. Office admin.\n"
                    "Doctor: Any fever, cough, leg swelling or calf pain recently?\n"
                    "Patient: No.\n"
                    "--- EXAMPLE EXPECTED JSON OUTPUT ---\n"
                    "{\n"
                    "  \"cc\": \"Chest pain × 1 hour\",\n"
                    "  \"hopi\": \"* Sudden onset central chest pain while climbing stairs at work\\n* Started approximately 1 hour prior to presentation\\n* Pain described as heavy/tight pressure sensation\\n* Severity 8/10\\n* Constant pain\\n* Radiates to left arm and jaw\\n* Mild relief with rest\\n* Associated diaphoresis\\n* Associated nausea without vomiting\\n* Mild shortness of breath and lightheadedness\\n* Previous intermittent mild chest discomfort but never this severe\",\n"
                    "  \"pmh\": \"* T2DM × 10 years\\n* Hypertension × 10 years\\n* Dyslipidaemia\",\n"
                    "  \"dh\": \"* Metformin\\n* Amlodipine\\n* Atorvastatin\\n* Medication compliance inconsistent\",\n"
                    "  \"fh\": \"* Brother had myocardial infarction at age 50\",\n"
                    "  \"sh\": \"* Smokes 1 pack/day since twenties\\n* Occasional alcohol\\n* Office admin worker\",\n"
                    "  \"allergies\": \"NKDA\",\n"
                    "  \"ros\": \"* No fever\\n* No cough\\n* No calf swelling\\n* No recent prolonged immobilisation\"\n"
                    "}\n"
                    "--- END OF EXAMPLE ---\n\n"
                    "Now, process the following user transcript and generate the JSON following the exact style, clinical tone, and medical terminology shown in the example."
                )
            },
            {"role": "user", "content": labeled_text}
        ],
        temperature=0.1 # Lowered temperature to force strict adherence to your format
    )
    
    # Safely turns the string response into a Python dictionary
    return json.loads(response.choices[0].message.content)

def run_post_consultation_pipeline(patient_id, mode="normal"):
    """Updated to accept patient_id so it can locate the chunk files."""
    # Step A & B: Get raw text from chunks and Diarize
    labeled = generate_diarized_transcript(patient_id, mode)
    
    # Step C: Extract medical notes (Leaving untouched as requested!)
    structured = process_clinical_tasks(labeled)

    return {
        "labeled_transcript": labeled,
        "medical_notes": structured 
    }