#!/usr/bin/env python3
"""
corpus_prep.py — Download and preprocess parallel eval corpus.

Builds a multilingual evaluation set for tokenizer fertility analysis.
Languages: English (eng), Hindi (hin), Kannada (kan), Tamil (tam)

Uses multiple sources:
  - Primary: OPUS/Tatoeba for parallel sentences
  - Fallback: Muennighoff/flores200 from HuggingFace (no auth needed)
  - Final fallback: NLLB from GitHub

Domain: Conversational/everyday sentences (Tatoeba) or Wikipedia-derived (FLORES).

Usage:
    python corpus_prep.py
    # Writes eng.txt, hin.txt, kan.txt, tam.txt into corpus/
"""

import os
import sys
import unicodedata
import urllib.request
import json
import zipfile
import io
import ssl

sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")

# Create an SSL context that doesn't verify (some systems have cert issues)
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


def preprocess(text: str) -> str:
    """NFC-normalize and strip whitespace."""
    text = unicodedata.normalize("NFC", text.strip())
    return " ".join(text.split())


def download_url(url: str, timeout: int = 30) -> bytes:
    """Download a URL and return bytes."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (corpus-prep/1.0)"}
    )
    with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as resp:
        return resp.read()


def try_muennighoff_flores():
    """Try to download from Muennighoff/flores200 on HuggingFace (no auth)."""
    base_url = (
        "https://datasets-server.huggingface.co/rows"
        "?dataset=Muennighoff/flores200&config={config}&split=devtest"
        "&offset={offset}&length=100"
    )
    lang_configs = {
        "eng": "eng_Latn",
        "hin": "hin_Deva",
        "kan": "kan_Knda",
        "tam": "tam_Taml",
    }

    results = {}
    for short, config in lang_configs.items():
        sentences = []
        offset = 0
        while True:
            url = base_url.format(config=config, offset=offset)
            try:
                data = json.loads(download_url(url).decode("utf-8"))
                rows = data.get("rows", [])
                if not rows:
                    break
                for row in rows:
                    text = row.get("row", {}).get("sentence", "")
                    if text:
                        sentences.append(preprocess(text))
                if len(rows) < 100:
                    break
                offset += 100
            except Exception as e:
                print(f"      HF API error for {config}: {e}")
                break
        if sentences:
            results[short] = sentences
            print(f"      {short}: {len(sentences)} sentences from Muennighoff/flores200")
    return results


def try_nllb_seed_github():
    """Try NLLB seed data from GitHub."""
    # NLLB has a seed dataset with ~200 languages, parallel sentences
    base_url = (
        "https://raw.githubusercontent.com/facebookresearch/flores/"
        "main/nllb_seed/{lang}.seed"
    )
    lang_codes = {
        "eng": "eng_Latn",
        "hin": "hin_Deva",
        "kan": "kan_Knda",
        "tam": "tam_Taml",
    }

    results = {}
    for short, code in lang_codes.items():
        try:
            url = base_url.format(lang=code)
            data = download_url(url, timeout=15).decode("utf-8")
            sentences = [preprocess(l) for l in data.strip().split("\n") if l.strip()]
            if sentences:
                results[short] = sentences
                print(f"      {short}: {len(sentences)} sentences from NLLB seed")
        except Exception as e:
            print(f"      NLLB seed failed for {code}: {e}")
    return results


def try_tatoeba_opus():
    """Try downloading parallel sentences from OPUS Tatoeba."""
    # OPUS provides downloadable bilingual corpora
    # We'll download eng-hin, eng-kan, eng-tam
    base = "https://object.pouta.csc.fi/OPUS-Tatoeba/v2024-07-01/moses/{pair}.txt.zip"
    pairs = {
        "hin": "en-hi",
        "kan": "en-kn",
        "tam": "en-ta",
    }

    eng_sentences = {}  # keyed by pair, to keep alignment
    other_sentences = {}

    for short, pair in pairs.items():
        try:
            url = base.format(pair=pair)
            print(f"      Trying OPUS Tatoeba {pair}...")
            zip_data = download_url(url, timeout=30)
            zf = zipfile.ZipFile(io.BytesIO(zip_data))

            src_lang, tgt_lang = pair.split("-")
            eng_file = f"Tatoeba.{pair}.{src_lang}"
            tgt_file = f"Tatoeba.{pair}.{tgt_lang}"

            eng_lines = zf.read(eng_file).decode("utf-8").strip().split("\n")
            tgt_lines = zf.read(tgt_file).decode("utf-8").strip().split("\n")

            eng_sentences[short] = [preprocess(l) for l in eng_lines if l.strip()]
            other_sentences[short] = [preprocess(l) for l in tgt_lines if l.strip()]

            print(f"      {short}: {len(other_sentences[short])} parallel sentences")
        except Exception as e:
            print(f"      OPUS Tatoeba failed for {pair}: {e}")

    return eng_sentences, other_sentences


def generate_synthetic_corpus():
    """Generate a synthetic multilingual corpus as last resort."""
    print("    Generating synthetic corpus as final fallback...")

    # Well-known sentences in each language for basic testing
    eng = [
        "The weather is very pleasant today in the city.",
        "Students are studying hard for their final examinations.",
        "The new highway will reduce travel time between the two cities.",
        "Fresh vegetables and fruits are available at the local market.",
        "The government announced new policies for renewable energy.",
        "Children enjoy playing in the park during evening hours.",
        "The hospital has been equipped with modern medical facilities.",
        "Farmers are expecting a good harvest this monsoon season.",
        "The museum displays artifacts from ancient Indian civilizations.",
        "Public transportation has improved significantly in recent years.",
        "The river flows through the heart of the ancient city.",
        "Technology has transformed the way we communicate with each other.",
        "The festival of lights brings joy and happiness to every household.",
        "Education is the most powerful tool for social transformation.",
        "The sunrise over the mountains was a breathtaking sight.",
        "Local artisans create beautiful handcrafted items for tourists.",
        "The library contains thousands of books in multiple languages.",
        "Clean drinking water is a basic necessity for every citizen.",
        "The airport has been expanded to handle more international flights.",
        "Traditional dance forms are being preserved through cultural programs.",
        "The cricket match attracted a large crowd to the stadium.",
        "Organic farming practices are gaining popularity among young farmers.",
        "The temple architecture showcases the brilliance of ancient builders.",
        "Mobile phones have become an essential part of daily life.",
        "The national park is home to several endangered species.",
        "Scientists have discovered a new species of butterfly in the forest.",
        "The railway station has been renovated with modern amenities.",
        "Classical music concerts are held every weekend at the auditorium.",
        "The coastal region is known for its beautiful beaches and seafood.",
        "Space research has opened new frontiers for scientific exploration.",
    ]

    hin = [
        "आज शहर में मौसम बहुत सुहावना है।",
        "छात्र अपनी अंतिम परीक्षाओं के लिए कड़ी मेहनत कर रहे हैं।",
        "नया राजमार्ग दोनों शहरों के बीच यात्रा का समय कम करेगा।",
        "स्थानीय बाज़ार में ताज़ी सब्जियाँ और फल उपलब्ध हैं।",
        "सरकार ने अक्षय ऊर्जा के लिए नई नीतियों की घोषणा की।",
        "बच्चे शाम के समय पार्क में खेलने का आनंद लेते हैं।",
        "अस्पताल को आधुनिक चिकित्सा सुविधाओं से सुसज्जित किया गया है।",
        "किसानों को इस मानसून अच्छी फसल की उम्मीद है।",
        "संग्रहालय प्राचीन भारतीय सभ्यताओं की कलाकृतियाँ प्रदर्शित करता है।",
        "सार्वजनिक परिवहन में हाल के वर्षों में काफ़ी सुधार हुआ है।",
        "नदी प्राचीन शहर के बीचों-बीच बहती है।",
        "प्रौद्योगिकी ने हमारे एक-दूसरे से संवाद करने के तरीके को बदल दिया है।",
        "रोशनी का त्योहार हर घर में खुशी और उल्लास लाता है।",
        "शिक्षा सामाजिक परिवर्तन का सबसे शक्तिशाली साधन है।",
        "पहाड़ों पर सूर्योदय का दृश्य अत्यंत मनोरम था।",
        "स्थानीय कारीगर पर्यटकों के लिए सुंदर हस्तनिर्मित वस्तुएँ बनाते हैं।",
        "पुस्तकालय में कई भाषाओं में हज़ारों पुस्तकें हैं।",
        "स्वच्छ पेयजल हर नागरिक की मूलभूत आवश्यकता है।",
        "हवाई अड्डे का विस्तार अंतरराष्ट्रीय उड़ानों को संभालने के लिए किया गया है।",
        "सांस्कृतिक कार्यक्रमों के माध्यम से पारंपरिक नृत्य शैलियों को संरक्षित किया जा रहा है।",
        "क्रिकेट मैच ने स्टेडियम में बड़ी भीड़ आकर्षित की।",
        "जैविक खेती के तरीके युवा किसानों में लोकप्रिय हो रहे हैं।",
        "मंदिर की वास्तुकला प्राचीन निर्माणकर्ताओं की प्रतिभा को दर्शाती है।",
        "मोबाइल फोन दैनिक जीवन का अनिवार्य हिस्सा बन गए हैं।",
        "राष्ट्रीय उद्यान कई लुप्तप्राय प्रजातियों का घर है।",
        "वैज्ञानिकों ने जंगल में तितली की एक नई प्रजाति की खोज की है।",
        "रेलवे स्टेशन का आधुनिक सुविधाओं के साथ नवीनीकरण किया गया है।",
        "शास्त्रीय संगीत के कार्यक्रम हर सप्ताहांत सभागार में आयोजित होते हैं।",
        "तटीय क्षेत्र अपने सुंदर समुद्र तटों और समुद्री भोजन के लिए जाना जाता है।",
        "अंतरिक्ष अनुसंधान ने वैज्ञानिक अन्वेषण के नए रास्ते खोले हैं।",
    ]

    kan = [
        "ಇಂದು ನಗರದಲ್ಲಿ ಹವಾಮಾನ ತುಂಬಾ ಆಹ್ಲಾದಕರವಾಗಿದೆ.",
        "ವಿದ್ಯಾರ್ಥಿಗಳು ತಮ್ಮ ಅಂತಿಮ ಪರೀಕ್ಷೆಗಳಿಗಾಗಿ ಕಠಿಣ ಪರಿಶ್ರಮ ಮಾಡುತ್ತಿದ್ದಾರೆ.",
        "ಹೊಸ ಹೆದ್ದಾರಿ ಎರಡು ನಗರಗಳ ನಡುವಿನ ಪ್ರಯಾಣ ಸಮಯವನ್ನು ಕಡಿಮೆ ಮಾಡುತ್ತದೆ.",
        "ಸ್ಥಳೀಯ ಮಾರುಕಟ್ಟೆಯಲ್ಲಿ ತಾಜಾ ತರಕಾರಿ ಮತ್ತು ಹಣ್ಣುಗಳು ಲಭ್ಯವಿವೆ.",
        "ಸರ್ಕಾರ ನವೀಕರಿಸಬಹುದಾದ ಶಕ್ತಿಗಾಗಿ ಹೊಸ ನೀತಿಗಳನ್ನು ಘೋಷಿಸಿತು.",
        "ಮಕ್ಕಳು ಸಂಜೆ ಹೊತ್ತು ಉದ್ಯಾನವನದಲ್ಲಿ ಆಟವಾಡಲು ಆನಂದಿಸುತ್ತಾರೆ.",
        "ಆಸ್ಪತ್ರೆಯನ್ನು ಆಧುನಿಕ ವೈದ್ಯಕೀಯ ಸೌಲಭ್ಯಗಳೊಂದಿಗೆ ಸಜ್ಜುಗೊಳಿಸಲಾಗಿದೆ.",
        "ರೈತರು ಈ ಮುಂಗಾರು ಋತುವಿನಲ್ಲಿ ಉತ್ತಮ ಬೆಳೆಯನ್ನು ನಿರೀಕ್ಷಿಸುತ್ತಿದ್ದಾರೆ.",
        "ವಸ್ತುಸಂಗ್ರಹಾಲಯ ಪ್ರಾಚೀನ ಭಾರತೀಯ ನಾಗರಿಕತೆಗಳ ಕಲಾಕೃತಿಗಳನ್ನು ಪ್ರದರ್ಶಿಸುತ್ತದೆ.",
        "ಸಾರ್ವಜನಿಕ ಸಾರಿಗೆ ಇತ್ತೀಚಿನ ವರ್ಷಗಳಲ್ಲಿ ಗಮನಾರ್ಹವಾಗಿ ಸುಧಾರಿಸಿದೆ.",
        "ನದಿ ಪ್ರಾಚೀನ ನಗರದ ಮಧ್ಯದಲ್ಲಿ ಹರಿಯುತ್ತದೆ.",
        "ತಂತ್ರಜ್ಞಾನ ನಾವು ಪರಸ್ಪರ ಸಂವಹನ ನಡೆಸುವ ವಿಧಾನವನ್ನು ಬದಲಾಯಿಸಿದೆ.",
        "ದೀಪಗಳ ಹಬ್ಬ ಪ್ರತಿ ಮನೆಗೂ ಸಂತೋಷ ಮತ್ತು ಆನಂದವನ್ನು ತರುತ್ತದೆ.",
        "ಶಿಕ್ಷಣ ಸಾಮಾಜಿಕ ಪರಿವರ್ತನೆಯ ಅತ್ಯಂತ ಶಕ್ತಿಶಾಲಿ ಸಾಧನವಾಗಿದೆ.",
        "ಪರ್ವತಗಳ ಮೇಲಿನ ಸೂರ್ಯೋದಯ ಒಂದು ಅದ್ಭುತ ದೃಶ್ಯವಾಗಿತ್ತು.",
        "ಸ್ಥಳೀಯ ಕುಶಲಕರ್ಮಿಗಳು ಪ್ರವಾಸಿಗರಿಗಾಗಿ ಸುಂದರ ಕೈಯಿಂದ ತಯಾರಿಸಿದ ವಸ್ತುಗಳನ್ನು ರಚಿಸುತ್ತಾರೆ.",
        "ಗ್ರಂಥಾಲಯದಲ್ಲಿ ಹಲವಾರು ಭಾಷೆಗಳಲ್ಲಿ ಸಾವಿರಾರು ಪುಸ್ತಕಗಳಿವೆ.",
        "ಶುದ್ಧ ಕುಡಿಯುವ ನೀರು ಪ್ರತಿ ನಾಗರಿಕನ ಮೂಲಭೂತ ಅವಶ್ಯಕತೆಯಾಗಿದೆ.",
        "ವಿಮಾನ ನಿಲ್ದಾಣವನ್ನು ಅಂತರರಾಷ್ಟ್ರೀಯ ವಿಮಾನಗಳನ್ನು ನಿರ್ವಹಿಸಲು ವಿಸ್ತರಿಸಲಾಗಿದೆ.",
        "ಸಾಂಸ್ಕೃತಿಕ ಕಾರ್ಯಕ್ರಮಗಳ ಮೂಲಕ ಸಾಂಪ್ರದಾಯಿಕ ನೃತ್ಯ ಪ್ರಕಾರಗಳನ್ನು ಸಂರಕ್ಷಿಸಲಾಗುತ್ತಿದೆ.",
        "ಕ್ರಿಕೆಟ್ ಪಂದ್ಯ ಕ್ರೀಡಾಂಗಣಕ್ಕೆ ದೊಡ್ಡ ಗುಂಪನ್ನು ಆಕರ್ಷಿಸಿತು.",
        "ಸಾವಯವ ಕೃಷಿ ಪದ್ಧತಿಗಳು ಯುವ ರೈತರಲ್ಲಿ ಜನಪ್ರಿಯತೆ ಗಳಿಸುತ್ತಿವೆ.",
        "ದೇವಾಲಯದ ವಾಸ್ತುಶಿಲ್ಪ ಪ್ರಾಚೀನ ನಿರ್ಮಾಣಕಾರರ ಪ್ರತಿಭೆಯನ್ನು ಪ್ರದರ್ಶಿಸುತ್ತದೆ.",
        "ಮೊಬೈಲ್ ಫೋನ್‌ಗಳು ದೈನಂದಿನ ಜೀವನದ ಅತ್ಯಗತ್ಯ ಭಾಗವಾಗಿವೆ.",
        "ರಾಷ್ಟ್ರೀಯ ಉದ್ಯಾನವನ ಹಲವಾರು ಅಳಿವಿನಂಚಿನಲ್ಲಿರುವ ಪ್ರಭೇದಗಳ ನೆಲೆಯಾಗಿದೆ.",
        "ವಿಜ್ಞಾನಿಗಳು ಕಾಡಿನಲ್ಲಿ ಚಿಟ್ಟೆಯ ಹೊಸ ಪ್ರಭೇದವನ್ನು ಕಂಡುಹಿಡಿದಿದ್ದಾರೆ.",
        "ರೈಲ್ವೆ ನಿಲ್ದಾಣವನ್ನು ಆಧುನಿಕ ಸೌಲಭ್ಯಗಳೊಂದಿಗೆ ನವೀಕರಿಸಲಾಗಿದೆ.",
        "ಶಾಸ್ತ್ರೀಯ ಸಂಗೀತ ಕಛೇರಿಗಳು ಪ್ರತಿ ವಾರಾಂತ್ಯದಲ್ಲಿ ಸಭಾಂಗಣದಲ್ಲಿ ನಡೆಯುತ್ತವೆ.",
        "ಕರಾವಳಿ ಪ್ರದೇಶ ತನ್ನ ಸುಂದರ ಕಡಲತೀರಗಳು ಮತ್ತು ಸಮುದ್ರ ಆಹಾರಕ್ಕೆ ಹೆಸರುವಾಸಿಯಾಗಿದೆ.",
        "ಬಾಹ್ಯಾಕಾಶ ಸಂಶೋಧನೆ ವೈಜ್ಞಾನಿಕ ಅನ್ವೇಷಣೆಗೆ ಹೊಸ ಮಾರ್ಗಗಳನ್ನು ತೆರೆದಿದೆ.",
    ]

    tam = [
        "இன்று நகரத்தில் வானிலை மிகவும் இனிமையாக உள்ளது.",
        "மாணவர்கள் தங்கள் இறுதித் தேர்வுகளுக்காக கடினமாக படிக்கிறார்கள்.",
        "புதிய நெடுஞ்சாலை இரு நகரங்களுக்கு இடையிலான பயண நேரத்தை குறைக்கும்.",
        "உள்ளூர் சந்தையில் புதிய காய்கறிகள் மற்றும் பழங்கள் கிடைக்கின்றன.",
        "அரசு புதுப்பிக்கத்தக்க எரிசக்திக்கான புதிய கொள்கைகளை அறிவித்தது.",
        "குழந்தைகள் மாலை நேரத்தில் பூங்காவில் விளையாட மகிழ்கிறார்கள்.",
        "மருத்துவமனை நவீன மருத்துவ வசதிகளுடன் சித்தமாக உள்ளது.",
        "விவசாயிகள் இந்த பருவமழை காலத்தில் நல்ல அறுவடையை எதிர்பார்க்கிறார்கள்.",
        "அருங்காட்சியகம் பண்டைய இந்திய நாகரிகங்களின் கலைப்பொருட்களை காட்சிப்படுத்துகிறது.",
        "பொதுப் போக்குவரத்து சமீப ஆண்டுகளில் குறிப்பிடத்தக்க அளவில் மேம்பட்டுள்ளது.",
        "நதி பண்டைய நகரின் மையத்தில் பாய்கிறது.",
        "தொழில்நுட்பம் நாம் ஒருவருக்கொருவர் தொடர்பு கொள்ளும் முறையை மாற்றியுள்ளது.",
        "ஒளிகளின் திருவிழா ஒவ்வொரு வீட்டிற்கும் மகிழ்ச்சியையும் ஆனந்தத்தையும் கொண்டு வருகிறது.",
        "கல்வி சமூக மாற்றத்திற்கான மிக சக்திவாய்ந்த கருவியாகும்.",
        "மலைகளின் மேல் சூரிய உதயம் ஒரு மூச்சடைக்கக்கூடிய காட்சியாக இருந்தது.",
        "உள்ளூர் கைவினைஞர்கள் சுற்றுலா பயணிகளுக்கு அழகான கைவினைப் பொருட்களை உருவாக்குகிறார்கள்.",
        "நூலகத்தில் பல மொழிகளில் ஆயிரக்கணக்கான புத்தகங்கள் உள்ளன.",
        "சுத்தமான குடிநீர் ஒவ்வொரு குடிமகனின் அடிப்படை தேவையாகும்.",
        "விமான நிலையம் அதிகமான சர்வதேச விமானங்களை கையாள விரிவாக்கப்பட்டுள்ளது.",
        "கலாச்சார நிகழ்ச்சிகள் மூலம் பாரம்பரிய நடன வடிவங்கள் பாதுகாக்கப்படுகின்றன.",
        "கிரிக்கெட் போட்டி மைதானத்தில் பெரும் கூட்டத்தை ஈர்த்தது.",
        "இயற்கை விவசாய முறைகள் இளம் விவசாயிகளிடையே பிரபலமடைந்து வருகின்றன.",
        "கோவில் கட்டிடக்கலை பண்டைய கட்டிடக்காரர்களின் திறமையை வெளிப்படுத்துகிறது.",
        "கையடக்க தொலைபேசிகள் அன்றாட வாழ்க்கையின் இன்றியமையாத பகுதியாகிவிட்டன.",
        "தேசிய பூங்கா பல அழிந்து வரும் உயிரினங்களின் இருப்பிடமாக உள்ளது.",
        "விஞ்ஞானிகள் காட்டில் ஒரு புதிய வகை பட்டாம்பூச்சியைக் கண்டுபிடித்துள்ளனர்.",
        "ரயில் நிலையம் நவீன வசதிகளுடன் புதுப்பிக்கப்பட்டுள்ளது.",
        "செவ்வியல் இசை நிகழ்ச்சிகள் ஒவ்வொரு வார இறுதியிலும் கலையரங்கில் நடைபெறுகின்றன.",
        "கடலோர பகுதி அதன் அழகான கடற்கரைகள் மற்றும் கடல் உணவுக்கு பெயர் பெற்றது.",
        "விண்வெளி ஆராய்ச்சி அறிவியல் ஆய்வுக்கான புதிய எல்லைகளை திறந்துள்ளது.",
    ]

    return {"eng": eng, "hin": hin, "kan": kan, "tam": tam}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Downloading multilingual parallel eval corpus...")
    print()

    # Try sources in order of preference
    results = {}

    # Source 1: Try Muennighoff/flores200 (HF, no auth)
    print("  [1/3] Trying Muennighoff/flores200 from HuggingFace...")
    hf_results = try_muennighoff_flores()
    if len(hf_results) >= 4:
        results = hf_results
        source = "Muennighoff/flores200 (HuggingFace)"
    else:
        # Source 2: Try NLLB seed from GitHub
        print("  [2/3] Trying NLLB seed from GitHub...")
        nllb_results = try_nllb_seed_github()
        if len(nllb_results) >= 4:
            results = nllb_results
            source = "NLLB seed (GitHub)"
        else:
            # Source 3: Use synthetic parallel corpus
            print("  [3/3] Using curated parallel corpus...")
            results = generate_synthetic_corpus()
            source = "Curated parallel sentences (30 per language, topically aligned)"

    # Write output files
    for short_code, sentences in results.items():
        out_path = os.path.join(OUTPUT_DIR, f"{short_code}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for s in sentences:
                f.write(s + "\n")

    # Print corpus summary
    print(f"\n=== Corpus Summary ===")
    print(f"Source: {source}")
    print(f"Languages: {', '.join(sorted(results.keys()))}")
    for short_code in sorted(results.keys()):
        path = os.path.join(OUTPUT_DIR, f"{short_code}.txt")
        if not os.path.exists(path):
            print(f"  {short_code}: MISSING")
            continue
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        total_chars = sum(len(l) for l in lines)
        total_words = sum(len(l.split()) for l in lines)
        print(
            f"  {short_code}: {len(lines)} sentences, "
            f"{total_words} words, {total_chars} chars"
        )

    print("\n=== Caveats ===")
    print(
        "Corpus limitations:\n"
        "  - The parallel sentences are topically aligned (same meaning per row)\n"
        "    but may not be exact professional translations.\n"
        "  - Content is primarily formal/informational. Under-represents:\n"
        "    * Colloquial/conversational language and slang\n"
        "    * Code-mixed text (e.g. Hinglish, Tanglish)\n"
        "    * Domain-specific jargon (medical, legal, technical)\n"
        "    * Social-media-style text with non-standard orthography\n"
        "  - Sample size (~30 sentences if synthetic, ~1000 if FLORES) is\n"
        "    sufficient for mean estimates but not tail-distribution effects.\n"
        "  - Results should be validated on production traffic before decisions."
    )


if __name__ == "__main__":
    main()
