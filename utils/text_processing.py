import re
from typing import Dict, List, Tuple
from difflib import get_close_matches


# Kamus kosakata emosi yang SANGAT LENGKAP
EMOTION_VOCABULARY = {
    "anger": [
        # English
        "angry", "furious", "rage", "hate", "worst", "terrible", "awful", "mad",
        "pissed", "livid", "enraged", "infuriated", "seething", "fuming", "wrathful",
        "irritated", "annoyed", "agitated", "provoked", "offended", "upset", "bitter",
        
        # Indonesian - Formal
        "marah", "kesal", "gemas", "benci", "buruk", "jelek", "kecewa", "jengkel",
        "emosi", "naik darah", "meradang", "dendam", "garang", "membara", "naik pitam",
        "sewot", "sebel", "dongkol", "ngambek", "kepengin", "pencara", "pukul",
        "berantem", "lawan", "gelisah", "geleng", "kapok", "sabar", "tabah",
        
        # Indonesian - Slang/Gaul
        "geram", "ngenes", "sengiti", "kepengin mencecehin", "gue pengen berantem",
        "jemot", "sebal", "bete", "kesel", "gerah", "emosi tinggi", "naik pitam",
        "manggung", "dongkol", "sewot", "ngamuk", "ngacir", "meledak", "labil", "kontol", "memek", "bangsat", "tolol"
        ,"goblok", "bego", "anjing", "babi", "titit", "kntol","kntl", "mmek", "mmk", "asu", "pecun", "bangke", "cukimay",
        "pendo", "ngentot", "bgst", "bgsat"       
        # Mixed expressions
        "badmood", "bad mood", "tidak baik", "sangat buruk", "amat benci", 
        "benar-benar marah", "sangat kesal", "benci banget", "dendam mendalam",
    ],
    
    "joy": [
        # English
        "happy", "joy", "love", "great", "wonderful", "amazing", "excellent", "awesome",
        "fantastic", "brilliant", "delighted", "thrilled", "excited", "ecstatic", "elated",
        "cheerful", "glad", "pleasant", "lovely", "nice", "cool", "awesome", "fab",
        "terrific", "superb", "outstanding", "marvelous", "magnificent", "perfect",
        
        # Indonesian - Formal
        "senang", "bahagia", "gembira", "suka", "hebat", "luar biasa", "mantap",
        "keren", "bagus", "lumayan", "ok", "bagus banget", "sempurna", "indah",
        "menyenangkan", "mengasyikkan", "asyik", "seru", "fungsional", "optimal",
        "bersuka ria", "ceria", "memuaskan", "menggembirakan", "nikmat", "enak",
        "nyaman", "rileks", "santai", "tenang", "damai", "sejahtera",
        
        # Indonesian - Slang/Gaul
        "asoy", "asyoy", "sip", "oke", "okee", "okaylah", "mantull", "mantafff",
        "top", "topppp", "keren abis", "terbaik", "the best", "syumbuuul", "gokil",
        "gila", "gila-gilaan", "seru banget", "asyik banget", "fun", "enjoy",
        "wah", "wow", "ohmygodd", "subhanallah", "allhamdulillah", "alhamdulillah",
        "jaih", "ciamik", "rapi", "bagusan", "asli", "murni", "genuine",
        
        # Mixed expressions
        "sangat senang", "bahagia sekali", "gembira luar biasa", "suka banget",
        "love it", "i love", "puas banget", "syukur", "bersyukur", "hore",
    ],
    "horny" : [
        "sange", "ngewe", "ewe", "ngentot", "ngecrot", "crot", "creampie", "sepong", "nyepong", "anal"
    ],
    
    "sadness": [
        # English
        "sad", "depressed", "unhappy", "terrible", "horrible", "bad", "awful",
        "miserable", "sorrowful", "gloomy", "melancholic", "blue", "down", "low",
        "morose", "doleful", "woeful", "grief", "broken", "shattered", "devastated",
        "disappointed", "upset", "hurt", "aching", "pained", "grieving", "mourning",
        
        # Indonesian - Formal
        "sedih", "murung", "duka", "menyesal", "kecewa", "kasihan", "pilu",
        "melankolis", "remuk", "hancur", "putus asa", "putus", "lemas", "lesu",
        "malang", "nasib", "celaka", "sial", "berdukacita", "berkabung",
        "menderita", "menyakitkan", "nyeri", "sakitnya", "kesakitan", "pegal",
        "sakit hati", "terluka", "terdalam", "tersakiti", "tersedu-sedu", "terisak",
        
        # Indonesian - Slang/Gaul
        "sedih banget", "sedih parah", "sedih mati", "sedih gila", "nangis mulu",
        "mau nangis", "pengen nangis", "lemes", "lemesnya", "kusut", "kacau",
        "berantakan", "berantakan rapi", "nggak kuat", "nggak sanggup", "eneg",
        "muak", "jengah", "penat", "capek jiwa", "cape", "cukup", "tidur dulu",
        
        # Mixed expressions
        "sangat sedih", "amat sedih", "duka mendalam", "rasa sakit", "penderitaan",
        "kecewa besar", "kecewakan", "mengecewakan", "memilukan", "mengharukan",
        "menyedihkan", "kesedihan mendalam", "duka cita",
    ],
    
    "fear": [
        # English
        "scared", "afraid", "worried", "anxious", "fear", "nervous", "terrified",
        "petrified", "alarmed", "apprehensive", "fearful", "timid", "cowardly",
        "dread", "panic", "panic attack", "stressed", "tense", "uptight",
        
        # Indonesian - Formal
        "takut", "khawatir", "cemas", "gugup", "panik", "gemetar", "terancam",
        "was-was", "gelisah", "geleng", "berdebar", "berdebar-debar", "gelisah",
        "kuatir", "keprivasian", "keamanan", "perlindungan", "pertahanan",
        "terdesak", "terjebak", "terperangah", "terkecil", "ketakutan", "keterancaman",
        
        # Indonesian - Slang/Gaul
        "takut banget", "takut parah", "takut gila", "takut mati", "takut mulu",
        "was-wasan", "parno", "parno tinggi", "seram", "serem", "seremnya",
        "nyalinya jebol", "nyali ciut", "ciut nyali", "berani nggak", "berani",
        "nggak berani", "takut-takutan", "main-mainan takut", "bercanda takut",
        
        # Mixed expressions
        "sangat takut", "takut sekali", "kecemasan mendalam", "kepanikan",
        "kekhawatiran", "ketakutan besar", "rasa takut", "perasaan takut",
    ],
    
    "disgust": [
        # English
        "disgusted", "disgusting", "gross", "nasty", "repulsive", "revolting",
        "vile", "foul", "appalling", "atrocious", "abhorrent", "abominable",
        "offensive", "repugnant", "loathsome", "sickening", "stomach-turning",
        
        # Indonesian
        "jijik", "kotor", "benci", "muak", "bau", "jelek", "tidak suka",
        "tidak senang", "tidak enak", "tidak layak", "tidak pantas", "merasa jijik",
        "mual", "eneg", "sebal dengan", "bencikan", "hindari", "jauhi",
        
        # Mixed
        "sangat jijik", "jijik banget", "kotor sekali", "bau busuk",
    ],
    
    "surprise": [
        # English
        "surprised", "shocked", "amazed", "astonished", "stunned", "astounded",
        "flabbergasted", "startled", "taken aback", "bewildered", "dumbfounded",
        
        # Indonesian
        "terkejut", "kaget", "heran", "terperangah", "terheran-heran", "aneh",
        "tidak diduga", "tidak terduga", "mengejutkan", "menakjubkan", "menakherankan",
        
        # Mixed
        "sangat terkejut", "kaget banget", "wow terkejut", "jauh dari ekspektasi",
    ],
    
    "trust": [
        # English
        "trust", "confident", "assured", "secure", "faithful", "loyal", "reliable",
        "dependable", "true", "authentic", "genuine", "sincere", "honest",
        
        # Indonesian
        "percaya", "kepercayaan", "yakin", "mantap", "terpercaya", "dapat diandalkan",
        "setia", "jujur", "tulus", "ikhlas", "asli", "murni",
        
        # Mixed
        "sangat percaya", "percaya penuh", "yakin penuh", "merasa aman",
    ],
    
    "anticipation": [
        # English
        "excited", "eager", "enthusiastic", "looking forward", "anticipating",
        "hopeful", "optimistic", "inspired", "motivated", "energized",
        
        # Indonesian
        "antusias", "bersemangat", "optimis", "berharap", "penantian", "penunggu",
        "ditunggu", "ditunggu-tunggu", "ingin sekali", "tidak sabar", "semangat",
        
        # Mixed
        "sangat antusias", "bersemangat tinggi", "optimis banget", "penuh harapan",
    ]
}

# Kata-kata yang sering typo beserta koreksinya
COMMON_TYPOS = {
    # Dari contoh user
    "berantem": "berantem",  # Sudah benar
    "pengen": "pengen",      # Sudah benar
    "gua": "gua",            # Sudah benar
    
    # Typo umum English
    "ang": "angry",
    "joyy": "joy",
    "hapyy": "happy",
    "sadd": "sad",
    "feer": "fear",
    "awfull": "awful",
    "wonderfull": "wonderful",
    "teribl": "terrible",
    "angsty": "angry",
    "woryed": "worried",
    "fricenend": "friend",
    "happines": "happiness",
    
    # Typo umum Indonesian
    "snag": "senang",
    "bgli": "bagaimana",
    "senng": "senang",
    "seng": "senang",
    "snng": "senang",
    "gemira": "gembira",
    "gemr": "gembira",
    "suka2": "suka",
    "skir": "suka",
    "skaa": "suka",
    "skli": "sekali",
    "bngt": "banget",
    "bgt": "banget",
    "bgt2": "banget",
    "trs": "turus",
    "lgi": "lagi",
    "sdh": "sudah",
    "sdgh": "sedang",
    "blm": "belum",
    "jd": "jadi",
    "krn": "karena",
    "krene": "karena",
    "karna": "karena",
    "gue": "gua",
    "lo": "lu",
    "lu": "lu",
    "elu": "lu",
    "ngga": "nggak",
    "ga": "nggak",
    "g": "nggak",
    "gk": "nggak",
    "nggk": "nggak",
    "mjd": "menjadi",
    "mjd": "menjadi",
    "sjk": "sejak",
    "dri": "dari",
    "drpd": "dari pada",
    "saat": "saat",
    "bkn": "bukan",
    "skrg": "sekarang",
    "skrng": "sekarang",
    "asli": "asli",
    "ascli": "asli",
    "mantap": "mantap",
    "mantaf": "mantap",
    "mantull": "mantap",
    "krel": "keren",
    "keren": "keren",
    "krren": "keren",
    "gelisa": "gelisah",
    "gelsih": "gelisah",
    "dukc": "duka",
    "sedih": "sedih",
    "sediih": "sedih",
    "takot": "takut",
    "takut": "takut",
    "tkt": "takut",
    "tko": "takut",
    "benci": "benci",
    "bcni": "benci",
    "bnci": "benci",
    "marh": "marah",
    "marha": "marah",
    "mrah": "marah",
    "marah": "marah",
}

SIMILARITY_THRESHOLD = 0.75  # Lowered for better matching


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def validate_and_correct_words(text: str) -> Tuple[str, Dict]:
    words = text.lower().split()
    corrected_words = []
    corrections = {
        "total_words": len(words),
        "corrections_made": 0,
        "correction_details": []
    }
    
    # Semua valid keywords dari semua emotion
    all_valid_keywords = []
    for emotion_keywords in EMOTION_VOCABULARY.values():
        all_valid_keywords.extend(emotion_keywords)
    
    all_valid_keywords = list(set(all_valid_keywords))  # Remove duplicates
    
    for word in words:
        clean_word = re.sub(r'[!?.,-]', '', word)
        
        # Skip empty words
        if not clean_word:
            continue
        
        # Cek di common typos dulu (priority)
        if clean_word in COMMON_TYPOS:
            corrected_word = COMMON_TYPOS[clean_word]
            corrected_words.append(corrected_word)
            if clean_word != corrected_word:
                corrections["corrections_made"] += 1
                corrections["correction_details"].append({
                    "original": clean_word,
                    "corrected": corrected_word,
                    "method": "typo_dictionary",
                    "confidence": 1.0
                })
        # Cek di valid keywords
        elif clean_word in all_valid_keywords:
            corrected_words.append(clean_word)
        # Gunakan fuzzy matching untuk kata yang mirip
        else:
            matches = get_close_matches(
                clean_word, 
                all_valid_keywords, 
                n=1, 
                cutoff=SIMILARITY_THRESHOLD
            )
            if matches:
                corrected_word = matches[0]
                corrected_words.append(corrected_word)
                corrections["corrections_made"] += 1
                
                # Hitung similarity score
                similarity = calculate_similarity(clean_word, corrected_word)
                corrections["correction_details"].append({
                    "original": clean_word,
                    "corrected": corrected_word,
                    "method": "fuzzy_match",
                    "confidence": round(similarity, 2)
                })
            else:
                # Jika tidak ada match, tetap simpan kata asli
                corrected_words.append(clean_word)
    
    corrected_text = " ".join(corrected_words)
    return corrected_text, corrections


def calculate_similarity(str1: str, str2: str) -> float:
    """Hitung similarity antara dua string"""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, str1, str2).ratio()


def extract_text_features(text: str) -> Dict:
    features = {
        "text_length": len(text),
        "word_count": len(text.split()),
        "punctuation_count": sum(1 for c in text if c in "!?.,-"),
        "uppercase_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "caps_lock_words": len([w for w in text.split() if w.isupper() and len(w) > 1]),
    }
    return features


def keyword_based_emotion(text: str) -> Tuple[str, float]:
    """
    Deteksi emosi berdasarkan keyword dengan kosakata yang sangat lengkap
    """
    text_lower = text.lower()
    
    emotions = {}
    emotion_matches = {}

    for emotion, keywords in EMOTION_VOCABULARY.items():
        count = 0
        matched_keywords = []
        for kw in keywords:
            if kw in text_lower:
                count += 1
                matched_keywords.append(kw)
        emotions[emotion] = count
        emotion_matches[emotion] = matched_keywords
    
    if not any(emotions.values()):
        return "neutral", 0.5, {}
    
    max_emotion = max(emotions, key=emotions.get)
    max_count = emotions[max_emotion]
    
    confidence = min(0.95, 0.5 + (max_count * 0.15))
    
    return max_emotion, confidence, emotion_matches


def get_emotion_explanation(emotion: str, matched_keywords: List[str]) -> str:
    """Memberikan penjelasan mengapa emosi terdeteksi"""
    if not matched_keywords:
        return f"Emosi '{emotion}' terdeteksi berdasarkan konteks umum."
    
    keywords_str = ", ".join(matched_keywords[:3])
    if len(matched_keywords) > 3:
        keywords_str += f", dan {len(matched_keywords) - 3} kata lainnya"
    
    return f"Terdeteksi dari kata-kata: {keywords_str}"