import re

def clean_text(text: str) -> str:
    """Membersihkan teks: lowercase dan hapus karakter spesial."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()