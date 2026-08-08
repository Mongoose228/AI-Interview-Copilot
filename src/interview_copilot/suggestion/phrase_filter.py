import re

_FILLER_BLACKLIST = {
    "uh", "uh-huh", "uh huh", "um", "hmm", "mm", "mm-hmm", "mm hmm", "yeah", "yep", "yup", 
    "yes", "no", "nope", "nah", "okay", "ok", "right", "sure", "alright", 
    "i see", "got it", "makes sense", "exactly", "ah", "oh", "wow", "interesting", "cool",
    "so", "well", "like", "you know"
}

def is_filler(text: str) -> bool:
    """
    Returns True if the text is a short filler phrase that shouldn't be sent to the LLM.
    We only drop it if it's very short (<= 3 words) AND exactly matches a known filler.
    """
    clean_text = re.sub(r'[^\w\s]', '', text.lower()).strip()
    if not clean_text:
        return True
    
    words = clean_text.split()
    if len(words) > 3:
        return False
        
    return clean_text in _FILLER_BLACKLIST
