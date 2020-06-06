from transformers import pipeline, AutoTokenizer

NER_MODEL = "mrm8488/bert-spanish-cased-finetuned-ner"
nlp_ner = pipeline("ner", model=NER_MODEL, tokenizer=(NER_MODEL, {"use_fast": False}))
tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert-spanish-cased-finetuned-ner")
MAX_SENT_LEN = 500


def is_sent_too_long_for_ner(text):
    return len(tokenizer.tokenize(text)) > MAX_SENT_LEN


def ner(text):
    return nlp_ner(text)
