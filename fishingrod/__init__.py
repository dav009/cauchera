from fishingrod.ner import ner, is_sent_too_long_for_ner
import es_core_news_sm

nlp = es_core_news_sm.load()


def split_into_sentences(text):
    """
    uses spaCy to chunk text into sentences
    """
    doc = nlp(text)
    return doc.sents


def split_text_if_needed(text):
    """
    splits big chunks of texts into paragraphs/sentences  which can be consumed by bert models
    """
    paragraphs = text.split("\n\n")
    for p in paragraphs:
        if is_sent_too_long_for_ner(p):
            for s in split_into_sentences(p):
                yield s.text
        else:
            yield p


def extract_relations(text):
    """
    returns a list of relations
    e1, r, e2
    """
    texts = split_text_if_needed(text)
    for t in texts:
        yield t, ner(t)

