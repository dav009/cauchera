from transformers import pipeline, AutoTokenizer

NER_MODEL = "mrm8488/bert-spanish-cased-finetuned-ner"
MAX_SENT_LEN = 500

nlp_ner = pipeline(
    "ner",
    model=NER_MODEL,
    grouped_entities=False,
    tokenizer=(NER_MODEL, {"use_fast": False}),
)
tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert-spanish-cased-finetuned-ner")


def is_sent_too_long_for_ner(text):
    return len(tokenizer.tokenize(text)) > MAX_SENT_LEN


def join_b_i_tags(entities):
    """
    fix over hugging face grouped entities bug
    remove this once these get solved:
    https://github.com/huggingface/transformers/issues/5077
    and https://github.com/huggingface/transformers/issues/4816
    """
    last_tag = ""
    current_entity = ""
    entity_type = ""
    output_entities = []
    for i in entities:
        current_tag = i["entity"]
        if current_tag.startswith("B"):
            if current_entity:
                output_entities.append((current_entity, entity_type))
            entity_type = current_tag.split("-")[1]

            current_entity = i["word"]
            if current_entity.startswith("##"):
                current_entity = current_entity.replace("##", "")
            last_tag = current_tag
        elif current_tag.startswith("I"):
            if i["word"].startswith("##"):
                current_entity = current_entity + i["word"].replace("#", "")
            else:
                current_entity = current_entity + " " + i["word"]
            last_tag = current_tag
    if last_tag:
        output_entities.append((current_entity, entity_type))
    return output_entities


def ner(text):
    return join_b_i_tags(nlp_ner(text))
