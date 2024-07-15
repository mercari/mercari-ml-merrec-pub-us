# Some parts derived from here: https://github.com/pytorch/serve/blob/master/ts/torch_handler/text_handler.py
import re
import string
import unicodedata

DEFAULT_TOKEN = "<unk>"
START_TOKEN = "<s>"
MASK_TOKEN = "<blank>"
END_TOKEN = "</s>"
CLEANUP_REGEX = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")


def remove_accented_characters(text):
    text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )
    return text


def custom_regex(text):
    clean_text = CLEANUP_REGEX.sub("", text)
    return clean_text


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def preprocess_text(text):
    return remove_accented_characters(custom_regex(
        remove_punctuation(text.casefold())))
