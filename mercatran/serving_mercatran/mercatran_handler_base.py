# Derived from here:
# https://github.com/pytorch/serve/blob/master/ts/torch_handler/text_handler.py
import importlib.util
import logging
import os
import re
import string
import unicodedata
from abc import ABC
from typing import Callable, List, Union

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from ts.torch_handler.base_handler import BaseHandler
from ts.torch_handler.contractions import CONTRACTION_MAP
from ts.utils.util import list_classes_from_module

import model_config

CLEANUP_REGEX = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
logger = logging.getLogger(__name__)


CONTRACTIONS_PATTERN = re.compile(
    "({})".format("|".join(CONTRACTION_MAP.keys())),
    flags=re.IGNORECASE | re.DOTALL,
)


class MercatranHandlerBase(BaseHandler, ABC):
    """
    Base class for all text based default handler.
    Contains various text based utility methods
    """

    def __init__(self):
        super().__init__()
        self.source_vocab = None
        self.input_text = None
        self.lig = None
        self.initialized = None

    def initialize(self, context):
        """
        Loads the model and Initializes the necessary artifacts
        """
        super().initialize(context)
        self.initialized = False
        source_vocab = (
            self.manifest["model"]["sourceVocab"]
            if "sourceVocab" in self.manifest["model"]
            else None
        )
        if source_vocab:
            # Backward compatibility
            self.source_vocab = Tokenizer.from_file(source_vocab)
        else:
            self.source_vocab = Tokenizer.from_file(
                self.get_source_vocab_path(context))

        # Captum initialization
        self.lig = None
        self.initialized = True

    def _load_pickled_model(self, model_dir, model_file, model_pt_path):
        """
        Loads the pickle file from the given model path.
        Args:
            model_dir (str): Points to the location of the model artefacts.
            model_file (.py): the file which contains the model class.
            model_pt_path (str): points to the location of the
            model pickle file.
        Raises:
            RuntimeError: It raises this error when the model.py
            file is missing.
            ValueError: Raises value error when there is more than
            one class in the label, since the mapping supports
            only one label per class.
        Returns:
            serialized model file: Returns the pickled pytorch model file
        """
        model_def_path = os.path.join(model_dir, model_file)
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model.py file")

        module = importlib.import_module(model_file.split(".")[0])
        model_class_definitions = list_classes_from_module(module)
        if len(model_class_definitions) != 1:
            raise ValueError(
                "Expected only one class as model definition. {}".format(
                    model_class_definitions
                )
            )

        model_class = model_class_definitions[0]
        model = model_class()
        if model_pt_path:
            state_dict = torch.load(model_pt_path, map_location=self.device)
            model.load_state_dict(state_dict)
        return model

    def get_source_vocab_path(self, ctx):
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        source_vocab_path = os.path.join(model_dir, "tokenizer.json")

        if os.path.isfile(source_vocab_path):
            return source_vocab_path
        else:
            raise Exception(
                "Missing the source_vocab file. Refer default handler "
                "documentation for details on using text_handler."
            )

    def _expand_contractions(self, text):
        """
        Expands the contracted words in the text
        """

        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = (
                CONTRACTION_MAP.get(match)
                if CONTRACTION_MAP.get(match)
                else CONTRACTION_MAP.get(match.lower())
            )
            expanded_contraction = first_char + expanded_contraction[1:]
            return expanded_contraction

        text = CONTRACTIONS_PATTERN.sub(expand_match, text)
        text = re.sub("'", "", text)
        return text

    def _remove_accented_characters(self, text):
        """
        Removes remove_accented_characters
        """
        text = (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("utf-8", "ignore")
        )
        return text

    def _remove_html_tags(self, text):
        """
        Removes html tags
        """
        clean_text = CLEANUP_REGEX.sub("", text)
        return clean_text

    def _remove_puncutation(self, *args, **kwargs):
        """
        Mispelled in original version. This is a compat layer
        """
        return self._remove_punctuation(*args, **kwargs)

    def _remove_punctuation(self, text):
        """
        Removes punctuation
        """
        return text.translate(str.maketrans("", "", string.punctuation))

    def _preprocess_text(self, text):
        return self._remove_accented_characters(
            self._remove_html_tags(self._remove_punctuation(text.casefold()))
        )

    def _tokenize(self, text):
        return self.tokenizer(text)

    def _add_token(self, collec, token_type=model_config.START_TOKEN):
        collec["tokens"].append(
            torch.tensor([self.source_vocab.token_to_id(token_type)])
        )
        collec["offsets"].append(1)  # size of the token

    def _add_row(
        self,
        collec,
        num_tokens=model_config.MODEL_SEQ_LEN + 2,  # start + end token
        token_type=model_config.MASK_TOKEN,
    ):
        for _ in range(num_tokens):
            self._add_token(collec=collec, token_type=token_type)

    def get_word_token(self, input_tokens):
        """
        Constructs word tokens from text
        """
        # Remove unicode space character from BPE Tokeniser
        tokens = [token.replace("Ġ", "") for token in input_tokens]
        return tokens

    def summarize_attributions(self, attributions):
        """
        Summarises the attribution across multiple runs
        """
        attributions = F.softmax(attributions)
        attributions_sum = attributions.sum(dim=-1)
        logger.info("attributions sum shape %d", attributions_sum.shape)
        attributions = attributions / torch.norm(attributions_sum)
        return attributions

    def bpe_text_pipeline(
        self,
        input: Union[List[str], str],
        tokenizer: Callable[[str], List[str]],
    ) -> List[int]:
        """A utility to convert a list of strings or a string into 
        tokens defined by the trained tokenizer"""
        return (
            tokenizer.encode(self._preprocess_text(input.casefold())).ids
            if input.strip()
            else [tokenizer.token_to_id(model_config.DEFAULT_TOKEN)]
        )

    def postprocess(self, data):
        res = {}
        for i, arr in enumerate(data):
            res[f"timestep_{i}"] = arr.tolist()
        return [res]
