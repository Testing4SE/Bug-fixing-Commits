from spacy.lang.en import English
from spacy.tokens import Doc
import spacy
from typing import List
from tqdm import tqdm
import re
import json
import numpy as np

class SpacyPreprocessor:
    def __init__(
        self,
        spacy_model=None,
        remove_numbers=False,
        remove_special=True,
        pos_to_remove=None,
        remove_stopwords=False,
        lemmatize=False,
    ):
        """
        Preprocesses text using spaCy
        :param remove_numbers: Whether to remove numbers from text
        :param remove_stopwords: Whether to remove stopwords from text
        :param remove_special: Whether to remove special characters (including numbers)
        :param pos_to_remove: list of PoS tags to remove
        :param lemmatize:  Whether to apply lemmatization
        """

        self._remove_numbers = remove_numbers
        self._pos_to_remove = pos_to_remove
        self._remove_stopwords = remove_stopwords
        self._remove_special = remove_special
        self._lemmatize = lemmatize

        if not spacy_model:
            self.model = spacy.load("en_core_web_sm")
        else:
            self.model = spacy_model

    @staticmethod
    def download_spacy_model(model="en_core_web_sm"):
        print(f"Downloading spaCy model {model}")
        spacy.cli.download(model)
        print(f"Finished downloading model")

    @staticmethod
    def load_model(model="en_core_web_sm"):
        return spacy.load(model, disable=["ner", "parser"])

    def tokenize(self, text) -> List[str]:
        """
        Tokenize text using a spaCy pipeline
        :param text: Text to tokenize
        :return: list of str
        """
        doc = self.model(text)
        return [token.text for token in doc]

    def preprocess_text(self, text) -> str:
        """
        Runs a spaCy pipeline and removes unwanted parts from text
        :param text: text string to clean
        :return: str, clean text
        """
        doc = self.model(text)
        return self.__clean(doc)

    def preprocess_text_list(self, texts=List[str]) -> List[str]:
        """
        Runs a spaCy pipeline and removes unwantes parts from a list of text.
        Leverages spaCy's `pipe` for faster batch processing.
        :param texts: List of texts to clean
        :return: List of clean texts
        """
        clean_texts = []
        for doc in tqdm(self.model.pipe(texts)):
            clean_texts.append(self.__clean(doc))

        return clean_texts

    def __clean(self, doc: Doc) -> str:

        tokens = []
        # POS Tags removal
        if self._pos_to_remove:
            for token in doc:
                if token.pos_ not in self._pos_to_remove:
                    tokens.append(token)
        else:
            tokens = doc

        # Remove Numbers
        if self._remove_numbers:
            tokens = [
                token for token in tokens if not (token.like_num or token.is_currency)
            ]

        # Remove Stopwords
        if self._remove_stopwords:
            tokens = [token for token in tokens if not token.is_stop]
        # remove unwanted tokens
        tokens = [
            token
            for token in tokens
            if not (
                token.is_punct or token.is_space or token.is_quote or token.is_bracket
            )
        ]

        # Remove empty tokens
        tokens = [token for token in tokens if token.text.strip() != ""]

        # Lemmatize
        if self._lemmatize:
            text = " ".join([token.lemma_ for token in tokens])
        else:
            text = " ".join([token.text for token in tokens])

        if self._remove_special:
            # Remove non alphabetic characters
            text = re.sub(r"[^a-zA-Z\']", " ", text)
        # remove non-Unicode characters
        text = re.sub(r"[^\x00-\x7F]+", "", text)

        text = text.lower()

        return text

bag_of_words = ['fix', 'fixing', 'fixed', 'fixes', 'error', 'errors', 'bug', 'bugs', 'buggy', 'mistake', 'mistakes', 'incorrect', 'fault', 'faulty',
                'defect', 'defects', 'flaw', 'flaws', 'repair', 'repairs', 'repaired', 'repairing']
                
def countterm(text):
    nlp = English()  
    doc = nlp(text)
    term_count = 0
    for token in doc:
        if str(token) in bag_of_words:
            term_count += 1
    return term_count
    
from SNgramExtractor import SNgramExtractor
def DependencyTree(text):
    SNgram_obj = SNgramExtractor(text, meta_tag='original', trigram_flag='yes', nlp_model=None)
    output = SNgram_obj.get_SNgram()
    return output['SNBigram']

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
def FetureExtr(corpus, gram_range):
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=gram_range, max_features=8000)
    X = vectorizer.fit_transform(corpus)
    transformer = TfidfTransformer()
    X = transformer.fit_transform(X)
    C = vectorizer.get_feature_names_out()
    return X, C



def load_cross_validation_split(use_filtered):
    if use_filtered==True:
        #fname = "../data/v4_10-folds_filtered.json" # This is the old version split
        assert False, "NOT support!"
    else:
        #fname = "../data/v4_10-folds.json"
        fname = "../data/v5_10-folds.json"
    with open(fname, "r") as f:
        d = json.load(f)
        labels_folds = d["label"]
        commits_folds = d["commit"]
        metrics_folds = d["metric"]
        return labels_folds, commits_folds, metrics_folds

bag_of_words = ['fix', 'fixing', 'fixed', 'fixes', 'error', 'errors', 'bug', 'bugs', 'buggy', 'mistake', 'mistakes', 'incorrect', 'fault', 'faulty',
                'defect', 'defects', 'flaw', 'flaws', 'repair', 'repairs', 'repaired', 'repairing']
seperators = [', ', '; ', ': ', '. ', ' - ']

nlp = spacy.load("en_core_web_sm")
def process_commit_info(text):
    # 1. split sentences
    doc = nlp(text)
    sents = [str(sent) for sent in doc.sents]
    for seperator in seperators:
        sub_sents = []
        for sent in sents:
            while sent != '':
                idx = sent.find(seperator)
                if idx != -1:
                    sub_sent = sent[:idx+1]
                    sent = sent[idx+1:]
                else:
                    sub_sent = sent
                    sent = ''
                sub_sents.append(sub_sent)
        sents = sub_sents
    sents = [sent for sent in sents if sent.strip()!=""]
    # 2. delete those not including keywords
    kept_sents = []
    for sent in sents:
        tokens = set([str(token).lower() for token in nlp(sent)])
        for word in bag_of_words:
            if word in tokens:
                kept_sents.append(sent)
                break
    # 3. space tokenize and combine sentences
    tokens = []
    for sent in kept_sents:
        tokens += sent.split()
    new_text = ' '.join(tokens)
    return new_text

def normalize_metrics(metrics):
    metrics = np.asarray(metrics)
    min_metric = np.min(metrics, axis=0)
    max_metric = np.max(metrics, axis=0)
    norm_metrics = (metrics - min_metric) / (max_metric - min_metric)
    return norm_metrics