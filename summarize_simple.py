"""
Extractive simmarization depend on the sentences.
Algoritm exract sentences that has most of the used words depending with sentence weight.
Algiritm calculate sum similarity for each sentence.
"""
import re
import os
from typing import List, Union
import math


def get_stop_words(filepath: Union[str, None]=None) -> List[str]:
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), 'stopwords.txt')
    with open(filepath, 'r') as f:
        return [w.strip().lower() for w in f.readlines()]

try:
    stop_words = get_stop_words()
except Exception as e:
    print(e)
    stop_words = []


def tokenize(text: str, stop_words: List[str]=stop_words)-> List[str]:
    """Tokenize text. Use raw stemming - first 4 letters."""
    text = text.lower()
    text = ' '.join([w for w in text.split() if w not in stop_words])
    parts = re.findall(r'\w{3,}', text)
    return [p[:4] for p in parts]


def extract_sentences(text: str)-> List[str]:
    """Extract sentences from text."""
    text = text.strip()
    text = text.replace('\n', '<sep>')  # new line separator
    text = re.sub(r'\.+', '.', text)
    text = re.sub(r'([a-zа-яё]{2,}\s*[.?!]+)\s*([A-ZА-ЯЁ])', '\g<1><sep>\g<2>', text)  # start of sentence
    text = re.sub(r'([a-zа-яё]{2,}[.?!]+)\s+(\d)', '\g<1><sep>\g<2>', text)  # start of numbered item
    sents = text.split('<sep>')
    res = []
    for s in sents:
        s = s.strip()
        if len(s) > 5:  # remove short sentences
            res.append(s)
    return res


def get_sentences_sim(s1: List[str], s2: List[str]) -> float:
    """calculate IOU similarity for two sentences
    Args:
        s1 (str): first sentence tolkenized
        s2 (str): second sentence tolkenized
    """
    s1_words = set(s1)
    s2_words = set(s2)

    try:
        rel = len(s1_words.intersection(s2_words)) / len(s1_words.union(s2_words))
    except ZeroDivisionError as e:
        rel = 0
    except Exception as e:
        print(e)
        rel = 0
    return rel


def get_matr_sim(sents: List[List[str]]) -> List[List[float]]:
    """
    Args:
        sents (List[List[str]]): list of tokenized sentences
    return: n*n matrix with similarity scores
    """
    n = len(sents)
    matr_sim = []
    for i in range(n):
        matr_sim.append([0] * n)

    for i in range(n):
        for j in range(i, n):
            if i == j:
                matr_sim[i][j] = 0
            else:
                sim = get_sentences_sim(sents[i], sents[j])
                matr_sim[i][j] = sim
                matr_sim[j][i] = sim

    return matr_sim


def correct_norm(x: float) -> float:
    return x if x > 0 else 0.0001


def calc_scores(matr_sim: List[List[float]]) -> List[float]:
    """ calculate importance score for each row
    Args:
        matr_sim (List[List[float]]): n*n matrix with similarity scores
    return: List[float] - importance score for each row

    Algorithm:
        1. calculate norms (sum of scores for each row)
        2. normalize matrix - normalize each row to relative similarity
        3. calculate sum of relative similarities by columns
    """
    # calculate norms (sum of scores for each row)
    norms = [0] * len(matr_sim)
    for i in range(len(matr_sim)):
        norms[i] = correct_norm(sum(matr_sim[i]))

    # normalize matrix - normalize each row to relative similarity
    for i in range(len(matr_sim)):
        for j in range(len(matr_sim)):
            matr_sim[i][j] = matr_sim[i][j] / norms[i]

    # calculate sum of relative similarities by columns
    scores = [0] * len(matr_sim)  # init scores
    for i in range(len(norms)):
        for j in range(len(norms)):
            scores[j] += matr_sim[i][j]
    return scores


def get_top_n_indexes(scores: List[float], n: int) -> List[int]:
    """ get indexes of top n by score """
    scores_places = list(zip(scores, range(len(scores))))
    scores_places = sorted(scores_places, key=lambda x: x[0], reverse=True)
    top_scores = scores_places[:n]

    indexes = sorted([i[1] for i in top_scores])
    return indexes


def extract_sum(text: str, n: int, stop_words:List[str]=stop_words) -> List[str]:
    """
    extract n snetences form text for maximazing summary information. (Extract most informative sentences from text)

    return List[str]
    """
    sents = extract_sentences(text)

    if n <= 0:
        return []

    if n >= len(sents):
        return sents

    tokenized_sents = [tokenize(s, stop_words=stop_words) for s in sents]

    data = list(zip(sents, tokenized_sents))
    data = [d for d in data if len(d[1]) > 2]  # remove short sentences
    sents = [d[0] for d in data]
    tokenized_sents = [d[1] for d in data]

    matr_sim = get_matr_sim(tokenized_sents)
    scores = calc_scores(matr_sim)
    indexes = get_top_n_indexes(scores, n)

    top_sents = [sents[i] for i in indexes]
    return top_sents


def summarize(text: str, n: Union[int, None]=None, first_n: int=4, last_n: int=2, stop_words: List[str]=stop_words) -> str:
    """
    Create summary from text.
    Args:
        text (str): text to summarize
        n (int): number of sentences to extract excluding the first_n and the last_n sentences.
                 If n is None: n = math.ceil(math.sqrt(len(sents) - first_n - last_n)) + 3
        first_n (int): number of sentences to extract from the beginning
        last_n (int): number of sentences to extract from the end
        stop_words (List[str]): list of stop words. Set [] for no stop words, use stop_words=stop_words by default.
    return: str
    """
    sents = extract_sentences(text)
    first_n = min(int(first_n), len(sents))
    first_n = max(first_n, 0)

    last_n = min(int(last_n), len(sents))
    last_n = max(last_n, 0)

    if first_n + last_n >= len(sents):
        return '\n'.join(sents)

    if n is None:
        n = math.ceil(math.sqrt(len(sents) - first_n - last_n)) + 3
        if n <= 0:
            n = 0

    n = min(n, len(sents) - first_n - last_n)
    n = max(n, 0)

    if first_n > 0:
        sents_first = sents[:first_n]
    else:
        sents_first = []

    if last_n > 0:
        sents_last = sents[-last_n:]
    else:
        sents_last = []

    if last_n == 0 and first_n == 0:
        sents_to_sum = sents
    elif last_n == 0:
        sents_to_sum = sents[first_n:]
    elif first_n == 0:
        sents_to_sum = sents[:-last_n]
    else:
        sents_to_sum = sents[first_n:-last_n]
    sents_sum = extract_sum('\n'.join(sents_to_sum), n)
    sents_sum = sents_first + sents_sum + sents_last
    return '\n'.join(sents_sum)
