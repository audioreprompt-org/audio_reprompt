from typing import Optional

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.tokenize import RegexpTokenizer


def get_synonyms(word: str, pos: Optional[str] = None) -> list:
    synonyms = set()
    synsets = wn.synsets(word, pos=pos) if pos else wn.synsets(word)

    for syn in synsets:
        for lemma in syn.lemmas():
            if lemma.name().lower() != word.lower():
                synonyms.add(lemma.name().replace("_", " "))

    return list(synonyms)


def remove_stopwords(sentence, lang: Optional[str] = "spanish") -> list[str]:
    tokenizer = RegexpTokenizer(r"[\w']+")
    words = tokenizer.tokenize(sentence)

    if not words:
        return []

    words = [word.lower() for word in words]
    return list(filter(lambda w: w not in stopwords.words(lang), words))


def get_salient_words(pos: str) -> bool:
    salient_pos = {
        "NN",
        "NNS",
        "NNP",
        "NNPS",  # nouns
        "VB",
        "VBD",
        "VBG",
        "VBN",
        "VBP",
        "VBZ",  # verbs
        "JJ",
        "JJR",
        "JJS",  # adjectives
        "RB",
        "RBR",
        "RBS",  # adverbs
    }
    return pos in salient_pos


def map_words_with_pos(words: list[str]) -> dict[str, str]:
    return {word_: tag for word_, tag in pos_tag(words) if get_salient_words(tag)}


def parse(sentence_list: list[str], lang: str) -> list[dict[str, str]]:
    results = []
    for sentence in sentence_list:
        tokens = remove_stopwords(sentence, lang=lang)
        words_pos_map = map_words_with_pos(tokens)

        print(words_pos_map)
        results.append(words_pos_map)

    return results


if __name__ == "__main__":
    sentences = [
        "Sugar tastes sweet.",
        "The lemonade is too sour.",
        "She likes a bitter aftertaste.",
        "The honey was very sweet.",
        "Dark chocolate has a bitter flavor.",
    ]

    spanish_sentences = ["Esta es una oración de ejemplo en español."]

    assert len(parse(sentences, "english")) >= 1
    assert len(parse(spanish_sentences, "spanish")) >= 1
