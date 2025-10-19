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


def parse_multi(sentence_list: list[str], lang: str) -> list[dict[str, str]]:
    return [parse(sentence, lang) for sentence in sentence_list]


def parse(sentence: str, lang: str) -> dict[str, str]:
    print(f"original sentence: `{sentence}`")

    tokens = remove_stopwords(sentence, lang=lang)
    words_pos_map = map_words_with_pos(tokens)

    print(f"result: {words_pos_map}")
    return words_pos_map


def concat_salient_words(word_pos_map: dict[str, str]) -> tuple[str, ...]:
    return tuple(" ".join(word_pos_map.keys()))


def test():
    sentences = [
        "Sugar tastes sweet.",
        "The lemonade is too sour.",
        "She likes a bitter aftertaste.",
        "The honey was very sweet.",
        "Dark chocolate has a bitter flavor.",
    ]

    spanish_sentences = ["Esta es una oración de ejemplo en español."]

    assert len(parse_multi(sentences, "english")) >= 1
    assert len(parse_multi(spanish_sentences, "spanish")) >= 1

    assert len(concat_salient_words(parse(sentences[0], "english"))) >= 1


if __name__ == "__main__":
    test()
