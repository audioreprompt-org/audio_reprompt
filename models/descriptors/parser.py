from typing import Optional

from nltk import pos_tag, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
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
    words = [word.lower() for word in tokenizer.tokenize(sentence)]

    if not words:
        return []

    stopwords_lang = stopwords.words(lang)
    return list(filter(lambda w: w not in stopwords_lang, words))


def get_salient_words(pos: str) -> bool:
    return pos.upper() in {"J", "N", "V", "R"}


def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }
    return tag_dict.get(tag, wordnet.NOUN)


def map_words_with_pos(words: list[str]) -> dict[str, str]:
    words_pos_map = {word: get_wordnet_pos(word) for word in words}
    return dict(filter(lambda item: get_salient_words(item[1]), words_pos_map.items()))


def parse_multi(sentence_list: list[str], lang: str) -> list[dict[str, str]]:
    return [parse(sentence, lang) for sentence in sentence_list]


def parse(sentence: str, lang: str) -> dict[str, str]:
    print(f"original sentence: `{sentence}`")

    tokens = remove_stopwords(sentence, lang=lang)
    words_pos_map = map_words_with_pos(tokens)

    lemmatized_words = {
        WordNetLemmatizer().lemmatize(word, pos): pos.upper()
        for word, pos in words_pos_map.items()
    }

    print(f"result: {lemmatized_words}")
    return lemmatized_words


def concat_salient_words(word_pos_map: dict[str, str]) -> str:
    return " ".join(word_pos_map.keys())


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
