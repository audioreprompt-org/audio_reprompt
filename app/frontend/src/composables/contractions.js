const ENGLISH_CONTRACTIONS = {
  // 1. Contacciones estándar
  "i'm": "i am",
  "i'll": "i will",
  "i'd": "i would",
  "i've": "i have",
  "you're": "you are",
  "you'll": "you will",
  "you'd": "you would",
  "you've": "you have",
  "he's": "he is",
  "he'll": "he will",
  "she's": "she is",
  "she'll": "she will",
  "it's": "it is",
  "it'll": "it will",
  "we're": "we are",
  "we'll": "we will",
  "we'd": "we would",
  "we've": "we have",
  "they're": "they are",
  "they'll": "they will",
  "they'd": "they would",
  "they've": "they have",

  // 2. Negaciones
  "aren't": "are not",
  "can't": "cannot",
  "couldn't": "could not",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hasn't": "has not",
  "haven't": "have not",
  "isn't": "is not",
  "mightn't": "might not",
  "mustn't": "must not",
  "shan't": "shall not",
  "shouldn't": "should not",
  "wasn't": "was not",
  "weren't": "were not",
  "won't": "will not",
  "wouldn't": "would not",
  "needn't": "need not",
  "ain't": "am not",
  "y'all'd've": "you all would have",

  // 3. Contraciones de 'have'/'had'/'would'
  "what'd": "what did",
  "who'd": "who would",
  "when'd": "when did",
  "where'd": "where did",
  "how'd": "how did",
  "here's": "here is",
  "there's": "there is",
  "that's": "that is",
  "what's": "what is",
  "who's": "who is",
  "when's": "when is",
  "where's": "where is",
  "how's": "how is",
  "let's": "let us",
  "y'all": "you all",
  "could've": "could have",
  "might've": "might have",
  "must've": "must have",
  "should've": "should have",
  "would've": "would have",
  "what'll": "what will",
  "who'll": "who will",

  // 4. Jerga Común que confunde a 'franc' con 'sco'
  gonna: "going to",
  wanna: "want to",
  gotta: "got to",
  kinda: "kind of",
  sorta: "sort of",
  hafta: "have to",
  oughta: "ought to",
  cuz: "because",
  lemme: "let me",
  gimme: "give me",
  cause: "because",
  dunno: "do not know",
  "y'all're": "you all are",
  "where'd'ya": "where did you",
  whatcha: "what are you",
};

// Mapeo de códigos ISO 639-3 a nombres legibles para el usuario.
export const LANGUAGE_MAP = {
  eng: "English",
  spa: "Spanish",
  fra: "French",
  deu: "German",
  por: "Portuguese",
  ita: "Italian",
  und: "an undetermined language",
  nld: "Dutch (Nederlands)",
  rus: "Russian",
  jpn: "Japanese",
  kor: "Korean",
  zho: "Chinese", // Genérico
  ara: "Arabic",
  hin: "Hindi",
  ben: "Bengali",
  tur: "Turkish",
  pol: "Polish",
  swe: "Swedish",
  nor: "Norwegian",
  fin: "Finnish",
  dan: "Danish",
  ell: "Greek",
  heb: "Hebrew",
  tha: "Thai",
  vie: "Vietnamese",
  ind: "Indonesian",
  cat: "Catalan",
  afr: "Afrikaans",
  sco: "Scots (Escocés)", // Importante para detectar el conflicto
  gla: "Gaelic (Escocés)",
  cym: "Welsh",

  // Códigos de sistemas (menos probables, pero útiles si Franc los devuelve):
  tlh: "Klingon (Fictional)",
  mul: "Multiple languages",
};

export const expandContractions = (text) => {
  let expandedText = text.toLowerCase();

  for (const contracted in ENGLISH_CONTRACTIONS) {
    const expanded = ENGLISH_CONTRACTIONS[contracted];
    // Usamos \b para asegurar límites de palabra y evitar reemplazos parciales
    const regex = new RegExp("\\b" + contracted + "\\b", "g");

    expandedText = expandedText.replace(regex, expanded);
  }

  return expandedText;
};
