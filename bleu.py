from decimal import DecimalException
import os
import re
import sys

import warnings
warnings.filterwarnings('ignore')

from nltk.translate.bleu_score import corpus_bleu
from num2words import num2words
import pandas as pd

ALLOWED_CHARS = {
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'ä', 'ö', 'ü',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    ' ',
}

WHITESPACE_REGEX = re.compile(r'[ \t]+')

NUMBER_REGEX = re.compile(r"^[0-9',.]+$")
NUMBER_DASH_REGEX = re.compile('[0-9]+[-\u2013\xad]')
DASH_NUMBER_REGEX = re.compile('[-\u2013\xad][0-9]+')


def preprocess_transcript(transcript):
    transcript = transcript.lower()
    transcript = transcript.replace('ß', 'ss')
    transcript = transcript.replace('ç', 'c')
    transcript = transcript.replace('á', 'a')
    transcript = transcript.replace('à', 'a')
    transcript = transcript.replace('â', 'a')
    transcript = transcript.replace('é', 'e')
    transcript = transcript.replace('è', 'e')
    transcript = transcript.replace('ê', 'e')
    transcript = transcript.replace('í', 'i')
    transcript = transcript.replace('ì', 'i')
    transcript = transcript.replace('î', 'i')
    transcript = transcript.replace('ó', 'o')
    transcript = transcript.replace('ò', 'o')
    transcript = transcript.replace('ô', 'o')
    transcript = transcript.replace('ú', 'u')
    transcript = transcript.replace('ù', 'u')
    transcript = transcript.replace('û', 'u')
    transcript = transcript.replace('-', ' ')
    transcript = transcript.replace('\u2013', ' ')
    transcript = transcript.replace('\xad', ' ')
    transcript = transcript.replace('/', ' ')
    transcript = WHITESPACE_REGEX.sub(' ', transcript)
    transcript = ''.join([char for char in transcript if char in ALLOWED_CHARS])
    transcript = WHITESPACE_REGEX.sub(' ', transcript)
    transcript = transcript.strip()

    return transcript


def score(df_true, df_pred):
    pred_path_to_sentence = {path: sentence for path, sentence in zip(df_pred['path'], df_pred['sentence'])}
    list_of_references = []
    hypotheses = []
    for true_path, true_sentence in zip(df_true['path'], df_true['sentence']):
        assert true_path in pred_path_to_sentence

        true_sentence_nums_to_words = sentence_nums_to_words(true_sentence)
        list_of_references.append([
            split(preprocess_transcript(true_sentence)),
            split(preprocess_transcript(true_sentence_nums_to_words)),
        ])
        hypotheses.append(split(preprocess_transcript(pred_path_to_sentence[true_path])))

    return corpus_bleu(list_of_references, hypotheses)


def sentence_nums_to_words(transcript):
    def transform_word(word):
        # num2words if the word is just one number
        if NUMBER_REGEX.match(word) is not None:
            try:
                if word.endswith('.'):
                    return num2words(word, lang='de', to='ordinal')
                else:
                    return num2words(word, lang='de', to='cardinal')
            except DecimalException:
                return word
        else:
            # num2words for the number part if the word contains a number followed by a dash
            match = NUMBER_DASH_REGEX.search(word)
            if match is not None:
                num = word[match.start():match.end() - 1]
                try:
                    num = num2words(num, lang='de', to='cardinal')
                except DecimalException:
                    pass
                word = word[:match.start()] + num + word[match.end() - 1:]

            # num2words for the number part if the word contains a dash followed by a number
            match = DASH_NUMBER_REGEX.search(word)
            if match is not None:
                num = word[match.start() + 1:match.end()]
                try:
                    num = num2words(num, lang='de', to='cardinal')
                except DecimalException:
                    pass
                word = word[:match.start() + 1] + num + word[match.end():]

            return word

    transcript = WHITESPACE_REGEX.sub(' ', transcript)
    transcript = transcript.strip()

    return ' '.join(transform_word(w) for w in split(transcript))


def split(transcript):
    return transcript.split(' ')


if __name__ == '__main__':
    path_to_script = os.path.dirname(sys.argv[0])
    df_submission = pd.read_csv(os.path.join(path_to_script, 'submission.csv'), sep=',', encoding='utf-8')
    df_public = pd.read_csv(os.path.join(path_to_script, 'public.csv'), sep=',', encoding='utf-8')
    df_private = pd.read_csv(os.path.join(path_to_script, 'private.csv'), sep=',', encoding='utf-8')

    score_public = score(df_public, df_submission)
    score_private = score(df_private, df_submission)

    print('%.20f' % score_public, ';', '%.20f' % score_private)
