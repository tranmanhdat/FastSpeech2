""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

from text import cmudict, pinyin

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]
_characters = ""
# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
# _arpabet = ["@" + s for s in cmudict.valid_symbols]
_pinyin = ["@" + s for s in pinyin.valid_symbols]

_ipa = ['ɯəj', 'ɤ̆j', 'ʷiə', 'ɤ̆w', 'ɯəw', 'ʷet', 'iəw', 'uəj', 'ʷen', 'tʰw', 'ʷɤ̆', 'ʷiu', 'kwi', 'ŋ͡m', 'k͡p', 'cw', 'jw', 'uə', 'eə', 'bw', 'oj', 'ʷi', 'vw', 'ăw', 'ʈw', 'ʂw', 'aʊ', 'fw', 'ɛu', 'tʰ', 'tʃ', 'ɔɪ', 'xw', 'ʷɤ', 'ɤ̆', 'ŋw', 'ʊə', 'zi', 'ʷă', 'dw', 'eɪ', 'aɪ', 'ew', 'iə', 'ɣw', 'zw', 'ɯj', 'ʷɛ', 'ɯw', 'ɤj', 'ɔ:', 'əʊ', 'ʷa', 'mw', 'ɑ:', 'hw', 'ɔj', 'uj', 'lw', 'ɪə', 'ăj', 'u:', 'aw', 'ɛj', 'iw', 'aj', 'ɜ:', 'kw', 'nw', 't∫', 'ɲw', 'eo', 'sw', 'tw', 'ʐw', 'iɛ', 'ʷe', 'i:', 'ɯə', 'dʒ', 'ɲ', 'θ', 'ʌ', 'l', 'w', '1', 'ɪ', 'ɯ', 'd', '∫', 'p', 'ə', 'u', 'o', '3', 'ɣ', '!', 'ð', 'ʧ', '6', 'ʒ', 'ʐ', 'z', 'v', 'g', 'ă', '_', 'æ', 'ɤ', '2', 'ʤ', 'i', '.', 'ɒ', 'b', 'h', 'n', 'ʂ', 'ɔ', 'ɛ', 'k', 'm', '5', ' ', 'c', 'j', 'x', 'ʈ', ',', '4', 'ʊ', 's', 'ŋ', 'a', 'ʃ', '?', 'r', ':', 'η', 'f', ';', 'e', 't', "'", 'sp', 'spn', 'sil']

_vie_phones =['a', 'b', 'c', 'ch', 'd', 'e', 'f', 'g', 'gh', 'gi', 'h', 'i', 'iê', 'iế', 'iề', 'iể', 'iễ', 'iệ', 'k', 'kh', 'kw', 'l', 'm', 'n', 'ng', 'ngh', 'nh', 'o', 'oa', 'oe', 'oo', 'oà', 'oá', 'oã', 'oè', 'oé', 'oò', 'oó', 'oõ', 'oă', 'oạ', 'oả', 'oắ', 'oằ', 'oẳ', 'oẵ', 'oặ', 'oẹ', 'oẻ', 'oẽ', 'oọ', 'oỏ', 'p', 'ph', 'q', 'r', 's', 'sh', 't', 'th', 'tr', 'u', 'uy', 'uyê', 'uyế', 'uyề', 'uyể', 'uyễ', 'uyệ', 'uâ', 'uê', 'uô', 'uý', 'uơ', 'uấ', 'uầ', 'uẩ', 'uẫ', 'uậ', 'uế', 'uề', 'uể', 'uễ', 'uệ', 'uố', 'uồ', 'uổ', 'uỗ', 'uộ', 'uớ', 'uờ', 'uở', 'uỡ', 'uợ', 'uỳ', 'uỵ', 'uỷ', 'uỹ', 'v', 'w', 'x', 'y', 'yê', 'yế', 'yề', 'yể', 'yễ', 'yệ', 'z', 'à', 'á', 'â', 'ã', 'è', 'é', 'ê', 'ì', 'í', 'ò', 'òa', 'ó', 'óa', 'ô', 'õ', 'ù', 'ú', 'ý', 'ă', 'đ', 'ĩ', 'ũ', 'ơ', 'ư', 'ươ', 'ướ', 'ườ', 'ưở', 'ưỡ', 'ượ', 'ạ', 'ả', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ', 'ẹ', 'ẻ', 'ẽ', 'ế', 'ề', 'ể', 'ễ', 'ệ', 'ỉ', 'ị', 'ọ', 'ọa', 'ỏ', 'ỏa', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ', 'ụ', 'ủ', 'ứ', 'ừ', 'ử', 'ữ', 'ự', 'ỳ', 'ỵ', 'ỷ', 'ỹ', 'sp', 'spn', 'sil']
# English symbols
# symbols = (
#     [_pad]
#     + list(_special)
#     + list(_punctuation)
#     + list(_letters)
#     + _arpabet
#     + _pinyin
#     + _silences
# )

# Vie IPA
# symbols = list(set(_ipa))

# My Vie phonemes
symbols = list(_vie_phones)
