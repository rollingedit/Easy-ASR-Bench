from app.scoring import normalize_words, score_against_reference, wer


def test_accents_are_preserved_in_normalized_words():
    assert normalize_words("cafe") != normalize_words("café")
    assert wer("café", "cafe", normalized=True) > 0


def test_curly_and_ascii_apostrophes_are_normalized_consistently():
    assert normalize_words("don't stop") == normalize_words("don’t stop")


def test_cjk_text_is_not_destroyed_by_normalization():
    assert normalize_words("你好世界") == ["你好世界"]
    assert wer("你好世界", "你好 世界", normalized=True) > 0


def test_combining_marks_survive_normalization():
    assert normalize_words("مَرْحَبًا") == normalize_words("مَرْحَبًا")


def test_punctuation_only_difference_affects_strict_not_normalized():
    scores = score_against_reference("hello, world", "hello world")

    assert scores["normalized_wer"] == 0
    assert scores["strict_wer"] > 0


def test_case_only_difference_affects_strict_not_normalized():
    scores = score_against_reference("Hello World", "hello world")

    assert scores["normalized_wer"] == 0
    assert scores["strict_wer"] > 0


def test_char_for_cjk_tokenizer_splits_cjk_without_spaces():
    text = "\u4f60\u597d\u4e16\u754c"

    assert normalize_words(text, tokenizer="char_for_cjk") == ["\u4f60", "\u597d", "\u4e16", "\u754c"]
    assert wer(text, "\u4f60\u597d\u4e16", normalized=True, tokenizer="char_for_cjk") == 0.25


def test_hebrew_niqqud_survives_normalization():
    text = "\u05e9\u05b8\u05c1\u05dc\u05d5\u05b9\u05dd"

    assert normalize_words(text) == [text]


def test_thai_no_space_sentence_can_use_char_tokenizer():
    text = "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35"

    assert len(normalize_words(text, tokenizer="char_for_cjk")) > 1
