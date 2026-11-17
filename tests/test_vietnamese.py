import unicodedata

from fastrag.text.vietnamese import (
    detect_legacy_encoding,
    keyword_terms,
    normalize_query,
    normalize_text,
    normalize_tone_placement,
    segment,
    strip_diacritics,
)


def test_nfc_and_whitespace():
    decomposed = unicodedata.normalize("NFD", "Căn hộ  cao   cấp")
    text, report = normalize_text(decomposed)
    assert text == "Căn hộ cao cấp"
    assert unicodedata.is_normalized("NFC", text)
    assert report.encoding == "unicode"


def test_tcvn3_is_converted():
    text, report = normalize_text("Hîp ®ång mua b¸n c¨n hé chung c")
    assert report.encoding == "tcvn3" and report.converted
    assert text.startswith("Hợp đồng mua bán căn hộ")


def test_unicode_vietnamese_is_not_misdetected():
    assert detect_legacy_encoding("Chính sách thanh toán: đặt cọc 50 triệu, ký HĐMB trong 7 ngày.") == "unicode"
    assert detect_legacy_encoding("Plain English text about the apartment.") == "unicode"


def test_tone_placement_canonical():
    assert normalize_tone_placement("hoà") == "hòa"
    assert normalize_tone_placement("thuỷ") == "thủy"
    assert normalize_tone_placement("khoẻ") == "khỏe"
    # Unchanged when a final consonant follows, or after "qu".
    assert normalize_tone_placement("hoàng") == "hoàng"
    assert normalize_tone_placement("quý") == "quý"
    assert normalize_query("Hoà Bình") == normalize_query("Hòa Bình")


def test_strip_diacritics():
    assert strip_diacritics("Đường Nguyễn Hữu Thọ, Quận 7") == "Duong Nguyen Huu Tho, Quan 7"


def test_segment_joins_compound_words():
    words = segment("Căn hộ chung cư gần trung tâm thương mại")
    assert "chung_cư" in [w.lower() for w in words]


def test_keyword_terms_cover_accentless_and_syllables():
    t = keyword_terms("Căn hộ chung cư").freqs
    assert "chung_cư" in t and "chung" in t and "cư" in t
    assert "~chung_cu" in t and "~chung" in t and "~cu" in t
    # Questions typed without diacritics share terms with the document.
    q = keyword_terms("can ho chung cu").freqs
    assert set(q) & {k for k in t if k.startswith("~")}


def test_stopwords_dropped_but_negation_kept():
    t = keyword_terms("Căn hộ này không có chỗ đậu xe").freqs
    assert "này" not in t and "không" in t


def test_query_words_drop_question_particles():
    from fastrag.text.vietnamese import query_words, word_in_terms

    assert query_words("Phí quản lý bao nhiêu?") == ["phí", "quản_lý"]
    assert query_words("Có được nuôi chó không?") == ["nuôi", "chó"]
    assert "không" in query_words("Căn hộ không có ban công")
    terms = set(keyword_terms("Đặt cọc 50 triệu").freqs)
    assert word_in_terms("đặt_cọc", terms) and word_in_terms("dat", terms) and word_in_terms("dat_coc", terms)


def test_query_words_drop_split_question_phrases():
    from fastrag.text.vietnamese import query_words

    assert query_words("dat coc bao nhieu") == ["dat", "coc"] or query_words("dat coc bao nhieu") == ["dat_coc"]
