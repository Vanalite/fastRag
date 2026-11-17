import pytest

from fastrag.text.values import (
    default_gazetteer,
    normalize_ward,
    parse_area,
    parse_bedrooms,
    parse_direction,
    parse_number,
    parse_price,
)


@pytest.mark.parametrize(
    ("text", "vnd"),
    [
        ("3,5 tỷ", 3_500_000_000),
        ("3.5 tỷ", 3_500_000_000),
        ("3 tỷ 5", 3_500_000_000),
        ("3ty5", 3_500_000_000),
        ("3 tỷ 200", 3_200_000_000),
        ("3 tỷ 200 triệu", 3_200_000_000),
        ("2 tỷ 75", 2_750_000_000),
        ("850tr", 850_000_000),
        ("850 triệu", 850_000_000),
        ("850 trieu", 850_000_000),
        ("1,2 tỷ VND", 1_200_000_000),
        ("3.500.000.000 đ", 3_500_000_000),
        ("3,500,000,000", 3_500_000_000),
        (3_500_000_000, 3_500_000_000),
        ("2.5 billion", 2_500_000_000),
    ],
)
def test_parse_price(text, vnd):
    assert parse_price(text).vnd == vnd


def test_price_per_m2():
    p = parse_price("35 triệu/m2")
    assert p.vnd == 35_000_000 and p.per_m2


@pytest.mark.parametrize("bad", ["thỏa thuận", "liên hệ", "12", 12, "", "rẻ", "khoảng vài tỷ"])
def test_price_rejects_instead_of_guessing(bad):
    with pytest.raises(ValueError):
        parse_price(bad)


@pytest.mark.parametrize(
    ("text", "m2"),
    [
        ("80m2", 80),
        ("80,5 m²", 80.5),
        ("5x20", 100),
        ("5 x 20m", 100),
        ("1 ha", 10000),
        ("1.000 m2", 1000),
        ("72 mét vuông", 72),
        (65, 65),
    ],
)
def test_parse_area(text, m2):
    assert parse_area(text) == m2


def test_area_rejects_garbage():
    with pytest.raises(ValueError):
        parse_area("rộng rãi thoáng mát")


def test_numbers():
    assert parse_number("3,5") == 3.5
    assert parse_number("1.200") == 1200
    assert parse_number("1.234,5") == 1234.5


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("Q7", "hcm-q7"),
        ("Q.7", "hcm-q7"),
        ("Quận 7", "hcm-q7"),
        ("quan 7", "hcm-q7"),
        ("District 7", "hcm-q7"),
        ("Quận 7, TP.HCM", "hcm-q7"),
        ("Bình Thạnh", "hcm-binh-thanh"),
        ("binh thanh", "hcm-binh-thanh"),
        ("Thủ Đức", "hcm-thu-duc"),
        ("TP Thủ Đức", "hcm-thu-duc"),
        ("Q2", "hcm-q2"),
        ("Cầu Giấy, Hà Nội", "hn-cau-giay"),
        ("Huyện Nhà Bè", "hcm-nha-be"),
    ],
)
def test_districts(text, code):
    assert default_gazetteer().district(text).code == code


def test_unknown_district_rejected():
    with pytest.raises(ValueError):
        default_gazetteer().district("Quận 99")


def test_thu_duc_includes_former_districts():
    assert set(default_gazetteer().descendants("hcm-thu-duc")) == {"hcm-thu-duc", "hcm-q2", "hcm-q9"}


def test_find_districts_in_text():
    found = default_gazetteer().find_in_text("Căn hộ Quận 7, gần cầu sang Bình Thạnh")
    assert found == ["hcm-q7", "hcm-binh-thanh"] or set(found) == {"hcm-q7", "hcm-binh-thanh"}


def test_tenant_aliases():
    g = default_gazetteer().with_aliases({"hcm-q7": ["Phú Mỹ Hưng"]})
    assert g.district("phu my hung").code == "hcm-q7"
    with pytest.raises(ValueError):
        default_gazetteer().with_aliases({"nowhere": ["x"]})


def test_wards():
    assert normalize_ward("P.5") == "phuong-5"
    assert normalize_ward("Phường 05") == "phuong-5"
    assert normalize_ward("Ward 5") == "phuong-5"
    assert normalize_ward("P. Thảo Điền") == "phuong-thao-dien"
    assert normalize_ward("Xã Tân Kiên") == "xa-tan-kien"


def test_bedrooms_and_direction():
    assert parse_bedrooms("2PN") == 2
    assert parse_bedrooms("3 phòng ngủ") == 3
    assert parse_bedrooms("studio") == 0
    assert parse_direction("Đông Nam") == "SE"
    assert parse_direction("hướng ĐN") == "SE"
    assert parse_direction("north-west") == "NW"
    with pytest.raises(ValueError):
        parse_direction("đẹp")
