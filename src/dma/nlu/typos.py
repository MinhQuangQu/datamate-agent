"""Telex / VNI leftovers in informal typing ("thangs", "hangf", "cuwar", "trowr leen", "tieenf", "naof").

A folded token that is NOT a known word is repaired when removing Telex artefacts (tone keys s/f/r/x/j,
doubled vowels aa/ee/oo/dd, the w modifier) yields a known word. Known words = a small core lexicon of
question words + every token of the rulebook, the shop README and the value index (entity names), so
product names are never "corrected" into something else.
"""
from __future__ import annotations

import re

CORE = """
thang tuan ngay nam quy hom nay qua truoc rui roi sau toi ke tu den het dau cuoi gan nhat lien do kia nua vua
doanh thu so ban duoc bao nhieu tien don hang mat san pham mon khach cua hang shop quan tiem nao gi la co khong
chay hot top liet ke nhung cac moi tung trung binh tong cong ty le phan tram ti trong tang giam hon kem chenh lech
huy tra no con ton kho nhap gia von lai loi lo nhan vien tho ca sang chieu toi gio thu chu nhat le tet dip mua
duoc nhieu it nhat cao thap lon nho dat re xem giup coi gium cho hoi minh voi va hay hoac ma thi khi nao bang
mot hai ba bon tu nam sau bay tam chin muoi tram nghin trieu ty dong ly chai lon thung loc hop bao goi vien vi
kg cai chiec cay tam met size dong xe phieu lan luot bill hoa thuoc lo han dung ke tien mat chuyen khoan the
kenh online tai cho giao ship phi voucher diem nhom loai dich vu sua thay nhot dang dang giao hoan thanh chua
tro len xuong ve ra vao duoi tren trong ngoai bi bo lay mua dem tinh ghi viet biet hien con vay nhi a ha
""".split()
_TONE = "sfrxj"


def build_lexicon(*texts: str, extra_tokens: set[str] | None = None) -> set[str]:
    from dma.textnorm import fold
    lex = set(CORE)
    for t in texts:
        if t:
            lex.update(fold(t).split())
    if extra_tokens:
        lex.update(extra_tokens)
    return lex


def _variants(tok: str) -> list[str]:
    out = []
    t = tok
    if len(t) > 2 and t[-1] in _TONE:
        out.append(t[:-1])
    base = re.sub(r"([aeiouy])w", r"\1", t)                         # cuwa -> cua, thuwr -> thur
    base = re.sub(r"(aa|ee|oo|dd)", lambda m: m.group(1)[0], base)   # leen -> len, ddi -> di
    if base != t:
        out.append(base)
        if len(base) > 2 and base[-1] in _TONE:
            out.append(base[:-1])
    m = re.match(r"^([a-z]+?)([sfrxj])([a-z]+)$", t)                 # tone key typed mid-word: "trowr" handled above, "hafng"
    if m and len(m.group(1)) >= 1:
        out.append(m.group(1) + m.group(3))
    return out


def fix_telex(folded: str, lexicon: set[str]) -> tuple[str, list[str]]:
    toks = folded.split()
    fixed = []
    changes = []
    for t in toks:
        if t in lexicon or t.isdigit() or any(ch.isdigit() for ch in t):
            fixed.append(t)
            continue
        rep = next((v for v in _variants(t) if v in lexicon), None)
        if rep:
            fixed.append(rep)
            changes.append(f"{t}->{rep}")
        else:
            fixed.append(t)
    return " ".join(fixed), changes


def fix_telex_text(text: str, lexicon: set[str]) -> tuple[str, list[str]]:
    """Same repair on accent-free text that KEEPS punctuation (the time resolver needs "3/2025")."""
    changes: list[str] = []

    def rep(m: re.Match) -> str:
        t = m.group(0)
        low = t.lower()
        if low in lexicon or any(ch.isdigit() for ch in low):
            return t
        v = next((v for v in _variants(low) if v in lexicon), None)
        if v:
            changes.append(f"{low}->{v}")
            return v
        return t
    return re.sub(r"[A-Za-z]+", rep, text), changes


_MARKS = (("dd", "đ"), ("uow", "ươ"), ("aa", "â"), ("ee", "ê"), ("oo", "ô"), ("aw", "ă"), ("ow", "ơ"), ("uw", "ư"))
_TONED = {"s": "áắấéếíóốớúứý", "f": "àằầèềìòồờùừỳ", "r": "ảẳẩẻểỉỏổởủửỷ", "x": "ãẵẫẽễĩõỗỡũữỹ", "j": "ạặậẹệịọộợụựỵ"}
_BASE = "aăâeêioôơuưy"
_VOWEL = re.compile(f"[{_BASE}]+")


def telex_to_vi(tok: str) -> str | None:
    """'soos' -> 'số', 'ddown' -> 'đơn', 'cos' -> 'có'. None when the token has no Telex keys."""
    t = tok.lower()
    tone = None
    if len(t) > 2 and t[-1] in _TONE:
        t, tone = t[:-1], t[-1]
    for a, b in _MARKS:
        t = t.replace(a, b)
    if tone is None and t == tok.lower():
        return None
    if tone:
        m = None
        for m in _VOWEL.finditer(t):        # the tone sits in the last vowel cluster
            pass
        if m is None:
            return None
        s, e = m.span()
        if t[max(0, s - 1):s + 1] in ("gi", "qu") and e - s > 1:
            s += 1                           # gi-/qu- glide is not the syllable nucleus
        cl = t[s:e]
        marked = [i for i, ch in enumerate(cl) if ch in "ăâêôơư"]
        if marked:
            i = marked[-1]
        elif len(cl) == 1 or e < len(t):
            i = len(cl) - 1 if e < len(t) else 0
        else:
            i = 0 if len(cl) == 2 else 1
        ch = cl[i]
        cl = cl[:i] + _TONED[tone][_BASE.index(ch)] + cl[i + 1:]
        t = t[:s] + cl + t[e:]
    return t


def retype_telex(text: str, fixes: list[str]) -> str:
    """Rewrite the tokens fix_telex_text repaired ('soos->so') in Vietnamese, so prompts read 'số', not 'soos'."""
    from dma.textnorm import fold
    want = dict(f.split("->", 1) for f in fixes)

    def rep(m: re.Match) -> str:
        low = m.group(0).lower()
        if low not in want:
            return m.group(0)
        vi = telex_to_vi(low)
        return vi if vi and fold(vi) == want[low] else m.group(0)
    return re.sub(r"[A-Za-z]+", rep, text)
