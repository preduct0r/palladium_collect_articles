#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_doi.py — ищет DOI для каждой библиографической записи.
Вход: .txt или .docx (каждая запись — отдельной строкой; можно "как есть" из списка литературы).
Выход: CSV с колонками: idx, raw_entry, doi, source, confidence, matched_title, authors, year, url.

Зависимости: requests, python-docx (только если читаете .docx)
    pip install requests python-docx
"""

import argparse, csv, json, os, re, sys, time, unicodedata
from typing import Dict, Tuple, Optional

try:
    import docx  # python-docx
except Exception:
    docx = None

import requests

CROSSREF_API = "https://api.crossref.org/works"
OPENALEX_API = "https://api.openalex.org/works"

# ---- Heurистики и утилиты ---------------------------------------------------

DOI_RE = re.compile(r'\b10\.\d{4,9}/[^\s"<>]+', re.IGNORECASE)

def norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("ё", "е").replace("Ё", "Е")
    s = re.sub(r'\s+', ' ', s)
    return s.strip().lower()

def token_set_similarity(a: str, b: str) -> float:
    """Простая метрика похожести по множеству токенов (0..1)."""
    A = set(re.findall(r'\w+', norm(a)))
    B = set(re.findall(r'\w+', norm(b)))
    if not A or not B:
        return 0.0
    inter = len(A & B)
    denom = max(len(A), len(B))
    return inter / denom

def extract_year(text: str) -> Optional[int]:
    years = re.findall(r'(19|20|21)\d{2}', text)
    if years:
        try:
            return int(years[0])
        except:
            return None
    return None

def guess_title_for_query(raw: str) -> str:
    """Пытается выделить заголовок для поискового запроса — но если сомнительно, шлём всю строку."""
    parts = re.split(r'//|—|–|-{2,}|::', raw)
    # Возьмём наиболее длинный фрагмент как "похоже на заголовок"
    cand = max(parts, key=len).strip()
    # Если слишком коротко/подозрительно — используем исходную строку
    return cand if len(cand) >= 10 else raw

# ---- Чтение входа ------------------------------------------------------------

def read_entries(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".docx":
        if docx is None:
            print("Ошибка: для чтения .docx установите пакет python-docx", file=sys.stderr)
            sys.exit(2)
        d = docx.Document(path)
        # Берём непустые абзацы построчно
        for p in d.paragraphs:
            line = p.text.strip()
            if line:
                yield line
    else:
        # .txt или что-то текстовое
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line

# ---- Поиск в Crossref и OpenAlex --------------------------------------------

def crossref_search(raw: str, mailto: Optional[str] = None) -> Optional[Dict]:
    params = {
        "query.bibliographic": raw,
        "rows": 5,
        "select": "DOI,title,author,issued,URL"
    }
    headers = {"User-Agent": f"doi-finder/1.0 (+https://example.org)"}
    if mailto:
        params["mailto"] = mailto
    try:
        r = requests.get(CROSSREF_API, params=params, headers=headers, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:
        return None
    items = data.get("message", {}).get("items", []) or []
    if not items:
        return None

    # Выбираем лучший результат по заголовку
    q = guess_title_for_query(raw)
    best, best_score = None, 0.0
    for it in items:
        title = " ".join(it.get("title") or [])
        score = token_set_similarity(q, title)
        # лёгкий бонус за совпадение года, если нашёлся
        year_q = extract_year(raw)
        year_hit = None
        try:
            year_hit = it.get("issued", {}).get("date-parts", [[None]])[0][0]
        except Exception:
            year_hit = None
        if year_q and year_hit and abs(year_q - year_hit) <= 1:
            score += 0.08
        if score > best_score:
            best_score = score
            best = it

    if best and "DOI" in best:
        return {
            "doi": best.get("DOI"),
            "title": " ".join(best.get("title") or []) or "",
            "authors": ", ".join(
                filter(None, [f"{a.get('family','')}" for a in (best.get("author") or [])])
            ),
            "year": (best.get("issued", {}).get("date-parts", [[None]])[0][0] if best.get("issued") else None),
            "url": best.get("URL"),
            "confidence": round(min(1.0, best_score) * 100),
            "source": "crossref"
        }
    return None

def openalex_search(raw: str) -> Optional[Dict]:
    params = {
        "search": raw,
        "per-page": 5,
        "select": "doi,title,authorships,publication_year,primary_location"
    }
    try:
        r = requests.get(OPENALEX_API, params=params, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:
        return None
    results = data.get("results", []) or []
    if not results:
        return None

    q = guess_title_for_query(raw)
    best, best_score = None, 0.0
    for it in results:
        title = it.get("title") or ""
        score = token_set_similarity(q, title)
        yq = extract_year(raw)
        yh = it.get("publication_year")
        if yq and yh and abs(yq - (yh or 0)) <= 1:
            score += 0.08
        if score > best_score:
            best_score = score
            best = it

    if best and best.get("doi"):
        authors = []
        for a in (best.get("authorships") or []):
            if a.get("author", {}).get("display_name"):
                authors.append(a["author"]["display_name"])
        url = None
        loc = best.get("primary_location") or {}
        if isinstance(loc, dict):
            url = (loc.get("landing_page_url")
                   or (loc.get("source") or {}).get("host_organization_lineage", [None])[-1])
        return {
            "doi": best.get("doi"),
            "title": best.get("title") or "",
            "authors": ", ".join(authors),
            "year": best.get("publication_year"),
            "url": url,
            "confidence": round(min(1.0, best_score) * 100),
            "source": "openalex"
        }
    return None

# ---- Основная логика ---------------------------------------------------------

def process_entries(entries, email: Optional[str] = None, pause: float = 0.4):
    for i, raw in enumerate(entries, 1):
        raw_clean = norm(raw)
        # 1) Уже указан DOI?
        m = DOI_RE.search(raw)
        if m:
            doi = m.group(0).rstrip(").,;")
            yield {
                "idx": i,
                "raw_entry": raw,
                "doi": doi,
                "source": "in-text",
                "confidence": 100,
                "matched_title": "",
                "authors": "",
                "year": "",
                "url": f"https://doi.org/{doi}"
            }
            continue

        # 2) Crossref
        cr = crossref_search(raw, mailto=email)
        if cr:
            yield {
                "idx": i,
                "raw_entry": raw,
                "doi": cr["doi"],
                "source": cr["source"],
                "confidence": cr["confidence"],
                "matched_title": cr["title"],
                "authors": cr["authors"],
                "year": cr["year"] or "",
                "url": cr["url"] or f"https://doi.org/{cr['doi']}",
            }
            time.sleep(pause)
            continue

        # 3) OpenAlex
        oa = openalex_search(raw)
        if oa:
            yield {
                "idx": i,
                "raw_entry": raw,
                "doi": oa["doi"],
                "source": oa["source"],
                "confidence": oa["confidence"],
                "matched_title": oa["title"],
                "authors": oa["authors"],
                "year": oa["year"] or "",
                "url": oa["url"] or f"https://doi.org/{oa['doi']}",
            }
            time.sleep(pause)
            continue

        # 4) Не нашли
        yield {
            "idx": i,
            "raw_entry": raw,
            "doi": "",
            "source": "not_found",
            "confidence": 0,
            "matched_title": "",
            "authors": "",
            "year": "",
            "url": ""
        }
        time.sleep(pause)

def main():
    ap = argparse.ArgumentParser(description="Найти DOI для списка источников")
    ap.add_argument("input", help="Путь к .txt или .docx с одной записью на строку")
    ap.add_argument("-o", "--output", default="dois.csv", help="CSV результат (по умолчанию dois.csv)")
    ap.add_argument("--mailto", default=None, help="Ваш email для параметра mailto в Crossref (желательно)")
    ap.add_argument("--pause", type=float, default=0.4, help="Пауза между запросами (сек.)")
    args = ap.parse_args()

    entries = list(read_entries(args.input))
    if not entries:
        print("Не нашёл ни одной записи во входном файле.", file=sys.stderr)
        sys.exit(1)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["idx","raw_entry","doi","source","confidence","matched_title","authors","year","url"]
        )
        writer.writeheader()
        for row in process_entries(entries, email=args.mailto, pause=args.pause):
            writer.writerow(row)

    print(f"Готово. Результат: {args.output} (обработано записей: {len(entries)})")

if __name__ == "__main__":
    main()