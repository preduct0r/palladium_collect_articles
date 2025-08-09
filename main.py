import argparse
import os
import re
import sqlite3
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, Generator, List, Optional, Set, Tuple

import requests
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OPENALEX_BASE_URL = "https://api.openalex.org"
DEFAULT_USER_AGENT = "collect-articles-bot/1.0 (mailto:example@example.com)"

# S3 Configuration
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY")


@dataclass
class ArticleMeta:
    doi: str
    openalex_id: Optional[str]
    title: Optional[str]
    year: Optional[int]
    is_open_access: Optional[bool]
    oa_status: Optional[str]
    best_pdf_url: Optional[str]
    cited_by_count: Optional[int]


def normalize_doi(doi: str) -> str:
    doi = doi.strip()
    doi = doi.lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    return doi


def init_s3_client() -> Optional[boto3.client]:
    """Initialize S3 client with credentials from environment variables."""
    if not all([S3_ENDPOINT_URL, S3_BUCKET_NAME, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY]):
        print("Warning: S3 credentials not fully configured. PDFs will not be uploaded to S3.")
        return None
    
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=S3_ENDPOINT_URL,
            aws_access_key_id=S3_ACCESS_KEY_ID,
            aws_secret_access_key=S3_SECRET_ACCESS_KEY
        )
        return s3_client
    except Exception as e:
        print(f"Error initializing S3 client: {e}")
        return None


def upload_file_to_s3(s3_client: boto3.client, file_path: str, s3_key: str) -> bool:
    """Upload a file to S3 bucket."""
    try:
        s3_client.upload_file(file_path, S3_BUCKET_NAME, s3_key)
        return True
    except ClientError as e:
        print(f"Error uploading {file_path} to S3: {e}")
        return False


def upload_content_to_s3(s3_client: boto3.client, content: bytes, s3_key: str) -> bool:
    """Upload content directly to S3 bucket."""
    try:
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=s3_key, Body=content)
        return True
    except ClientError as e:
        print(f"Error uploading content to S3 key {s3_key}: {e}")
        return False


def init_db(db_path: str) -> sqlite3.Connection:
    # Ensure parent directory exists for the database file
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS articles (
            doi TEXT PRIMARY KEY,
            openalex_id TEXT,
            title TEXT,
            year INTEGER,
            is_open_access INTEGER,
            oa_status TEXT,
            cited_by_count INTEGER,
            pdf_url TEXT,
            pdf_path TEXT,
            source_pdf TEXT,
            distance INTEGER,
            found_from_doi TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT
        );
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS relations (
            from_doi TEXT NOT NULL,
            to_doi TEXT NOT NULL,
            relation TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (from_doi, to_doi),
            FOREIGN KEY (from_doi) REFERENCES articles(doi) ON DELETE CASCADE,
            FOREIGN KEY (to_doi) REFERENCES articles(doi) ON DELETE CASCADE
        );
        """
    )

    conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_distance ON articles(distance);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_doi);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_relations_to ON relations(to_doi);")

    return conn


def now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def upsert_article(
    conn: sqlite3.Connection,
    meta: ArticleMeta,
    distance: Optional[int],
    found_from_doi: Optional[str],
    pdf_url: Optional[str] = None,
    pdf_path: Optional[str] = None,
    source_pdf: Optional[str] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO articles (
            doi, openalex_id, title, year, is_open_access, oa_status,
            cited_by_count, pdf_url, pdf_path, source_pdf, distance, found_from_doi, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(doi) DO UPDATE SET
            openalex_id=excluded.openalex_id,
            title=COALESCE(excluded.title, articles.title),
            year=COALESCE(excluded.year, articles.year),
            is_open_access=COALESCE(excluded.is_open_access, articles.is_open_access),
            oa_status=COALESCE(excluded.oa_status, articles.oa_status),
            cited_by_count=COALESCE(excluded.cited_by_count, articles.cited_by_count),
            pdf_url=COALESCE(excluded.pdf_url, articles.pdf_url),
            pdf_path=COALESCE(excluded.pdf_path, articles.pdf_path),
            source_pdf=COALESCE(excluded.source_pdf, articles.source_pdf),
            distance=COALESCE(articles.distance, excluded.distance),
            found_from_doi=COALESCE(articles.found_from_doi, excluded.found_from_doi),
            updated_at=excluded.updated_at
        ;
        """,
        (
            meta.doi,
            meta.openalex_id,
            meta.title,
            meta.year,
            int(meta.is_open_access) if meta.is_open_access is not None else None,
            meta.oa_status,
            meta.cited_by_count,
            pdf_url or meta.best_pdf_url,
            pdf_path,
            source_pdf,
            distance,
            found_from_doi,
            now_iso(),
            now_iso(),
        ),
    )
    conn.commit()


def insert_relation(conn: sqlite3.Connection, from_doi: str, to_doi: str, relation: str) -> None:
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO relations (from_doi, to_doi, relation) VALUES (?, ?, ?);
            """,
            (from_doi, to_doi, relation),
        )
        conn.commit()
    except sqlite3.Error:
        pass


def build_headers(mailto: Optional[str]) -> Dict[str, str]:
    ua = DEFAULT_USER_AGENT
    if mailto and "example@example.com" in ua:
        ua = f"collect-articles-bot/1.0 (mailto:{mailto})"
    return {"User-Agent": ua}


def get_openalex_work_by_doi(doi: str, mailto: Optional[str]) -> Optional[Dict]:
    norm = normalize_doi(doi)
    url = f"{OPENALEX_BASE_URL}/works/doi:{norm}"
    params = {}
    if mailto:
        params["mailto"] = mailto
    try:
        resp = requests.get(url, headers=build_headers(mailto), params=params, timeout=20)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


def extract_meta_from_openalex(work: Dict, fallback_doi: Optional[str]) -> ArticleMeta:
    def safe_get(d: Dict, path: List[str]) -> Optional[object]:
        cur = d
        try:
            for p in path:
                if cur is None:
                    return None
                cur = cur.get(p)
            return cur
        except Exception:
            return None

    doi = (work.get("doi") or fallback_doi or "").strip()
    doi = normalize_doi(doi) if doi else doi

    best_oa = work.get("best_oa_location") or {}
    best_pdf = best_oa.get("url_for_pdf") or best_oa.get("pdf_url")

    openalex_id = work.get("id")
    title = work.get("title")
    year = work.get("publication_year") or safe_get(work, ["from_publication_date"]) or None

    is_oa = None
    oa_status = None
    if "open_access" in work and isinstance(work["open_access"], dict):
        is_oa = work["open_access"].get("is_oa")
        oa_status = work["open_access"].get("oa_status")

    cited_by_count = work.get("cited_by_count")

    return ArticleMeta(
        doi=doi,
        openalex_id=openalex_id,
        title=title,
        year=int(year) if isinstance(year, int) else None,
        is_open_access=bool(is_oa) if is_oa is not None else None,
        oa_status=oa_status,
        best_pdf_url=best_pdf,
        cited_by_count=int(cited_by_count) if isinstance(cited_by_count, int) else None,
    )


def iter_citations_for_doi(
    doi: str,
    mailto: Optional[str],
    per_parent_limit: int,
    request_pause_s: float,
) -> Generator[Dict, None, None]:
    work = get_openalex_work_by_doi(doi, mailto)
    if not work:
        return
    cited_by_url = work.get("cited_by_api_url")
    if not cited_by_url:
        return

    params = {"per_page": min(200, max(25, per_parent_limit))}
    if mailto:
        params["mailto"] = mailto

    count_yielded = 0
    next_url = cited_by_url

    while next_url and count_yielded < per_parent_limit:
        try:
            resp = requests.get(next_url, headers=build_headers(mailto), params=params, timeout=30)
            if resp.status_code != 200:
                break
            data = resp.json()
            results = data.get("results", [])
            for item in results:
                if count_yielded >= per_parent_limit:
                    break
                yield item
                count_yielded += 1
            next_url = data.get("meta", {}).get("next_url")
            if next_url:
                time.sleep(request_pause_s)
        except Exception:
            break


def safe_filename_from_doi(doi: str) -> str:
    # Replace slashes and other problematic characters
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", doi)
    if not name.endswith(".pdf"):
        name = f"{name}.pdf"
    return name


def download_pdf_to_s3(url: str, s3_client: Optional[boto3.client], s3_key: str, verify_tls: bool) -> bool:
    """Download PDF and upload directly to S3."""
    if not s3_client:
        return False
    
    try:
        with requests.get(url, stream=True, timeout=60, verify=verify_tls) as r:
            if r.status_code != 200 or "pdf" not in (r.headers.get("Content-Type") or "").lower():
                # Even if content-type not pdf, try saving based on URL ending
                if not url.lower().endswith(".pdf") and r.status_code != 200:
                    return False
            
            # Collect all content in memory
            content = b''
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    content += chunk
            
            # Upload to S3
            return upload_content_to_s3(s3_client, content, s3_key)
    except Exception as e:
        print(f"Error downloading PDF from {url}: {e}")
        return False


def try_download_pdf_for_article(
    doi: str,
    direct_pdf_url: Optional[str],
    s3_client: Optional[boto3.client],
    use_scihub: bool,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    # Returns: (pdf_url_used, s3_key, source_pdf)
    filename = safe_filename_from_doi(doi)

    # Try direct OA link first
    if direct_pdf_url:
        success = download_pdf_to_s3(direct_pdf_url, s3_client, filename, verify_tls=True)
        if success:
            return (direct_pdf_url, filename, "openalex")

    if not use_scihub:
        return (None, None, None)

    # Lazy import Sci-Hub only when needed to avoid import-time errors
    SciHubSearcher = None
    try:
        from sci_hub import SciHubSearcher as _SciHubSearcher  # Import from local sci_hub.py
        SciHubSearcher = _SciHubSearcher
    except Exception:
        SciHubSearcher = None

    if SciHubSearcher is None:
        return (None, None, None)

    try:
        searcher = SciHubSearcher()
        res = searcher.search_paper_by_doi(doi)
        if res and res.get("status") == "success" and res.get("pdf_url"):
            scihub_pdf_url = res.get("pdf_url")
            success = download_pdf_to_s3(scihub_pdf_url, s3_client, filename, verify_tls=False)
            if success:
                return (scihub_pdf_url, filename, "scihub")
    except Exception:
        pass

    return (None, None, None)


def process_article(
    conn: sqlite3.Connection,
    doi: str,
    parent_doi: Optional[str],
    distance: int,
    s3_client: Optional[boto3.client],
    mailto: Optional[str],
    per_parent_limit: int,
    request_pause_s: float,
    use_scihub: bool,
) -> List[str]:
    norm_doi = normalize_doi(doi)

    # Fetch own metadata from OpenAlex
    work = get_openalex_work_by_doi(norm_doi, mailto)
    if work:
        meta = extract_meta_from_openalex(work, norm_doi)
    else:
        meta = ArticleMeta(
            doi=norm_doi,
            openalex_id=None,
            title=None,
            year=None,
            is_open_access=None,
            oa_status=None,
            best_pdf_url=None,
            cited_by_count=None,
        )

    # Try to download PDF
    pdf_url_used, s3_key, source_pdf = try_download_pdf_for_article(
        norm_doi, meta.best_pdf_url, s3_client, use_scihub
    )

    # Persist self
    upsert_article(
        conn,
        meta,
        distance=distance,
        found_from_doi=parent_doi,
        pdf_url=pdf_url_used,
        pdf_path=s3_key,  # This now contains the S3 key instead of local path
        source_pdf=source_pdf,
    )

    # Explore citations (i.e., papers that cite this one)
    children_dois: List[str] = []
    for item in iter_citations_for_doi(norm_doi, mailto, per_parent_limit, request_pause_s):
        child_meta = extract_meta_from_openalex(item, None)
        child_doi = child_meta.doi
        if not child_doi:
            continue  # skip if no DOI
        children_dois.append(child_doi)
        # Upsert minimal child row (distance assigned when processed in BFS)
        upsert_article(conn, child_meta, distance=None, found_from_doi=None)
        insert_relation(conn, from_doi=norm_doi, to_doi=child_doi, relation="cited_by")

    return children_dois


def bfs_crawl(
    seed_doi: str,
    db_path: str,
    mailto: Optional[str],
    max_depth: int,
    max_total: Optional[int],
    per_parent_limit: int,
    request_pause_s: float,
    use_scihub: bool,
) -> None:
    conn = init_db(db_path)
    s3_client = init_s3_client()

    visited: Set[str] = set()
    queue: Deque[Tuple[str, Optional[str], int]] = deque()  # (doi, parent_doi, distance)

    seed_norm = normalize_doi(seed_doi)
    queue.append((seed_norm, None, 0))

    processed = 0

    while queue:
        doi, parent, dist = queue.popleft()
        if max_total is not None and processed >= max_total:
            break
        if doi in visited:
            continue
        visited.add(doi)

        try:
            children = process_article(
                conn,
                doi,
                parent_doi=parent,
                distance=dist,
                s3_client=s3_client,
                mailto=mailto,
                per_parent_limit=per_parent_limit,
                request_pause_s=request_pause_s,
                use_scihub=use_scihub,
            )
        except Exception as e:
            print(f"Error processing {doi}: {e}")
            children = []

        processed += 1

        if dist < max_depth:
            for child in children:
                if child not in visited:
                    queue.append((child, doi, dist + 1))

        # Be nice to APIs
        time.sleep(request_pause_s)

    conn.close()


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Crawl citations (cited_by) starting from a DOI, download PDFs to S3, and store metadata.")
    p.add_argument("--doi", required=True, help="Seed DOI or URL, e.g., 10.1038/s41586-020-2649-2 or https://doi.org/...")
    p.add_argument("--db", default=os.path.abspath("./articles.db"), help="Path to SQLite database file")
    p.add_argument("--mailto", default=None, help="Email for OpenAlex polite usage header")
    p.add_argument("--max-depth", type=int, default=1, help="BFS depth (0=only seed, 1=seed + direct citers)")
    p.add_argument("--max-total", type=int, default=None, help="Max total articles to process in this run")
    p.add_argument("--per-parent-limit", type=int, default=50, help="Max citing papers to fetch per parent")
    p.add_argument("--sleep", type=float, default=1.0, help="Pause between requests (seconds)")
    p.add_argument("--no-scihub", action="store_true", help="Disable Sci-Hub fallback")
    return p.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)

    print(
        f"Starting crawl from DOI={normalize_doi(args.doi)}, depth={args.max_depth}, per_parent_limit={args.per_parent_limit}, max_total={args.max_total}"
    )
    print(f"PDFs will be saved to S3 bucket: {S3_BUCKET_NAME}")

    bfs_crawl(
        seed_doi=args.doi,
        db_path=args.db,
        mailto=args.mailto,
        max_depth=args.max_depth,
        max_total=args.max_total,
        per_parent_limit=args.per_parent_limit,
        request_pause_s=args.sleep,
        use_scihub=not args.no_scihub,
    )

    print("Done.")


if __name__ == "__main__":
    main(sys.argv[1:])
