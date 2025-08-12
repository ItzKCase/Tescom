import argparse
import csv
import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd


# -----------------------------
# Logging
# -----------------------------


class OneLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        return re.sub(r"\s+", " ", message).strip()


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    fmt = "%(levelname)s | %(message)s"
    handler.setFormatter(OneLineFormatter(fmt))
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)


# -----------------------------
# Constants and normalization
# -----------------------------


ALIAS_MANUFACTURER_FILENAME = "manufacturer_alias.csv"
ALIAS_MODEL_FILENAME = "model_alias.csv"

# Expanded, conservative manufacturer alias seeds
SEED_MANUFACTURER_ALIASES = [
    # Keysight / Agilent / HP family
    {"canonical": "KEYSIGHT", "alias": "AGILENT"},
    {"canonical": "KEYSIGHT", "alias": "AGILENT TECHNOLOGIES"},
    {"canonical": "KEYSIGHT", "alias": "HEWLETT PACKARD"},
    {"canonical": "KEYSIGHT", "alias": "HEWLETT-PACKARD"},
    {"canonical": "KEYSIGHT", "alias": "H-P"},
    {"canonical": "KEYSIGHT", "alias": "H P"},
    {"canonical": "KEYSIGHT", "alias": "HP"},
    {"canonical": "KEYSIGHT", "alias": "HP/AGILENT"},
    {"canonical": "KEYSIGHT", "alias": "HP-AGILENT"},

    # Tektronix / Keithley
    {"canonical": "TEKTRONIX", "alias": "TEK"},
    {"canonical": "KEITHLEY", "alias": "TEKTRONIX KEITHLEY"},
    {"canonical": "KEITHLEY", "alias": "KEITHLEY INSTRUMENTS"},

    # Rohde & Schwarz
    {"canonical": "ROHDE & SCHWARZ", "alias": "R&S"},
    {"canonical": "ROHDE & SCHWARZ", "alias": "ROHDE AND SCHWARZ"},
    {"canonical": "ROHDE & SCHWARZ", "alias": "ROHDE+SCHWARZ"},
    {"canonical": "ROHDE & SCHWARZ", "alias": "R AND S"},

    # Teledyne LeCroy
    {"canonical": "TELEDYNE LECROY", "alias": "LECROY"},
    {"canonical": "TELEDYNE LECROY", "alias": "TELEDYNE-LECROY"},

    # B&K Precision variants
    {"canonical": "BK PRECISION", "alias": "B&K"},
    {"canonical": "BK PRECISION", "alias": "B & K"},
    {"canonical": "BK PRECISION", "alias": "BK"},
    {"canonical": "BK PRECISION", "alias": "B&K PRECISION"},
    {"canonical": "BK PRECISION", "alias": "B K PRECISION"},

    # National Instruments
    {"canonical": "NATIONAL INSTRUMENTS", "alias": "NI"},
    {"canonical": "NATIONAL INSTRUMENTS", "alias": "NATIONAL INSTRUMENTS (NI)"},

    # Anritsu / Wiltron
    {"canonical": "ANRITSU", "alias": "WILTRON"},

    # Viavi / JDSU / IFR / Marconi / Aeroflex family
    {"canonical": "VIAVI SOLUTIONS", "alias": "JDSU"},
    {"canonical": "VIAVI SOLUTIONS", "alias": "JDS UNIPHASE"},
    {"canonical": "VIAVI SOLUTIONS", "alias": "IFR"},
    {"canonical": "VIAVI SOLUTIONS", "alias": "IFR SYSTEMS"},
    {"canonical": "VIAVI SOLUTIONS", "alias": "MARCONI"},
    {"canonical": "VIAVI SOLUTIONS", "alias": "MARCONI INSTRUMENTS"},
    {"canonical": "VIAVI SOLUTIONS", "alias": "AEROFLEX"},
    {"canonical": "VIAVI SOLUTIONS", "alias": "COBHAM"},

    # Fluke families
    {"canonical": "FLUKE", "alias": "FLUKE CORPORATION"},
    {"canonical": "FLUKE CALIBRATION", "alias": "RUSKA"},
    {"canonical": "FLUKE CALIBRATION", "alias": "HART SCIENTIFIC"},
    {"canonical": "FLUKE CALIBRATION", "alias": "DH INSTRUMENTS"},
    {"canonical": "FLUKE CALIBRATION", "alias": "DHI"},
    {"canonical": "FLUKE BIOMEDICAL", "alias": "RAYSAFE"},
    {"canonical": "FLUKE BIOMEDICAL", "alias": "UNFORS RAYSAFE"},
    {"canonical": "FLUKE BIOMEDICAL", "alias": "METRON"},

    # Druck family
    {"canonical": "DRUCK", "alias": "GE DRUCK"},
    {"canonical": "DRUCK", "alias": "GE SENSING"},
    {"canonical": "DRUCK", "alias": "Baker Hughes Druck"},
    {"canonical": "DRUCK", "alias": "BHGE DRUCK"},
    {"canonical": "DRUCK", "alias": "DRUCK LTD"},

    # GW Instek / Good Will
    {"canonical": "GW INSTEK", "alias": "GOOD WILL INSTEK"},
    {"canonical": "GW INSTEK", "alias": "GOOD WILL INSTRUMENT"},
    {"canonical": "GW INSTEK", "alias": "INSTEK"},

    # Aim-TTi / Thurlby Thandar
    {"canonical": "AIM-TTI", "alias": "AIM TTI"},
    {"canonical": "AIM-TTI", "alias": "THURLBY THANDAR"},
    {"canonical": "AIM-TTI", "alias": "TTI"},

    # Chroma
    {"canonical": "CHROMA", "alias": "CHROMA ATE"},

    # Yokogawa
    {"canonical": "YOKOGAWA", "alias": "YOKOGAWA ELECTRIC"},
    {"canonical": "YOKOGAWA", "alias": "YOKOGAWA T&M"},

    # Rigol / Siglent / Pico / Hioki
    {"canonical": "RIGOL", "alias": "RIGOL TECHNOLOGIES"},
    {"canonical": "SIGLENT", "alias": "SIGLENT TECHNOLOGIES"},
    {"canonical": "PICO TECHNOLOGY", "alias": "PICO"},
    {"canonical": "HIOKI", "alias": "HIOKI E.E. CORPORATION"},

    # Stanford Research Systems
    {"canonical": "STANFORD RESEARCH SYSTEMS", "alias": "SRS"},

    # Megger family
    {"canonical": "MEGGER", "alias": "AVO"},
    {"canonical": "MEGGER", "alias": "BIDDLE"},

    # Omega
    {"canonical": "OMEGA ENGINEERING", "alias": "OMEGA"},
    {"canonical": "OMEGA ENGINEERING", "alias": "OMEGA ENG"},

    # TDK-Lambda / Lambda
    {"canonical": "TDK-LAMBDA", "alias": "LAMBDA"},
    {"canonical": "TDK-LAMBDA", "alias": "TDK LAMBDA"},

    # Kepco
    {"canonical": "KEPCO", "alias": "KEPCO INC"},
    {"canonical": "KEPCO", "alias": "KEPCO POWER SUPPLIES"},

    # WIKA / Mensor (keep MENSOR canonical if used that way)
    {"canonical": "MENSOR", "alias": "WIKA MENSOR"},
    {"canonical": "WIKA", "alias": "WIKA INSTRUMENT"},

    # Teledyne (umbrella, sometimes appears on labels)
    {"canonical": "TELEDYNE", "alias": "TELEDYNE TECHNOLOGIES"},
]

SEED_MODEL_ALIASES = [
    # DMMs / DAQs (Keysight/Agilent/HP family)
    {"canonical_key": "KEYSIGHT::34401A", "alias_model": "HP34401A"},
    {"canonical_key": "KEYSIGHT::34401A", "alias_model": "AGILENT34401A"},
    {"canonical_key": "KEYSIGHT::34401A", "alias_model": "34401"},

    {"canonical_key": "KEYSIGHT::34410A", "alias_model": "AGILENT34410A"},
    {"canonical_key": "KEYSIGHT::34461A", "alias_model": "AGILENT34461A"},
    {"canonical_key": "KEYSIGHT::34970A", "alias_model": "HP34970A"},
    {"canonical_key": "KEYSIGHT::34970A", "alias_model": "AGILENT34970A"},
    {"canonical_key": "KEYSIGHT::34972A", "alias_model": "AGILENT34972A"},
    {"canonical_key": "KEYSIGHT::E3631A", "alias_model": "AGILENTE3631A"},
    {"canonical_key": "KEYSIGHT::E3631A", "alias_model": "E 3631 A"},

    # Tektronix scopes
    {"canonical_key": "TEKTRONIX::MSO2024B", "alias_model": "MSO 2024B"},
    {"canonical_key": "TEKTRONIX::DPO2024B", "alias_model": "DPO 2024B"},

    # Fluke
    {"canonical_key": "FLUKE::87V", "alias_model": "87-V"},
    {"canonical_key": "FLUKE::8846A", "alias_model": "8846 A"},

    # NI
    {"canonical_key": "NATIONAL INSTRUMENTS::USB-6211", "alias_model": "USB 6211"},
    {"canonical_key": "NATIONAL INSTRUMENTS::USB-6211", "alias_model": "NI USB-6211"},

    # Druck / Mensor
    {"canonical_key": "DRUCK::DPI610", "alias_model": "DPI-610"},
    {"canonical_key": "MENSOR::CPC6000", "alias_model": "CPC 6000"},

    # R&S / Rigol / Siglent
    {"canonical_key": "ROHDE & SCHWARZ::FSV7", "alias_model": "R&S FSV7"},
    {"canonical_key": "RIGOL::DS1054Z", "alias_model": "DS 1054Z"},
    {"canonical_key": "SIGLENT::SDS1104X-E", "alias_model": "SDS1104XE"},
]



def collapse_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def normalize_manufacturer(value: str) -> str:
    if value is None:
        return ""
    value = collapse_spaces(str(value))
    return value.upper()


def normalize_model(value: str) -> str:
    if value is None:
        return ""
    value = str(value).strip().upper()
    # remove trivial separators: spaces, hyphens, underscores
    value = re.sub(r"[\s\-_]+", "", value)
    return value


def to_int_accredited(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        value_upper = value.upper().strip()
        if value_upper == "YES":
            return 1
        elif value_upper == "NO":
            return 0
    return 0  # Default to 0 for any other value


def to_int_flag(value: Any) -> int:
    """Generic boolean-to-int converter for spreadsheet flags."""
    return to_int_accredited(value)


def parse_price(value: Any) -> Optional[float]:
    """Parse a price field from Excel to float (REAL). Returns None if not parseable."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # Treat negative or zero as valid values; return float
        return float(value)
    s = str(value).strip()
    if s == "":
        return None
    s_lower = s.lower()
    if s_lower in {"na", "n/a", "none", "null", "-", "tbd", "unknown"}:
        return None
    # Remove currency symbols and thousand separators
    s_clean = re.sub(r"[,$]", "", s)
    try:
        return float(s_clean)
    except Exception:
        return None
    if isinstance(value, (int, bool)):
        return 1 if bool(value) else 0
    s = str(value).strip().lower()
    truthy = {"yes", "y", "true", "t", "1", "accredited"}
    falsy = {"no", "n", "false", "f", "0", "not accredited", "na", "null", "none"}
    if s in truthy:
        return 1
    if s in falsy or s == "":
        return 0
    # fallback: try numeric
    try:
        return 1 if float(s) > 0 else 0
    except Exception:
        return 0


# -----------------------------
# Aliases loading and seeding
# -----------------------------


@dataclass
class Aliases:
    manufacturer_alias_to_canonical: Dict[str, str]
    # Mapping: manufacturer_norm -> { alias_model_norm -> canonical_model_norm }
    model_alias_by_manufacturer: Dict[str, Dict[str, str]]


def ensure_alias_files(alias_dir: Path) -> None:
    alias_dir.mkdir(parents=True, exist_ok=True)
    manu_path = alias_dir / ALIAS_MANUFACTURER_FILENAME
    model_path = alias_dir / ALIAS_MODEL_FILENAME

    if not manu_path.exists():
        with manu_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["canonical", "alias"])
            writer.writeheader()
            for row in SEED_MANUFACTURER_ALIASES:
                writer.writerow(row)

    if not model_path.exists():
        with model_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["canonical_key", "alias_model"])
            writer.writeheader()
            for row in SEED_MODEL_ALIASES:
                writer.writerow(row)


def load_aliases(alias_dir: Path) -> Aliases:
    ensure_alias_files(alias_dir)

    manu_map: Dict[str, str] = {}
    manu_csv = alias_dir / ALIAS_MANUFACTURER_FILENAME
    df_manu = pd.read_csv(manu_csv)
    for _, r in df_manu.iterrows():
        canonical = normalize_manufacturer(r.get("canonical", ""))
        alias = normalize_manufacturer(r.get("alias", ""))
        if alias:
            manu_map[alias] = canonical
        if canonical and canonical not in manu_map:
            # canonical maps to itself for convenience
            manu_map[canonical] = canonical

    # Model aliases grouped by canonical manufacturer
    model_map: Dict[str, Dict[str, str]] = {}
    model_csv = alias_dir / ALIAS_MODEL_FILENAME
    df_model = pd.read_csv(model_csv)
    for _, r in df_model.iterrows():
        canonical_key = str(r.get("canonical_key", "")).strip()
        alias_model = normalize_model(r.get("alias_model", ""))
        if not canonical_key:
            continue
        if "::" not in canonical_key:
            continue
        manu_part, canonical_model = canonical_key.split("::", 1)
        manu_canon = normalize_manufacturer(manu_part)
        canonical_model_norm = normalize_model(canonical_model)
        if manu_canon not in model_map:
            model_map[manu_canon] = {}
        if alias_model:
            model_map[manu_canon][alias_model] = canonical_model_norm
        # also allow canonical to map to itself
        model_map[manu_canon][canonical_model_norm] = canonical_model_norm

    return Aliases(manu_map, model_map)


def apply_manufacturer_alias(name: str, alias: Aliases) -> str:
    norm = normalize_manufacturer(name)
    return alias.manufacturer_alias_to_canonical.get(norm, norm)


def apply_model_alias(manufacturer_norm: str, model: str, alias: Aliases) -> str:
    model_norm = normalize_model(model)
    mapping = alias.model_alias_by_manufacturer.get(manufacturer_norm, {})
    return mapping.get(model_norm, model_norm)


# -----------------------------
# Excel ingestion and cleaning
# -----------------------------


def find_column(df: pd.DataFrame, target: str) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    return cols.get(target.lower())


def ingest_excel(excel_path: Path, sample: int = 0) -> pd.DataFrame:
    df = pd.read_excel(excel_path)
    if sample and sample > 0:
        df = df.head(sample)

    # Standardize column names (case-insensitive fetch)
    col_manufacturer = find_column(df, "MANUFACTURER")
    col_model = find_column(df, "MODEL")
    col_description = find_column(df, "DESCRIPTION")
    col_accredited = find_column(df, "Accredited") or find_column(df, "Accredited?")
    # Optional fields
    col_price_l1 = (
        find_column(df, "Price Level 1")
        or find_column(df, "Price L1")
        or find_column(df, "Price1")
        or find_column(df, "L1 Price")
    )
    col_price_l2 = (
        find_column(df, "Price Level 2")
        or find_column(df, "Price L2")
        or find_column(df, "Price2")
        or find_column(df, "L2 Price")
    )
    col_price_l3 = (
        find_column(df, "Price Level 3")
        or find_column(df, "Price L3")
        or find_column(df, "Price3")
        or find_column(df, "L3 Price")
    )
    col_level4_only = (
        find_column(df, "Level 4 only")
        or find_column(df, "Level4 only")
        or find_column(df, "Level4 Only")
        or find_column(df, "Level 4 Only")
        or find_column(df, "Level 4 Only?")
        or find_column(df, "L4 Only")
    )
    col_comments = find_column(df, "Comments") or find_column(df, "Comment")

    required = [col_manufacturer, col_model, col_description, col_accredited]
    if any(c is None for c in required):
        missing = [n for n, c in zip(["MANUFACTURER", "MODEL", "DESCRIPTION", "Accredited"], required) if c is None]
        raise ValueError(f"Missing required columns in Excel: {', '.join(missing)}")

    # Rename to standard internal names
    df = df.rename(columns={
        col_manufacturer: "MANUFACTURER",
        col_model: "MODEL",
        col_description: "DESCRIPTION",
        col_accredited: "Accredited",
    })

    # Bring optional columns under standard internal names if they exist
    if col_price_l1:
        df = df.rename(columns={col_price_l1: "Price_L1"})
    if col_price_l2:
        df = df.rename(columns={col_price_l2: "Price_L2"})
    if col_price_l3:
        df = df.rename(columns={col_price_l3: "Price_L3"})
    if col_level4_only:
        df = df.rename(columns={col_level4_only: "Level4_Only"})
    if col_comments:
        df = df.rename(columns={col_comments: "Comments"})

    # Ensure string types and clean whitespace
    for c in ["MANUFACTURER", "MODEL", "DESCRIPTION"]:
        df[c] = df[c].astype(str).fillna("").map(lambda s: collapse_spaces(str(s)))
    df["Accredited"] = df["Accredited"].map(to_int_accredited)

    # Normalize optional fields
    if "Level4_Only" in df.columns:
        df["Level4_Only"] = df["Level4_Only"].map(to_int_flag)
    else:
        df["Level4_Only"] = 0

    for c in ["Price_L1", "Price_L2", "Price_L3"]:
        if c in df.columns:
            df[c] = df[c].map(parse_price)
        else:
            df[c] = None

    if "Comments" not in df.columns:
        df["Comments"] = ""

    # Capture any additional columns into a JSON blob for preservation
    standard_cols = {"MANUFACTURER", "MODEL", "DESCRIPTION", "Accredited", "Price_L1", "Price_L2", "Price_L3", "Level4_Only", "Comments"}
    extra_cols = [c for c in df.columns if c not in standard_cols]

    def _row_extra_json(row: pd.Series) -> str:
        extra: Dict[str, Any] = {}
        for col in extra_cols:
            val = row.get(col, None)
            # Skip NaN/None/empty-like
            if pd.isna(val):
                continue
            if isinstance(val, str):
                sval = collapse_spaces(val)
                if not sval or sval.lower() in {"none", "na", "n/a", "null"}:
                    continue
                extra[col] = sval
            else:
                extra[col] = val
        try:
            return json.dumps(extra, ensure_ascii=False)
        except Exception:
            # Fallback: coerce non-serializable values to string
            safe_extra: Dict[str, Any] = {}
            for k, v in extra.items():
                try:
                    json.dumps(v)
                    safe_extra[k] = v
                except Exception:
                    safe_extra[k] = str(v)
            return json.dumps(safe_extra, ensure_ascii=False)

    if extra_cols:
        df["extra_json"] = df.apply(_row_extra_json, axis=1)
    else:
        df["extra_json"] = "{}"

    return df


def clean_and_normalize(df: pd.DataFrame, aliases: Aliases) -> pd.DataFrame:
    # Apply manufacturer aliases first
    df["manufacturer_canon"] = df["MANUFACTURER"].map(lambda s: apply_manufacturer_alias(s, aliases))
    df["manufacturer_norm"] = df["manufacturer_canon"].map(normalize_manufacturer)

    # Model normalization and alias by canonical manufacturer
    df["model_norm"] = df["MODEL"].map(normalize_model)
    df["model_norm"] = df.apply(
        lambda r: apply_model_alias(r["manufacturer_norm"], r["model_norm"], aliases), axis=1
    )

    # For storage, we persist canonical manufacturer and canonical model (from aliases)
    df["manufacturer"] = df["manufacturer_canon"].map(lambda s: collapse_spaces(s))

    # If a model alias changed the model_norm to a canonical, set display model to that canonical
    df["model"] = df["model_norm"]

    # Keep original description
    df["description"] = df["DESCRIPTION"].astype(str)
    
    # Convert YES/NO to 1/0 for boolean fields
    df["accredited"] = df["Accredited"]  # Already converted by to_int_accredited function
    df["level4_only"] = df["Level4_Only"].map({"YES": 1, "NO": 0}).fillna(0).astype(int)
    df["price_level1"] = df["Price_L1"].astype(float) if "Price_L1" in df.columns else None
    df["price_level2"] = df["Price_L2"].astype(float) if "Price_L2" in df.columns else None
    df["price_level3"] = df["Price_L3"].astype(float) if "Price_L3" in df.columns else None
    df["comments"] = df["Comments"].astype(str) if "Comments" in df.columns else ""

    cleaned = df[[
        "manufacturer",
        "model",
        "description",
        "accredited",
        "level4_only",
        "price_level1",
        "price_level2",
        "price_level3",
        "comments",
        "manufacturer_norm",
        "model_norm",
        "extra_json",
    ]].copy()

    return cleaned


# -----------------------------
# SQLite schema and loading
# -----------------------------


SCHEMA_SQL = r"""
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS equipment (
    id INTEGER PRIMARY KEY,
    manufacturer TEXT,
    model TEXT,
    description TEXT,
    accredited INTEGER NOT NULL,
    level4_only INTEGER NOT NULL DEFAULT 0,
    price_level1 REAL,
    price_level2 REAL,
    price_level3 REAL,
    comments TEXT,
    manufacturer_norm TEXT,
    model_norm TEXT,
    extra_json TEXT
);

-- Unique index for idempotency and fast exact lookups
CREATE UNIQUE INDEX IF NOT EXISTS idx_equipment_exact ON equipment(manufacturer_norm, model_norm);

-- FTS5 virtual table linked to content table
CREATE VIRTUAL TABLE IF NOT EXISTS equipment_fts USING fts5(
    description, manufacturer, model,
    content='equipment', content_rowid='id'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS equipment_ai AFTER INSERT ON equipment BEGIN
  INSERT INTO equipment_fts(rowid, description, manufacturer, model)
  VALUES (new.id, new.description, new.manufacturer, new.model);
END;

CREATE TRIGGER IF NOT EXISTS equipment_ad AFTER DELETE ON equipment BEGIN
  DELETE FROM equipment_fts WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS equipment_au AFTER UPDATE ON equipment BEGIN
  DELETE FROM equipment_fts WHERE rowid = old.id;
  INSERT INTO equipment_fts(rowid, description, manufacturer, model)
  VALUES (new.id, new.description, new.manufacturer, new.model);
END;
"""


def drop_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.executescript(
        """
        DROP TRIGGER IF EXISTS equipment_ai;
        DROP TRIGGER IF EXISTS equipment_ad;
        DROP TRIGGER IF EXISTS equipment_au;
        DROP TABLE IF EXISTS equipment_fts;
        DROP TABLE IF EXISTS equipment;
        """
    )
    conn.commit()


def ensure_schema(conn: sqlite3.Connection, rebuild: bool) -> None:
    if rebuild:
        logging.info("Dropping and recreating schema (--rebuild)")
        drop_schema(conn)
    conn.executescript(SCHEMA_SQL)
    conn.commit()

    # Ensure new columns exist on older databases (non-rebuild path)
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(equipment)")
        cols = {row[1] for row in cur.fetchall()}  # name is at index 1
        # Add columns if missing
        if "level4_only" not in cols:
            logging.info("Adding missing column 'level4_only' to equipment table")
            cur.execute("ALTER TABLE equipment ADD COLUMN level4_only INTEGER NOT NULL DEFAULT 0")
        if "price_level1" not in cols:
            logging.info("Adding missing column 'price_level1' to equipment table")
            cur.execute("ALTER TABLE equipment ADD COLUMN price_level1 REAL")
        if "price_level2" not in cols:
            logging.info("Adding missing column 'price_level2' to equipment table")
            cur.execute("ALTER TABLE equipment ADD COLUMN price_level2 REAL")
        if "price_level3" not in cols:
            logging.info("Adding missing column 'price_level3' to equipment table")
            cur.execute("ALTER TABLE equipment ADD COLUMN price_level3 REAL")
        if "comments" not in cols:
            logging.info("Adding missing column 'comments' to equipment table")
            cur.execute("ALTER TABLE equipment ADD COLUMN comments TEXT")
        if "extra_json" not in cols:
            logging.info("Adding missing column 'extra_json' to equipment table")
            cur.execute("ALTER TABLE equipment ADD COLUMN extra_json TEXT")
            conn.commit()
    except Exception as e:
        logging.warning(f"Schema check/upgrade skipped or failed: {e}")


def insert_dataframe(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    rows = [
        (
            r["manufacturer"],
            r["model"],
            r["description"],
            int(r["accredited"]),
            int(r["level4_only"]),
            (None if pd.isna(r["price_level1"]) else float(r["price_level1"])) if r.get("price_level1") is not None else None,
            (None if pd.isna(r["price_level2"]) else float(r["price_level2"])) if r.get("price_level2") is not None else None,
            (None if pd.isna(r["price_level3"]) else float(r["price_level3"])) if r.get("price_level3") is not None else None,
            r.get("comments", ""),
            r["manufacturer_norm"],
            r["model_norm"],
            r.get("extra_json", "{}"),
        )
        for _, r in df.iterrows()
    ]

    sql = (
        "INSERT INTO equipment (manufacturer, model, description, accredited, level4_only, price_level1, price_level2, price_level3, comments, manufacturer_norm, model_norm, extra_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(manufacturer_norm, model_norm) DO UPDATE SET "
        "manufacturer=excluded.manufacturer, model=excluded.model, description=excluded.description, accredited=excluded.accredited, "
        "level4_only=excluded.level4_only, price_level1=excluded.price_level1, price_level2=excluded.price_level2, price_level3=excluded.price_level3, comments=excluded.comments, "
        "extra_json=excluded.extra_json"
    )
    cur = conn.cursor()
    cur.executemany(sql, rows)
    conn.commit()

    # Ensure FTS index is up-to-date
    cur.execute("INSERT INTO equipment_fts(equipment_fts) VALUES('rebuild')")
    conn.commit()


# -----------------------------
# Validation and reporting
# -----------------------------


def dataframe_report(df: pd.DataFrame) -> Dict[str, Any]:
    total = len(df)
    distinct_manufacturer = df["manufacturer_norm"].nunique()
    distinct_model = df["model_norm"].nunique()
    null_pct = {col: float(df[col].isna().mean() * 100.0) for col in df.columns}

    # collisions on (manufacturer_norm, model_norm)
    dupes = (
        df.groupby(["manufacturer_norm", "model_norm"]).size().reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    collisions = dupes[dupes["count"] > 1].head(20)

    result = {
        "rows": int(total),
        "distinct_manufacturers": int(distinct_manufacturer),
        "distinct_models": int(distinct_model),
        "null_pct": null_pct,
        "collisions_top20": collisions.to_dict(orient="records"),
    }
    return result


def log_report(report: Dict[str, Any]) -> None:
    logging.info(
        f"Ingested rows={report['rows']} | manufacturers={report['distinct_manufacturers']} | models={report['distinct_models']}"
    )
    if report["collisions_top20"]:
        logging.warning("Duplicate collisions on (manufacturer_norm, model_norm): showing top 20")
        for row in report["collisions_top20"]:
            logging.warning(
                f"collision: {row['manufacturer_norm']} | {row['model_norm']} | count={row['count']}"
            )


# -----------------------------
# Search API
# -----------------------------


def _open_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def _manufacturer_family(manufacturer_norm: str) -> List[str]:
    # Known vendor family for Keysight
    if manufacturer_norm in {"KEYSIGHT", "AGILENT", "HEWLETT PACKARD", "HP"}:
        return ["KEYSIGHT", "AGILENT", "HEWLETT PACKARD", "HP"]
    return [manufacturer_norm]


def _fts_confidence_from_bm25(rank: float) -> float:
    # bm25: lower is better. Map typical small values to 0.8 and larger to 0.6
    try:
        r = float(rank)
    except Exception:
        return 0.6
    if r <= 1.0:
        return 0.8
    if r <= 3.0:
        return 0.72
    if r <= 6.0:
        return 0.66
    return 0.6


def _row_to_item(row: sqlite3.Row, confidence: float) -> Dict[str, Any]:
    return {
        "manufacturer": row["manufacturer"],
        "model": row["model"],
        "description": row["description"],
        "accredited": int(row["accredited"]),
        "confidence": float(confidence),
    }


def search_equipment(
    db_path: str,
    manufacturer_or_query: str = "",
    model: str = "",
    use_semantic: bool = False,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Behavior:
    1) Normalize & apply aliases.
    2) If both manufacturer and model are provided -> exact on (manufacturer_norm, model_norm).
    3) Else try near-exact heuristics (strip separators, vendor family).
    4) Else FTS5 on description/model/manufacturer tokens.
    Returns: {
      "matches": [ { "manufacturer", "model", "description", "accredited", "confidence" } ],
      "alternates": [...],  # similar models
      "query_norm": {...}
    }
    """

    db_dir = Path(db_path).resolve().parent
    aliases = load_aliases(db_dir)

    manufacturer_in = manufacturer_or_query or ""
    model_in = model or ""

    # Apply normalization and aliases
    manufacturer_canon = apply_manufacturer_alias(manufacturer_in, aliases) if manufacturer_in else ""
    manufacturer_norm = normalize_manufacturer(manufacturer_canon) if manufacturer_canon else ""
    model_norm = normalize_model(model_in) if model_in else ""
    if manufacturer_norm and model_norm:
        model_norm = apply_model_alias(manufacturer_norm, model_norm, aliases)

    result: Dict[str, Any] = {
        "matches": [],
        "alternates": [],
        "query_norm": {
            "manufacturer": manufacturer_canon,
            "manufacturer_norm": manufacturer_norm,
            "model_norm": model_norm,
        },
    }

    with _open_conn(db_path) as conn:
        cur = conn.cursor()

        # 1) Exact match if both provided
        if manufacturer_norm and model_norm:
            cur.execute(
                """
                SELECT manufacturer, model, description, accredited
                FROM equipment
                WHERE manufacturer_norm = ? AND model_norm = ?
                LIMIT ?
                """,
                (manufacturer_norm, model_norm, limit),
            )
            rows = cur.fetchall()
            if rows:
                result["matches"] = [_row_to_item(r, 1.0) for r in rows]
                return result

        # 2) Near-exact: try vendor family and separator variants if model provided
        if model_norm:
            family = _manufacturer_family(manufacturer_norm) if manufacturer_norm else None
            candidates: List[sqlite3.Row] = []
            if family:
                placeholders = ",".join(["?"] * len(family))
                # Match exact model_norm across vendor family
                cur.execute(
                    f"""
                    SELECT manufacturer, model, description, accredited
                    FROM equipment
                    WHERE manufacturer_norm IN ({placeholders}) AND model_norm = ?
                    LIMIT ?
                    """,
                    (*family, model_norm, limit),
                )
                candidates = cur.fetchall()
            if candidates:
                result["matches"] = [_row_to_item(r, 0.85) for r in candidates]
                return result

        # 3) FTS5 full-text search
        query_text = manufacturer_in
        if model_in:
            if query_text:
                query_text = f"{query_text} {model_in}"
            else:
                query_text = model_in
        query_text = query_text.strip()

        if query_text:
            cur.execute(
                """
                SELECT e.manufacturer, e.model, e.description, e.accredited, bm25(equipment_fts) AS rank
                FROM equipment_fts
                JOIN equipment e ON e.id = equipment_fts.rowid
                WHERE equipment_fts MATCH ?
                ORDER BY rank ASC
                LIMIT ?
                """,
                (query_text, limit),
            )
            rows = cur.fetchall()
            result["matches"] = [
                _row_to_item(r, _fts_confidence_from_bm25(r["rank"])) for r in rows
            ]

            # Alternates: similar models without exact name match
            alt_models: List[str] = []
            for r in rows:
                m = str(r["model"]).upper()
                if model_norm and normalize_model(m) == model_norm:
                    continue
                if m not in alt_models:
                    alt_models.append(m)
                if len(alt_models) >= max(0, limit - len(result["matches"])):
                    break
            result["alternates"] = alt_models
            return result

        # Nothing to search with
        return result


# -----------------------------
# CLI
# -----------------------------


def export_parquet(df: pd.DataFrame, out_path: Path) -> None:
    df.to_parquet(out_path, engine="pyarrow", index=False)


def build_database(
    excel: Path,
    db_path: Path,
    alias_dir: Path,
    rebuild: bool,
    sample: int,
    report: bool,
) -> None:
    logging.info(f"Loading Excel: {excel}")
    df_raw = ingest_excel(excel, sample=sample)

    logging.info("Loading alias files and normalizing data")
    aliases = load_aliases(alias_dir)
    df_clean = clean_and_normalize(df_raw, aliases)

    # Reporting
    rep = dataframe_report(df_clean)
    if report:
        log_report(rep)

    # Build database
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _open_conn(str(db_path)) as conn:
        ensure_schema(conn, rebuild=rebuild)
        logging.info("Inserting rows into SQLite (idempotent upsert)")
        insert_dataframe(conn, df_clean)

    # Export parquet
    parquet_path = db_path.with_suffix(".parquet")
    logging.info(f"Writing Parquet: {parquet_path}")
    export_parquet(df_clean, parquet_path)
    logging.info("Done")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build an optimized SQLite equipment database from Excel, with alias resolution and FTS5."
    )
    p.add_argument("--excel", default="./Tescom_new_list.xlsx", help="Path to Excel file")
    p.add_argument("--db", default="equipment.db", help="Output SQLite DB path")
    p.add_argument(
        "--aliases", default="./", help="Directory containing alias CSVs (created if absent)"
    )
    p.add_argument("--rebuild", action="store_true", help="Drop & recreate tables")
    p.add_argument("--sample", type=int, default=0, help="Sample N rows; 0 = all")
    p.add_argument("--report", action="store_true", help="Print summary stats")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return p.parse_args()


def _demo_searches(db_path: str) -> None:
    print("\nDemo searches (JSON):")
    examples = [
        {"manufacturer_or_query": "Keysight", "model": "34401A"},
        {"manufacturer_or_query": "HP", "model": "34401A"},
        {"manufacturer_or_query": "Agilent", "model": "HP34401A"},
        {"manufacturer_or_query": "vacuum gauge", "model": ""},
    ]
    for ex in examples:
        res = search_equipment(db_path=db_path, **ex, limit=5)
        print(json.dumps(res, indent=2))


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    excel = Path(args.excel)
    db_path = Path(args.db)
    alias_dir = Path(args.aliases)

    build_database(
        excel=excel,
        db_path=db_path,
        alias_dir=alias_dir,
        rebuild=args.rebuild,
        sample=args.sample,
        report=args.report,
    )

    # Demo block to show example searches
    _demo_searches(str(db_path))


if __name__ == "__main__":
    main()


