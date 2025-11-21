#Topic: data prep for bertopic modelling
#Notes:
# - Input: parquet file
# - Output: cleaned dataframe saved as parquet file, with an additional column 'cleaned_text' and 'embed_id'
# steps include: remove urls, duplicates, and NAs 

from __future__ import annotations
import argparse
import re
import sys
from typing import Optional

import pandas as pd
from langdetect import detect, LangDetectException
from tqdm import tqdm


# enable tqdm on pandas
tqdm.pandas()

# ===== PRE-COMPILE ALL REGEX PATTERNS (ONE TIME ONLY) =====
URL_PATTERN = re.compile(r'https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
PARTIAL_URL_PATTERN = re.compile(r'https?://\S*')
DOMAIN_PATTERN = re.compile(r'\s(?:www\.|(?:[\w-]+\.)+(?:com|net|org|edu|gov|mil|biz|info|io|me|tv|[\w]{2,}))\S*')
FRAGMENTS_PATTERN = re.compile(r'(?:press\.coop|gab\.com|youtube\.com|bitchute\.com|imdb\.com)\/\S*')
ASCII_PATTERN = re.compile(r'[^\x00-\x7F]+')
SPACE_PATTERN = re.compile(r'\s+')


def remove_non_english(text: str | None) -> Optional[str]:
    """
    Return the text only if detected language is English.
    If detection fails, returns the original text.
    Short/non-string inputs are returned unchanged or None according to original logic.
    """
    if not isinstance(text, str) or len(text.strip()) < 3:
        return text

    try:
        detected_lang = detect(text)
        if detected_lang != 'en':
            return None
        return text
    except LangDetectException:
        # If detection fails, keep the text (same as original)
        return text


def remove_urls(text: str | None) -> Optional[str]:
    """Remove URLs and related fragments using precompiled regex patterns."""
    if not isinstance(text, str):
        return text
    text = URL_PATTERN.sub('', text)
    text = PARTIAL_URL_PATTERN.sub('', text)
    text = DOMAIN_PATTERN.sub(' ', text)
    text = FRAGMENTS_PATTERN.sub('', text)
    return text


def clean_text(text: str | None) -> Optional[str]:
    """
    Comprehensive text cleaning function with optimized regex.
    Returns None for non-English or empty/invalid results.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return None

    # First check if text is English
    text = remove_non_english(text)
    if text is None:
        return None

    # Remove URLs/fragments/domains
    text = remove_urls(text)

    # Remove emojis and non-ASCII characters
    text = ASCII_PATTERN.sub('', text)

    # Collapse multiple spaces and trim
    text = SPACE_PATTERN.sub(' ', text).strip()

    return text if len(text) > 0 else None


def run(input_path: str, output_path: str) -> None:
    # Read input
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        print(f"ERROR: failed to read parquet at {input_path}: {e}", file=sys.stderr)
        sys.exit(2)

    # Basic checks and stats (based on 'post_text' column)
    if 'post_text' not in df.columns:
        print("ERROR: input file does not contain a 'post_text' column.", file=sys.stderr)
        sys.exit(3)

    total = df['post_text'].shape[0]
    na_rows = df['post_text'].isna().sum()
    missing_rate = na_rows / total * 100 if total > 0 else 0.0
    print(f"The dataset contains: {total} records")
    print(f"{na_rows} records are missing 'post_text' ({missing_rate:.2f}% missing rate)")

    # Platform distribution if 'platform' exists
    if 'platform' in df.columns:
        platform_counts = df['platform'].value_counts()
        print("Platform distribution:")
        print(platform_counts.to_string())
    else:
        print("No 'platform' column found — skipping platform distribution.")

    # Keep behavior similar to your original script, but be explicit: drop rows missing post_text or id (if present)
    # If 'id' not present, just drop missing post_text
    subset_for_drop = ['post_text']
    if 'id' in df.columns:
        subset_for_drop.append('id')
    df = df.dropna(subset=subset_for_drop)

    # Drop duplicate IDs if present
    if 'id' in df.columns:
        before_dup = df.shape[0]
        df = df.drop_duplicates(subset=['id'])
        after_dup = df.shape[0]
        print(f"Dropped {before_dup - after_dup} duplicate rows based on 'id' column.")
    else:
        print("No 'id' column found — skipping duplicate drop by 'id'.")

    # Apply cleaning to post_text -> new column post_clean
    print("Cleaning text (this may take a while) ...")
    df['post_clean'] = df['post_text'].progress_apply(clean_text)

    # Drop rows where cleaning resulted in None (non-English or empty after cleaning)
    before_clean_drop = df.shape[0]
    df = df[df['post_clean'].notna()].copy()
    after_clean_drop = df.shape[0]
    print(f"Dropped {before_clean_drop - after_clean_drop} rows where post_clean is None/NA.")

    # Reset index and set embed_id as index string
    df = df.reset_index(drop=True)
    df['embed_id'] = df.index.astype(str)

    # Save cleaned data
    try:
        df.to_parquet(output_path, index=False)
        print(f"Saved cleaned parquet to: {output_path} ({df.shape[0]} rows)")
    except Exception as e:
        print(f"ERROR: failed to write parquet to {output_path}: {e}", file=sys.stderr)
        sys.exit(4)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean post_text in a parquet file and save cleaned parquet.")
    p.add_argument("--input", "-i", required=True, help="Input parquet file path")
    p.add_argument("--output", "-o", required=True, help="Output parquet file path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output)