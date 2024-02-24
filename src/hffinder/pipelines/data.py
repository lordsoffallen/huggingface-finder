from typing import Optional, Any
from datasets import Dataset
from hffinder.extras.iso_639 import LANGUAGES_ISO_639_1, LANGUAGES_ISO_639_3
from bs4 import BeautifulSoup
from unstructured.partition.html import partition_html
from markdown_it import MarkdownIt
from .regex import code_blocks, citation, placeholder, urls, emojis, \
    hidden_comment_blocks

import re
import json


MD = MarkdownIt("commonmark").enable('table')


def convert_tags_to_str(tags: list) -> str:
    """ Take a list of tags and convert them text format with # appended as suffix """
    tag_str = "TAGS\n"

    if len(tags) == 0:
        return ""

    if len(tags) == 1 and isinstance(tags[0], list):
        tags = tags[0]

    for tag in tags:
        if ":" in tag:
            kv_pair = tag.split(":")

            if kv_pair[0] == "language":
                # Replace language iso code with language names
                if len(kv_pair[1]) == 2:
                    try:
                        lang_name = LANGUAGES_ISO_639_1[kv_pair[1]]['name']
                    except KeyError:
                        lang_name = kv_pair[1]
                elif len(kv_pair[1]) == 3:
                    try:
                        lang_name = LANGUAGES_ISO_639_3[kv_pair[1]]['name']
                    except KeyError:
                        lang_name = kv_pair[1]
                elif kv_pair[1] in ['code', 'multilingual']:
                    lang_name = kv_pair[1]
                else:
                    raise ValueError(
                        f"Unexpected language length={len(kv_pair[1])}, "
                        f"lang={kv_pair[1]}"
                    )
                tag = f"{kv_pair[0]}:{lang_name}"

            tag_str += "#" + tag.replace(":", "-") + " "
        else:
            tag_str += "#" + tag + " "

    tag_str += "\n"

    return tag_str


def extract_arxiv_number(tags: list) -> Optional[str]:
    if len(tags) == 0:
        return None

    if len(tags) == 1 and isinstance(tags[0], list):
        tags = tags[0]

    for tag in tags:
        if ":" in tag:
            kv_pair = tag.split(":")
            if kv_pair[0] == "arxiv":
                return kv_pair[1]

    return None


def extract_languages(tags: list) -> list:
    if len(tags) == 0:
        return []

    if len(tags) == 1 and isinstance(tags[0], list):
        tags = tags[0]

    langs = []

    for tag in tags:
        if ":" in tag:
            kv_pair = tag.split(":")
            if kv_pair[0] == "language":
                languages = kv_pair[1]
                if isinstance(languages, list):
                    langs.append(*languages)
                elif isinstance(languages, str):
                    langs.append(languages)
                else:
                    raise ValueError

    return langs


def coalesce_null_langs(x):
    """Sometimes metadata has lang information but tag not. """

    if len(x['languages']) == 0:
        parsed = json.loads(x['metadata']).get('language')
        if parsed is not None:
            return parsed if isinstance(parsed, list) else [parsed]
    return x['languages']


def _preprocess_text(text: str) -> str:
    text = re.sub(code_blocks, "", text)
    text = re.sub(placeholder, "", text)
    text = re.sub(citation, "", text)
    text = re.sub(hidden_comment_blocks, "", text)
    text = re.sub(emojis, "", text)
    text = text.replace("`", "'")

    return text


def _parse_tables(soup: BeautifulSoup) -> str:
    """ Convert tables into more understandable format """
    # Extract column names
    columns = [th.get_text(strip=True) for th in soup.find('tr').find_all('th')]

    # Extract and format the table content
    table_rows = []
    for row in soup.find_all('tr')[1:]:  # Skip the first row containing headers
        row_data = [
            f"{column} - {cell.get_text(strip=True)}"
            for column, cell in zip(columns, row.find_all('td'))
        ]
        table_rows.append("\n".join(row_data))

    # Create a dense text representation
    return "\n".join(table_rows)


def process_markdown(text: str):
    html = BeautifulSoup(MD.render(text), 'lxml')

    # Replace URLs
    for a in html.find_all('a'):
        a.replace_with("[URL]")

    # Better table data
    for t in html.find_all('table'):
        try:
            t.replace_with(_parse_tables(t))
        except AttributeError:
            # No parsing the table so we drop
            t.decompose()

    if len(html) == 0:
        return ""

    # Extract text elements from html
    text = '\n'.join([h.text for h in partition_html(text=str(html))])     # noqa

    # Replace URLs within text
    text = re.sub(urls, "[URL]", text)

    return text


def preprocess_datasets(ds: Dataset, n_jobs: int = 10) -> Dataset:
    # Process Tags
    ds = ds.map(lambda x: {"arxiv": extract_arxiv_number(x)}, input_columns="tags")
    ds = ds.map(lambda x: {"languages": extract_languages(x)}, input_columns="tags")
    ds = ds.map(
        lambda x: {"languages": coalesce_null_langs(x)},
        input_columns=["tags", "languages"]
    )
    ds = ds.map(lambda x: {"tags_str": convert_tags_to_str(x)}, input_columns="tags")

    # Process Texts
    ds = ds.map(lambda x: {"text_str": _preprocess_text(x)}, input_columns="text")
    # Filter empty markdown files
    ds = ds.filter(lambda x: True if len(x) > 0 else False, input_columns="text_str")
    ds = ds.map(
        lambda x: {"text_str": process_markdown(x)},
        input_columns="text_str",
        num_proc=n_jobs,
    )

    return ds


def preprocess_models(ds: Dataset) -> Dataset:
    pass
