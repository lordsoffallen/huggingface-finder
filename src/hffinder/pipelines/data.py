from datasets import Dataset
from bs4 import BeautifulSoup
from unstructured.partition.html import partition_html
from markdown_it import MarkdownIt
from .regex import code_blocks, citation, placeholder, urls, emojis, \
    hidden_comment_blocks
from iso639.language import Language, LanguageNotFoundError

import logging
import re
import json


logger = logging.getLogger(__file__)
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
                try:
                    lang_name = Language.match(kv_pair[1]).name
                except LanguageNotFoundError:
                    lang_name = kv_pair[1]

                tag = f"{kv_pair[0]}:{lang_name}"

            tag_str += "#" + tag.replace(":", "-") + " "
        else:
            tag_str += "#" + tag + " "

    tag_str += "\n"

    return tag_str


def extract_arxiv_number(tags: list) -> list:
    if len(tags) == 0:
        return []

    if len(tags) == 1 and isinstance(tags[0], list):
        tags = tags[0]

    links = []

    for tag in tags:
        if ":" in tag:
            kv_pair = tag.split(":")
            if kv_pair[0] == "arxiv":
                links.append(kv_pair[1])

    return links


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


def coalesce_null_langs(x) -> list:
    """Sometimes metadata has lang information but tag not. """

    langs = x['languages']

    if len(langs) == 0:
        parsed = json.loads(x['metadata']).get('language')

        if parsed is not None:
            if isinstance(parsed, list):
                # YAML `no` issue gets converted to false and [{}] issue
                for idx, lang in enumerate(parsed):
                    if lang is False:
                        parsed[idx] = "no"
                    elif isinstance(lang, dict):
                        parsed.pop(idx)     # remove dicts
                    elif not isinstance(lang, str):
                        logger.warning(f"List contains unusual data type={lang}")
                return parsed
            elif isinstance(parsed, str):
                return [parsed]
            elif isinstance(parsed, bool) and not parsed:
                # YAML `no` issue gets converted to false
                return ['no']
            else:
                logger.warning(f"Couldn't parse metadata language={parsed}")
                return []
    return langs


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


def preprocess(ds: Dataset, n_jobs: int | str = 10) -> Dataset:
    # Process Tags
    ds = ds.map(lambda x: {"arxiv": extract_arxiv_number(x)}, input_columns="tags")
    ds = ds.map(lambda x: {"languages": extract_languages(x)}, input_columns="tags")
    ds = ds.map(lambda x: {"languages": coalesce_null_langs(x)})
    ds = ds.map(lambda x: {"tags_str": convert_tags_to_str(x)}, input_columns="tags")

    # Process Texts
    ds = ds.map(lambda x: {"text_str": _preprocess_text(x)}, input_columns="text")
    # Filter empty markdown files
    ds = ds.filter(lambda x: True if len(x) > 0 else False, input_columns="text_str")
    ds = ds.map(
        lambda x: {"text_str": process_markdown(x)},
        input_columns="text_str",
        num_proc=n_jobs if isinstance(n_jobs, int) else int(n_jobs),
    )
    ds = ds.map(
        lambda txt, tag: {"processed_text": f"{txt}\n{tag}"},
        input_columns=["text_str", "tags_str"]
    )

    return ds
