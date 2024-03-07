from datasets import Dataset
from markdown_it import MarkdownIt
from .regex import code_blocks, citation, placeholder, urls, emojis, \
    hidden_comment_blocks, markdown_links, html_table, html_table_columns, \
    html_table_data, markdown_headers
from iso639.language import Language, LanguageNotFoundError
from markdownify import markdownify
from . import BLACK_LIST
from .. import TransformerModel

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


def _preprocess_text(text: str, url_token: str) -> str:
    text = re.sub(hidden_comment_blocks, "", text)
    text = re.sub(code_blocks, "", text)
    text = re.sub(citation, "", text)
    text = re.sub(markdown_links, r'\1', text)
    text = re.sub(placeholder, "", text)
    text = re.sub(emojis, "", text)
    text = re.sub(urls, url_token, text)
    text = text.replace("`", "'")
    text = text.replace("**", "")   # Bold text

    return text


def _parse_tables(text: str) -> str:
    """ Check if there are Markdown tables present in the data. Parse them if so."""

    def _parse_single_table(table: str) -> str:
        columns = re.findall(html_table_columns, table)
        data = re.findall(html_table_data, table)
        rows = [
            ', '.join(list(map(lambda x, y: f"{x}: {y}", columns, d))) for d in data
        ]
        all_rows = "\n".join(rows)
        # Return html format to parse it back
        return f"<p>{all_rows}</p>"

    md = MD.render(text)

    if len(re.findall(html_table, md)) > 0:
        parsed_html = re.sub(html_table, lambda t: _parse_single_table(t[0]), md)
        # Convert from html back to markdown
        try:
            return markdownify(parsed_html)
        except RecursionError:
            logger.warning("Recursion Error is encountered, setting output to empty")
            return ""
    else:
        return text


def process_markdown(text: str) -> list[str]:
    """ Take Markdown file and split into header + context as a list of strings """
    # Use regex to capture headings and their content
    headers_and_contents = re.findall(markdown_headers, text)

    # Extract captured groups from the regex match
    return [group.strip() for group in headers_and_contents]


def preprocess(ds: Dataset, url_token: str, n_jobs: int | str = 10) -> Dataset:
    # filter bad dataset/model ids
    ds = ds.filter(lambda x: True if x not in BLACK_LIST else False, input_columns='id')

    # Process Tags
    ds = ds.map(lambda x: {"arxiv": extract_arxiv_number(x)}, input_columns="tags")
    ds = ds.map(lambda x: {"languages": extract_languages(x)}, input_columns="tags")
    ds = ds.map(lambda x: {"languages": coalesce_null_langs(x)})
    ds = ds.map(lambda x: {"tags_str": convert_tags_to_str(x)}, input_columns="tags")

    # Process Texts
    ds = ds.map(
        lambda x: {"text_str": _preprocess_text(x, url_token)},
        input_columns="text",
        num_proc=n_jobs if isinstance(n_jobs, int) else int(n_jobs),
    )
    # Filter empty markdown files
    ds = ds.filter(lambda x: True if len(x) > 0 else False, input_columns="text_str")
    # Parse Markdown tables if they exist
    ds = ds.map(
        lambda x: {"text_str": _parse_tables(x)},
        input_columns="text_str",
        num_proc=n_jobs if isinstance(n_jobs, int) else int(n_jobs),
    )
    ds = ds.map(
        lambda x: {"text_lists": process_markdown(x)},
        input_columns="text_str",
        num_proc=n_jobs if isinstance(n_jobs, int) else int(n_jobs),
    )
    ds = ds.map(
        lambda txt, tag: {"processed_texts": [tag] + txt},
        input_columns=["text_lists", "tags_str"]
    )

    return ds


def prepare_for_tokenizer(
    model_and_tokenizer: TransformerModel,
    ds: Dataset,
    token_prefix: str = '',
    n_jobs: int | str = 10
) -> Dataset:
    """ Take clean data and apply tokenizer """
    def _batch_or_single_text(
        texts: list[str], token_lengths: list[int], max_length: int
    ) -> list[str]:

        if sum(token_lengths) < max_length:
            return [token_prefix + ''.join(texts)]   # Fits into model context
        else:
            # long context, decide how to split
            size = 0
            batch_text = []
            start_idx = 0

            for i, s in enumerate(token_lengths):
                if (size < max_length) and ((size + s) < max_length):
                    size += s
                else:
                    # get the current chunk
                    batch_text.append(
                        token_prefix + ''.join(texts[start_idx:i])
                    )
                    start_idx = i
                    size = 0

            return batch_text

    tokenizer = model_and_tokenizer.tokenizer

    # Compute sections token lengths
    ds = ds.map(
        lambda x: {"tokens_length": [len(tokenizer.tokenize(i)) for i in x]},
        input_columns='processed_texts',
        num_proc=n_jobs if isinstance(n_jobs, int) else int(n_jobs),
    )

    # Texts are batched together if they are long.
    max_length = tokenizer.model_max_length - len(tokenizer.tokenize(token_prefix)) - 2
    ds = ds.map(
        lambda texts, token_length: {
            "input_texts": _batch_or_single_text(texts, token_length, max_length)
        },
        input_columns=['processed_texts', 'tokens_length'],
        num_proc=n_jobs if isinstance(n_jobs, int) else int(n_jobs),
    )

    return ds
