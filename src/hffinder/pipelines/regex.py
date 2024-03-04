from pregex.core.pre import Pregex
from pregex.core.classes import AnyWhitespace, AnyButWhitespace
from pregex.core.quantifiers import Optional, Indefinite, OneOrMore
from pregex.core.operators import Either
from pregex.meta import HttpUrl

import sys
import re


# Code Blocks
_anything_in_between = Indefinite(
    Pregex(
        f'[{AnyWhitespace() + AnyButWhitespace()}]', escape=False
    ),
    is_greedy=False
)
_code_blocks = Pregex('```') + _anything_in_between + Pregex('```')
code_blocks = _code_blocks.get_compiled_pattern()


_hidden_comment_blocks = Pregex('<!--') + _anything_in_between + Pregex('-->')
hidden_comment_blocks = _hidden_comment_blocks.get_compiled_pattern()

# Dummy Placeholders
_placeholder = Either(
    Optional('[') + "More Information Needed" + Optional(']'),
    Optional('[') + "Needs More Information" + Optional(']')
)
placeholder = _placeholder.get_compiled_pattern()

# Citation Text
_citation = OneOrMore('#') + Indefinite(AnyWhitespace()) + \
    "Citation" + Indefinite(AnyWhitespace()) + Optional("Information")
citation = _citation.get_compiled_pattern()


# URLs
urls = HttpUrl().get_compiled_pattern()

# Markdown Links [text](url)
markdown_links = re.compile(r'\[([^]]+)]\(([^)]+)\)')

markdown_headers = re.compile(r'(#{1,6} .*?)(?=\n#|$)', flags=re.DOTALL)

html_table = re.compile(r'<table>(.+?)</table>', re.DOTALL)
html_table_columns = re.compile(r'<th>(.+?)</th>')
html_table_data = re.compile(
    r'<tr>\s*<td>(.*?)</td>\s*<td>(.*?)</td>\s*<td>(.*?)</td>\s*</tr>'
)

if sys.maxunicode < 0x10FFFF:
    emojis = re.compile(
        r"[\u2600-\u26FF\u2700-\u27BF]",
        flags=re.IGNORECASE,
    )
else:
    emojis = re.compile(
        r"[\u2600-\u26FF\u2700-\u27BF\U0001F300-\U0001F5FF\U0001F600-\U0001F64F"
        r"\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF]",
        flags=re.IGNORECASE,
    )

