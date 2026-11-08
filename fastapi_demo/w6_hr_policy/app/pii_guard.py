"""Mask common identifiers before model calls and before storing session turns."""
import re

IDENTIFIERS = re.compile(
    r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"
    r"|(?<!\d)1[3-9]\d{9}(?!\d)"
    r"|(?<!\d)62\d{14,17}(?!\d)"
    r"|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)

def mask_identifiers(question):
    return IDENTIFIERS.sub("[标识已隐藏]", question)
