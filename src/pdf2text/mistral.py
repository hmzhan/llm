
import base64
import pycountry
from enum import Enum
from pathlib import Path
from pydantic import BaseModel

languages = {
    lang.alpha_2: lang.name for lang in pycountry.languages if hasattr(lang, 'alpha_2')
}


class LanguageMeta(Enum.__class__):
    def __new__(metacls, cls, bases, classdict):
        for code, name in languages.items():
            classdict[name.upper().replace(' ', '_')] = name
        return super().__new__(metacls, cls, bases, classdict)


class Language(Enum, metaclass=LanguageMeta):
    pass


class StructuredOCR(BaseModel):
    file_name: str
    topics: list[str]
    languages: list[Language]
    ocr_contents: dict


