"""
Carrega o dataset StereoSet pt-BR no formato JSON gerado por convert_csv_to_json.py.
"""

import json
from dataclasses import dataclass, field
from typing import List


@dataclass
class Sentence:
    ID: str
    sentence: str
    gold_label: str


@dataclass
class Example:
    ID: str
    target: str
    bias_type: str
    context: str
    sentences: List[Sentence] = field(default_factory=list)


class StereoSet:
    def __init__(self, path: str):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)["data"]

        self._intrasentence = [
            self._parse(item) for item in data.get("intrasentence", [])
        ]
        self._intersentence = [
            self._parse(item) for item in data.get("intersentence", [])
        ]

    def _parse(self, item: dict) -> Example:
        sentences = [
            Sentence(ID=s["id"], sentence=s["sentence"], gold_label=s["gold_label"])
            for s in item["sentences"]
        ]
        return Example(
            ID=item["id"],
            target=item["target"],
            bias_type=item["bias_type"],
            context=item["context"],
            sentences=sentences,
        )

    def get_intrasentence_examples(self) -> List[Example]:
        return self._intrasentence

    def get_intersentence_examples(self) -> List[Example]:
        return self._intersentence
