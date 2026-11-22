import os
import re
from functools import cache

import pandas as pd
from datasets import load_dataset
from docx import Document
from docx.table import Table
from pptx import Presentation

# Dataset configuration

SPACE_AUTHOR_NAME = os.getenv("SPACE_AUTHOR_NAME", "bstraehle")

DATASET_TYPE_GAIA = "gaia"
DATASET_TYPE_HLE  = "hle"

# OpenAI project keys (sk-proj-...) are longer than 150 characters
MAX_API_KEY_LENGTH = 300

FEN_PATTERN = re.compile(r"[rnbqkpRNBQKP1-8/]+\s+[wb]\s+(?:-|[KQkq]+)\s+(?:-|[a-h][36])\s+\d+\s+\d+")

# Dataset processing

@cache
def _load_validation_dataset() -> pd.DataFrame:
    return load_dataset(f"{SPACE_AUTHOR_NAME}/validation", split="validation").to_pandas()


def get_dataset(dataset_type, level):
    """Examples as [question, final answer, file name] rows. GAIA levels are 1-3; HLE is level 0."""
    if (dataset_type == DATASET_TYPE_HLE) != (level == 0):
        return []

    df = _load_validation_dataset()
    df = df[df["Level"] == level]

    return df[["Question", "Final answer", "file_name"]].values.tolist()

# File processing

def is_ext(file_path, ext):
    return os.path.splitext(file_path)[1].lower() == ext.lower()


def read_file_json(file_path):
    df = None

    if is_ext(file_path, ".csv"):
        df = pd.read_csv(file_path)
    elif is_ext(file_path, ".xls") or is_ext(file_path, ".xlsx"):
        df = pd.read_excel(file_path)
    elif is_ext(file_path, ".json") or is_ext(file_path, ".jsonl"):
        df = pd.read_json(file_path, lines=is_ext(file_path, ".jsonl"))

    return "" if df is None else df.to_json()


def read_docx_text(file_path):
    text = []

    for block in Document(file_path).iter_inner_content():
        if isinstance(block, Table):
            text.extend(" | ".join(cell.text.strip() for cell in row.cells) for row in block.rows)
        elif block.style.name.startswith("Heading"):
            text.append("\n**" + block.text + "**\n")
        elif block.text:
            text.append(block.text)

    return "\n".join(text)


def read_pptx_text(file_path):
    return "\n\n".join(
        "\n".join(shape.text for shape in slide.shapes if hasattr(shape, "text"))
        for slide in Presentation(file_path).slides
    )

# Input validation

def validate_input(question, openai_api_key, gemini_api_key, anthropic_api_key):
    if (len(question) > 500
        or len(openai_api_key) > MAX_API_KEY_LENGTH
        or len(gemini_api_key) > MAX_API_KEY_LENGTH
        or len(anthropic_api_key) > MAX_API_KEY_LENGTH):
        return False

    sanitized_question = re.sub(r'[^\w\s.,!?\'\-()@$%&+/:;"=\[\]]', '', question)

    return sanitized_question == question
