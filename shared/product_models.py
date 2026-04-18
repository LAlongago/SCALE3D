from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from shared.schemas import ProductModelDefinition


def _product_model_dir() -> Path:
    return Path(__file__).resolve().parent / "product_models"


@lru_cache(maxsize=1)
def load_product_models() -> dict[str, ProductModelDefinition]:
    models: dict[str, ProductModelDefinition] = {}
    for path in sorted(_product_model_dir().glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = ProductModelDefinition.model_validate(payload)
        models[model.product_model_id] = model
    return models


def get_product_model(product_model_id: str) -> ProductModelDefinition:
    models = load_product_models()
    if product_model_id not in models:
        available = ", ".join(sorted(models))
        raise KeyError(f"Unknown product model '{product_model_id}'. Available: {available}")
    return models[product_model_id]
