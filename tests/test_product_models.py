from shared.product_models import get_product_model, load_product_models


def test_product_models_load():
    models = load_product_models()
    assert "byme-36parts" in models
    assert models["byme-36parts"].num_parts == 36


def test_product_model_lookup():
    model = get_product_model("byme-36parts")
    assert model.part_names["0"] == "part_00"
