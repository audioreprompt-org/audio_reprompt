from typing import Any

from models.food_prompts.typedefs import FoodItemAggCrossModal, FoodItemCrossModal


def parse_item_result(item: dict[str, Any]) -> FoodItemAggCrossModal | None:
    """
    assume that an item follows the shape:
    {custom_id},{food_item},{dimension_results},{model},{input_tokens},{output_tokens},{total_tokens}
    dimension results are ordered according to dimensions tuple
    """
    dimensions = (
        "chemical_flavor",
        "human_response",
        "temperature",
        "texture",
        "emotion",
        "color",
        "taste",
    )

    food_item = item["food_item"]
    food_captions = item["food_captions"]

    if food_captions == "No Label":
        return None

    dimension_results = [dim_values.split(",") for dim_values in food_captions.split("|")]
    return {"food_item": food_item, "values": list(zip(dimensions, dimension_results))}


def map_to_fooditem_crossmodal(result: dict[str, Any]) -> list[FoodItemCrossModal]:
    if food_item_agg:= parse_item_result(result):
        food_item = food_item_agg["food_item"]
        return [
            {"food_item": food_item, "dimension": dimension, "descriptor": val.strip().lower()}
            for dimension, values in food_item_agg["values"]
            for val in values
        ]

    return []
