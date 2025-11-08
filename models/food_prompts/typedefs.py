from typing import TypedDict


class FoodItemCrossModal(TypedDict):
    food_item: str
    dimension: str
    value: str


class FoodItemCrossModalEncoded(FoodItemCrossModal):
    embeddings: list[float]


class FoodItemAggCrossModal(TypedDict):
    food_item: str
    values: list[tuple[str, list[str]]]
