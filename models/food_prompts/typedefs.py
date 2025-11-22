from typing import TypedDict


class FoodItemCrossModal(TypedDict):
    food_item: str
    dimension: str
    descriptor: str


class FoodItemCrossModalEncoded(FoodItemCrossModal):
    text_embedding: list[float]


class FoodItemAggCrossModal(TypedDict):
    food_item: str
    values: list[tuple[str, list[str]]]
