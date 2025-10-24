import csv
import json
from typing import Any

from torch.utils.data import Dataset

from models.musicgen.typedefs import PromptItemLoader


def read_json(file_path: str) -> dict[str, Any] | list[dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def read_csv(file_path: str) -> list[list[str]]:
    with open(file_path, "r", encoding="utf-8") as f:
        csv_reader = csv.reader(f)
        next(csv_reader)

    return [row for row in csv_reader]


class DatasetLoader(Dataset):
    """
    Dataset para representar input data para el modelo `taste-music-dataset`.
    Attributes:
        file_path (str): La ruta al archivo JSON que contiene los datos de Spanio.
    """

    def __init__(self, file_path, format_: str = "json"):
        """
        son_file_path (str): Ruta del archivo input con prompts.
        """
        super().__init__()
        self.file_path = file_path
        self.format = format_
        self.records: list[PromptItemLoader] = []
        self._load_data(self.format_)

    def _load_data(self, file_format: str) -> None:
        read_functions = {
            "json": read_json,
            "csv": read_csv,
        }
        try:
            self.records = read_functions[file_format](self.file_path)
        except Exception as e:
            print(f"Error cargando {self.json_file_path}: {e}")

        if file_format == "csv":
            self.records = [
                {"id": record[0], "prompt": record[1]} for record in self.records
            ]

    def __len__(self) -> int:
        """
        Returns:
            int: Número de registros disponibles.
        """
        return len(self.records)

    def __getitem__(self, idx) -> PromptItemLoader:
        """
        Parameters:
            idx (int): Índice del registro a retornar.
        Returns
            El item del dataset en la posicion recibida por parametro.
        """
        if not 0 <= idx < len(self.records):
            raise IndexError(
                f"indice {idx} fuera de rango para dataset: {len(self.records)}."
            )
        return self.records[idx]
