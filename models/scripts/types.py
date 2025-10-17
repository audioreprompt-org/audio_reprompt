from dataclasses import dataclass
from torch.utils.data import Dataset
import json
import csv

# Tipos
@dataclass
class MusicGenData:
    """Estructura de datos para resultados de generación musical."""

    id: str
    instrument: str
    taste: str
    description: str
    audio_path: str



@dataclass
class MusicGenCLAPResult(MusicGenData):
    """Extiende MusicGenData para incluir el CLAP Score."""

    clap_score: float

class LoadSpanioDataset(Dataset):
    """
    Clase `Dataset` para cargar, transformar y exportar descripciones musicales
    del conjunto de datos de Spanio (`taste-music-dataset`).

    Esta clase permite leer un archivo JSON con estructura de columnas o lista
    de registros, convertirlo a una lista de diccionarios individuales,
    acceder a sus elementos por índice y exportarlos a CSV.

    Attributes:
        json_file_path (str): La ruta al archivo JSON que contiene los datos de Spanio.
    """

    def __init__(self, json_file_path):
        """
        Inicializa la clase `LoadSpanioDataset` cargando el contenido del archivo JSON.

        Parameters:
            son_file_path (str): La ruta al archivo JSON que contiene los datos de Spanio.
        """
        super().__init__()
        self.json_file_path = json_file_path
        self.records = []
        self._load_data()

    def _load_data(self):
        """
        Carga los datos del archivo JSON y normaliza su estructura.
        """
        try:
            with open(self.json_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Normalizar estructura (lista o dict).
                self.records = (
                    data
                    if isinstance(data, list)
                    else [{"id": k, **v} for k, v in data.items()]
                )
            print(f"Cargados {len(self.records)} registros desde {self.json_file_path}")
        except Exception as e:
            print(f"Error cargando {self.json_file_path}: {e}")

    def __len__(self):
        """
        Retorna el número total de registros (extractos) en el dataset.

        Returns:
            int: Número de registros disponibles.
        """
        return len(self.records)

    def __getitem__(self, idx):
        """
        Retorna un registro específico del dataset por índice.

        Parameters:
            idx (int): Índice del registro a retornar.

        Returns
            dict: Diccionario con las llaves `id`, `instrument` y `description`.
        """
        if not 0 <= idx < len(self.records):
            raise IndexError(
                f"indice {idx} fuera de rango para dataset: {len(self.records)}."
            )
        return self.records[idx]

    def map_records(self):
        """
        Mapea los registros de self.records() a un diccionario.

        Cada clave del diccionario es el 'id' del registro, y su valor es otro
        diccionario con el 'content' y 'title' del registro.

        Returns

        dict:
            - Diccionario donde las llaves son los `id` de los registros y los valores.
            - son diccionarios con `instrument` y `description`.

        """
        return {
            doc["id"]: {
                "instrument": doc["instrument"],
                "description": doc["description"],
            }
            for doc in self.records
        }

    def to_csv(self, output_path="spanio_prompts.csv"):
        """
        Exporta los registros del dataset a un archivo CSV con columnas:
        `id`, `instrument`, `description`.

        Parameters
            output_path (str): Ruta de salida donde se guardará el archivo CSV.

        """
        try:
            with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(
                    csvfile, fieldnames=["id", "instrument", "description"]
                )
                writer.writeheader()
                writer.writerows(self.records)
            print(f"CSV generado en: {output_path}")
        except Exception as e:
            print(f"Error exportando a CSV: {e}")

