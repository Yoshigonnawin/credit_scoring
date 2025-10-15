from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import re
import pandas as pd


class Preprocessor:
    column_transformer: ColumnTransformer
    categorized_columns: dict[str, list[str]]
    transformer_params: dict[str, list]

    def __init__(self, column_transformer: ColumnTransformer) -> None:
        self.column_transformer = column_transformer

    def _categorize_columns(self, data: pd.DataFrame) -> None:
        numeric_cols = []
        boolean_cols = []
        categorical_cols = []

        for col in data.columns:
            # Булевые (bool или содержат только 0/1, True/False)
            if data[col].dtype == "bool":
                boolean_cols.append(col)
            # Числовые (int, float)
            elif pd.api.types.is_numeric_dtype(data[col]):
                # Проверяем, не является ли числовая колонка на самом деле булевой
                unique_vals = data[col].dropna().unique()
                if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
                    boolean_cols.append(col)
                else:
                    numeric_cols.append(col)
            # Категориальные (object, category, datetime)
            else:
                categorical_cols.append(col)

        self.categorized_columns = {
            "numeric": numeric_cols,
            "boolean": boolean_cols,
            "categorical": categorical_cols,
        }

    @staticmethod
    def _fill_missing_values(
        data: pd.DataFrame, num_cols: list[str], cat_cols: list[str]
    ) -> pd.DataFrame:
        data = data.copy()

        for col in num_cols:
            if data[col].isna().any():
                data[col].fillna(data[col].mean(), inplace=True)

        for col in cat_cols:
            if data[col].isna().any():
                mode_value = data[col].mode()
                if not mode_value.empty:
                    data[col].fillna(mode_value.iloc[0], inplace=True)

        return data

    @staticmethod
    def _remove_prefixes(columns: list[str]) -> list[str]:
        pattern = r"^(cats__|numscaler__|remainder__)"
        return [re.sub(pattern, "", col) for col in columns]

    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        self._categorize_columns(data)

        step_1 = self._fill_missing_values(
            data,
            self.categorized_columns.get("numeric"),
            self.categorized_columns.get("categorical"),
        )
        step_1["def_45"] = [None for _ in range(data.shape[0])]
        step_1["application_datetime"] = [None for _ in range(data.shape[0])]

        step_2_array = self.column_transformer.transform(step_1)
        step_2 = pd.DataFrame(
            step_2_array,
            columns=self._remove_prefixes(
                self.column_transformer.get_feature_names_out()
            ),
        )

        return step_2.drop(columns=["def_45", "application_datetime"])

    def get_column_dtypes(self) -> dict[str, str]:
        """
        Возвращает маппинг колонка -> тип данных из ColumnTransformer.

        Returns:
            Словарь {col_name: dtype}
        """
        column_dtypes = {}

        for name, transformer, columns in self.column_transformer.transformers_:
            if name == "remainder":
                continue

            col_list = (
                list(columns) if isinstance(columns, (list, tuple)) else [columns]
            )

            # Определяем dtype по типу трансформера
            if isinstance(transformer, OrdinalEncoder):
                dtype = "categorical"
            elif isinstance(transformer, StandardScaler):
                dtype = "numeric"
            else:
                dtype = type(transformer).__name__

            for col in col_list:
                column_dtypes[col] = dtype

        return column_dtypes
