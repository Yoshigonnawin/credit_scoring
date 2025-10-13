import pandas as pd
import re


def split_data(
    data: pd.DataFrame, split_column: str, ratio: float = 0.8
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    dates = pd.DataFrame(
        pd.to_datetime(data[split_column]).dt.date.unique(), columns=["dates"]
    )
    cutoff = str(dates.quantile(ratio).values[0])
    part_1 = data.query(f"{split_column}<'{cutoff}'")
    part_2 = data.query(f"{split_column}>='{cutoff}'")
    return part_1, part_2


def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    dat = data.copy()
    return dat.drop_duplicates().reset_index(drop=True)


def fill_missing_values(
    data: pd.DataFrame, num_cols: pd.Series | list[str], cat_cols: pd.Series | list[str]
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


def categorize_columns(data: pd.DataFrame) -> dict[str, list[str]]:
    """
    Разделяет колонки DataFrame на числовые, булевые и категориальные.

    Returns:
        dict с ключами 'numeric', 'boolean', 'categorical'
    """
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

    return {
        "numeric": numeric_cols,
        "boolean": boolean_cols,
        "categorical": categorical_cols,
    }


def remove_prefixes(columns: list[str]) -> list[str]:
    """Удаляет подстроки cats__.*, numscaler__.*, remainder__.*"""
    pattern = r"^(cats__|numscaler__|remainder__)"
    return [re.sub(pattern, "", col) for col in columns]
