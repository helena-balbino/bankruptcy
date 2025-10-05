from __future__ import annotations

from typing import Iterable, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


def _header(title: str) -> None:
    print(f"\n{title}\n" + "-" * 80)

def _ensure_target_in_data(data: pd.DataFrame, target: str) -> None:
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in data.columns.")

def _numeric_only(data: pd.DataFrame) -> pd.DataFrame:
    """Return numeric-only DataFrame, replacing inf/-inf with NaN."""

    num = data.select_dtypes(include=[np.number]).copy()
    return num.replace([np.inf, -np.inf], np.nan)

def _fd_bins(x: pd.Series) -> int:
    """Calculate number of bins using Freedman–Diaconis rule (fallback to sqrt)."""

    x = x.dropna().values
    n = x.size
    if n <= 1:
        return 1

    iqr = np.subtract(*np.percentile(x, [75, 25]))
    if iqr == 0:
        return min(30, max(1, int(np.sqrt(n))))

    h = 2 * iqr * (n ** (-1/3))
    bins = int(np.ceil((x.max() - x.min()) / h)) if h > 0 else min(30, max(1, int(np.sqrt(n))))
    return max(bins, 1)

def _format_title(s: str) -> str:
    return s.replace(".", " ").strip().title()

def _plot_hist_and_violin(data: pd.DataFrame, feature: str, target: str) -> None:
    """Plot histogram (left) and violinplot by class (right) for a numeric feature."""
    _ensure_target_in_data(data, target)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    x = data[feature].dropna()
    bins = _fd_bins(x)

    sns.histplot(x, kde=True, bins=bins, ax=axes[0])
    axes[0].set_title(f"Distribution of {_format_title(feature)}")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Frequency")

    sns.violinplot(x=data[target], y=data[feature], ax=axes[1], inner="box")
    axes[1].set_title(f"Violinplot of {_format_title(feature)} by {_format_title(target)}")
    axes[1].set_xlabel(_format_title(target))
    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.show()


# EDA functions

def text_normalization(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names: strip, lowercase, replace spaces with dots,
    remove special characters, and ensure uniqueness.
    """
    cols = (data.columns
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", ".", regex=True)
            .str.replace(r"[()%¥/]", "", regex=True)
            )

    seen = {}
    new_cols = []

    for c in cols:
      if c not in seen:
        seen[c] = 0
        new_cols.append(c)
      else:
        seen[c] += 1
        new_cols.append(f"{c}.{seen[c]}")

    data.columns = new_cols
    # return data


def basic_eda(data: pd.DataFrame, target: Optional[str] = None) -> Dict[str, object]:
    """
    Perform basic exploratory data analysis and return a dict of artifacts.
    """
    results: Dict[str, object] = {}

    _header("Basic Exploratory Data Analysis")
    data.info()

    results["shape"] = data.shape
    results["dtypes"] = data.dtypes
    results["missing"] = data.isna().sum().sort_values(ascending=False)
    results["duplicated_count"] = int(data.duplicated().sum())

    _header("Shape")
    print(results["shape"])

    _header("Dtypes")
    print(results["dtypes"])

    _header("Missing Values (per column)")
    print(results["missing"])

    _header("Duplicate Rows")
    print(results["duplicated_count"])

    if target is not None and target in data.columns:
        vc = data[target].value_counts(normalize=True, dropna=False).sort_index()
        results["target_distribution"] = vc
        _header(f"Target Distribution: {target}")
        print(vc)

    #return results


def statistical_eda(data: pd.DataFrame, include: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Compute statistical summary (describe) plus examples.
    """
    summary = pd.DataFrame({
        "Data Type": data.dtypes.astype(str),
        "Non-Null Count": data.notnull().sum(),
        "Null Count": data.isnull().sum(),
        "Unique Values": data.nunique()
    })

    numeric_cols = data.select_dtypes(include=["number"])

    for stat in ["Mean", "Median", "Mode", "Std", "Min", "Max", "IQR", "Skew"]:
        summary[stat] = "-"

    summary["Examples"] = data.apply(lambda col: ", ".join(map(str, col.dropna().unique()[:3])))

    for col in numeric_cols.columns:
        s = numeric_cols[col].dropna()
        if s.empty:
            continue
        summary.loc[col, "Mean"]   = f"{s.mean():.2f}"
        summary.loc[col, "Median"] = f"{s.median():.2f}"
        summary.loc[col, "Mode"]   = f"{s.mode().iloc[0]:.2f}" if not s.mode().empty else "-"
        summary.loc[col, "Std"]    = f"{s.std():.2f}"
        summary.loc[col, "Min"]    = f"{s.min():.2f}"
        summary.loc[col, "Max"]    = f"{s.max():.2f}"
        summary.loc[col, "IQR"]    = f"{s.quantile(0.75) - s.quantile(0.25):.2f}"
        summary.loc[col, "Skew"]   = f"{s.skew():.2f}"

    if include:
        # evita KeyError
        keep = [c for c in include if c in summary.index]
        summary = summary.loc[keep]

    return summary


def correlation_analysis(data: pd.DataFrame, annot: bool = False, figsize: Tuple[int, int] = (20, 16)) -> None:
    """
    Plot correlation heatmap for numeric variables with upper triangle masked.
    """
    num = _numeric_only(data)
    valid_cols = [c for c in num.columns if num[c].nunique(dropna=True) > 1]
    num = num[valid_cols]

    if num.shape[1] < 2:
        raise ValueError("Not enough numeric features with more than one unique value for correlation analysis.")

    corr = num.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr, mask=mask, vmin=-1, vmax=1, center=0,
        cmap="Blues", square=True, linewidths=.5,
        annot=annot, fmt=".2f", cbar_kws={"shrink": .8}
    )
    plt.title("Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def top_correlations(data: pd.DataFrame, target: str, n: int = 10, plot: bool = True) -> pd.Series:
    """
    List top correlations (positive and negative) with target column.
    Optionally plot histogram + violinplot for the top positively correlated features.
    """
    _ensure_target_in_data(data, target)
    num = _numeric_only(data)
    valid_cols = [c for c in num.columns if num[c].nunique(dropna=True) > 1]
    num = num[valid_cols]

    if target not in num.columns:
        raise ValueError(f"Target '{target}' must be numeric and have >1 unique value.")

    corr = num.corr(numeric_only=True)[target].dropna().sort_values(ascending=False)

    _header("Correlation Analysis")
    print(f"\nTop {n} POSITIVE correlations with '{target}':\n", corr.drop(labels=[target]).head(n))
    print(f"\nTop {n} NEGATIVE correlations with '{target}':\n", corr.drop(labels=[target]).tail(n))

    if plot:
        features = corr.index[corr.index != target][:n].tolist()
        for f in features:
            if f in data.columns:
                _plot_hist_and_violin(data, f, target)
    else:
        display(corr)

    return corr


def class_imbalance(data: pd.DataFrame, target: str, labels_map: Optional[Dict[object, str]] = None) -> None:
    """
    Plot target class distribution (countplot) with counts and percentages.
    """
    _ensure_target_in_data(data, target)
    x = data[target].map(labels_map) if labels_map else data[target]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x=x, ax=ax)

    total = x.dropna().shape[0]  # garante denominador consistente com o plot
    for p in ax.patches:
        height = p.get_height()
        percent = 100 * height / total if total else 0
        ax.text(
            p.get_x() + p.get_width() / 2.,
            height,
            f'{height} | {percent:.1f}%',
            ha='center', va='bottom', fontsize=11, color='black'
        )

    ax.set_title(f"Class Distribution: {_format_title(target)}", fontsize=16, fontweight='bold')
    ax.set_xlabel(_format_title(target))
    ax.set_ylabel("")
    ax.yaxis.set_ticks([])

    plt.tight_layout()
    plt.show()


def pairplots(data: pd.DataFrame, target: str, n: int = 6, kind: str = "scatter") -> None:
    """
    Create a pairplot for the N features most positively correlated with the target.
    kind: 'scatter' ou 'kde' (repasado a pairplot).
    """
    _ensure_target_in_data(data, target)
    num = _numeric_only(data)
    if target not in num.columns:
        raise ValueError(f"Target '{target}' must be numeric to select correlations.")

    corr = num.corr(numeric_only=True)[target].dropna().sort_values(ascending=False)
    features = [c for c in corr.index if c != target][:n]
    cols = features + [target]

    diag_kind = "kde" if kind == "kde" else "hist"
    g = sns.pairplot(data[cols].dropna(), hue=target, diag_kind=diag_kind, kind=kind)
    g.fig.suptitle("Pairplot (top correlated features)", y=1.02)

    plt.tight_layout()
    plt.show()


## Normalização e Escolanamento

def splitdata(df, seed, target, testsize=0.20, use_stratify=True):
    from sklearn.model_selection import train_test_split
    
    #Embaralha todas as amostras para evitar qualquer ordenação já existente
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    # Serpa as variaveis preditora da variavel alvo
    X = df.drop(columns=[target])
    y = df[target]

    # Determina se os dados serão separados garantindo a proporção da variavel alvo
    stratify_opt = y if use_stratify else None

    # Separa 80% dos dados para um conjunto de treino e 20% para o conjunto temporário
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=testsize,
        stratify=stratify_opt,
        random_state=seed
    )

    stratify_temp = y_temp if use_stratify else None

    # Separa novamente o conjunto temporário em 80% para validação e 20% para teste
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=testsize,
        stratify=stratify_temp,
        random_state=seed
    )

    # Retorna os conjunto de treino, validação e teste
    return X_train, X_val, X_test, y_train, y_val, y_test


def check_signal(data, exclude=None):
    """
    Check numeric columns to see if they are positive for log1p.

    Parameters
    ----------
    data : pd.DataFrame
    exclude : list, optional
        Columns to ignore (e.g. target)

    Returns
    -------
    dict
        {
          'positive': [cols with min >= 0],
          'positive_and_negative': {col: shift_value for min >= -1 and < 0},
          'negative': [cols with min < -1]
        }
    """
    if exclude is None:
        exclude = []
    # Seleciona colunas numéricas, excluindo a variavel alvo
    num = data.select_dtypes(include=[np.number]).drop(columns=exclude, errors='ignore')
    num = num.replace([np.inf, -np.inf], np.nan)

    # Inicializa ods indices
    positive, positive_and_negative, negative = [], {}, []

    # Identifica os valores minimos de cada coluna
    for c in num.columns:
        min_val = num[c].min(skipna=True)
        if min_val >= 0:
            positive.append(c)
        elif min_val >= -1:
            positive_and_negative[c] = -min_val + 1e-9
        else:
            negative.append(c)

    # Retornar os indices com o nomes das colunas com valores positivos, negativo e ambos
    return {'positive': positive, 'positive_and_negative': positive_and_negative, 'negative': negative}

def find_right_skewed(data, cols, skew_threshold=1.0, target=None, exclude=None, min_unique=2):
    """
    Return columns (from `cols`) that are right-skewed (skew >= threshold),
    excluding the target and any columns in `exclude`.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    cols : list
        Columns to test (candidates).
    skew_threshold : float, default=1.0
        Minimum skewness to consider right-skewed.
    target : str or None, default=None
        Target column name to exclude from consideration.
    exclude : list or None, default=None
        Additional columns to exclude.
    min_unique : int, default=2
        Keep only columns with at least this many distinct values.

    Returns
    -------
    list
        Subset of columns that are numeric, varying, not excluded,
        and have skew >= `skew_threshold`.
    """
    exclude_set = set(exclude or [])
    if target is not None:
        exclude_set.add(target)

    # Filtra colunas candidatas
    candidates = [c for c in cols if (c in data.columns) and (c not in exclude_set)]

    # Mantém apenas colunas com variedade mínima de valores
    num = data[candidates].select_dtypes(include=[np.number]).copy()
    num = num.replace([np.inf, -np.inf], np.nan)
    varying = [c for c in num.columns if num[c].nunique(dropna=True) >= min_unique]
    if not varying:
        return []

    # Calcula a assimetria das colunas candidatas
    skew = num[varying].skew(numeric_only=True)
    right_skewed = [c for c in varying if (pd.notna(skew.get(c)) and skew[c] >= skew_threshold)]

    # Retorna as colunas candidatas
    return right_skewed


def log1p_transform(X, cols):
    """
    Apply log1p transformation to a predefined set of skewed columns.

    Parameters
    ----------
    X : pandas.DataFrame
        Input DataFrame with numeric features.

    Returns
    -------
    pandas.DataFrame
        Transformed DataFrame.
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=X_train.columns)
    X = X.copy()
    for c in cols:
        X[c] = np.log1p(X[c].clip(lower=0))
    return X



