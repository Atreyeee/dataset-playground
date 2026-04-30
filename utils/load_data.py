import pandas as pd


def load_csv(file) -> tuple:
    """
    Load a CSV file from a Streamlit UploadedFile object.

    Returns
    -------
    (DataFrame, None)  on success
    (None, error_str)  on failure
    """
    try:
        df = pd.read_csv(file)
        if df.empty:
            return None, "The file is empty."
        return df, None
    except pd.errors.ParserError as e:
        return None, f"CSV parsing error: {e}"
    except pd.errors.EmptyDataError:
        return None, "The file contains no data."
    except UnicodeDecodeError:
        # try latin-1 fallback
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding="latin-1")
            return df, None
        except Exception as e2:
            return None, f"Encoding error: {e2}"
    except Exception as e:
        return None, str(e)
