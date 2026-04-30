import pandas as pd
import urllib.request

def load_csv(file) -> tuple:
    """Load a CSV file from a Streamlit UploadedFile object."""
    try:
        df = pd.read_csv(file)
        if df.empty:
            return None, "The file is empty."
        return df, None
    except Exception as e:
        return None, str(e)

def load_sample_dataset() -> tuple:
    """Loads the Titanic sample dataset."""
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    try:
        df = pd.read_csv(url)
        return df, None
    except Exception as e:
        return None, f"Failed to load sample data: {e}"
