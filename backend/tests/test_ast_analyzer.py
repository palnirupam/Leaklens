from app.detectors.ast_analyzer import analyze_script


LEAKY = """
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv')
X = df.drop('y', axis=1)
y = df['y']
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
"""

CLEAN = """
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('data.csv')
X = df.drop('y', axis=1)
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
"""


def test_leaky_script_is_flagged():
    a = analyze_script(LEAKY)
    assert a.saw_split is True
    high = [f for f in a.findings if f.pattern == "preprocessing_before_split"]
    assert len(high) >= 1
    assert high[0].severity == "high"


def test_clean_script_no_preprocessing_flag():
    a = analyze_script(CLEAN)
    assert a.saw_split is True
    names = [f.pattern for f in a.findings]
    assert "preprocessing_before_split" not in names


def test_parse_error_surfaces():
    a = analyze_script("def broken(:")
    assert a.parse_error is not None
