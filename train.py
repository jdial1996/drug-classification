import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import (
    ColumnTransformer,
)  # lets us link different pipelines together

# Encoder is to used to encode categorical data
from sklearn.preprocessing import (
    StandardScaler,
    OrdinalEncoder,
)  # preproccessing = cleaning the data / preparing it for the model
from sklearn.impute import SimpleImputer  # handle missing data
from sklearn.ensemble import RandomForestClassifier  # the models
from sklearn.metrics import accuracy_score, f1_score  # test/evaluate model
import skops.io as skops


# Load Data
drug_df = pd.read_csv("./datasets/drug.csv")

# Shuffle Data

drug_df = drug_df.sample(frac=1)

# # Independent variable
X = drug_df.drop("Drug", axis=1).values

# # Dependent variable
Y = drug_df.Drug.values

# # This will return 4 datasets.  X_train (70% of X), X_test (30% of X) Y_train (70% of Y), Y_test (30% of Y)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=125
)

numerical_columns = [0, 4]
categorical_columns = [1, 2, 3]


# Use pipeline to apply sequential steps. Use column transformer to apply steps in parallel

transform = ColumnTransformer(
    [
        ("encoder", OrdinalEncoder(), categorical_columns),
        ("num_imputer", SimpleImputer(strategy="median"), numerical_columns),
        ("num_scaler", StandardScaler(), numerical_columns),
    ]
)

pipeline = Pipeline(
    [
        ("preprocessor", transform),
        (
            "model",
            RandomForestClassifier(n_estimators=100, random_state=125),
        ),  # data will end up here
    ]
)

pipeline.fit(X_train, Y_train)

predictions = pipeline.predict(X_test)

accuracy = accuracy_score(Y_test, predictions)
f1 = f1_score(Y_test, predictions, average="macro")

model_metrics = {
    "accuracy_score": f"{str(round(accuracy, 2) * 100)}%",
    "F1_score": round(f1, 2),
}

with open("./results/metrics.json", "w") as f:
    json.dump(model_metrics, f)

skops.dump(pipeline, "model/drug_pipeline.skops")
