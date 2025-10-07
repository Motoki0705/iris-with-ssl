from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load dataset
iris_df = pd.read_csv("iris.csv")

# Separate features and target variable
X = iris_df.drop("target", axis=1)
y = iris_df["target"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Standardize feature values for MLP stability
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and train MLP classifier with different parameters
mlp = MLPClassifier(
    hidden_layer_sizes=(10,),  # Simpler model
    solver='lbfgs',             # Often good for small datasets
    max_iter=1000,
    random_state=42,
)
mlp.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = mlp.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
