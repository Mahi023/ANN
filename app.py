import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from Filter import hyperparameter_filters
from Connector import load_and_train_model
from ModelSummary import model_summary_to_df

st.set_page_config(page_title="Student Depression ANN Dashboard", layout="wide")

st.title("Student Depression with ANN 🏥")

# Get hyperparameter selections from sidebar
hyperparams = hyperparameter_filters()

# Load dataset and train model
model, history, X_test, y_test = load_and_train_model("Student Depression Dataset.csv", "Depression", hyperparams)

# Display model summary
st.write("### Model Summary")
summary_string = []
model.summary(print_fn=lambda x: summary_string.append(x), line_length=1000)
st.code("\n".join(summary_string), language="plaintext")

# Display training history
st.write("### Training History")

# Convert history to DataFrame
history_df = pd.DataFrame(history.history)
st.line_chart(history_df)

# Load test dataset for evaluation
st.write("### Model Evaluation")

# Get model predictions
y_pred = model.predict(X_test)

# Convert predictions to binary format
y_pred_binary = (y_pred > 0.5).astype(int)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_binary)

# Plot Confusion Matrix
st.write("### Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Student Depression", "Student Depression"], yticklabels=["No Student Depression", "Student Depression"])
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
st.pyplot(fig)

# Compute classification report
st.write("### Classification Report")
report = classification_report(y_test, y_pred_binary, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# Compute ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
st.write("### ROC Curve")
fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("Receiver Operating Characteristic (ROC) Curve")
ax_roc.legend(loc="lower right")
st.pyplot(fig_roc)