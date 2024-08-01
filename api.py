import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

results = {
    'Logistic Regression': {
        'class_0': {'precision': 0.79, 'recall': 0.64, 'f1-score': 0.71, 'support': 512},
        'class_1': {'precision': 0.96, 'recall': 0.98, 'f1-score': 0.97, 'support': 4436},
        'accuracy': 0.95,
        'macro avg': {'precision': 0.87, 'recall': 0.81, 'f1-score': 0.84, 'support': 4948},
        'weighted avg': {'precision': 0.94, 'recall': 0.95, 'f1-score': 0.94, 'support': 4948}
    },
    'Decision Tree': {
        'class_0': {'precision': 0.74, 'recall': 0.60, 'f1-score': 0.66, 'support': 512},
        'class_1': {'precision': 0.95, 'recall': 0.98, 'f1-score': 0.96, 'support': 4436},
        'accuracy': 0.94,
        'macro avg': {'precision': 0.85, 'recall': 0.79, 'f1-score': 0.81, 'support': 4948},
        'weighted avg': {'precision': 0.93, 'recall': 0.94, 'f1-score': 0.93, 'support': 4948}
    },
    'SVM': {
        'class_0': {'precision': 0.82, 'recall': 0.61, 'f1-score': 0.70, 'support': 512},
        'class_1': {'precision': 0.96, 'recall': 0.98, 'f1-score': 0.97, 'support': 4436},
        'accuracy': 0.95,
        'macro avg': {'precision': 0.89, 'recall': 0.80, 'f1-score': 0.84, 'support': 4948},
        'weighted avg': {'precision': 0.94, 'recall': 0.95, 'f1-score': 0.94, 'support': 4948}
    },
    'KNN': {
        'class_0': {'precision': 0.80, 'recall': 0.57, 'f1-score': 0.66, 'support': 512},
        'class_1': {'precision': 0.95, 'recall': 0.98, 'f1-score': 0.97, 'support': 4436},
        'accuracy': 0.94,
        'macro avg': {'precision': 0.87, 'recall': 0.77, 'f1-score': 0.81, 'support': 4948},
        'weighted avg': {'precision': 0.94, 'recall': 0.94, 'f1-score': 0.94, 'support': 4948}
    }
}


# Streamlit app
st.title('Classifier Performance Comparison')
st.write("""
This app demonstrates the performance of different classifiers using precision, recall, and F1-score metrics.
""")

metrics = ['precision', 'recall', 'f1-score']
classifiers = list(results.keys())
classes = ['class_0', 'class_1']

# Prepare data for grouped bar chart
def prepare_data(results, metrics, classifiers, classes):
    data = []
    for metric in metrics:
        for classifier in classifiers:
            for cls in classes:
                data.append({
                    'Classifier': classifier,
                    'Class': cls,
                    'Metric': metric,
                    'Value': results[classifier][cls][metric]
                })
    return pd.DataFrame(data)

# Convert results to DataFrame
df = prepare_data(results, metrics, classifiers, classes)

# Plotting function
def plot_grouped_bar_chart(df, metric):
    fig, ax = plt.subplots(figsize=(10, 6))
    for cls in classes:
        subset = df[(df['Metric'] == metric) & (df['Class'] == cls)]
        ax.bar(subset['Classifier'], subset['Value'], label=f'{cls} ({metric})')
    ax.set_title(f'{metric.capitalize()} Comparison')
    ax.set_ylabel(metric.capitalize())
    ax.legend()
    st.pyplot(fig)

# Display metrics
for metric in metrics:
    st.header(metric.capitalize())
    plot_grouped_bar_chart(df, metric)

# Display raw data (optional)
if st.checkbox('Show raw data'):
    st.subheader('Raw Data')
    st.write(pd.DataFrame(results))

