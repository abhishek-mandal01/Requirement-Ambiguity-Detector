import matplotlib
matplotlib.use('Agg') # Headless mode for backend
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

def generate_bar_chart(metrics_a, metrics_b):
    """
    Generates a grouped bar chart for Mode A vs Mode B metrics and returns base64.
    """
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    a_vals = [metrics_a['accuracy'], metrics_a['precision'], metrics_a['recall'], metrics_a['f1_score']]
    b_vals = [metrics_b['accuracy'], metrics_b['precision'], metrics_b['recall'], metrics_b['f1_score']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width/2, a_vals, width, label='Mode A (Raw)', color='#a3b1c6')
    ax.bar(x + width/2, b_vals, width, label='Mode B (Preprocessed)', color='#4a90e2')
    
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_confusion_matrix_chart(cm, title):
    """
    Generates a visual confusion matrix and returns base64.
    """
    if cm is None:
        return ""
        
    fig, ax = plt.subplots(figsize=(4, 4))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='large')
            
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(title, pad=10)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Clear', 'Ambig'])
    ax.set_yticklabels(['Clear', 'Ambig'])
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
