import pickle
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Set style for professional looking plots
sns.set_theme(style="whitegrid")

def compare_models():
    models = ['lstm', 'gru', 'cnn']
    
    plt.figure(figsize=(12, 6))
    
    found_data = False
    
    # Colors for differentiation
    colors = {'lstm': 'blue', 'gru': 'green', 'cnn': 'red'}
    
    print("📊 Generating Combined Comparison Plot...")
    
    for model in models:
        filename = f"history_{model}.pkl"
        
        if os.path.exists(filename):
            found_data = True
            print(f"   Loading {filename}...")
            with open(filename, 'rb') as f:
                history = pickle.load(f)
            
            # Get Validation Loss (The most important metric for generalization)
            val_loss = history['val_loss']
            epochs = range(1, len(val_loss) + 1)
            
            # Plot
            plt.plot(epochs, val_loss, label=f'{model.upper()} Val Loss', color=colors[model], linewidth=2)
            
            # Optional: Plot Train loss as dashed line (if you want to see overfitting)
            # plt.plot(epochs, history['loss'], linestyle='--', color=colors[model], alpha=0.5)

    if not found_data:
        print("❌ No history files found. Run 'python src/train.py --model [name]' first.")
        return

    plt.title('Model Comparison: Validation Loss over Epochs', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    output_file = "model_comparison_v2.png"
    plt.savefig(output_file, dpi=300)
    print(f"✅ Combined plot saved to {output_file}")

if __name__ == "__main__":
    compare_models()