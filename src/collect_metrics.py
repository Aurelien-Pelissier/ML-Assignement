import argparse
import tensorflow as tf
from pathlib import Path

from main import (
    Imagenet64,
    FragmentEncoder,
    nt_xent_loss,
    pairwise_bce_loss,
    evaluate_performances,
    EMBEDDING_DIM  # You can also replace with a CLI arg if needed
)

def main(data_path, model_checkpoint):
    data_path = Path(data_path)
    model_checkpoint = Path(model_checkpoint)

    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")
    if not model_checkpoint.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_checkpoint}")

    is_bce = "BCE" in model_checkpoint.name

    # Load model
    encoder = tf.keras.models.load_model(
        str(model_checkpoint),  
        custom_objects={"FragmentEncoder": FragmentEncoder}
    )

    # Loss function
    loss_func = pairwise_bce_loss if is_bce else nt_xent_loss
    
    # Load dataset
    dataset = Imagenet64(str(data_path))

    # Evaluate
    val_loss, val_mcc, val_auc, val_ari = evaluate_performances(
        dataset=dataset,
        encoder=encoder,
        loss_func=loss_func,
        sample_size=10,
        repeat=50, #sample 50 time 10 images to more robustly evaluate (note that max is repeat = 500)
        step="eval",
        plot_clusters=False,
        show_inline = True
    )

    # Output results
    print("\nEvaluation Results:")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  MCC: {val_mcc:.4f}, AUC: {val_auc:.4f}, Mean ARI: {val_ari:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained fragment encoder on validation data.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to validation data directory")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to saved model weights (.weights.h5)")

    args = parser.parse_args()
    main(args.data_path, args.model_checkpoint)
