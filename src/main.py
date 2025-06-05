import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from tensorflow.keras import layers, models
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import pickle
import sys
import os
import random
import umap.umap_ as umap
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib
matplotlib.rcParams.update({'font.size': 18})

from data import Imagenet64
from data import normalize_img

# Constants
IMG_SIZE = 64
FRAG_PER_DIM = 4
FRAG_SIZE = IMG_SIZE // FRAG_PER_DIM
N_FRAGMENTS = FRAG_PER_DIM * FRAG_PER_DIM
TEMPERATURE = 0.1

EMBEDDING_DIM = 8

METRIC = 'euclidean'

METRIC = 'cosine'

#Euclidean vs Cosine



def evaluate_performances(dataset, encoder, loss_func, sample_size=10, repeat = 10, step = 0, plot_clusters = False, clust_metric = 'cosine', method = 'pca', show_inline = False):
    
    #A predictable generator for val estimation
    
    # Set up accumulators
    val_loss = 0
    val_ari = 0
    
    embs = []
    true_labs = []
    

    for i in range(repeat):
        print(f"Repeat {i+1}/{repeat}")
        
        fragments, labels = sample_and_fragment_val_images(dataset, sample_size=sample_size, seed=42 + i)
        frag_batch = tf.convert_to_tensor(fragments, dtype=tf.float32)
        label_batch = tf.convert_to_tensor(labels, dtype=tf.int32)

        embeddings = encoder(frag_batch)
        loss = loss_func(embeddings, label_batch, temp = TEMPERATURE, metric = METRIC)

        ari, pred_labels = evaluate_ari(embeddings, label_batch)

        val_loss += loss
        val_ari += ari

        true_labs += list(labels + i*sample_size) # different labels for each batch
        embs += list(embeddings)

        
    # Convert for visualization
    embs = np.stack([e.numpy() for e in embs])
    true_labs = np.array(true_labs)
    val_mcc, val_auc = compute_pair_metrics(embs, true_labs, plot_path="Figure/hist_plot-%s.png" % step, step = step, num_pairs=len(labels)*100, show_inline = show_inline)
    
    
    
    if plot_clusters:
    
        # Optional: visualize only the first 100 fragments sorted by label
        sorted_indices = np.argsort(true_labs)[:160] # First 10 images
        embs = embs[sorted_indices]
        true_labs = true_labs[sorted_indices]
        #pred_labs = pred_labs[sorted_indices]
        
    
        # Step 4: UMAP plot
        plot_path = "Figure/umap_true_labels_%s.png" % step
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        
        if method == 'pca':
            reducer = PCA(n_components=2)
            reduced = reducer.fit_transform(embs)
            xlabel, ylabel = "PC 1", "PC 2"
            title = f"PCA of Embeddings (ARI = {ari:.3f}), step = {step}"
        elif method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
            reduced = reducer.fit_transform(embs)
            xlabel, ylabel = "UMAP 1", "UMAP 2"
            title = f"UMAP of Embeddings (ARI = {ari:.3f}), step = {step}"
        else:
            raise ValueError(f"Unsupported method: {method}")
            
        plt.figure(figsize=(6, 5))
        plt.scatter(reduced[:, 0], reduced[:, 1], c=true_labs, s=20, alpha=0.7, cmap='tab10')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(plot_path)
        if show_inline:
            plt.show()
        else:
            plt.close()
        
    
        
        from scipy.spatial.distance import pdist, squareform
        
        if clust_metric == 'euclidean':
            dist_matrix = squareform(pdist(embs, metric='euclidean'))
        else:
            dist_matrix = squareform(pdist(embs, metric='cosine'))
            
        dist_matrix = dist_matrix - np.min(dist_matrix)
        sim_matrix = 1 - dist_matrix/np.max(dist_matrix)
        
        sim_matrix[sim_matrix<0.5] = 0.5
            
            
        title = f"Similarity matrix for 100 fragments (ARI = {ari:.3f}), step = {step}"    
        plot_path = f"Figure/imshow_sim.{step}.png"
        
        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[20, 1], hspace=0.1)
        
        # Main similarity matrix
        ax0 = plt.subplot(gs[0])
        im = ax0.imshow(sim_matrix, origin='lower', cmap='inferno')
        ax0.set_title(title, fontsize = 15)
        ax0.set_xticks([])
        ax0.set_yticks([])
        #fig.colorbar(im, ax=ax0)
        
        # Label color bar
        ax1 = plt.subplot(gs[1], sharex=ax0)
        ax1.imshow(true_labs[np.newaxis, :], aspect='auto', cmap='tab10', extent=[0, len(true_labs), 0, 1])
        ax1.set_yticks([])
        ax1.set_xticks([])
        
        plt.savefig(plot_path)
        if show_inline:
            plt.show()
        else:
            plt.close()
        
        
        plot_path = f"Figure/imshow_sim_clustered.{step}.png"
        k = len(np.unique(true_labs))  # or set to known number of original images
        cluster_labels = KMeans(n_clusters=k, n_init=10).fit_predict(embs)
        
        sorted_indices = np.argsort(cluster_labels)
        sim_matrix_sorted = sim_matrix[sorted_indices][:, sorted_indices]
        true_labs_sorted = true_labs[sorted_indices]
        fig = plt.figure(figsize=(8, 8), constrained_layout=True)
        gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])
        sim_matrix_sorted[sim_matrix_sorted < 0.5] = 0.5
        

        ax0 = plt.subplot(gs[0])
        im = ax0.imshow(sim_matrix_sorted, origin='lower', cmap='inferno')
        ax0.set_title(title)
        ax0.set_xticks([])
        ax0.set_yticks([])

        ax1 = plt.subplot(gs[1], sharex=ax0)
        ax1.imshow(true_labs_sorted[np.newaxis, :], aspect='auto', cmap='tab10', extent=[0, len(true_labs_sorted), 0, 1])
        ax1.set_yticks([])
        ax1.set_xticks([])
        
        plt.savefig(plot_path)
        if show_inline:
            plt.show()
        else:
            plt.close()
        
        
        
        
    return val_loss/repeat, val_mcc, val_auc, val_ari/repeat





def get_all_val_images(dataset):
    x_val = normalize_img(dataset.data["x_val"])
    y_val = dataset.data["y_val"]
    return x_val, y_val


def sample_and_fragment_val_images(dataset, sample_size=5000, seed=42):
    x_val, y_val = get_all_val_images(dataset)
    assert sample_size <= len(x_val)

    np.random.seed(seed)
    indices = np.random.choice(len(x_val), size=sample_size, replace=False)

    indices_tf = tf.convert_to_tensor(indices, dtype=tf.int32)
    sampled_imgs = tf.gather(x_val, indices_tf)

    all_fragments = []
    fragment_labels = []

    for idx in range(sample_size):
        img = sampled_imgs[idx]
        fragments = fragment_image(img)
        unique_label = idx  # ← unique label per image
        fragment_labels.extend([unique_label] * len(fragments))
        all_fragments.extend(fragments)

    all_fragments = np.array(all_fragments)
    fragment_labels = np.array(fragment_labels)

    perm = np.random.permutation(len(all_fragments))
    return all_fragments[perm], fragment_labels[perm]

          


def fragment_image(image):
    fragments = []
    for i in range(0, IMG_SIZE, FRAG_SIZE):
        for j in range(0, IMG_SIZE, FRAG_SIZE):
            fragment = image[i:i+FRAG_SIZE, j:j+FRAG_SIZE, :]
            fragments.append(fragment)
    return fragments


def sample_and_fragment_batch_from_generator(generator, sample_size=10):
    # Use generator to fetch a single batch large enough to include sample_size full images
    x_batch, _ = next(generator)
    #x_batch = x_batch[:sample_size]

    all_fragments = []
    fragment_labels = []

    for idx, img in enumerate(x_batch):
        fragments = fragment_image(img)
        all_fragments.extend(fragments)
        fragment_labels.extend([idx] * len(fragments))

    all_fragments = np.array(all_fragments)
    fragment_labels = np.array(fragment_labels)

    perm = np.random.permutation(len(all_fragments))
    return all_fragments[perm], fragment_labels[perm]

@tf.keras.utils.register_keras_serializable()
class FragmentEncoder(tf.keras.Model):
    def __init__(self, embedding_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.encoder = tf.keras.Sequential([
            layers.Conv2D(32, kernel_size=3, strides=1, padding='valid', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, kernel_size=2, strides=1, activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(embedding_dim)
        ])
        self.embedding_dim = embedding_dim  # store this for config

    def call(self, x):
        return self.encoder(x)

    def get_config(self):
        config = super().get_config()
        config.update({"embedding_dim": self.embedding_dim})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    
    
def evaluate_ari(embeddings, true_labels, constrained_Kmeans=False): 
    

    n_clusters = len(np.unique(true_labels))
    embeddings = np.asarray(embeddings)
    true_labels = np.asarray(true_labels)

    if constrained_Kmeans:
        from sklearn_extra.cluster import KMeansConstrained
        
        # Compute balanced size
        total_samples = len(true_labels)
        size_per_cluster = total_samples // n_clusters
        remainder = total_samples % n_clusters

        # e.g., [5,5,5,6] if 21 samples, 4 clusters
        size_constraints = [size_per_cluster + 1] * remainder + \
                           [size_per_cluster] * (n_clusters - remainder)

        pred_labels = KMeansConstrained(
            n_clusters=n_clusters,
            size_min=min(size_constraints),
            size_max=max(size_constraints),
            random_state=0
        ).fit_predict(embeddings)
    else:
        pred_labels = KMeans(
            n_clusters=n_clusters,
            n_init="auto",  # for scikit-learn ≥1.4; use n_init=10 otherwise
            random_state=0
        ).fit_predict(embeddings)

    ari = adjusted_rand_score(true_labels, pred_labels)
    return ari, pred_labels




def compute_pair_metrics(embeddings, labels, num_pairs=None, plot_path="Figure/hist_plot.png", step = 0, show_inline = False):
    # Normalize embeddings
    
    labels = labels.numpy() if hasattr(labels, 'numpy') else labels
    embeddings = tf.math.l2_normalize(embeddings, axis=1)
    embeddings_np = embeddings.numpy()
    N = len(labels)

    if num_pairs is None:
        num_pairs = N  # default: N pairs total (N/2 pos, N/2 neg)

    # Create lookup table: label → list of indices
    label_to_indices = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(idx)

    # Sample positive pairs
    pos_pairs = []
    for label, idxs in label_to_indices.items():
        if len(idxs) < 2:
            continue
        sampled = [tuple(random.sample(idxs, 2)) for _ in range(min(len(idxs) // 2, num_pairs // 2))]
        pos_pairs.extend(sampled)
        if len(pos_pairs) >= num_pairs // 2:
            break
    pos_pairs = pos_pairs[:num_pairs // 2]

    # Sample negative pairs
    neg_pairs = []
    labels_list = list(label_to_indices.keys())
    while len(neg_pairs) < num_pairs // 2:
        l1, l2 = random.sample(labels_list, 2)
        i = random.choice(label_to_indices[l1])
        j = random.choice(label_to_indices[l2])
        neg_pairs.append((i, j))

    # Compute cosine similarities and map to [0, 1]
    pos_scores = [np.dot(embeddings_np[i], embeddings_np[j]) for i, j in pos_pairs]
    neg_scores = [np.dot(embeddings_np[i], embeddings_np[j]) for i, j in neg_pairs]
    
    # Combine
    sim_scores = np.array(pos_scores + neg_scores)
    pair_labels = np.array([1] * len(pos_scores) + [0] * len(neg_scores))

    # Metrics
    thresholds = np.linspace(0, 1, 25)
    mcc_scores = [matthews_corrcoef(pair_labels, sim_scores > t) for t in thresholds]
    best_mcc = max(mcc_scores)

    try:
        auc = roc_auc_score(pair_labels, sim_scores)
    except ValueError:
        auc = float('nan')

    # Plot histogram
    bins = np.linspace(-1,1,50)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.hist(pos_scores, bins=bins, alpha=0.45, label='Same Image', color='green', density = True)
    plt.hist(neg_scores, bins=bins, alpha=0.45, label='Different Image', color='red', density = True)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("PDF")
    plt.title("Pairwise Similarity of step %s (AUC: %.2f, MCC: %.2f)" % (step, auc, best_mcc), fontsize = 13)
    plt.legend(fontsize = 13)
    #plt.tight_layout()
    plt.savefig(plot_path)
    if show_inline:
        plt.show()
    else:
        plt.close()

    return best_mcc, auc


def nt_xent_loss(embeddings, labels, temp = 0.1, metric='cosine'):
    if metric == 'cosine':
        # L2 normalize for cosine similarity
        embeddings = tf.math.l2_normalize(embeddings, axis=1)
        logits = tf.matmul(embeddings, embeddings, transpose_b=True) / temp
    elif metric == 'euclidean':
        # Compute pairwise Euclidean distances (squared)
        # dist(i, j)^2 = ||ei - ej||^2 = ||ei||^2 + ||ej||^2 - 2*ei·ej
        dot_product = tf.matmul(embeddings, embeddings, transpose_b=True)
        square_norm = tf.reduce_sum(tf.square(embeddings), axis=1, keepdims=True)
        dist_squared = square_norm - 2 * dot_product + tf.transpose(square_norm)
        # Convert distances to similarities (negative distance for contrastive objective)
        logits = -dist_squared / temp
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    labels = tf.cast(labels, dtype=tf.int32)
    mask = tf.equal(tf.expand_dims(labels, 1), tf.expand_dims(labels, 0))
    logits_mask = tf.ones_like(mask, dtype=tf.float32) - tf.eye(tf.shape(labels)[0])
    positives_mask = tf.cast(mask, tf.float32) * logits_mask

    exp_logits = tf.exp(logits) * logits_mask
    log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-10)

    mean_log_prob_pos = tf.reduce_sum(positives_mask * log_prob, axis=1) / tf.reduce_sum(positives_mask, axis=1)
    loss = -mean_log_prob_pos
    return tf.reduce_mean(loss)



def pairwise_bce_loss(embeddings, labels, pos_weight=9, neg_weight=1, temp = None, metric = None):
    """
    Compute binary cross-entropy loss on all fragment pairs.

    Args:
        embeddings: Tensor of shape (N, D) where N = num fragments, D = embedding dim.
        labels: Tensor of shape (N,) with integer labels indicating image ID (0 to sample_size-1).
        pos_weight: Weight for positive (same-image) pairs.
        neg_weight: Weight for negative (different-image) pairs.

    Returns:
        Scalar weighted binary cross-entropy loss.
    """
    # Normalize embeddings
    embeddings = tf.math.l2_normalize(embeddings, axis=1)

    # Compute pairwise cosine similarities
    sim_matrix = tf.matmul(embeddings, embeddings, transpose_b=True)  # (N, N)

    # Build pairwise label matrix: 1 if same class, 0 otherwise
    label_matrix = tf.cast(tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1)), tf.float32)

    # Binary targets and logits
    targets = tf.reshape(label_matrix, [-1])  # (N*N,)
    logits = tf.reshape(sim_matrix, [-1])     # cosine similarity as logits

    # Compute weights per pair
    weights = tf.where(targets > 0.5,
                   tf.constant(pos_weight, dtype=tf.float32),
                   tf.constant(neg_weight, dtype=tf.float32))

    # Binary cross entropy (from_logits=False since cosine similarity is [-1, 1])
    bce = tf.keras.losses.binary_crossentropy(targets, (logits + 1.0) / 2.0)  # map to [0,1]
    weighted_loss = tf.reduce_mean(weights * bce)
    
    return weighted_loss






    
    
    






# === Script 1: Training with nt_xent ===
def train_model_CL(
    data_path,
    checkpoint_folder,
    total_steps=1000,          # Total number of training steps
    sample_size=10,             # Number of full images sampled per batch
    embedding_dim=16,           # Dimensionality of the fragment embeddings
    patience=1000,              # Number of steps to wait without improvement before early stopping
    eval_interval=50           # Evaluate model every N steps
):
    # Load the dataset using the Imagenet64 wrapper
    dataset = Imagenet64(data_path)

    # Initialize data generators for training and validation
    train_gen = dataset.datagen_cls(batch_size=sample_size, ds="train", augmentation=False)
    #val_gen = dataset.datagen_cls(batch_size=sample_size, ds="val", augmentation=False)

    # Initialize the encoder model that maps fragments to embeddings
    encoder = FragmentEncoder(embedding_dim=embedding_dim)

    # Use the Adam optimizer for training
    optimizer = tf.keras.optimizers.Adam(1e-3)

    # Setup for early stopping
    best_val_ari = -1                   # Track the best validation ARI achieved
    best_val_loss = float('inf')
    steps_without_improvement = 0       # Counter for early stopping
    
    
    store_train_loss = []
    store_val_loss = []
    store_val_mcc = []
    store_val_auc = []
    store_val_ari = []
    x_step = []
    

    # Training loop across total steps
    loss_history = []
    for step in tqdm(range(1, total_steps + 1), desc="Training"):
        fragments, labels = sample_and_fragment_batch_from_generator(train_gen)
        fragments = tf.convert_to_tensor(fragments, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)

        with tf.GradientTape() as tape:
            embeddings = encoder(fragments)
            loss = nt_xent_loss(embeddings, labels, temp = TEMPERATURE, metric = METRIC)

        grads = tape.gradient(loss, encoder.trainable_weights)
        optimizer.apply_gradients(zip(grads, encoder.trainable_weights))
        loss_history.append(float(loss.numpy()))

        # Periodic evaluation
        if step % eval_interval == 0 or step == 1:
            
            # Compute and print average training loss for the last interval
            avg_loss = sum(loss_history) / len(loss_history)
            print(f"\nStep {step}: Avg training loss = {avg_loss:.4f}")
            loss_history = [] #Rinitialize the training loss
            
            print(f"Step {step}: Evaluating val metrics...")
            
            val_loss, val_mcc, val_auc, val_ari = evaluate_performances(dataset, encoder, nt_xent_loss, sample_size=10, repeat = 5, step = step)
            
            print(f"Step {step}: Validation loss = {val_loss:.4f}")
            print(f"Step {step}: Validation MCC = {val_mcc:.4f}, AUC = {val_auc:.4f}")
            print(f"Step {step}: Validation ARI = {val_ari:.4f}")

            # Check if the model improved
            if val_ari > best_val_ari or val_loss < best_val_loss:
                best_val_ari = max(val_ari,best_val_ari)
                best_val_loss = min(best_val_loss, val_loss)
                steps_without_improvement = 0  # Reset early stopping counter

                # Save the best model
                checkpoint_path = f"{checkpoint_folder}/best_model_CL_%s.weights.h5" % EMBEDDING_DIM
                encoder.save_weights(checkpoint_path)
                print(f"Improved ARI or val loss. Model saved to {checkpoint_path}")
            else:
                # No improvement, increment counter
                steps_without_improvement += eval_interval
                print(f"No improvement for {steps_without_improvement} steps.")

            # Early stopping condition
            if steps_without_improvement >= patience:
                print("Early stopping triggered.")
                break
            
            print()
            
            
            
            x_step.append(step)
            store_train_loss.append(avg_loss)
            store_val_loss.append(val_loss)
            store_val_mcc.append(val_mcc)
            store_val_auc.append(val_auc)
            store_val_ari.append(val_ari)
            
    
    
    #Save final encoder if we reach the end
    #encoder.save(checkpoint_path)
    
    
    plt.plot(x_step, store_train_loss, label = 'Train_loss')
    plt.plot(x_step, store_val_loss, label = 'Val_loss')
    plt.ylabel('XT-Ent Loss')
    plt.xlabel('Training Steps')
    plt.legend()
    plt.savefig('Loss_progression_CL')
    plt.close()
    
    
    plt.plot(x_step, store_val_ari, label = 'ARI')
    plt.plot(x_step, store_val_mcc, label = 'MCC')
    plt.ylabel('Metric value')
    plt.xlabel('Training Steps')
    plt.legend()
    plt.savefig('ARI_MCC_progression_CL')
    plt.close()
    
    
    plt.plot(x_step, store_val_auc, label = 'AUC')
    plt.ylabel('Metric value')
    plt.xlabel('Training Steps')
    plt.legend()
    plt.savefig('AUC_progression_CL')
    plt.close()




    
    
# === Script 1: Training with CE ===
def train_model_BCE(
    data_path,
    checkpoint_folder,
    total_steps=1000,          # Total number of training steps
    sample_size=10,             # Number of full images sampled per batch
    embedding_dim=16,           # Dimensionality of the fragment embeddings
    patience=200000000000,              # Number of steps to wait without improvement before early stopping
    eval_interval=50           # Evaluate model every N steps
):
    dataset = Imagenet64(data_path)

    # Initialize data generators
    train_gen = dataset.datagen_cls(batch_size=sample_size, ds="train", augmentation=False)
    #val_gen = dataset.datagen_cls(batch_size=sample_size, ds="val", augmentation=False)

    # Define encoder only
    encoder = FragmentEncoder(embedding_dim=embedding_dim)
    optimizer = tf.keras.optimizers.Adam(1e-3)

    best_val_ari = -1
    best_val_loss = float('inf')
    steps_without_improvement = 0
    loss_history = []
    
    
    store_train_loss = []
    store_val_loss = []
    store_val_mcc = []
    store_val_auc = []
    store_val_ari = []
    x_step = []

    for step in tqdm(range(1, total_steps + 1), desc="Training"):
        fragments, labels = sample_and_fragment_batch_from_generator(train_gen)
        fragments = tf.convert_to_tensor(fragments, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)

        with tf.GradientTape() as tape:
            embeddings = encoder(fragments)
            loss = pairwise_bce_loss(embeddings, labels, pos_weight=10.0, neg_weight=1.0)

        grads = tape.gradient(loss, encoder.trainable_weights)
        optimizer.apply_gradients(zip(grads, encoder.trainable_weights))
        loss_history.append(float(loss.numpy()))
        
        

    

        # Periodic evaluation
        if step % eval_interval == 0 or step == 1:
            
            # Compute and print average training loss for the last interval
            avg_loss = sum(loss_history) / len(loss_history)
            print(f"\nStep {step}: Avg training loss = {avg_loss:.4f}")
            loss_history = [] #Rinitialize the training loss
            
            print(f"Step {step}: Evaluating val metrics...")
            
            val_loss, val_mcc, val_auc, val_ari = evaluate_performances(dataset, encoder, pairwise_bce_loss, sample_size=10, repeat = 5, step = step)
            
            print(f"Step {step}: Validation loss = {val_loss:.4f}")
            print(f"Step {step}: Validation MCC = {val_mcc:.4f}, AUC = {val_auc:.4f}")
            print(f"Step {step}: Validation ARI = {val_ari:.4f}")

            # Check if the model improved
            if val_ari > best_val_ari or val_loss < best_val_loss:
                best_val_ari = max(val_ari,best_val_ari)
                best_val_loss = min(best_val_loss, val_loss)
                steps_without_improvement = 0  # Reset early stopping counter

                # Save the best model
                checkpoint_path = f"{checkpoint_folder}/best_model_BCE_%s.weights.h5" % EMBEDDING_DIM
                encoder.save_weights(checkpoint_path)
                print(f"Improved ARI or val loss. Model saved to {checkpoint_path}")
            else:
                # No improvement, increment counter
                steps_without_improvement += eval_interval
                print(f"No improvement for {steps_without_improvement} steps.")

            # Early stopping condition
            if steps_without_improvement >= patience:
                print("Early stopping triggered.")
                break
            
            print()
            
            
            
            x_step.append(step)
            store_train_loss.append(avg_loss)
            store_val_loss.append(val_loss)
            store_val_mcc.append(val_mcc)
            store_val_auc.append(val_auc)
            store_val_ari.append(val_ari)
            
            
    
    
    plt.plot(x_step, store_train_loss, label = 'Train_loss')
    plt.plot(x_step, store_val_loss, label = 'Val_loss')
    plt.ylabel('XT-Ent Loss')
    plt.xlabel('Training Steps')
    plt.legend()
    plt.savefig('Loss_progression_BCE')
    plt.close()
    
    
    plt.plot(x_step, store_val_ari, label = 'ARI')
    plt.plot(x_step, store_val_mcc, label = 'MCC')
    plt.ylabel('Metric value')
    plt.xlabel('Training Steps')
    plt.legend()
    plt.savefig('ARI_MCC_progression_BCE')
    plt.close()
    
    
    plt.plot(x_step, store_val_auc, label = 'AUC')
    plt.ylabel('Metric value')
    plt.xlabel('Training Steps')
    plt.legend()
    plt.savefig('AUC_progression_BCE')
    plt.close()









if __name__ == "__main__":
    mode = "metrics"  # Choose from: "train_CL", "train_BCE", "metrics", "visualize"
    
    mode = "visualize"
    
    #mode = "visualize"
    
    #mode = "train_CL"
    
    # Editable parameters
    data_path = "data"
    model_checkpoint = "model/best_model_CL_%s.weights.h5" % EMBEDDING_DIM # or "model/best_model_BCE.keras"
    
    if mode == "train_BCE":
        train_model_BCE(data_path, "model", embedding_dim=EMBEDDING_DIM)
    
    elif mode == "train_CL":
        train_model_CL(data_path, "model", embedding_dim=EMBEDDING_DIM)
    
    elif mode in {"metrics", "visualize"}:
        # Load data and model
        dataset = Imagenet64(data_path)
        encoder = FragmentEncoder(embedding_dim=EMBEDDING_DIM)
        encoder.build(input_shape=(None, 16, 16, 3))
        encoder.load_weights(model_checkpoint)

        # Determine appropriate loss function
        if "BCE" in model_checkpoint:
            loss_func = pairwise_bce_loss
        else:
            loss_func = nt_xent_loss

        # Evaluation configuration
        if mode == "metrics":
            print("Running metrics collection on large validation set...")
            val_loss, val_mcc, val_auc, val_ari = evaluate_performances(
                dataset=dataset,
                encoder=encoder,
                loss_func=loss_func,
                sample_size=10,
                repeat=5,  #sample 25 time 10 images to more robustly evaluate 
                step="final",
                plot_clusters=False,
                show_inline = True
            )
        elif mode == "visualize":
            print("Generating visualizations on small batch...")
            val_loss, val_mcc, val_auc, val_ari = evaluate_performances(
                dataset=dataset,
                encoder=encoder,
                loss_func=loss_func,
                sample_size=10,
                repeat=1,  #only sample 10 images a single time to visualize
                step="viz",
                plot_clusters=True,
                show_inline = True
            )

        print("\nEvaluation Results:")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"MCC: {val_mcc:.4f}, AUC: {val_auc:.4f}, ARI: {val_ari:.4f}")
    
    else:
        raise ValueError("Unknown mode. Use 'train_CL', 'train_BCE', 'metrics', or 'visualize'.")