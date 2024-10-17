import base64
import io

import matplotlib

matplotlib.use('Agg')
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from flask import Flask, render_template
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def generate_fake_data():
    try:
        np.random.seed(42)
        n_products = 200
        names = [f"Product {i}" for i in range(1, n_products + 1)]
        prices = np.concatenate([
            np.random.normal(50, 10, 140),
            np.random.normal(100, 20, 40),
            np.random.normal(20, 5, 20)
        ])
        prices = np.clip(prices, 10, 200)
        features = np.column_stack((
            prices,
            np.random.rand(n_products) * 100,  # Random feature 1
            np.random.rand(n_products) * 50    # Random feature 2
        ))
        return pd.DataFrame(features, columns=['price', 'feature1', 'feature2'], index=names)
    except Exception as e:
        logger.error(f"Error generating fake data: {e}")
        return pd.DataFrame()

def create_pca_plot(df, n_components=2):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)
    
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_features)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=df['price'], cmap='viridis')
    plt.colorbar(label='Price')
    plt.title('PCA of Product Features')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance explained)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance explained)')
    return plt

def create_tsne_plot(df):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)
    
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(scaled_features)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=df['price'], cmap='viridis')
    plt.colorbar(label='Price')
    plt.title('t-SNE of Product Features')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    return plt

def create_kmeans_plot(df):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis')
    plt.colorbar(scatter, label='Cluster')
    plt.title('K-means Clustering of Products')
    plt.xlabel('PCA feature 1')
    plt.ylabel('PCA feature 2')
    return plt

def create_isolation_forest_plot(df):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)
    
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outlier_labels = iso_forest.fit_predict(scaled_features)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=outlier_labels, cmap='RdYlGn')
    plt.colorbar(label='Outlier (-1) vs. Inlier (1)')
    plt.title('Isolation Forest: Outlier Detection')
    plt.xlabel('PCA feature 1')
    plt.ylabel('PCA feature 2')
    return plt

def save_plot_to_base64(plt):
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return plot_data

@app.route('/')
def index():
    try:
        df = generate_fake_data()
        
        if df.empty:
            return render_template('index.html', error="Failed to generate data.")

        pca_plot = create_pca_plot(df)
        pca_plot_data = save_plot_to_base64(pca_plot)

        tsne_plot = create_tsne_plot(df)
        tsne_plot_data = save_plot_to_base64(tsne_plot)

        kmeans_plot = create_kmeans_plot(df)
        kmeans_plot_data = save_plot_to_base64(kmeans_plot)

        isolation_forest_plot = create_isolation_forest_plot(df)
        isolation_forest_plot_data = save_plot_to_base64(isolation_forest_plot)

        return render_template('index.html', 
                               pca_plot=pca_plot_data,
                               tsne_plot=tsne_plot_data,
                               kmeans_plot=kmeans_plot_data,
                               isolation_forest_plot=isolation_forest_plot_data)
    except Exception as e:
        logger.exception(f"An error occurred in the index route: {e}")
        return render_template('index.html', error="An unexpected error occurred.")

if __name__ == '__main__':
    app.run(debug=True)