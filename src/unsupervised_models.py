"""
Unsupervised Learning Models for Flash Flood Risk Analysis
Addresses SDG 13 (Climate Action) and SDG 11 (Sustainable Cities and Communities)
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
import joblib

class UnsupervisedFloodAnalyzer:
    """
    Unsupervised learning system for flood risk analysis and town clustering
    """
    
    def __init__(self):
        self.kmeans_model = None
        self.pca_model = None
        self.scaler = StandardScaler()
        self.cluster_labels = None
        self.pca_components = None
        self.cluster_centers = None
        self.feature_names = None
        
    def perform_pca_analysis(self, X, feature_names, n_components=None, explained_variance_threshold=0.95):
        """
        Perform Principal Component Analysis for dimensionality reduction
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
            n_components: Number of components (if None, use variance threshold)
            explained_variance_threshold: Threshold for explained variance
            
        Returns:
            tuple: (X_pca, explained_variance_ratio, pca_components)
        """
        print("📊 Performing Principal Component Analysis...")
        
        self.feature_names = feature_names
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine number of components if not specified
        if n_components is None:
            pca_temp = PCA()
            pca_temp.fit(X_scaled)
            cumulative_variance = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= explained_variance_threshold) + 1
            print(f"Selected {n_components} components to explain {explained_variance_threshold*100}% of variance")
        
        # Perform PCA
        self.pca_model = PCA(n_components=n_components, random_state=42)
        X_pca = self.pca_model.fit_transform(X_scaled)
        
        # Store results
        self.pca_components = X_pca
        explained_variance_ratio = self.pca_model.explained_variance_ratio_
        
        print(f"✅ PCA completed: {X.shape[1]} features → {n_components} components")
        print(f"Explained variance ratio: {explained_variance_ratio}")
        print(f"Total explained variance: {np.sum(explained_variance_ratio):.3f}")
        
        return X_pca, explained_variance_ratio, self.pca_model.components_
    
    def find_optimal_clusters(self, X, max_clusters=10):
        """
        Find optimal number of clusters using elbow method and silhouette analysis
        
        Args:
            X: Feature matrix (can be original or PCA-reduced)
            max_clusters: Maximum number of clusters to test
            
        Returns:
            int: Optimal number of clusters
        """
        print("🔍 Finding optimal number of clusters...")
        
        # Calculate distortions and silhouette scores for different k values
        distortions = []
        silhouette_scores = []
        calinski_scores = []
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            distortions.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))
            calinski_scores.append(calinski_harabasz_score(X, kmeans.labels_))
        
        # Find elbow point
        # Calculate the rate of change of distortion
        rate_of_change = np.diff(distortions)
        rate_of_change_rate = np.diff(rate_of_change)
        elbow_k = np.argmax(rate_of_change_rate) + 2  # +2 because we start from k=2
        
        # Find best silhouette score
        best_silhouette_k = K_range[np.argmax(silhouette_scores)]
        
        # Find best Calinski-Harabasz score
        best_calinski_k = K_range[np.argmax(calinski_scores)]
        
        print(f"Elbow method suggests: {elbow_k} clusters")
        print(f"Silhouette analysis suggests: {best_silhouette_k} clusters")
        print(f"Calinski-Harabasz suggests: {best_calinski_k} clusters")
        
        # Use silhouette score as primary criterion, with elbow as tiebreaker
        optimal_k = best_silhouette_k
        if abs(silhouette_scores[best_silhouette_k-2] - silhouette_scores[elbow_k-2]) < 0.01:
            optimal_k = elbow_k
        
        print(f"✅ Optimal number of clusters: {optimal_k}")
        
        return optimal_k, distortions, silhouette_scores, calinski_scores
    
    def perform_kmeans_clustering(self, X, n_clusters=None, feature_names=None):
        """
        Perform K-means clustering on the data
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters (if None, find optimal)
            feature_names: List of feature names for interpretation
            
        Returns:
            tuple: (cluster_labels, cluster_centers, model)
        """
        print("🎯 Performing K-means clustering...")
        
        if feature_names is not None:
            self.feature_names = feature_names
        
        # Find optimal number of clusters if not specified
        if n_clusters is None:
            n_clusters, _, _, _ = self.find_optimal_clusters(X)
        
        # Perform clustering
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans_model.fit_predict(X)
        cluster_centers = self.kmeans_model.cluster_centers_
        
        # Store results
        self.cluster_labels = cluster_labels
        self.cluster_centers = cluster_centers
        
        # Calculate clustering quality metrics
        silhouette_avg = silhouette_score(X, cluster_labels)
        calinski_avg = calinski_harabasz_score(X, cluster_labels)
        
        print(f"✅ K-means clustering completed with {n_clusters} clusters")
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        print(f"Calinski-Harabasz Score: {calinski_avg:.3f}")
        
        return cluster_labels, cluster_centers, self.kmeans_model
    
    def analyze_cluster_characteristics(self, X, cluster_labels, feature_names=None):
        """
        Analyze characteristics of each cluster
        
        Args:
            X: Feature matrix
            cluster_labels: Cluster assignments
            feature_names: List of feature names
            
        Returns:
            pd.DataFrame: Cluster characteristics summary
        """
        print("📈 Analyzing cluster characteristics...")
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        
        # Create DataFrame with features and cluster labels
        df = pd.DataFrame(X, columns=feature_names)
        df['cluster'] = cluster_labels
        
        # Calculate cluster statistics
        cluster_stats = []
        unique_clusters = np.unique(cluster_labels)
        
        for cluster in unique_clusters:
            cluster_data = df[df['cluster'] == cluster]
            
            stats = {
                'cluster': cluster,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df) * 100
            }
            
            # Add mean values for each feature
            for feature in feature_names:
                stats[f'{feature}_mean'] = cluster_data[feature].mean()
                stats[f'{feature}_std'] = cluster_data[feature].std()
            
            cluster_stats.append(stats)
        
        cluster_summary = pd.DataFrame(cluster_stats)
        
        print(f"✅ Cluster analysis completed for {len(unique_clusters)} clusters")
        
        return cluster_summary
    
    def identify_risk_profiles(self, cluster_summary, feature_names):
        """
        Identify risk profiles for each cluster based on key flood risk features
        
        Args:
            cluster_summary: Cluster characteristics summary
            feature_names: List of feature names
            
        Returns:
            dict: Risk profile for each cluster
        """
        print("⚠️ Identifying risk profiles for each cluster...")
        
        # Define key risk features and their risk direction
        risk_features = {
            'rainfall_24h': 'high',  # Higher values = higher risk
            'rainfall_48h': 'high',
            'rainfall_72h': 'high',
            'elevation': 'low',      # Lower values = higher risk
            'slope': 'high',         # Higher values = higher risk
            'distance_to_river': 'low',  # Lower values = higher risk
            'distance_to_lake': 'low',
            'population_density': 'high',
            'impervious_surface': 'high',
            'flood_history_1y': 'high',
            'flood_history_5y': 'high',
            'flood_history_10y': 'high'
        }
        
        risk_profiles = {}
        
        for _, cluster_row in cluster_summary.iterrows():
            cluster_id = cluster_row['cluster']
            risk_score = 0
            risk_factors = []
            
            for feature, direction in risk_features.items():
                if feature in feature_names:
                    feature_mean_col = f'{feature}_mean'
                    if feature_mean_col in cluster_row.index:
                        value = cluster_row[feature_mean_col]
                        
                        # Calculate risk contribution
                        if direction == 'high':
                            if value > np.percentile([cluster_row[f'{feature}_mean'] for _, cluster_row in cluster_summary.iterrows()], 75):
                                risk_score += 1
                                risk_factors.append(f"High {feature}")
                        else:  # direction == 'low'
                            if value < np.percentile([cluster_row[f'{feature}_mean'] for _, cluster_row in cluster_summary.iterrows()], 25):
                                risk_score += 1
                                risk_factors.append(f"Low {feature}")
            
            # Determine risk level
            if risk_score >= 8:
                risk_level = "Very High"
            elif risk_score >= 6:
                risk_level = "High"
            elif risk_score >= 4:
                risk_level = "Medium"
            elif risk_score >= 2:
                risk_level = "Low"
            else:
                risk_level = "Very Low"
            
            risk_profiles[cluster_id] = {
                'risk_level': risk_level,
                'risk_score': risk_score,
                'risk_factors': risk_factors,
                'cluster_size': cluster_row['size'],
                'percentage': cluster_row['percentage']
            }
        
        print("✅ Risk profiles identified for all clusters")
        return risk_profiles
    
    def plot_clustering_results(self, X, cluster_labels, save_path='results/clustering_analysis.png'):
        """
        Create comprehensive visualization of clustering results
        
        Args:
            X: Feature matrix
            cluster_labels: Cluster assignments
            save_path: Path to save the plot
        """
        print("📊 Creating clustering visualization...")
        
        # If data is high-dimensional, use PCA for visualization
        if X.shape[1] > 2:
            pca_viz = PCA(n_components=2, random_state=42)
            X_viz = pca_viz.fit_transform(X)
            x_label = f'PC1 ({pca_viz.explained_variance_ratio_[0]:.2%} variance)'
            y_label = f'PC2 ({pca_viz.explained_variance_ratio_[1]:.2%} variance)'
        else:
            X_viz = X
            x_label = 'Feature 1'
            y_label = 'Feature 2'
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Unsupervised Learning Analysis: Town Clustering for Flood Risk', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Cluster scatter plot
        scatter = axes[0, 0].scatter(X_viz[:, 0], X_viz[:, 1], c=cluster_labels, 
                                   cmap='viridis', alpha=0.7, s=50)
        axes[0, 0].set_xlabel(x_label)
        axes[0, 0].set_ylabel(y_label)
        axes[0, 0].set_title('Town Clusters')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add cluster centers if available
        if self.cluster_centers is not None:
            if X.shape[1] > 2:
                centers_viz = pca_viz.transform(self.cluster_centers)
            else:
                centers_viz = self.cluster_centers
            axes[0, 0].scatter(centers_viz[:, 0], centers_viz[:, 1], 
                             c='red', marker='x', s=200, linewidths=3, label='Cluster Centers')
            axes[0, 0].legend()
        
        # Plot 2: Cluster size distribution
        unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
        axes[0, 1].bar(unique_clusters, counts, color='skyblue', alpha=0.7)
        axes[0, 1].set_xlabel('Cluster ID')
        axes[0, 1].set_ylabel('Number of Towns')
        axes[0, 1].set_title('Cluster Size Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Feature importance in clustering (if PCA was used)
        if self.pca_model is not None and self.feature_names is not None:
            # Get feature importance from first two principal components
            feature_importance = np.abs(self.pca_model.components_[:2, :])
            avg_importance = np.mean(feature_importance, axis=0)
            
            # Get top 10 features
            top_indices = np.argsort(avg_importance)[-10:]
            top_features = [self.feature_names[i] for i in top_indices]
            top_importance = avg_importance[top_indices]
            
            axes[1, 0].barh(range(len(top_features)), top_importance, color='lightgreen', alpha=0.7)
            axes[1, 0].set_yticks(range(len(top_features)))
            axes[1, 0].set_yticklabels(top_features)
            axes[1, 0].set_xlabel('Feature Importance')
            axes[1, 0].set_title('Top 10 Features in Clustering')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Silhouette analysis
        from sklearn.metrics import silhouette_samples
        silhouette_vals = silhouette_samples(X, cluster_labels)
        
        y_ticks = []
        y_lower, y_upper = 0, 0
        
        for i in unique_clusters:
            cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
            cluster_silhouette_vals.sort()
            size_cluster_i = cluster_silhouette_vals.shape[0]
            y_upper += size_cluster_i
            color = plt.cm.viridis(i / len(unique_clusters))
            axes[1, 1].barh(range(y_lower, y_upper), cluster_silhouette_vals, 
                           height=1.0, edgecolor='none', color=color)
            y_ticks.append((y_lower + y_upper) / 2)
            y_lower += size_cluster_i
        
        axes[1, 1].axvline(x=silhouette_score(X, cluster_labels), color="red", linestyle="--")
        axes[1, 1].set_yticks(y_ticks)
        axes[1, 1].set_yticklabels(unique_clusters)
        axes[1, 1].set_xlabel('Silhouette Coefficient')
        axes[1, 1].set_ylabel('Cluster')
        axes[1, 1].set_title('Silhouette Analysis')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Clustering visualization saved to {save_path}")
    
    def save_models(self, filepath_prefix='models/unsupervised_models'):
        """
        Save unsupervised learning models
        
        Args:
            filepath_prefix: Prefix for model files
        """
        print("💾 Saving unsupervised learning models...")
        
        if self.kmeans_model is not None:
            joblib.dump(self.kmeans_model, f"{filepath_prefix}_kmeans.pkl")
            print("✅ K-means model saved")
        
        if self.pca_model is not None:
            joblib.dump(self.pca_model, f"{filepath_prefix}_pca.pkl")
            print("✅ PCA model saved")
        
        if self.scaler is not None:
            joblib.dump(self.scaler, f"{filepath_prefix}_scaler.pkl")
            print("✅ Scaler saved")
    
    def complete_unsupervised_analysis(self, X, feature_names, n_clusters=None):
        """
        Complete unsupervised learning analysis pipeline
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
            n_clusters: Number of clusters (if None, find optimal)
            
        Returns:
            dict: Complete analysis results
        """
        print("🚀 Starting complete unsupervised learning analysis...")
        
        # Step 1: PCA for dimensionality reduction
        X_pca, explained_variance, pca_components = self.perform_pca_analysis(X, feature_names)
        
        # Step 2: Find optimal number of clusters
        if n_clusters is None:
            optimal_k, distortions, silhouette_scores, calinski_scores = self.find_optimal_clusters(X_pca)
            n_clusters = optimal_k
        
        # Step 3: Perform clustering
        cluster_labels, cluster_centers, kmeans_model = self.perform_kmeans_clustering(X_pca, n_clusters, feature_names)
        
        # Step 4: Analyze cluster characteristics
        cluster_summary = self.analyze_cluster_characteristics(X, cluster_labels, feature_names)
        
        # Step 5: Identify risk profiles
        risk_profiles = self.identify_risk_profiles(cluster_summary, feature_names)
        
        # Step 6: Create visualizations
        self.plot_clustering_results(X_pca, cluster_labels)
        
        # Step 7: Save models
        self.save_models()
        
        # Compile results
        results = {
            'pca_results': {
                'explained_variance': explained_variance,
                'n_components': X_pca.shape[1],
                'original_features': X.shape[1]
            },
            'clustering_results': {
                'n_clusters': n_clusters,
                'cluster_labels': cluster_labels,
                'cluster_centers': cluster_centers,
                'silhouette_score': silhouette_score(X_pca, cluster_labels),
                'calinski_score': calinski_harabasz_score(X_pca, cluster_labels)
            },
            'cluster_summary': cluster_summary,
            'risk_profiles': risk_profiles
        }
        
        print("🎉 Complete unsupervised learning analysis finished!")
        return results

if __name__ == "__main__":
    # Example usage
    from data_preprocessing import FloodDataPreprocessor
    
    # Load and preprocess data
    preprocessor = FloodDataPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_pipeline(
        'data/sample_data.csv',
        'data/town_characteristics.csv'
    )
    
    # Combine train and test for unsupervised analysis
    X_combined = np.vstack([X_train, X_test])
    
    # Perform unsupervised analysis
    analyzer = UnsupervisedFloodAnalyzer()
    results = analyzer.complete_unsupervised_analysis(X_combined, feature_names)
    
    # Display results
    print("\n" + "="*60)
    print("UNSUPERVISED LEARNING RESULTS")
    print("="*60)
    print(f"Number of clusters: {results['clustering_results']['n_clusters']}")
    print(f"Silhouette Score: {results['clustering_results']['silhouette_score']:.3f}")
    print(f"Calinski-Harabasz Score: {results['clustering_results']['calinski_score']:.3f}")
    
    print("\nRisk Profiles:")
    for cluster_id, profile in results['risk_profiles'].items():
        print(f"Cluster {cluster_id}: {profile['risk_level']} Risk "
              f"(Score: {profile['risk_score']}, Size: {profile['cluster_size']} towns)")
        print(f"  Risk factors: {', '.join(profile['risk_factors'][:3])}...") 