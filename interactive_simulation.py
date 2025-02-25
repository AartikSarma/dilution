import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt

# Import simulation functions
# Assuming the core simulation module is imported as follows:
from biomarker_dilution_sim import *
from visualization_module import *

# Set page config
st.set_page_config(
    page_title="Biomarker Dilution Simulation",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define functions for the interactive app

def run_interactive_simulation(params):
    """
    Run a simulation with the given parameters and return results.
    """
    # Generate dataset
    dataset = generate_dataset(**params)
    
    # Apply normalization methods
    X_true = dataset['X_true']
    X_obs = dataset['X_obs']
    y = dataset['y']
    
    norm_methods = [
        'none', 'total_sum', 'pqn', 'clr', 'ilr', 
        'reference', 'quantile'
    ]
    
    # Run analysis
    results = run_single_simulation(params, norm_methods)
    
    return dataset, results


def plot_interactive_dilution_effect(dataset):
    """
    Create interactive plotly visualization of dilution effect.
    """
    X_true = dataset['X_true']
    X_obs = dataset['X_obs']
    dilution_factors = dataset['dilution_factors']
    
    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("True vs Observed Concentration", "Dilution Factor Distribution")
    )
    
    # Add scatter plot
    fig.add_trace(
        go.Scatter(
            x=X_true[:, 0],
            y=X_obs[:, 0],
            mode='markers',
            marker=dict(
                size=8,
                color=dilution_factors,
                colorscale='Viridis',
                colorbar=dict(title="Dilution Factor"),
                showscale=True
            ),
            hovertemplate="True: %{x:.2f}<br>Observed: %{y:.2f}<br>Dilution: %{marker.color:.2f}"
        ),
        row=1, col=1
    )
    
    # Add reference line y=x
    max_val = max(X_true[:, 0].max(), X_obs[:, 0].max())
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add histogram of dilution factors
    fig.add_trace(
        go.Histogram(
            x=dilution_factors,
            nbinsx=30,
            marker_color='rgba(0, 0, 255, 0.5)'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        title_text="Effect of Dilution on Biomarker Concentration",
        hovermode="closest"
    )
    
    fig.update_xaxes(title_text="True Concentration", row=1, col=1)
    fig.update_yaxes(title_text="Observed Concentration", row=1, col=1)
    fig.update_xaxes(title_text="Dilution Factor", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    
    return fig


def plot_interactive_pca(dataset, normalized_data):
    """
    Create interactive PCA plot comparing different normalization methods.
    """
    X_true = dataset['X_true']
    X_obs = dataset['X_obs']
    y = dataset['y']
    
    # Initialize figure
    fig = make_subplots(
        rows=2, cols=len(normalized_data),
        subplot_titles=[f"{method} - PCA" for method in normalized_data.keys()] + 
                       [f"{method} - Group Separation" for method in normalized_data.keys()],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )
    
    # Define colors for groups
    group_colors = px.colors.qualitative.Set1
    unique_groups = np.unique(y)
    n_groups = len(unique_groups)
    
    # For each normalization method
    for i, (method, X_norm) in enumerate(normalized_data.items()):
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_norm)
        
        # Plot PCA colored by group
        for g in range(n_groups):
            mask = y == g
            fig.add_trace(
                go.Scatter(
                    x=X_pca[mask, 0],
                    y=X_pca[mask, 1],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=group_colors[g % len(group_colors)]
                    ),
                    name=f'Group {g}',
                    legendgroup=f'Group {g}',
                    showlegend=(i == 0)
                ),
                row=1, col=i+1
            )
        
        # Create separate boxplots for each group
        for g in range(n_groups):
            mask = y == g
            fig.add_trace(
                go.Box(
                    x=[f'Group {g}'] * np.sum(mask),  # Same group name for all points
                    y=X_pca[mask, 0],  # PC1 values for this group
                    name=f'Group {g}',
                    legendgroup=f'Group {g}',
                    showlegend=False,
                    marker_color=group_colors[g % len(group_colors)]  # Single color for boxplot
                ),
                row=2, col=i+1
            )
        
        # Update axes
        var_explained = pca.explained_variance_ratio_
        fig.update_xaxes(title_text=f"PC1 ({var_explained[0]:.1%})", row=1, col=i+1)
        fig.update_yaxes(title_text=f"PC2 ({var_explained[1]:.1%})", row=1, col=i+1)
        fig.update_xaxes(title_text="Group", row=2, col=i+1)
        fig.update_yaxes(title_text="PC1 Value", row=2, col=i+1)
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="PCA Visualization by Normalization Method",
        hovermode="closest"
    )
    
    return fig


def plot_interactive_correlation(dataset, normalized_data):
    """
    Create interactive correlation heatmaps for different normalization methods.
    """
    # Create figure with subplots
    n_methods = len(normalized_data)
    fig = make_subplots(
        rows=1, cols=n_methods,
        subplot_titles=[method for method in normalized_data.keys()],
        horizontal_spacing=0.03
    )
    
    # For each normalization method
    for i, (method, X_norm) in enumerate(normalized_data.items()):
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X_norm.T)
        
        # Add heatmap
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix,
                x=[f'B{j+1}' for j in range(X_norm.shape[1])],
                y=[f'B{j+1}' for j in range(X_norm.shape[1])],
                colorscale='RdBu_r',
                zmid=0,
                zmin=-1,
                zmax=1,
                colorbar=dict(
                    title="Correlation",
                    len=0.6,
                    y=0.5,
                    yanchor="middle",
                    thickness=15,
                    x=1.02 if i == n_methods - 1 else None,
                ),
                showscale=(i == n_methods - 1)
            ),
            row=1, col=i+1
        )
    
    # Update layout
    fig.update_layout(
        height=500,
        title_text="Correlation Structure by Normalization Method",
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig


def plot_performance_comparison(results):
    """
    Create interactive bar charts comparing method performance.
    """
    # Extract metrics for each normalization method
    methods = list(results['univariate'].keys())
    
    # Prepare data for various metrics
    power_data = [results['univariate'][method]['power'] for method in methods]
    type_i_error_data = [results['univariate'][method]['type_i_error'] for method in methods]
    corr_recovery_data = [results['correlation'][method]['rank_correlation'] for method in methods]
    pca_recovery_data = [results['pca'][method]['rv_coefficient'] for method in methods]
    clustering_data = [results['clustering'][method]['adjusted_rand_index'] for method in methods]
    classification_data = [results['classification'][method]['accuracy'] for method in methods]
    
    # Create figure with subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Statistical Power", "Type I Error Rate",
            "Correlation Recovery", "PCA Subspace Recovery",
            "Clustering Performance", "Classification Accuracy"
        ),
        vertical_spacing=0.1
    )
    
    # Add bar charts
    fig.add_trace(go.Bar(x=methods, y=power_data, marker_color='royalblue'), row=1, col=1)
    fig.add_trace(go.Bar(x=methods, y=type_i_error_data, marker_color='salmon'), row=1, col=2)
    fig.add_trace(go.Bar(x=methods, y=corr_recovery_data, marker_color='forestgreen'), row=2, col=1)
    fig.add_trace(go.Bar(x=methods, y=pca_recovery_data, marker_color='purple'), row=2, col=2)
    fig.add_trace(go.Bar(x=methods, y=clustering_data, marker_color='orange'), row=3, col=1)
    fig.add_trace(go.Bar(x=methods, y=classification_data, marker_color='teal'), row=3, col=2)
    
    # Add reference line for type I error
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=0.05,
        x1=len(methods) - 0.5,
        y1=0.05,
        line=dict(color="red", width=2, dash="dash"),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=900,
        title_text="Performance Metrics by Normalization Method",
        showlegend=False
    )
    
    fig.update_yaxes(range=[0, 1], row=1, col=1)  # Power
    fig.update_yaxes(range=[0, 0.2], row=1, col=2)  # Type I error
    fig.update_yaxes(range=[0, 1], row=2, col=1)  # Correlation
    fig.update_yaxes(range=[0, 1], row=2, col=2)  # PCA
    fig.update_yaxes(range=[0, 1], row=3, col=1)  # Clustering
    fig.update_yaxes(range=[0, 1], row=3, col=2)  # Classification
    
    return fig


def interactive_simulation_app():
    """
    Main function for the interactive simulation app.
    """
    st.title("Biomarker Dilution Simulation")
    
    st.markdown("""
    This interactive tool simulates the effects of variable dilution on biomarker measurements in respiratory samples
    and evaluates different normalization strategies. Adjust the parameters to explore different scenarios and see
    how well various methods recover the true biological signals.
    """)
    
    # Sidebar for parameters
    st.sidebar.header("Simulation Parameters")
    
    # Sample parameters
    n_subjects = st.sidebar.slider("Number of Subjects", 20, 500, 100)
    n_groups = st.sidebar.slider("Number of Groups", 2, 5, 2)
    n_biomarkers = st.sidebar.slider("Number of Biomarkers", 3, 100, 10)
    
    # Correlation structure
    st.sidebar.subheader("Correlation Structure")
    correlation_type = st.sidebar.selectbox(
        "Correlation Type",
        ["none", "low", "moderate", "high", "block"],
        index=2
    )
    
    block_size = None
    if correlation_type == "block":
        block_size = st.sidebar.slider("Block Size", 2, min(10, n_biomarkers), 3)
    
    # Effect size
    effect_size = st.sidebar.slider("Effect Size", 0.1, 2.0, 0.8)
    
    # Distribution
    distribution = st.sidebar.selectbox(
        "Biomarker Distribution",
        ["normal", "lognormal", "mixed"],
        index=1
    )
    
    # Dilution parameters
    st.sidebar.subheader("Dilution Parameters")
    dilution_severity = st.sidebar.selectbox(
        "Dilution Severity",
        ["Mild", "Moderate", "Severe"],
        index=1
    )
    
    # Set alpha/beta based on severity
    if dilution_severity == "Mild":
        dilution_alpha, dilution_beta = 8.0, 2.0
    elif dilution_severity == "Moderate":
        dilution_alpha, dilution_beta = 5.0, 5.0
    else:  # Severe
        dilution_alpha, dilution_beta = 2.0, 8.0
    
    # LOD parameters
    st.sidebar.subheader("Limit of Detection")
    lod_percentile = st.sidebar.slider(
        "LOD Percentile",
        0.0, 0.5, 0.1,
        help="Fraction of true values below LOD"
    )
    
    lod_handling = st.sidebar.selectbox(
        "LOD Handling Method",
        ["substitute", "zero", "min"],
        index=0
    )
    
    # Build parameter dictionary
    params = {
        'n_subjects': n_subjects,
        'n_biomarkers': n_biomarkers,
        'n_groups': n_groups,
        'correlation_type': correlation_type,
        'effect_size': effect_size,
        'distribution': distribution,
        'dilution_alpha': dilution_alpha,
        'dilution_beta': dilution_beta,
        'lod_percentile': lod_percentile,
        'lod_handling': lod_handling
    }
    
    if block_size is not None:
        params['block_size'] = block_size
    
    # Button to run simulation
    run_button = st.sidebar.button("Run Simulation")
    
    if run_button:
        # Run simulation with progress bar
        with st.spinner("Running simulation..."):
            dataset, results = run_interactive_simulation(params)
        
        st.success("Simulation complete!")
        
        # Tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "Dilution Effects", 
            "Normalization Methods", 
            "Performance Comparison",
            "Detailed Results"
        ])
        
        with tab1:
            st.header("Visualizing Dilution Effects")
            
            # Plotly figure of dilution effects
            fig_dilution = plot_interactive_dilution_effect(dataset)
            st.plotly_chart(fig_dilution, use_container_width=True)
            
            # Additional explanations
            st.markdown("""
            **Observations:**
            - The scatter plot shows how the observed concentration relates to the true concentration.
            - Each point represents a sample, with color indicating the dilution factor.
            - The diagonal line shows where points would fall with no dilution.
            - The histogram shows the distribution of dilution factors across samples.
            """)
        
        with tab2:
            st.header("Comparing Normalization Methods")
            
            # Apply various normalizations
            X_true = dataset['X_true']
            X_obs = dataset['X_obs']
            
            normalized_data = {
                "True": X_true,
                "Raw (Diluted)": X_obs,
                "Total Sum": normalize_total_sum(X_obs),
                "PQN": normalize_probabilistic_quotient(X_obs),
                "CLR": centered_log_ratio(X_obs),
                "Reference": normalize_reference_biomarker(X_obs, 0),
                "Quantile": normalize_quantile(X_obs)
            }
            
            # PCA visualization
            st.subheader("Principal Component Analysis")
            fig_pca = plot_interactive_pca(dataset, normalized_data)
            st.plotly_chart(fig_pca, use_container_width=True)
            
            # Correlation structure
            st.subheader("Correlation Structure")
            fig_corr = plot_interactive_correlation(dataset, normalized_data)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Explanations
            st.markdown("""
            **PCA Visualization:**
            - The top row shows the first two principal components for each method.
            - Colors represent different groups.
            - The bottom row shows boxplots of PC1 values by group, illustrating group separation.
            
            **Correlation Structure:**
            - Heatmaps show the correlation between biomarkers for each method.
            - Blue indicates negative correlation, red indicates positive correlation.
            - Dilution introduces artificial correlations that normalization methods try to correct.
            """)
        
        with tab3:
            st.header("Performance Metrics Comparison")
            
            # Performance comparison
            fig_performance = plot_performance_comparison(results)
            st.plotly_chart(fig_performance, use_container_width=True)
            
            # Interpretations
            st.markdown("""
            **Metric Definitions:**
            - **Statistical Power**: Ability to detect true differences between groups (higher is better)
            - **Type I Error Rate**: Rate of false positives (should be ≤ 0.05)
            - **Correlation Recovery**: How well the method recovers true correlation structure (higher is better)
            - **PCA Subspace Recovery**: Similarity between true and recovered principal component subspaces (higher is better)
            - **Clustering Performance**: Agreement between true groups and clustering results (higher is better)
            - **Classification Accuracy**: Accuracy in predicting group membership (higher is better)
            """)
            
            # Method recommendations
            st.subheader("Method Recommendations")
            
            # Determine best methods based on metrics
            power_best = max(results['univariate'].items(), key=lambda x: x[1]['power'])[0]
            corr_best = max(results['correlation'].items(), key=lambda x: x[1]['rank_correlation'])[0]
            pca_best = max(results['pca'].items(), key=lambda x: x[1]['rv_coefficient'])[0]
            clustering_best = max(results['clustering'].items(), key=lambda x: x[1]['adjusted_rand_index'])[0]
            classif_best = max(results['classification'].items(), key=lambda x: x[1]['accuracy'])[0]
            
            # Create recommendation table
            recommendation_data = {
                'Metric': ['Statistical Power', 'Correlation Recovery', 'PCA Recovery', 'Clustering', 'Classification'],
                'Best Method': [power_best, corr_best, pca_best, clustering_best, classif_best]
            }
            
            st.table(pd.DataFrame(recommendation_data))
            
            # Overall recommendation
            methods_count = {}
            for method in [power_best, corr_best, pca_best, clustering_best, classif_best]:
                methods_count[method] = methods_count.get(method, 0) + 1
            
            overall_best = max(methods_count.items(), key=lambda x: x[1])[0]
            
            st.info(f"For this specific scenario, **{overall_best}** normalization appears to be the most robust overall method.")
        
        with tab4:
            st.header("Detailed Results")
            
            # Raw parameter values
            st.subheader("Simulation Parameters")
            st.json(params)
            
            # Raw performance metrics
            st.subheader("Performance Metrics")
            
            # Create expandable sections for each category
            with st.expander("Univariate Analysis Metrics"):
                st.table(pd.DataFrame({k: v for k, v in results['univariate'].items()}))
            
            with st.expander("Correlation Analysis Metrics"):
                st.table(pd.DataFrame({k: v for k, v in results['correlation'].items()}))
            
            with st.expander("PCA Metrics"):
                st.table(pd.DataFrame({k: v for k, v in results['pca'].items()}))
            
            with st.expander("Clustering Metrics"):
                st.table(pd.DataFrame({k: v for k, v in results['clustering'].items()}))
            
            with st.expander("Classification Metrics"):
                st.table(pd.DataFrame({k: v for k, v in results['classification'].items()}))
    
    else:
        # Default view
        st.info("Adjust the parameters and click 'Run Simulation' to start.")
        
        # Example visualizations
        st.header("Example Visualizations")
        
        # Load pre-computed examples
        example_params = {
            'n_subjects': 100,
            'n_biomarkers': 10,
            'n_groups': 2,
            'correlation_type': 'moderate',
            'effect_size': 0.8,
            'distribution': 'lognormal',
            'dilution_alpha': 2.0,
            'dilution_beta': 8.0,
            'lod_percentile': 0.1,
            'lod_handling': 'substitute'
        }
        
        # Generate example dataset
        example_dataset = generate_dataset(**example_params)
        
        # Show dilution effect
        st.subheader("Example: Dilution Effect")
        example_fig = plot_interactive_dilution_effect(example_dataset)
        st.plotly_chart(example_fig, use_container_width=True)
        
        # Explanation of the problem
        st.header("The Biomarker Dilution Problem")
        st.markdown("""
        ## Why is this important?
        
        In pulmonary research, biomarker concentrations in respiratory samples (BAL, tracheal aspirates, etc.) 
        are affected by variable dilution during the collection process. This dilution:
        
        - Introduces noise and reduces statistical power
        - Creates artificial correlations between biomarkers
        - Affects machine learning outcomes (clustering, classification)
        - Can lead to inconsistent or irreproducible findings
        
        ## How do normalization methods help?
        
        Normalization methods attempt to correct for dilution effects by:
        
        - **Total Sum Normalization**: Converts to relative abundances (compositional data)
        - **Probabilistic Quotient Normalization (PQN)**: Uses a reference spectrum and median scaling
        - **Centered Log-Ratio (CLR)**: Log-ratio transformation for compositional data
        - **Reference Biomarker**: Normalizes to a reference biomarker (e.g., urea)
        - **Quantile Normalization**: Forces identical distributions across samples
        
        This simulation tool helps researchers understand which methods work best in different scenarios.
        """)


if __name__ == "__main__":
    interactive_simulation_app()
