import sys
import os
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure project root is in PYTHONPATH
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Enhanced SHAP loading with version checking
SHAP_AVAILABLE = False
SHAP_ERROR = None
SHAP_VERSION = None

def check_shap_compatibility():
    """Check SHAP compatibility with current environment"""
    global SHAP_AVAILABLE, SHAP_ERROR, SHAP_VERSION
    
    try:
        import shap
        SHAP_VERSION = shap.__version__
        
        # Test basic SHAP functionality
        import numpy as np
        test_data = np.random.rand(2, 3)
        
        # Try creating a simple explainer to test compatibility
        from sklearn.ensemble import RandomForestRegressor
        test_model = RandomForestRegressor(n_estimators=2, random_state=42)
        test_model.fit(test_data, [1, 2])
        
        # Test explainer creation
        test_explainer = shap.Explainer(test_model)
        
        SHAP_AVAILABLE = True
        logger.info(f"SHAP {SHAP_VERSION} loaded and tested successfully")
        return True
        
    except ImportError as e:
        SHAP_ERROR = f"Import error: {str(e)}"
        logger.warning(f"SHAP not available: {e}")
        return False
    except Exception as e:
        SHAP_ERROR = f"Compatibility error: {str(e)}"
        logger.warning(f"SHAP compatibility issue: {e}")
        return False

# Check SHAP compatibility
check_shap_compatibility()

# Import from src module with better error handling
try:
    from src.config import CUSTOMER_FEATS, SHAP_EXPLAINER, XGB_MODEL, SCALER_F, FEATURES_F
    config_loaded = True
    logger.info("Config loaded successfully")
except ImportError as e:
    logger.error(f"Could not import from src.config: {e}")
    
    # Define fallback paths
    CUSTOMER_FEATS = "artifacts/customer_features.csv"
    SHAP_EXPLAINER = "artifacts/shap_explainer.joblib"
    XGB_MODEL = "artifacts/xgb_model.joblib"
    SCALER_F = "artifacts/scaler.joblib"
    FEATURES_F = "artifacts/feature_cols.joblib"
    config_loaded = False

st.set_page_config(page_title="CLV Dashboard", layout="wide")
st.title("🎯 Customer Lifetime Value Dashboard")

# Enhanced system status sidebar
with st.sidebar:
    st.header("🔧 System Status")
    st.write(f"**Config Loaded:** {'✅' if config_loaded else '❌'}")
    st.write(f"**SHAP Available:** {'✅' if SHAP_AVAILABLE else '❌'}")
    
    if SHAP_AVAILABLE:
        st.write(f"**SHAP Version:** {SHAP_VERSION}")
    elif SHAP_ERROR:
        st.error(f"**SHAP Error:** {SHAP_ERROR}")
    
    # Environment info
    st.write(f"**Python:** {sys.version.split()[0]}")
    st.write(f"**NumPy:** {np.__version__}")
    st.write(f"**Pandas:** {pd.__version__}")
    
    st.divider()
    
    files_to_check = {
        "Customer Data": CUSTOMER_FEATS,
        "XGB Model": XGB_MODEL,
        "SHAP Explainer": SHAP_EXPLAINER,
        "Scaler": SCALER_F,
        "Features": FEATURES_F
    }
    
    st.write("**📁 File Status:**")
    all_files_exist = True
    for name, path in files_to_check.items():
        exists = os.path.exists(path)
        if not exists:
            all_files_exist = False
        st.write(f"- {name}: {'✅' if exists else '❌'}")
    
    if not all_files_exist:
        st.warning("Some files are missing. Run training pipeline first.")

@st.cache_data
def load_customer_data():
    """Load customer data with comprehensive error handling"""
    if not os.path.exists(CUSTOMER_FEATS):
        logger.warning(f"Customer features file not found: {CUSTOMER_FEATS}")
        return None
    try:
        df = pd.read_csv(CUSTOMER_FEATS)
        logger.info(f"Loaded {len(df)} customer records")
        return df
    except Exception as e:
        logger.error(f"Error loading customer data: {e}")
        st.error(f"Error loading customer data: {e}")
        return None

@st.cache_resource
def load_models():
    """Load all models with comprehensive error handling"""
    models = {}
    
    try:
        if os.path.exists(XGB_MODEL):
            models['xgb'] = joblib.load(XGB_MODEL)
            logger.info("XGBoost model loaded successfully")
        else:
            logger.warning(f"XGBoost model not found: {XGB_MODEL}")
            
        if os.path.exists(SCALER_F):
            models['scaler'] = joblib.load(SCALER_F)
            logger.info("Scaler loaded successfully")
        else:
            logger.warning(f"Scaler not found: {SCALER_F}")
            
        if os.path.exists(FEATURES_F):
            models['features'] = joblib.load(FEATURES_F)
            logger.info(f"Feature columns loaded: {len(models['features'])} features")
        else:
            logger.warning(f"Features file not found: {FEATURES_F}")
            
        # Only load SHAP explainer if SHAP is working
        if os.path.exists(SHAP_EXPLAINER) and SHAP_AVAILABLE:
            try:
                models['shap_explainer'] = joblib.load(SHAP_EXPLAINER)
                logger.info("SHAP explainer loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load SHAP explainer: {e}")
        else:
            if not SHAP_AVAILABLE:
                logger.info("SHAP explainer not loaded due to SHAP unavailability")
            else:
                logger.warning(f"SHAP explainer file not found: {SHAP_EXPLAINER}")
                
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        st.error(f"Error loading models: {e}")
    
    return models

def safe_shap_analysis(models, df, feature_cols, sample_size=5):
    """Safely perform SHAP analysis with comprehensive fallbacks"""
    if not SHAP_AVAILABLE:
        return None, "SHAP not available in environment"
    
    if 'shap_explainer' not in models:
        return None, "SHAP explainer not loaded"
    
    try:
        import shap
        
        # Use very small sample for stability
        X_sample = df[feature_cols].head(sample_size)
        logger.info(f"Performing SHAP analysis on {len(X_sample)} samples")
        
        # Scale the data if scaler is available
        if 'scaler' in models:
            try:
                X_scaled = models['scaler'].transform(X_sample)
                logger.info("Data scaled successfully for SHAP")
            except Exception as e:
                logger.warning(f"Scaling failed, using raw data: {e}")
                X_scaled = X_sample.values
        else:
            X_scaled = X_sample.values
        
        explainer = models['shap_explainer']
        
        # Try different SHAP calculation methods
        methods_to_try = [
            ("explainer(X_scaled)", lambda: explainer(X_scaled)),
            ("explainer.shap_values(X_scaled)", lambda: explainer.shap_values(X_scaled))
        ]
        
        for method_name, method in methods_to_try:
            try:
                logger.info(f"Trying SHAP method: {method_name}")
                shap_values = method()
                logger.info(f"SHAP calculation successful with {method_name}")
                return shap_values, None
            except Exception as e:
                logger.warning(f"SHAP method {method_name} failed: {e}")
                continue
        
        return None, "All SHAP calculation methods failed"
                
    except Exception as e:
        logger.error(f"SHAP analysis error: {e}")
        return None, f"SHAP analysis error: {e}"

def create_xgb_feature_importance_plot(models):
    """Create XGBoost feature importance plot with error handling"""
    try:
        feature_cols = models['features']
        xgb_model = models['xgb']
        
        # Get feature importance
        importance_scores = xgb_model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance_scores
        }).sort_values('importance', ascending=False).head(15)
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(len(importance_df)), importance_df['importance'], 
                      color='steelblue', alpha=0.7)
        
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.set_xlabel('Feature Importance')
        ax.set_title('XGBoost Feature Importance (Top 15 Features)')
        ax.invert_yaxis()
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, importance_df['importance'])):
            ax.text(bar.get_width() + max(importance_df['importance']) * 0.01, 
                   bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', va='center', ha='left', fontsize=10)
        
        plt.tight_layout()
        return fig, None
        
    except Exception as e:
        return None, f"Error creating feature importance plot: {e}"

def initialize_app():
    """Run initialization if needed"""
    required_files = [CUSTOMER_FEATS, XGB_MODEL]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        st.warning(f"Missing required files: {missing_files}")
        st.warning("Running initialization...")
        
        with st.spinner("Initializing application..."):
            try:
                # Try to run startup script
                if os.path.exists("startup.py"):
                    import subprocess
                    result = subprocess.run([sys.executable, "startup.py"], 
                                          capture_output=True, text=True, cwd=project_root)
                    if result.returncode == 0:
                        st.success("✅ Initialization completed successfully!")
                        st.experimental_rerun()
                    else:
                        st.error(f"❌ Initialization failed: {result.stderr}")
                        if result.stdout:
                            st.code(result.stdout)
                else:
                    # Try to run train_pipeline.py
                    if os.path.exists("train_pipeline.py"):
                        import subprocess
                        result = subprocess.run([sys.executable, "train_pipeline.py"], 
                                              capture_output=True, text=True, cwd=project_root)
                        if result.returncode == 0:
                            st.success("✅ Training pipeline completed successfully!")
                            st.experimental_rerun()
                        else:
                            st.error(f"❌ Training pipeline failed: {result.stderr}")
                    else:
                        st.error("❌ No initialization script found. Please run the training pipeline manually.")
            except Exception as e:
                st.error(f"❌ Failed to run initialization: {e}")
        st.stop()

# Initialize if needed
initialize_app()

# Load data and models
df = load_customer_data()
models = load_models()

if df is None:
    st.error("❌ No customer data available. Please ensure the training pipeline has been run.")
    st.info("💡 Run `python train_pipeline.py` to generate the required data.")
    st.stop()

# Main Dashboard Content
st.subheader("📊 Customer Overview")
if len(df) > 0:
    # Organize columns for display
    cols_order = ["customer_id", "predicted_clv", "segment_label", "segment_name"]
    available_cols = [c for c in cols_order if c in df.columns]
    remaining = [c for c in df.columns if c not in available_cols]
    
    display_df = df[available_cols + remaining].head(25)
    
    # Format numeric columns for better display
    if "predicted_clv" in display_df.columns:
        display_df["predicted_clv"] = display_df["predicted_clv"].round(2)
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    with col2:
        if "predicted_clv" in df.columns:
            avg_clv = df['predicted_clv'].mean()
            st.metric("Avg CLV", f"${avg_clv:,.0f}")
    with col3:
        if "segment_name" in df.columns:
            unique_segments = df['segment_name'].nunique()
            st.metric("Segments", unique_segments)
    with col4:
        if "predicted_clv" in df.columns:
            total_clv = df['predicted_clv'].sum()
            st.metric("Total CLV", f"${total_clv:,.0f}")
else:
    st.warning("⚠️ No customer data available.")

# CLV by segment visualization
if "segment_name" in df.columns and "predicted_clv" in df.columns and len(df) > 0:
    st.subheader("📈 CLV Analysis by Customer Segment")
    
    try:
        # Calculate segment statistics
        segment_stats = df.groupby("segment_name").agg({
            "predicted_clv": ["mean", "sum", "count"],
            "customer_id": "count"
        }).round(2)
        
        segment_stats.columns = ["avg_clv", "total_clv", "clv_count", "customer_count"]
        segment_stats = segment_stats.reset_index()
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Average CLV bar chart
        colors = plt.cm.Set3(range(len(segment_stats)))
        bars1 = ax1.bar(segment_stats["segment_name"], segment_stats["avg_clv"], 
                        color=colors, alpha=0.8)
        ax1.set_title("Average CLV by Segment")
        ax1.set_ylabel("Average CLV ($)")
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars1, segment_stats["avg_clv"]):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value * 0.01,
                    f"${value:,.0f}", ha="center", va="bottom", fontweight='bold')
        
        # Customer count pie chart
        ax2.pie(segment_stats["customer_count"], labels=segment_stats["segment_name"], 
               colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title("Customer Distribution by Segment")
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Display segment statistics table
        st.subheader("📋 Segment Statistics")
        st.dataframe(segment_stats, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error creating segment analysis: {e}")
        logger.error(f"Segment analysis error: {e}")

# Feature Importance Analysis
st.subheader("🌟 Feature Importance Analysis")
if 'xgb' in models and 'features' in models:
    try:
        fig, error = create_xgb_feature_importance_plot(models)
        if fig:
            st.pyplot(fig)
            plt.close()
            
            # Try SHAP as advanced option
            if SHAP_AVAILABLE and 'shap_explainer' in models:
                with st.expander("🔍 Advanced SHAP Analysis", expanded=False):
                    st.info("🔄 Computing SHAP values... This may take a moment.")
                    
                    with st.spinner("Analyzing feature interactions..."):
                        shap_values, shap_error = safe_shap_analysis(
                            models, df, models['features'], sample_size=3
                        )
                    
                    if shap_values is not None:
                        try:
                            import shap
                            X_sample = df[models['features']].head(3)
                            
                            # Try summary plot
                            try:
                                fig, ax = plt.subplots(figsize=(10, 8))
                                shap.summary_plot(shap_values, features=X_sample, 
                                                feature_names=models['features'], 
                                                show=False, ax=ax, max_display=10)
                                st.pyplot(fig)
                                plt.close()
                                st.success("✅ SHAP analysis completed successfully!")
                            except Exception as plot_error:
                                st.warning(f"⚠️ SHAP visualization failed: {plot_error}")
                        except Exception as viz_error:
                            st.warning(f"⚠️ SHAP visualization error: {viz_error}")
                    else:
                        st.warning(f"⚠️ SHAP analysis failed: {shap_error}")
            else:
                if not SHAP_AVAILABLE:
                    st.info("ℹ️ SHAP analysis not available due to compatibility issues.")
                else:
                    st.info("ℹ️ SHAP explainer not available. Run training pipeline to generate.")
        else:
            st.error(f"❌ {error}")
            
    except Exception as e:
        st.error(f"❌ Error in feature importance analysis: {e}")
        logger.error(f"Feature importance error: {e}")
else:
    st.warning("⚠️ Models not available for feature importance analysis")
    st.info("💡 Run the training pipeline to generate required models.")

# Individual Customer Analysis
st.subheader("👤 Individual Customer Analysis")
if len(df) > 0 and 'features' in models:
    try:
        # Limit customer list for performance
        customer_list = df["customer_id"].astype(str).tolist()[:100]
        
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_customer = st.selectbox(
                "🔍 Select Customer ID", 
                customer_list,
                help="Choose a customer to analyze their CLV prediction details"
            )

        if selected_customer:
            cust_info = df[df["customer_id"].astype(str) == selected_customer].iloc[0]
            
            with col2:
                # Customer summary metrics
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    if "predicted_clv" in df.columns:
                        clv_value = cust_info['predicted_clv']
                        st.metric("Predicted CLV", f"${clv_value:,.2f}")
                with metrics_col2:
                    if "segment_name" in df.columns:
                        st.metric("Segment", cust_info['segment_name'])
                with metrics_col3:
                    if "frequency" in cust_info:
                        st.metric("Transactions", f"{int(cust_info['frequency'])}")
            
            st.divider()
            
            # Detailed feature analysis
            col_details1, col_details2 = st.columns(2)
            
            with col_details1:
                st.write("**📊 Key Metrics:**")
                key_features = ['recency_days', 'frequency', 'monetary', 'avg_order_value']
                for feature in key_features:
                    if feature in cust_info:
                        value = cust_info[feature]
                        if feature == 'monetary' or feature == 'avg_order_value':
                            st.write(f"**{feature.replace('_', ' ').title()}:** ${value:,.2f}")
                        else:
                            st.write(f"**{feature.replace('_', ' ').title()}:** {value:.2f}")
            
            with col_details2:
                st.write("**🎯 Advanced Features:**")
                feature_cols = models['features']
                advanced_features = [f for f in feature_cols if f not in ['recency_days', 'frequency', 'monetary', 'avg_order_value']][:6]
                
                for feature in advanced_features:
                    if feature in cust_info:
                        value = cust_info[feature]
                        st.write(f"**{feature.replace('_', ' ').title()}:** {value:.2f}")
            
            # SHAP explanation for individual customer
            if SHAP_AVAILABLE and 'shap_explainer' in models and 'scaler' in models:
                with st.expander("🔬 Individual SHAP Explanation", expanded=False):
                    try:
                        with st.spinner("Computing individual SHAP values..."):
                            cust_row = df[df["customer_id"].astype(str) == selected_customer][models['features']]
                            
                            if 'scaler' in models:
                                cust_scaled = models['scaler'].transform(cust_row)
                            else:
                                cust_scaled = cust_row.values
                            
                            explainer = models['shap_explainer']
                            
                            # Try to get SHAP values for this customer
                            import shap
                            cust_shap_values = explainer(cust_scaled)
                            
                            fig, ax = plt.subplots(figsize=(10, 8))
                            shap.waterfall_plot(cust_shap_values[0], show=False)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                            
                            st.success("✅ Individual SHAP explanation generated!")
                            
                    except Exception as e:
                        st.warning(f"⚠️ Individual SHAP explanation failed: {e}")
                        st.info("This often occurs due to version compatibility issues between numpy and SHAP.")
            else:
                st.info("ℹ️ Individual SHAP explanations not available.")

    except Exception as e:
        st.error(f"❌ Error in customer analysis: {e}")
        logger.error(f"Customer analysis error: {e}")

# System diagnostics
with st.expander("🔧 System Diagnostics & Information"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📈 Application Status:**")
        st.write("- Status: ✅ Running")
        if df is not None:
            st.write(f"- Data Shape: {df.shape}")
            st.write(f"- Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        st.write("**🤖 Models Available:**")
        model_status = [
            ("XGBoost Model", 'xgb' in models),
            ("Feature Scaler", 'scaler' in models),
            ("Feature Columns", 'features' in models),
            ("SHAP Explainer", 'shap_explainer' in models and SHAP_AVAILABLE)
        ]
        
        for model_name, available in model_status:
            st.write(f"- {model_name}: {'✅' if available else '❌'}")
    
    with col2:
        st.write("**🔧 Environment Details:**")
        st.write(f"- Python: {sys.version.split()[0]}")
        st.write(f"- NumPy: {np.__version__}")
        st.write(f"- Pandas: {pd.__version__}")
        st.write(f"- Streamlit: {st.__version__}")
        if SHAP_AVAILABLE:
            st.write(f"- SHAP: {SHAP_VERSION} ✅")
        else:
            st.write(f"- SHAP: ❌ ({SHAP_ERROR})")
        
        st.write("**📁 File Paths:**")
        st.code(f"Customer Data: {CUSTOMER_FEATS}")
        st.code(f"Models Dir: {os.path.dirname(XGB_MODEL)}")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <strong>🎯 CLV Dashboard</strong> | Powered by XGBoost & Machine Learning<br>
    <small>Built with Streamlit • Enhanced Error Handling • Production Ready</small>
</div>
""", unsafe_allow_html=True)
