"""
Quality Control System - Streamlit Demo Application

Interactive web application for quality control analysis including control charts,
process capability analysis, and anomaly detection.

DISCLAIMER: This is an experimental research/educational tool. Do not use for
automated quality control decisions without human review and validation.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from omegaconf import OmegaConf

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data import QualityDataGenerator
from src.eval import QualityControlEvaluator
from src.quality_control import (
    ProcessCapabilityAnalysis,
    QualityAnomalyDetector,
    QualityControlChart,
)
from src.viz import QualityControlVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Quality Control System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .success-metric {
        color: #28a745;
        font-weight: bold;
    }
    .warning-metric {
        color: #ffc107;
        font-weight: bold;
    }
    .danger-metric {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    if config_path.exists():
        return OmegaConf.load(config_path)
    else:
        # Return default configuration
        return {
            "data": {
                "n_samples": 100,
                "target_mean": 100.0,
                "target_std": 2.0,
                "anomaly_rate": 0.05,
                "random_state": 42,
            },
            "control_chart": {
                "control_limit_multiplier": 3.0,
                "subgroup_size": 1,
            },
            "capability": {
                "specification_lower": 94.0,
                "specification_upper": 106.0,
                "target_value": 100.0,
            },
            "anomaly_detection": {
                "contamination": 0.1,
                "random_state": 42,
            },
        }


def generate_sample_data(
    n_samples: int,
    target_mean: float,
    target_std: float,
    anomaly_rate: float,
    data_type: str,
) -> Tuple[List[float], List[bool], Dict[str, Any]]:
    """Generate sample quality control data."""
    generator = QualityDataGenerator(random_state=42)
    
    if data_type == "Control Chart":
        return generator.generate_control_chart_data(
            n_samples=n_samples,
            target_mean=target_mean,
            target_std=target_std,
            anomaly_rate=anomaly_rate,
        )
    elif data_type == "Time Series":
        return generator.generate_time_series_quality_data(
            n_samples=n_samples,
            target_mean=target_mean,
            target_std=target_std,
            anomaly_rate=anomaly_rate,
        )
    elif data_type == "Manufacturing":
        df, labels, metadata = generator.generate_realistic_manufacturing_data(
            n_samples=n_samples,
            process_type="machining",
        )
        # Use first column as primary measurement
        measurements = df.iloc[:, 0].tolist()
        return measurements, labels, metadata
    else:
        # Default to control chart data
        return generator.generate_control_chart_data(
            n_samples=n_samples,
            target_mean=target_mean,
            target_std=target_std,
            anomaly_rate=anomaly_rate,
        )


def main() -> None:
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">📊 Quality Control System</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ DISCLAIMER:</strong> This is an experimental research/educational tool. 
        Do not use for automated quality control decisions without human review and validation.
        All results should be interpreted by qualified quality control professionals.
    </div>
    """, unsafe_allow_html=True)
    
    # Load configuration
    config = load_config()
    
    # Sidebar for parameters
    st.sidebar.header("🔧 Configuration Parameters")
    
    # Data generation parameters
    st.sidebar.subheader("📊 Data Generation")
    data_type = st.sidebar.selectbox(
        "Data Type",
        ["Control Chart", "Time Series", "Manufacturing"],
        help="Select the type of quality control data to generate"
    )
    
    n_samples = st.sidebar.slider(
        "Number of Samples",
        min_value=50,
        max_value=500,
        value=config["data"]["n_samples"],
        help="Number of quality measurements to generate"
    )
    
    target_mean = st.sidebar.number_input(
        "Target Mean",
        min_value=0.0,
        max_value=1000.0,
        value=config["data"]["target_mean"],
        step=0.1,
        help="Target process mean value"
    )
    
    target_std = st.sidebar.number_input(
        "Target Standard Deviation",
        min_value=0.1,
        max_value=50.0,
        value=config["data"]["target_std"],
        step=0.1,
        help="Target process standard deviation"
    )
    
    anomaly_rate = st.sidebar.slider(
        "Anomaly Rate",
        min_value=0.0,
        max_value=0.2,
        value=config["data"]["anomaly_rate"],
        step=0.01,
        help="Proportion of anomalous samples"
    )
    
    # Control chart parameters
    st.sidebar.subheader("📈 Control Chart")
    control_limit_multiplier = st.sidebar.slider(
        "Control Limit Multiplier",
        min_value=2.0,
        max_value=4.0,
        value=config["control_chart"]["control_limit_multiplier"],
        step=0.1,
        help="Multiplier for control limits (typically 3.0 for 3-sigma)"
    )
    
    # Process capability parameters
    st.sidebar.subheader("🎯 Process Capability")
    spec_lower = st.sidebar.number_input(
        "Lower Specification Limit",
        min_value=0.0,
        max_value=1000.0,
        value=config["capability"]["specification_lower"],
        step=0.1,
        help="Lower specification limit for process capability"
    )
    
    spec_upper = st.sidebar.number_input(
        "Upper Specification Limit",
        min_value=0.0,
        max_value=1000.0,
        value=config["capability"]["specification_upper"],
        step=0.1,
        help="Upper specification limit for process capability"
    )
    
    # Anomaly detection parameters
    st.sidebar.subheader("🔍 Anomaly Detection")
    contamination = st.sidebar.slider(
        "Expected Contamination",
        min_value=0.01,
        max_value=0.3,
        value=config["anomaly_detection"]["contamination"],
        step=0.01,
        help="Expected proportion of anomalies"
    )
    
    # Generate data button
    if st.sidebar.button("🔄 Generate New Data", type="primary"):
        st.rerun()
    
    # Generate sample data
    measurements, true_anomalies, metadata = generate_sample_data(
        n_samples=n_samples,
        target_mean=target_mean,
        target_std=target_std,
        anomaly_rate=anomaly_rate,
        data_type=data_type,
    )
    
    # Store data in session state
    if 'measurements' not in st.session_state:
        st.session_state.measurements = measurements
        st.session_state.true_anomalies = true_anomalies
        st.session_state.metadata = metadata
    
    # Update session state if parameters changed
    if (st.session_state.get('n_samples') != n_samples or 
        st.session_state.get('target_mean') != target_mean or
        st.session_state.get('target_std') != target_std or
        st.session_state.get('anomaly_rate') != anomaly_rate):
        
        st.session_state.measurements = measurements
        st.session_state.true_anomalies = true_anomalies
        st.session_state.metadata = metadata
        st.session_state.n_samples = n_samples
        st.session_state.target_mean = target_mean
        st.session_state.target_std = target_std
        st.session_state.anomaly_rate = anomaly_rate
    
    # Get data from session state
    measurements = st.session_state.measurements
    true_anomalies = st.session_state.true_anomalies
    metadata = st.session_state.metadata
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "📈 Control Chart", "🎯 Capability Analysis", 
        "🔍 Anomaly Detection", "📋 Performance Report"
    ])
    
    with tab1:
        st.header("📊 Data Overview")
        
        # Data summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Sample Count",
                len(measurements),
                help="Total number of quality measurements"
            )
        
        with col2:
            st.metric(
                "Mean",
                f"{np.mean(measurements):.2f}",
                help="Average measurement value"
            )
        
        with col3:
            st.metric(
                "Standard Deviation",
                f"{np.std(measurements, ddof=1):.2f}",
                help="Process variation"
            )
        
        with col4:
            st.metric(
                "Anomaly Rate",
                f"{sum(true_anomalies) / len(true_anomalies) * 100:.1f}%",
                help="Proportion of anomalous samples"
            )
        
        # Data visualization
        st.subheader("📈 Data Visualization")
        
        # Create a simple time series plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(measurements) + 1)),
            y=measurements,
            mode='lines+markers',
            name='Measurements',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        # Highlight anomalies
        anomaly_indices = [i for i, is_anomaly in enumerate(true_anomalies) if is_anomaly]
        anomaly_values = [measurements[i] for i in anomaly_indices]
        
        if anomaly_values:
            fig.add_trace(go.Scatter(
                x=[i + 1 for i in anomaly_indices],
                y=anomaly_values,
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=10, symbol='x')
            ))
        
        fig.update_layout(
            title="Quality Measurements Over Time",
            xaxis_title="Sample Number",
            yaxis_title="Measurement Value",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data statistics
        st.subheader("📊 Data Statistics")
        
        df_stats = pd.DataFrame({
            'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', 'Max', '25%', '50%', '75%'],
            'Value': [
                len(measurements),
                f"{np.mean(measurements):.3f}",
                f"{np.std(measurements, ddof=1):.3f}",
                f"{np.min(measurements):.3f}",
                f"{np.max(measurements):.3f}",
                f"{np.percentile(measurements, 25):.3f}",
                f"{np.percentile(measurements, 50):.3f}",
                f"{np.percentile(measurements, 75):.3f}",
            ]
        })
        
        st.dataframe(df_stats, use_container_width=True)
    
    with tab2:
        st.header("📈 Control Chart Analysis")
        
        # Initialize control chart
        control_chart = QualityControlChart(
            control_limit_multiplier=control_limit_multiplier
        )
        control_chart.add_measurements(measurements)
        
        # Calculate control limits
        center_line, ucl, lcl = control_chart.calculate_control_limits()
        
        # Detect out-of-control points
        violations_df = control_chart.detect_out_of_control()
        
        # Display control chart metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Center Line", f"{center_line:.2f}")
        
        with col2:
            st.metric("Upper Control Limit", f"{ucl:.2f}")
        
        with col3:
            st.metric("Lower Control Limit", f"{lcl:.2f}")
        
        with col4:
            oc_count = violations_df['out_of_control'].sum()
            st.metric("Out of Control Points", oc_count)
        
        # Control chart visualization
        visualizer = QualityControlVisualizer()
        
        out_of_control_indices = violations_df[violations_df['out_of_control']]['sample'].tolist()
        
        fig = visualizer.plot_control_chart(
            measurements=measurements,
            center_line=center_line,
            upper_control_limit=ucl,
            lower_control_limit=lcl,
            title="Quality Control Chart",
            interactive=True,
            out_of_control_points=out_of_control_indices,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Rule violations
        st.subheader("🚨 Control Chart Rule Violations")
        
        rule_violations = violations_df[violations_df['out_of_control']]
        
        if not rule_violations.empty:
            st.dataframe(rule_violations[['sample', 'measurement', 'rule_1_violation', 
                                        'rule_2_violation', 'rule_3_violation', 'rule_4_violation']], 
                        use_container_width=True)
        else:
            st.success("✅ No control chart rule violations detected!")
    
    with tab3:
        st.header("🎯 Process Capability Analysis")
        
        # Initialize capability analysis
        capability_analysis = ProcessCapabilityAnalysis(
            specification_lower=spec_lower,
            specification_upper=spec_upper,
            target_value=target_mean,
        )
        
        # Calculate capability indices
        capability_indices = capability_analysis.calculate_capability_indices(measurements)
        
        # Display capability metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cpk = capability_indices['Cpk']
            color = "success-metric" if cpk >= 1.33 else "warning-metric" if cpk >= 1.0 else "danger-metric"
            st.markdown(f'<div class="metric-card"><span class="{color}">Cpk: {cpk:.2f}</span></div>', 
                       unsafe_allow_html=True)
        
        with col2:
            cp = capability_indices['Cp']
            color = "success-metric" if cp >= 1.33 else "warning-metric" if cp >= 1.0 else "danger-metric"
            st.markdown(f'<div class="metric-card"><span class="{color}">Cp: {cp:.2f}</span></div>', 
                       unsafe_allow_html=True)
        
        with col3:
            yield_rate = capability_indices['yield']
            color = "success-metric" if yield_rate >= 0.99 else "warning-metric" if yield_rate >= 0.95 else "danger-metric"
            st.markdown(f'<div class="metric-card"><span class="{color}">Yield: {yield_rate:.1%}</span></div>', 
                       unsafe_allow_html=True)
        
        with col4:
            ppm = capability_indices['ppm_defects']
            color = "success-metric" if ppm <= 10000 else "warning-metric" if ppm <= 50000 else "danger-metric"
            st.markdown(f'<div class="metric-card"><span class="{color}">PPM: {ppm:.0f}</span></div>', 
                       unsafe_allow_html=True)
        
        # Capability visualization
        fig = visualizer.plot_capability_analysis(
            measurements=measurements,
            spec_lower=spec_lower,
            spec_upper=spec_upper,
            capability_indices=capability_indices,
            title="Process Capability Analysis",
            interactive=True,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Capability assessment
        st.subheader("📊 Capability Assessment")
        
        assessment = capability_indices['capability_assessment']
        st.info(f"**Process Capability Assessment:** {assessment}")
        
        # Detailed capability indices
        st.subheader("📈 Detailed Capability Indices")
        
        capability_df = pd.DataFrame([
            {'Index': 'Cp', 'Value': capability_indices['Cp'], 'Description': 'Process capability index'},
            {'Index': 'Cpk', 'Value': capability_indices['Cpk'], 'Description': 'Process capability index (centered)'},
            {'Index': 'Cpu', 'Value': capability_indices['Cpu'], 'Description': 'Upper capability index'},
            {'Index': 'Cpl', 'Value': capability_indices['Cpl'], 'Description': 'Lower capability index'},
            {'Index': 'Yield', 'Value': f"{capability_indices['yield']:.1%}", 'Description': 'Process yield'},
            {'Index': 'PPM Defects', 'Value': f"{capability_indices['ppm_defects']:.0f}", 'Description': 'Defects per million'},
        ])
        
        st.dataframe(capability_df, use_container_width=True)
    
    with tab4:
        st.header("🔍 Anomaly Detection Analysis")
        
        # Initialize anomaly detector
        anomaly_detector = QualityAnomalyDetector(
            contamination=contamination,
            random_state=42,
        )
        
        # Train the detector
        anomaly_detector.fit(measurements)
        
        # Detect anomalies
        anomaly_results = anomaly_detector.detect_anomalies(measurements)
        
        # Display anomaly detection metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            detected_anomalies = anomaly_results['is_anomaly'].sum()
            st.metric("Detected Anomalies", detected_anomalies)
        
        with col2:
            true_anomalies_count = sum(true_anomalies)
            st.metric("True Anomalies", true_anomalies_count)
        
        with col3:
            if true_anomalies_count > 0:
                precision = precision_score(true_anomalies, anomaly_results['is_anomaly'])
                st.metric("Precision", f"{precision:.2f}")
            else:
                st.metric("Precision", "N/A")
        
        with col4:
            if true_anomalies_count > 0:
                recall = recall_score(true_anomalies, anomaly_results['is_anomaly'])
                st.metric("Recall", f"{recall:.2f}")
            else:
                st.metric("Recall", "N/A")
        
        # Anomaly detection visualization
        fig = visualizer.plot_anomaly_detection(
            measurements=measurements,
            anomaly_scores=anomaly_results['anomaly_score'].tolist(),
            anomaly_predictions=anomaly_results['is_anomaly'].tolist(),
            title="Anomaly Detection Results",
            interactive=True,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly details
        st.subheader("🔍 Anomaly Details")
        
        detected_anomalies_df = anomaly_results[anomaly_results['is_anomaly']]
        
        if not detected_anomalies_df.empty:
            st.dataframe(detected_anomalies_df[['sample', 'measurement', 'anomaly_score', 'anomaly_probability']], 
                        use_container_width=True)
        else:
            st.success("✅ No anomalies detected!")
    
    with tab5:
        st.header("📋 Performance Report")
        
        # Initialize evaluator
        evaluator = QualityControlEvaluator()
        
        # Prepare data for evaluation
        control_chart_data = {
            'center_line': center_line,
            'upper_control_limit': ucl,
            'lower_control_limit': lcl,
        }
        
        capability_data = {
            'capability_indices': capability_indices,
        }
        
        anomaly_data = {
            'true_labels': true_anomalies,
            'predicted_labels': anomaly_results['is_anomaly'].tolist(),
            'anomaly_scores': anomaly_results['anomaly_score'].tolist(),
        }
        
        # Create performance report
        performance_report = evaluator.create_performance_report(
            measurements=measurements,
            control_chart_data=control_chart_data,
            capability_data=capability_data,
            anomaly_data=anomaly_data,
        )
        
        # Display performance metrics by category
        categories = performance_report['Category'].unique()
        
        for category in categories:
            st.subheader(f"📊 {category}")
            
            category_data = performance_report[performance_report['Category'] == category]
            
            # Create metrics display
            cols = st.columns(min(len(category_data), 4))
            
            for i, (_, row) in enumerate(category_data.iterrows()):
                with cols[i % 4]:
                    value = row['Value']
                    if isinstance(value, float):
                        if value >= 0.8:
                            color = "success-metric"
                        elif value >= 0.6:
                            color = "warning-metric"
                        else:
                            color = "danger-metric"
                        st.markdown(f'<div class="metric-card"><span class="{color}">{row["Metric"]}: {value:.3f}</span></div>', 
                                   unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="metric-card"><strong>{row["Metric"]}:</strong> {value}</div>', 
                                   unsafe_allow_html=True)
        
        # Overall performance summary
        st.subheader("🎯 Overall Performance Summary")
        
        overall_metrics = performance_report[performance_report['Category'] == 'Overall Performance']
        
        overall_score = overall_metrics[overall_metrics['Metric'] == 'overall_score']['Value'].iloc[0]
        quality_grade = overall_metrics[overall_metrics['Metric'] == 'quality_grade']['Value'].iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            if overall_score >= 0.8:
                color = "success-metric"
            elif overall_score >= 0.6:
                color = "warning-metric"
            else:
                color = "danger-metric"
            
            st.markdown(f'<div class="metric-card"><span class="{color}">Overall Score: {overall_score:.3f}</span></div>', 
                       unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="metric-card"><strong>Quality Grade:</strong> {quality_grade}</div>', 
                       unsafe_allow_html=True)
        
        # Recommendations
        recommendations = overall_metrics[overall_metrics['Metric'] == 'recommendations']['Value'].iloc[0]
        
        st.subheader("💡 Recommendations")
        
        for i, recommendation in enumerate(recommendations, 1):
            st.write(f"{i}. {recommendation}")
        
        # Download report
        st.subheader("📥 Download Report")
        
        csv = performance_report.to_csv(index=False)
        st.download_button(
            label="Download Performance Report (CSV)",
            data=csv,
            file_name="quality_control_performance_report.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
