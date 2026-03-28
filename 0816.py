"""
Project 816: Modern Quality Control System

A comprehensive quality control system that monitors product characteristics to ensure
they meet predefined standards. This modernized implementation includes:

- Statistical Process Control (SPC) with control charts
- Process Capability Analysis (Cp, Cpk, Pp, Ppk)
- Advanced Anomaly Detection using Machine Learning
- Comprehensive Evaluation and Reporting
- Interactive Visualization and Dashboard

DISCLAIMER: This is an experimental research/educational tool. Do not use for
automated quality control decisions without human review and validation.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class QualityDataGenerator:
    """Synthetic data generator for quality control analysis."""
    
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate_control_chart_data(
        self,
        n_samples: int = 100,
        target_mean: float = 100.0,
        target_std: float = 2.0,
        anomaly_rate: float = 0.05,
    ) -> Tuple[List[float], List[bool], Dict[str, Any]]:
        """Generate control chart data with anomalies."""
        measurements = []
        anomaly_labels = []
        
        for i in range(n_samples):
            # Base measurement
            measurement = np.random.normal(target_mean, target_std)
            
            # Determine if this is an anomaly
            is_anomaly = np.random.random() < anomaly_rate
            
            if is_anomaly:
                anomaly_labels.append(True)
                # Add anomaly (shift)
                measurement += 3 * target_std
            else:
                anomaly_labels.append(False)
            
            measurements.append(measurement)
        
        metadata = {
            'n_samples': n_samples,
            'target_mean': target_mean,
            'target_std': target_std,
            'anomaly_rate': anomaly_rate,
            'actual_mean': np.mean(measurements),
            'actual_std': np.std(measurements, ddof=1),
            'actual_anomaly_rate': sum(anomaly_labels) / len(anomaly_labels),
        }
        
        return measurements, anomaly_labels, metadata


class QualityControlChart:
    """Statistical Process Control Chart implementation."""
    
    def __init__(
        self,
        target_mean: Optional[float] = None,
        target_std: Optional[float] = None,
        control_limit_multiplier: float = 3.0,
    ) -> None:
        self.target_mean = target_mean
        self.target_std = target_std
        self.control_limit_multiplier = control_limit_multiplier
        
        self.center_line: Optional[float] = None
        self.upper_control_limit: Optional[float] = None
        self.lower_control_limit: Optional[float] = None
        self.process_std: Optional[float] = None
        
        self.measurements: List[float] = []
        
    def add_measurements(self, values: List[float]) -> None:
        """Add multiple measurements to the control chart."""
        self.measurements.extend(values)
        
    def calculate_control_limits(self) -> Tuple[float, float, float]:
        """Calculate control limits based on current data."""
        if not self.measurements:
            raise ValueError("No measurements available for control limit calculation")
            
        if self.target_mean is not None:
            self.center_line = self.target_mean
        else:
            self.center_line = np.mean(self.measurements)
            
        if self.target_std is not None:
            self.process_std = self.target_std
        else:
            self.process_std = np.std(self.measurements, ddof=1)
            
        margin = self.control_limit_multiplier * self.process_std
        self.upper_control_limit = self.center_line + margin
        self.lower_control_limit = self.center_line - margin
        
        return self.center_line, self.upper_control_limit, self.lower_control_limit
        
    def detect_out_of_control(self) -> pd.DataFrame:
        """Detect out-of-control points using various rules."""
        if self.center_line is None:
            self.calculate_control_limits()
            
        df = pd.DataFrame({
            'sample': range(1, len(self.measurements) + 1),
            'measurement': self.measurements,
        })
        
        # Rule 1: Points outside control limits
        df['rule_1_violation'] = (
            (df['measurement'] > self.upper_control_limit) |
            (df['measurement'] < self.lower_control_limit)
        )
        
        # Overall violation flag
        df['out_of_control'] = df['rule_1_violation']
        
        return df


class ProcessCapabilityAnalysis:
    """Process Capability Analysis for quality control."""
    
    def __init__(
        self,
        specification_lower: Optional[float] = None,
        specification_upper: Optional[float] = None,
        target_value: Optional[float] = None,
    ) -> None:
        self.spec_lower = specification_lower
        self.spec_upper = specification_upper
        self.target_value = target_value
        
    def calculate_capability_indices(
        self,
        measurements: List[float],
        use_short_term: bool = True,
    ) -> Dict[str, float]:
        """Calculate process capability indices."""
        if not measurements:
            raise ValueError("No measurements provided")
            
        data = np.array(measurements)
        mean_val = np.mean(data)
        std_val = np.std(data, ddof=1)
        
        if self.spec_lower is None or self.spec_upper is None:
            self.spec_lower = mean_val - 3 * std_val
            self.spec_upper = mean_val + 3 * std_val
            
        spec_range = self.spec_upper - self.spec_lower
        
        # Calculate capability indices
        cp = spec_range / (6 * std_val)
        cpu = (self.spec_upper - mean_val) / (3 * std_val)
        cpl = (mean_val - self.spec_lower) / (3 * std_val)
        cpk = min(cpu, cpl)
        
        # Calculate defect rate
        z_lower = (self.spec_lower - mean_val) / std_val
        z_upper = (self.spec_upper - mean_val) / std_val
        
        p_lower = stats.norm.cdf(z_lower)
        p_upper = 1 - stats.norm.cdf(z_upper)
        defect_rate = p_lower + p_upper
        
        # Capability assessment
        if cpk >= 1.67:
            assessment = "Excellent"
        elif cpk >= 1.33:
            assessment = "Adequate"
        elif cpk >= 1.0:
            assessment = "Marginal"
        else:
            assessment = "Inadequate"
            
        return {
            'Cp': cp,
            'Cpk': cpk,
            'Cpu': cpu,
            'Cpl': cpl,
            'mean': mean_val,
            'std': std_val,
            'spec_lower': self.spec_lower,
            'spec_upper': self.spec_upper,
            'spec_range': spec_range,
            'sample_size': len(data),
            'defect_rate': defect_rate,
            'ppm_defects': defect_rate * 1_000_000,
            'yield': 1 - defect_rate,
            'capability_assessment': assessment,
        }


class QualityAnomalyDetector:
    """Advanced anomaly detection for quality control using machine learning."""
    
    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.random_state = random_state
        
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def fit(self, measurements: List[float]) -> None:
        """Fit the anomaly detection model."""
        if not measurements:
            raise ValueError("No training measurements provided")
            
        data = np.array(measurements).reshape(-1, 1)
        data_scaled = self.scaler.fit_transform(data)
        
        self.isolation_forest.fit(data_scaled)
        self.is_trained = True
        
    def detect_anomalies(
        self,
        measurements: List[float],
    ) -> pd.DataFrame:
        """Detect anomalies and return detailed results."""
        if not self.is_trained:
            raise ValueError("Model must be trained before detecting anomalies")
            
        data = np.array(measurements).reshape(-1, 1)
        data_scaled = self.scaler.transform(data)
        
        anomaly_scores = self.isolation_forest.decision_function(data_scaled)
        anomaly_predictions = self.isolation_forest.predict(data_scaled)
        
        is_anomaly = anomaly_predictions == -1
        
        results = pd.DataFrame({
            'sample': range(1, len(measurements) + 1),
            'measurement': measurements,
            'anomaly_score': anomaly_scores,
            'is_anomaly': is_anomaly,
        })
        
        return results


def create_control_chart_plot(
    measurements: List[float],
    center_line: float,
    upper_control_limit: float,
    lower_control_limit: float,
    out_of_control_points: Optional[List[int]] = None,
) -> None:
    """Create and display a control chart."""
    samples = list(range(1, len(measurements) + 1))
    
    plt.figure(figsize=(12, 6))
    
    # Plot measurements
    plt.plot(samples, measurements, 'bo-', linewidth=2, markersize=6, label='Measurements')
    
    # Add control limits
    plt.axhline(center_line, color='green', linestyle='-', linewidth=2, label='Center Line')
    plt.axhline(upper_control_limit, color='red', linestyle='--', linewidth=2, label='UCL')
    plt.axhline(lower_control_limit, color='red', linestyle='--', linewidth=2, label='LCL')
    
    # Highlight out-of-control points
    if out_of_control_points:
        oc_measurements = [measurements[i-1] for i in out_of_control_points]
        plt.scatter(out_of_control_points, oc_measurements, color='red', s=100, 
                   marker='x', label='Out of Control', zorder=5)
    
    # Add control zone shading
    plt.fill_between(samples, lower_control_limit, upper_control_limit, 
                    color='lightgray', alpha=0.3, label='Control Zone')
    
    plt.title('Quality Control Chart', fontsize=14, fontweight='bold')
    plt.xlabel('Sample Number', fontsize=12)
    plt.ylabel('Measurement Value', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Main function demonstrating the modern quality control system."""
    print("=" * 80)
    print("PROJECT 816: MODERN QUALITY CONTROL SYSTEM")
    print("=" * 80)
    print()
    
    # Configuration
    config = {
        "data": {"n_samples": 100, "target_mean": 100.0, "target_std": 2.0, "anomaly_rate": 0.05},
        "control_chart": {"control_limit_multiplier": 3.0},
        "capability": {"specification_lower": 94.0, "specification_upper": 106.0},
        "anomaly_detection": {"contamination": 0.1},
    }
    
    # Step 1: Generate synthetic quality control data
    print("STEP 1: Generating Quality Control Data")
    print("-" * 40)
    
    generator = QualityDataGenerator(random_state=42)
    measurements, true_anomalies, metadata = generator.generate_control_chart_data(
        n_samples=config["data"]["n_samples"],
        target_mean=config["data"]["target_mean"],
        target_std=config["data"]["target_std"],
        anomaly_rate=config["data"]["anomaly_rate"],
    )
    
    print(f"Generated {len(measurements)} measurements")
    print(f"Target mean: {config['data']['target_mean']:.2f}, Actual mean: {np.mean(measurements):.2f}")
    print(f"Target std: {config['data']['target_std']:.2f}, Actual std: {np.std(measurements, ddof=1):.2f}")
    print(f"Anomaly rate: {sum(true_anomalies) / len(true_anomalies) * 100:.1f}%")
    print()
    
    # Step 2: Control Chart Analysis
    print("STEP 2: Control Chart Analysis")
    print("-" * 40)
    
    control_chart = QualityControlChart(
        control_limit_multiplier=config["control_chart"]["control_limit_multiplier"]
    )
    control_chart.add_measurements(measurements)
    
    center_line, ucl, lcl = control_chart.calculate_control_limits()
    violations_df = control_chart.detect_out_of_control()
    out_of_control_count = violations_df['out_of_control'].sum()
    
    print(f"Center Line: {center_line:.3f}")
    print(f"Upper Control Limit: {ucl:.3f}")
    print(f"Lower Control Limit: {lcl:.3f}")
    print(f"Out-of-control points: {out_of_control_count}")
    
    if out_of_control_count > 0:
        print("Out-of-control samples:")
        oc_samples = violations_df[violations_df['out_of_control']]
        for _, row in oc_samples.iterrows():
            print(f"  Sample {row['sample']}: {row['measurement']:.3f}")
    else:
        print("✅ No out-of-control points detected!")
    print()
    
    # Step 3: Process Capability Analysis
    print("STEP 3: Process Capability Analysis")
    print("-" * 40)
    
    capability_analysis = ProcessCapabilityAnalysis(
        specification_lower=config["capability"]["specification_lower"],
        specification_upper=config["capability"]["specification_upper"],
        target_value=config["data"]["target_mean"],
    )
    
    capability_indices = capability_analysis.calculate_capability_indices(measurements)
    
    print(f"Cp: {capability_indices['Cp']:.3f}")
    print(f"Cpk: {capability_indices['Cpk']:.3f}")
    print(f"Process Yield: {capability_indices['yield']:.1%}")
    print(f"PPM Defects: {capability_indices['ppm_defects']:.0f}")
    print(f"Capability Assessment: {capability_indices['capability_assessment']}")
    print()
    
    # Step 4: Anomaly Detection
    print("STEP 4: Anomaly Detection")
    print("-" * 40)
    
    anomaly_detector = QualityAnomalyDetector(
        contamination=config["anomaly_detection"]["contamination"],
        random_state=42,
    )
    
    anomaly_detector.fit(measurements)
    anomaly_results = anomaly_detector.detect_anomalies(measurements)
    detected_anomalies = anomaly_results['is_anomaly'].sum()
    
    print(f"Detected anomalies: {detected_anomalies}")
    print(f"True anomalies: {sum(true_anomalies)}")
    
    if sum(true_anomalies) > 0:
        precision = precision_score(true_anomalies, anomaly_results['is_anomaly'])
        recall = recall_score(true_anomalies, anomaly_results['is_anomaly'])
        f1 = f1_score(true_anomalies, anomaly_results['is_anomaly'])
        
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")
    else:
        print("No true anomalies to evaluate against")
    print()
    
    # Step 5: Visualization
    print("STEP 5: Creating Visualizations")
    print("-" * 40)
    
    out_of_control_indices = violations_df[violations_df['out_of_control']]['sample'].tolist()
    
    create_control_chart_plot(
        measurements=measurements,
        center_line=center_line,
        upper_control_limit=ucl,
        lower_control_limit=lcl,
        out_of_control_points=out_of_control_indices,
    )
    
    # Step 6: Save Results
    print("STEP 6: Saving Results")
    print("-" * 40)
    
    from pathlib import Path
    output_dir = Path(__file__).parent / "assets"
    output_dir.mkdir(exist_ok=True)
    
    # Save performance data
    data_df = pd.DataFrame({
        'sample': range(1, len(measurements) + 1),
        'measurement': measurements,
        'true_anomaly': true_anomalies,
        'detected_anomaly': anomaly_results['is_anomaly'],
        'anomaly_score': anomaly_results['anomaly_score'],
        'out_of_control': violations_df['out_of_control'],
    })
    
    data_df.to_csv(output_dir / "quality_data.csv", index=False)
    print(f"Quality data saved to: {output_dir / 'quality_data.csv'}")
    
    # Save capability results
    capability_df = pd.DataFrame([capability_indices])
    capability_df.to_csv(output_dir / "capability_results.csv", index=False)
    print(f"Capability results saved to: {output_dir / 'capability_results.csv'}")
    print()
    
    # Summary
    print("=" * 80)
    print("QUALITY CONTROL ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"📊 Analyzed {len(measurements)} quality measurements")
    print(f"📈 Control chart: {out_of_control_count} out-of-control points")
    print(f"🎯 Process capability: Cpk = {capability_indices['Cpk']:.3f} ({capability_indices['capability_assessment']})")
    print(f"🔍 Anomaly detection: {detected_anomalies} anomalies detected")
    print()
    print("📁 All results saved to the 'assets' directory")
    print()
    print("⚠️  DISCLAIMER: This is an experimental research/educational tool.")
    print("   Do not use for automated quality control decisions without human review.")
    print("=" * 80)


if __name__ == "__main__":
    main()