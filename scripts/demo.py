#!/usr/bin/env python3
"""
Quality Control System - Quick Demo

This script demonstrates the key features of the modern quality control system.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import the quality control classes
from quality_control import (
    QualityDataGenerator,
    QualityControlChart,
    ProcessCapabilityAnalysis,
    QualityAnomalyDetector,
)

def quick_demo():
    """Run a quick demonstration of the quality control system."""
    print("🔬 Quality Control System - Quick Demo")
    print("=" * 50)
    
    # 1. Generate sample data
    print("\n1. Generating sample quality data...")
    generator = QualityDataGenerator(random_state=42)
    measurements, anomalies, metadata = generator.generate_control_chart_data(
        n_samples=50,
        target_mean=100.0,
        target_std=2.0,
        anomaly_rate=0.1
    )
    
    print(f"   Generated {len(measurements)} measurements")
    print(f"   Mean: {np.mean(measurements):.2f}, Std: {np.std(measurements, ddof=1):.2f}")
    print(f"   Anomalies: {sum(anomalies)} ({sum(anomalies)/len(anomalies)*100:.1f}%)")
    
    # 2. Control Chart Analysis
    print("\n2. Control Chart Analysis...")
    control_chart = QualityControlChart()
    control_chart.add_measurements(measurements)
    center_line, ucl, lcl = control_chart.calculate_control_limits()
    
    violations_df = control_chart.detect_out_of_control()
    oc_count = violations_df['out_of_control'].sum()
    
    print(f"   Center Line: {center_line:.2f}")
    print(f"   Control Limits: {lcl:.2f} - {ucl:.2f}")
    print(f"   Out-of-control points: {oc_count}")
    
    # 3. Process Capability Analysis
    print("\n3. Process Capability Analysis...")
    capability = ProcessCapabilityAnalysis(
        specification_lower=94.0,
        specification_upper=106.0
    )
    indices = capability.calculate_capability_indices(measurements)
    
    print(f"   Cp: {indices['Cp']:.3f}")
    print(f"   Cpk: {indices['Cpk']:.3f}")
    print(f"   Yield: {indices['yield']:.1%}")
    print(f"   Assessment: {indices['capability_assessment']}")
    
    # 4. Anomaly Detection
    print("\n4. Anomaly Detection...")
    detector = QualityAnomalyDetector(contamination=0.1)
    detector.fit(measurements)
    results = detector.detect_anomalies(measurements)
    
    detected_count = results['is_anomaly'].sum()
    print(f"   Detected anomalies: {detected_count}")
    
    if sum(anomalies) > 0:
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(anomalies, results['is_anomaly'])
        recall = recall_score(anomalies, results['is_anomaly'])
        f1 = f1_score(anomalies, results['is_anomaly'])
        
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1-Score: {f1:.3f}")
    
    # 5. Create visualization
    print("\n5. Creating visualization...")
    plt.figure(figsize=(10, 6))
    
    samples = list(range(1, len(measurements) + 1))
    plt.plot(samples, measurements, 'bo-', linewidth=2, markersize=4, label='Measurements')
    
    plt.axhline(center_line, color='green', linestyle='-', linewidth=2, label='Center Line')
    plt.axhline(ucl, color='red', linestyle='--', linewidth=2, label='UCL')
    plt.axhline(lcl, color='red', linestyle='--', linewidth=2, label='LCL')
    
    # Highlight anomalies
    anomaly_indices = [i+1 for i, is_anomaly in enumerate(anomalies) if is_anomaly]
    anomaly_values = [measurements[i] for i, is_anomaly in enumerate(anomalies) if is_anomaly]
    
    if anomaly_values:
        plt.scatter(anomaly_indices, anomaly_values, color='red', s=100, 
                   marker='x', label='True Anomalies', zorder=5)
    
    plt.title('Quality Control Chart - Demo', fontsize=14, fontweight='bold')
    plt.xlabel('Sample Number')
    plt.ylabel('Measurement Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_dir = Path(__file__).parent / "assets"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "demo_control_chart.png", dpi=300, bbox_inches='tight')
    print(f"   Chart saved to: {output_dir / 'demo_control_chart.png'}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DEMO SUMMARY")
    print("=" * 50)
    print(f"✅ Generated {len(measurements)} quality measurements")
    print(f"✅ Control chart analysis completed")
    print(f"✅ Process capability: Cpk = {indices['Cpk']:.3f}")
    print(f"✅ Anomaly detection: {detected_count} anomalies found")
    print(f"✅ Visualization created and saved")
    print()
    print("⚠️  Remember: This is for educational purposes only!")
    print("   Always consult quality control professionals for production decisions.")


if __name__ == "__main__":
    quick_demo()
