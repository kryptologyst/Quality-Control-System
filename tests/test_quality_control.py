"""
Tests for Quality Control System

Unit tests for the quality control system components.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

# Import modules to test
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data import QualityDataGenerator
from quality_control import (
    QualityControlChart,
    ProcessCapabilityAnalysis,
    QualityAnomalyDetector,
)
from eval import QualityControlEvaluator


class TestQualityDataGenerator:
    """Test cases for QualityDataGenerator."""
    
    def test_generate_control_chart_data(self):
        """Test control chart data generation."""
        generator = QualityDataGenerator(random_state=42)
        measurements, anomalies, metadata = generator.generate_control_chart_data(
            n_samples=50, target_mean=100.0, target_std=2.0
        )
        
        assert len(measurements) == 50
        assert len(anomalies) == 50
        assert isinstance(measurements, list)
        assert isinstance(anomalies, list)
        assert all(isinstance(x, bool) for x in anomalies)
        assert metadata['n_samples'] == 50
        assert metadata['target_mean'] == 100.0
    
    def test_generate_time_series_data(self):
        """Test time series data generation."""
        generator = QualityDataGenerator(random_state=42)
        df, anomalies, metadata = generator.generate_time_series_quality_data(
            n_samples=100, target_mean=50.0, target_std=1.0
        )
        
        assert len(df) == 100
        assert len(anomalies) == 100
        assert 'timestamp' in df.columns
        assert 'measurement' in df.columns
        assert metadata['n_samples'] == 100
    
    def test_generate_manufacturing_data(self):
        """Test manufacturing data generation."""
        generator = QualityDataGenerator(random_state=42)
        df, anomalies, metadata = generator.generate_realistic_manufacturing_data(
            n_samples=200, process_type="machining"
        )
        
        assert len(df) == 200
        assert len(anomalies) == 200
        assert len(df.columns) >= 4  # Should have multiple quality variables
        assert metadata['process_type'] == "machining"


class TestQualityControlChart:
    """Test cases for QualityControlChart."""
    
    def test_control_chart_initialization(self):
        """Test control chart initialization."""
        chart = QualityControlChart(target_mean=100.0, target_std=2.0)
        
        assert chart.target_mean == 100.0
        assert chart.target_std == 2.0
        assert chart.control_limit_multiplier == 3.0
        assert len(chart.measurements) == 0
    
    def test_add_measurements(self):
        """Test adding measurements to control chart."""
        chart = QualityControlChart()
        measurements = [100.0, 101.0, 99.0, 102.0]
        
        chart.add_measurements(measurements)
        
        assert len(chart.measurements) == 4
        assert chart.measurements == measurements
    
    def test_calculate_control_limits(self):
        """Test control limit calculation."""
        chart = QualityControlChart()
        measurements = [100.0, 101.0, 99.0, 102.0, 100.5]
        chart.add_measurements(measurements)
        
        center_line, ucl, lcl = chart.calculate_control_limits()
        
        assert center_line == np.mean(measurements)
        assert ucl > center_line
        assert lcl < center_line
        assert ucl - center_line == center_line - lcl  # Symmetric limits
    
    def test_detect_out_of_control(self):
        """Test out-of-control detection."""
        chart = QualityControlChart()
        # Create data with some extreme values
        measurements = [100.0] * 20 + [110.0, 90.0]  # Two outliers
        chart.add_measurements(measurements)
        
        violations_df = chart.detect_out_of_control()
        
        assert len(violations_df) == len(measurements)
        assert 'out_of_control' in violations_df.columns
        assert 'rule_1_violation' in violations_df.columns


class TestProcessCapabilityAnalysis:
    """Test cases for ProcessCapabilityAnalysis."""
    
    def test_capability_analysis_initialization(self):
        """Test capability analysis initialization."""
        analysis = ProcessCapabilityAnalysis(
            specification_lower=95.0,
            specification_upper=105.0,
            target_value=100.0
        )
        
        assert analysis.spec_lower == 95.0
        assert analysis.spec_upper == 105.0
        assert analysis.target_value == 100.0
    
    def test_calculate_capability_indices(self):
        """Test capability indices calculation."""
        analysis = ProcessCapabilityAnalysis(
            specification_lower=94.0,
            specification_upper=106.0
        )
        
        # Create normally distributed data
        measurements = np.random.normal(100.0, 2.0, 100)
        indices = analysis.calculate_capability_indices(measurements)
        
        assert 'Cp' in indices
        assert 'Cpk' in indices
        assert 'yield' in indices
        assert 'ppm_defects' in indices
        assert 'capability_assessment' in indices
        assert 0 <= indices['yield'] <= 1
        assert indices['ppm_defects'] >= 0
    
    def test_capability_assessment(self):
        """Test capability assessment logic."""
        analysis = ProcessCapabilityAnalysis()
        
        # Test different Cpk values
        measurements_good = np.random.normal(100.0, 1.0, 100)  # Good process
        measurements_poor = np.random.normal(100.0, 5.0, 100)  # Poor process
        
        indices_good = analysis.calculate_capability_indices(measurements_good)
        indices_poor = analysis.calculate_capability_indices(measurements_poor)
        
        # Good process should have higher Cpk
        assert indices_good['Cpk'] > indices_poor['Cpk']


class TestQualityAnomalyDetector:
    """Test cases for QualityAnomalyDetector."""
    
    def test_anomaly_detector_initialization(self):
        """Test anomaly detector initialization."""
        detector = QualityAnomalyDetector(contamination=0.1, random_state=42)
        
        assert detector.contamination == 0.1
        assert detector.random_state == 42
        assert not detector.is_trained
    
    def test_fit_and_predict(self):
        """Test anomaly detector training and prediction."""
        detector = QualityAnomalyDetector(contamination=0.1, random_state=42)
        
        # Create training data
        measurements = np.random.normal(100.0, 2.0, 100).tolist()
        
        # Train detector
        detector.fit(measurements)
        assert detector.is_trained
        
        # Make predictions
        test_measurements = [100.0, 110.0, 90.0]  # Include some potential anomalies
        scores, predictions = detector.predict(test_measurements)
        
        assert len(scores) == 3
        assert len(predictions) == 3
        assert all(pred in [-1, 1] for pred in predictions)
    
    def test_detect_anomalies(self):
        """Test anomaly detection with detailed results."""
        detector = QualityAnomalyDetector(contamination=0.1, random_state=42)
        
        # Create training data
        measurements = np.random.normal(100.0, 2.0, 100).tolist()
        detector.fit(measurements)
        
        # Detect anomalies
        results = detector.detect_anomalies(measurements)
        
        assert isinstance(results, pd.DataFrame)
        assert 'sample' in results.columns
        assert 'measurement' in results.columns
        assert 'is_anomaly' in results.columns
        assert 'anomaly_score' in results.columns
        assert len(results) == len(measurements)


class TestQualityControlEvaluator:
    """Test cases for QualityControlEvaluator."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = QualityControlEvaluator()
        
        assert isinstance(evaluator.evaluation_results, dict)
        assert len(evaluator.evaluation_results) == 0
    
    def test_evaluate_control_chart_performance(self):
        """Test control chart performance evaluation."""
        evaluator = QualityControlEvaluator()
        
        measurements = np.random.normal(100.0, 2.0, 100).tolist()
        control_limits = {
            'center_line': 100.0,
            'upper_control_limit': 106.0,
            'lower_control_limit': 94.0,
        }
        
        metrics = evaluator.evaluate_control_chart_performance(
            measurements, control_limits
        )
        
        assert 'points_outside_limits' in metrics
        assert 'percentage_outside_limits' in metrics
        assert 'control_limit_coverage' in metrics
        assert metrics['control_limit_coverage'] >= 0
    
    def test_evaluate_process_capability(self):
        """Test process capability evaluation."""
        evaluator = QualityControlEvaluator()
        
        measurements = np.random.normal(100.0, 2.0, 100).tolist()
        capability_indices = {
            'Cpk': 1.5,
            'Cp': 1.6,
            'yield': 0.99,
            'defect_rate': 0.01,
            'ppm_defects': 10000,
        }
        
        metrics = evaluator.evaluate_process_capability(
            measurements, capability_indices
        )
        
        assert 'cpk_achievement' in metrics
        assert 'quality_level' in metrics
        assert 'process_stability' in metrics
        assert isinstance(metrics['cpk_achievement'], bool)
    
    def test_evaluate_anomaly_detection(self):
        """Test anomaly detection evaluation."""
        evaluator = QualityControlEvaluator()
        
        true_labels = [False] * 80 + [True] * 20
        predicted_labels = [False] * 85 + [True] * 15
        anomaly_scores = np.random.random(100).tolist()
        
        metrics = evaluator.evaluate_anomaly_detection(
            true_labels, predicted_labels, anomaly_scores
        )
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def test_complete_workflow(self):
        """Test the complete quality control workflow."""
        # Generate data
        generator = QualityDataGenerator(random_state=42)
        measurements, true_anomalies, metadata = generator.generate_control_chart_data(
            n_samples=50, target_mean=100.0, target_std=2.0, anomaly_rate=0.1
        )
        
        # Control chart analysis
        control_chart = QualityControlChart()
        control_chart.add_measurements(measurements)
        center_line, ucl, lcl = control_chart.calculate_control_limits()
        violations_df = control_chart.detect_out_of_control()
        
        # Process capability analysis
        capability_analysis = ProcessCapabilityAnalysis(
            specification_lower=94.0,
            specification_upper=106.0
        )
        capability_indices = capability_analysis.calculate_capability_indices(measurements)
        
        # Anomaly detection
        anomaly_detector = QualityAnomalyDetector(random_state=42)
        anomaly_detector.fit(measurements)
        anomaly_results = anomaly_detector.detect_anomalies(measurements)
        
        # Evaluation
        evaluator = QualityControlEvaluator()
        control_chart_data = {
            'center_line': center_line,
            'upper_control_limit': ucl,
            'lower_control_limit': lcl,
        }
        capability_data = {'capability_indices': capability_indices}
        anomaly_data = {
            'true_labels': true_anomalies,
            'predicted_labels': anomaly_results['is_anomaly'].tolist(),
            'anomaly_scores': anomaly_results['anomaly_score'].tolist(),
        }
        
        performance_report = evaluator.create_performance_report(
            measurements, control_chart_data, capability_data, anomaly_data
        )
        
        # Verify all components worked together
        assert len(measurements) == 50
        assert center_line is not None
        assert ucl is not None
        assert lcl is not None
        assert 'Cpk' in capability_indices
        assert len(anomaly_results) == 50
        assert len(performance_report) > 0
        assert 'Overall Performance' in performance_report['Category'].values


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
