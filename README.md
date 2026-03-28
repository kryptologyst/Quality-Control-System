# Quality Control System

A comprehensive quality control system for statistical process control (SPC), process capability analysis, and anomaly detection. This project provides both programmatic APIs and interactive web interfaces for quality control analysis.

## ⚠️ DISCLAIMER

**This is an experimental research/educational tool. Do not use for automated quality control decisions without human review and validation. All results should be interpreted by qualified quality control professionals.**

## Features

### Core Capabilities

- **Statistical Process Control (SPC)**
  - X-bar control charts with configurable control limits
  - Multiple control chart rules (Nelson rules)
  - Out-of-control point detection
  - Process stability assessment

- **Process Capability Analysis**
  - Cp, Cpk, Pp, Ppk indices calculation
  - Process yield and defect rate analysis
  - Specification limit management
  - Capability assessment and grading

- **Advanced Anomaly Detection**
  - Machine learning-based anomaly detection (Isolation Forest)
  - Statistical anomaly detection
  - Configurable sensitivity parameters
  - Performance evaluation metrics

- **Comprehensive Evaluation**
  - Multi-dimensional performance assessment
  - Quality grade calculation
  - Improvement recommendations
  - Detailed reporting

### Data Generation

- **Synthetic Data Generation**
  - Control chart data with various patterns
  - Time series data with seasonality and trends
  - Multivariate quality data
  - Batch quality data with batch effects
  - Realistic manufacturing scenarios

### Visualization

- **Interactive Visualizations**
  - Control charts with rule violations
  - Process capability analysis plots
  - Anomaly detection results
  - Comprehensive dashboards

- **Static Visualizations**
  - High-quality matplotlib plots
  - Publication-ready figures
  - Customizable styling

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager

### Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Optional Dependencies

```bash
# For development
pip install -e ".[dev]"

# For ML tracking
pip install -e ".[tracking]"
```

## Quick Start

### Command Line Usage

Run the main quality control analysis:

```bash
python 0816.py
```

This will:
1. Generate synthetic quality control data
2. Perform control chart analysis
3. Calculate process capability indices
4. Detect anomalies using ML methods
5. Generate comprehensive performance report
6. Create visualizations
7. Save all results to the `assets/` directory

### Interactive Web Application

Launch the Streamlit dashboard:

```bash
streamlit run demo/app.py
```

The web application provides:
- Interactive parameter configuration
- Real-time data generation
- Multiple analysis views
- Downloadable reports
- Comprehensive visualizations

## Project Structure

```
quality-control-system/
├── src/                          # Source code
│   ├── data/                     # Data generation
│   ├── quality_control/          # Core QC algorithms
│   ├── eval/                     # Evaluation metrics
│   └── viz/                      # Visualization
├── configs/                      # Configuration files
├── demo/                         # Web application
├── assets/                       # Output files
├── tests/                        # Unit tests
├── scripts/                      # Utility scripts
├── notebooks/                    # Jupyter notebooks
├── requirements.txt              # Dependencies
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## Configuration

The system uses YAML configuration files for parameter management. Key configuration sections:

### Data Generation
```yaml
data:
  n_samples: 100
  target_mean: 100.0
  target_std: 2.0
  anomaly_rate: 0.05
```

### Control Chart
```yaml
control_chart:
  control_limit_multiplier: 3.0
  subgroup_size: 1
```

### Process Capability
```yaml
capability:
  specification_lower: 94.0
  specification_upper: 106.0
  target_value: 100.0
```

## API Usage

### Basic Quality Control Analysis

```python
from src.quality_control import QualityControlChart, ProcessCapabilityAnalysis
from src.data import QualityDataGenerator

# Generate data
generator = QualityDataGenerator(random_state=42)
measurements, anomalies, metadata = generator.generate_control_chart_data(
    n_samples=100, target_mean=100.0, target_std=2.0
)

# Control chart analysis
control_chart = QualityControlChart()
control_chart.add_measurements(measurements)
center_line, ucl, lcl = control_chart.calculate_control_limits()
violations = control_chart.detect_out_of_control()

# Process capability analysis
capability = ProcessCapabilityAnalysis(
    specification_lower=94.0,
    specification_upper=106.0
)
indices = capability.calculate_capability_indices(measurements)
print(f"Cpk: {indices['Cpk']:.3f}")
```

### Anomaly Detection

```python
from src.quality_control import QualityAnomalyDetector

# Train anomaly detector
detector = QualityAnomalyDetector(contamination=0.1)
detector.fit(measurements)

# Detect anomalies
results = detector.detect_anomalies(measurements)
anomalies = results[results['is_anomaly']]
print(f"Detected {len(anomalies)} anomalies")
```

### Visualization

```python
from src.viz import QualityControlVisualizer

visualizer = QualityControlVisualizer()

# Create control chart
fig = visualizer.plot_control_chart(
    measurements=measurements,
    center_line=center_line,
    upper_control_limit=ucl,
    lower_control_limit=lcl,
    interactive=True
)
fig.show()
```

## Data Schema

### Quality Measurements
- `sample`: Sample number (1, 2, 3, ...)
- `measurement`: Measured value
- `timestamp`: Time of measurement (for time series)
- `batch_id`: Batch identifier (for batch data)
- `true_anomaly`: Ground truth anomaly label

### Process Capability Results
- `Cp`: Process capability index
- `Cpk`: Process capability index (centered)
- `Cpu`: Upper capability index
- `Cpl`: Lower capability index
- `yield`: Process yield percentage
- `ppm_defects`: Defects per million
- `capability_assessment`: Qualitative assessment

### Control Chart Results
- `center_line`: Process center line
- `upper_control_limit`: Upper control limit
- `lower_control_limit`: Lower control limit
- `rule_violations`: Control chart rule violations
- `out_of_control`: Overall out-of-control status

## Evaluation Metrics

### Control Chart Performance
- Points outside control limits
- Average run length (ARL)
- False alarm rate
- Detection sensitivity

### Process Capability
- Capability indices (Cp, Cpk, Pp, Ppk)
- Process yield
- Defect rate (PPM)
- Process centering

### Anomaly Detection
- Precision, Recall, F1-Score
- ROC-AUC, Average Precision
- False positive/negative rates
- Detection efficiency

### Overall Performance
- Weighted quality score
- Quality grade (Excellent/Good/Satisfactory/Marginal/Poor)
- Stability assessment
- Improvement recommendations

## Advanced Features

### Multivariate Quality Control
- Multiple quality variables
- Correlation analysis
- Multivariate control charts
- Cross-variable anomaly detection

### Time Series Analysis
- Seasonal patterns
- Trend analysis
- Cyclic variations
- Missing data handling

### Batch Analysis
- Batch-to-batch variation
- Within-batch variation
- Batch effect detection
- Batch-specific capability

### Manufacturing Scenarios
- Machining processes
- Injection molding
- Welding processes
- Custom process types

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ tests/
ruff check src/ tests/
```

### Type Checking

```bash
mypy src/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Montgomery, D.C. (2019). Introduction to Statistical Quality Control, 8th Edition
- Wheeler, D.J. (2019). Understanding Statistical Process Control, 4th Edition
- ISO 9001:2015 Quality Management Systems
- Six Sigma methodologies and tools

## Support

For questions, issues, or contributions, please:
1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed description
4. Contact the maintainers

---

**Remember: This tool is for educational and research purposes. Always consult with qualified quality control professionals before making production decisions.**
# Quality-Control-System
