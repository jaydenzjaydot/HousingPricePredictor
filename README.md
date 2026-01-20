# Housing Price Predictor

A comprehensive data science project for predicting housing prices using machine learning models.

## Project Overview

This project follows a structured approach to building a predictive model for housing prices, incorporating data exploration, cleaning, feature engineering, model development, and evaluation.

## Task Management

### Approach
This project was managed using a modular, sequential workflow that follows data science best practices:

- **Notebook-based Development**: Jupyter notebooks were used for exploratory analysis and iterative development
- **Code Organization**: Core functionality was extracted into separate Python modules for reusability
- **Version Control**: Regular commits tracked progress through each phase
- **Configuration Management**: Model parameters and settings stored in `model_config.json`

### Key Principles
1. **Separation of Concerns**: Data processing, feature engineering, modeling, and visualization are in separate modules
2. **Reproducibility**: All steps can be reproduced from raw data through to final predictions
3. **Testing**: Unit tests validate core functions before deployment
4. **Documentation**: Code and analysis are documented for team understanding and future maintenance

## Step-by-Step Project Guide

### Phase 1: Data Exploration
**File**: [notebooks/01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb)

- Load raw housing dataset from `data/raw/HousingPriceDataSet.csv`
- Examine dataset structure, shape, and basic statistics
- Identify data types and missing values
- Analyze target variable distribution
- Create initial visualizations to understand data patterns
- Document key findings and insights

**Deliverable**: Understanding of data characteristics and potential issues

---

### Phase 2: Data Cleaning
**File**: [notebooks/02_data_cleaning.ipynb](notebooks/02_data_cleaning.ipynb)

**Tasks**:
- Handle missing values (removal or imputation)
- Remove or treat outliers
- Standardize data formats and types
- Fix inconsistencies and data quality issues
- Validate data integrity after cleaning

**Tools**: 
- Source: [src/data_processing.py](src/data_processing.py)
- Data I/O: [data.py](data.py)

**Output**: Cleaned dataset saved to `data/processed/cleaned_data.csv`

---

### Phase 3: Feature Engineering
**File**: [notebooks/03_feature_engineering.ipynb](notebooks/03_feature_engineering.ipynb)

**Tasks**:
- Create new features from existing variables
- Apply transformations (scaling, encoding categorical variables)
- Feature selection based on relevance and importance
- Handle multicollinearity
- Create polynomial or interaction features as needed

**Tools**:
- Implementation: [src/feature_engineering.py](src/feature_engineering.py)
- Visualization: [src/visualization.py](src/visualization.py)

**Output**: Feature-engineered dataset ready for modeling

---

### Phase 4: Modeling
**File**: [notebooks/04_modeling.ipynb](notebooks/04_modeling.ipynb)

**Tasks**:
- Split data into training and testing sets
- Build baseline models
- Train multiple model architectures (Linear Regression, Tree-based, etc.)
- Hyperparameter tuning using cross-validation
- Compare model performance metrics

**Tools**:
- Implementation: [src/modeling.py](src/modeling.py)
- Configuration: [models/model_config.json](models/model_config.json)

**Output**: Trained models with hyperparameter settings

---

### Phase 5: Evaluation
**File**: [notebooks/05_evaluation.ipynb](notebooks/05_evaluation.ipynb)

**Tasks**:
- Evaluate model performance on test set
- Calculate relevant metrics (MAE, RMSE, R² score)
- Create visualizations of predictions vs actual values
- Analyze residuals and error patterns
- Generate final model selection recommendation

**Tools**:
- Visualization: [src/visualization.py](src/visualization.py)
- Analysis results: [reports/figures/](reports/figures/)

**Output**: Performance report and visualizations saved to `reports/`

---

### Phase 6: Reporting
**File**: [reports/technical_report.md](reports/technical_report.md)

- Summarize methodology and approach
- Present key findings from each phase
- Document model selection rationale
- Provide recommendations for deployment or further improvements
- Include visualizations and performance metrics

---

## Project Structure

```
HousingPricePredictor/
├── notebooks/              # Jupyter notebooks for each phase
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_evaluation.ipynb
│   └── reports/
│       └── figures/
├── src/                    # Core Python modules
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── visualization.py
├── data/                   # Data directory
│   ├── raw/               # Original dataset
│   ├── processed/         # Cleaned and prepared data
│   └── external/          # External data sources
├── models/                # Model configurations and artifacts
│   └── model_config.json
├── reports/               # Final reports and visualizations
│   ├── technical_report.md
│   └── figures/
├── tests/                 # Unit tests
│   └── test_functions.py
├── requirements.txt       # Python dependencies
├── environment.yml        # Conda environment file
└── README.md             # This file

```

## Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Required packages listed in `requirements.txt`

### Installation
```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Project
1. Navigate to the notebooks directory
2. Open Jupyter and start with `01_data_exploration.ipynb`
3. Execute notebooks sequentially through `05_evaluation.ipynb`
4. Review the technical report in `reports/technical_report.md`

## Key Outputs

- **Cleaned Dataset**: `data/processed/cleaned_data.csv`
- **Trained Models**: `models/` directory
- **Model Configuration**: `models/model_config.json`
- **Visualizations**: `reports/figures/` and `notebooks/reports/figures/`
- **Final Report**: `reports/technical_report.md`

## Testing

Run unit tests to validate core functions:
```bash
python -m pytest tests/
```

## Author Notes

This project demonstrates a professional data science workflow with clear separation between exploration (notebooks) and production code (src/), comprehensive documentation, and reproducible results.

## License

See [LICENSE](LICENSE) for details.
