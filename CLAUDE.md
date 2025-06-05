# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment

Always use virtual environment for Python execution:
```bash
# Activate virtual environment
source .venv/bin/activate

# Or use the helper script
./run_venv.sh python <your_script.py>
```

## Common Development Commands

### Setup and Installation
```bash
# Initial setup (creates .venv and installs dependencies)
./setup_env.sh

# Activate virtual environment manually
source .venv/bin/activate

# Install/update dependencies
pip install -r requirements.txt
```

### Main Application Commands
```bash
# Unified CLI entry point (main.py)
python main.py backtest                    # Run backtest with default settings
python main.py collect --start-year 2024  # Collect data for specific year
python main.py encode --start-year 2022   # Encode data for machine learning
python main.py train                       # Train the model

# Backtest with options
python main.py backtest --min-ev 1.2 --no-trifecta --train-years 2022 2023 --test-years 2024 2025
```

### Data Collection and Processing
```bash
# Enhanced scraping with payout data (recommended)
python src/data_processing/enhanced_scraping.py --year 2024

# Checkpoint-based scraping (resumable)
python src/data_processing/data_scraping_with_checkpoint.py --start 2024 --end 2024

# Data encoding (v2 with payout support)
python src/data_processing/data_encoding_v2.py --start 2022 --end 2023
```

### Testing
```bash
# Run specific test files (no formal test runner configured)
python test_payout_parsing.py
python debug_backtest.py

# Code quality tools (dependencies available)
black src/         # Code formatting
flake8 src/        # Linting  
mypy src/          # Type checking
```

### Visualization and Analysis
```bash
# Enhanced backtest with visualization
python enhanced_visual_backtest.py --train-years 2022 --test-years 2024 2025 --min-ev 1.2
```

## Project Architecture

### Core Components

1. **Main Entry Point**: `main.py` - Unified CLI for all operations
2. **Configuration**: `src/core/config.py` - Centralized config management using `config/config.yaml`
3. **Utilities**: `src/core/utils.py` - Common data loading, feature processing, model management
4. **Base Strategy**: `src/strategies/base.py` - Abstract base class for all betting strategies

### Data Flow Architecture

```
Raw Data Collection → Data Encoding → Model Training → Backtesting → Results
     (scraping)         (features)      (LightGBM)     (strategies)   (metrics)
```

### Directory Structure Pattern

- `src/` - All source code organized by functionality
- `data/` - Raw scraped data (year-based Excel files)
- `data_with_payout/` - Enhanced data with payout information
- `encoded/` - Processed data ready for machine learning
- `models/` - Trained LightGBM models
- `results/` - Backtest results and analysis
- `config/` - Configuration files and parameters

### Strategy Pattern Implementation

All betting strategies inherit from `BaseStrategy` and implement:
- `create_additional_features()` - Strategy-specific feature engineering
- `train_model()` - Model training logic
- `select_bets()` - Bet selection algorithm
- `calculate_bet_amount()` - Position sizing
- `_calculate_profit()` - Profit/loss calculation

### Data Pipeline Architecture

1. **Scraping Layer**: Collects race data from netkeiba.com with checkpoint support
2. **Processing Layer**: Encodes categorical variables, creates features, handles missing data
3. **ML Layer**: LightGBM with Optuna optimization, SHAP interpretability
4. **Strategy Layer**: Kelly criterion, expected value calculation, risk management
5. **Backtest Layer**: Time-series validation, comprehensive performance metrics

## Configuration Management

All settings are centralized in `config/config.yaml`:
- Scraping parameters (years, workers, timeout)
- Model parameters (LightGBM settings, optimization trials)
- Backtest parameters (capital, thresholds, risk limits)
- Feature engineering settings

Access config via `from src.core.config import config` in code.

## Data Handling Patterns

### Loading Data
Use the unified `DataLoader` class which handles both regular and payout-enhanced data automatically:
```python
from src.core.utils import DataLoader
loader = DataLoader()
data = loader.load_race_data(years=[2022, 2023], use_payout_data=True)
```

### Feature Processing  
Use `FeatureProcessor` for consistent feature engineering:
```python
from src.core.utils import FeatureProcessor
processor = FeatureProcessor()
data = processor.prepare_basic_features(data)
data = processor.create_target_variables(data)
```

## Key Implementation Notes

- Always use virtual environment for Python execution
- Data files follow naming pattern: `{year}.xlsx` or `{year}_with_payout.xlsx`
- Race IDs format: YYYYMMDDCCR (year/month/day/course/race)
- Target variables: `is_win`, `is_place` (3rd or better), `is_exacta` (2nd or better)
- All betting strategies support multiple bet types: trifecta, quinella, wide, win/place
- Backtest uses time-series split (train on early years, test on later years)
- Kelly criterion with risk limits applied for position sizing

## Coding Practices

- Edit existing files rather than creating new ones when improving code
- Use the centralized config system rather than hardcoded values
- Follow the established patterns for data loading and feature processing
- Leverage the base strategy class for consistent backtest implementation
- Use the logger from `src.core.utils.setup_logger()` for consistent logging

## Development Guidelines

- ファイルの修正を行う際は新しいファイルを作らず同じファイルを更新してください
- コードの改善を行う際は極力新しいファイルを作らないこと
- 関数名やファイル名はわかりやすく名付けること

## Important Memory
- 決してデモデータを用意しないこと