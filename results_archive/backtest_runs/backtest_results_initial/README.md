# Enhanced Visual Backtest Results

This directory contains the output from the enhanced visual backtesting system.

## Files Generated

1. **backtest_comprehensive_analysis.png** - Main visualization with 9 subplots showing:
   - Capital evolution over time
   - Monthly performance
   - Expected value distribution
   - Win rate by horse popularity
   - Profit distribution
   - Rolling performance metrics
   - Drawdown analysis
   - Bet amount distribution
   - Summary statistics table

2. **backtest_detailed_analysis.png** - Additional detailed analysis showing:
   - Expected value vs actual returns scatter plot
   - Model calibration curve
   - Performance by odds range
   - Time-based analysis (if available)

3. **backtest_results.json** - Complete results in JSON format including:
   - Summary metrics
   - Configuration parameters
   - Full trade history
   - Monthly results

4. **trade_history.csv** - Detailed record of every trade made during backtesting

## How to Run

### Basic Usage

```bash
# Run with default settings (2022 training, 2023-2025 testing)
python enhanced_visual_backtest.py

# Specify custom date range
python enhanced_visual_backtest.py --start-year 2020 --end-year 2024 --train-years 2020 2021 --test-years 2022 2023 2024

# Adjust betting parameters
python enhanced_visual_backtest.py --bet-fraction 0.02 --min-ev 1.1
```

### Command Line Options

- `--start-year`: Start year for loading data (default: 2022)
- `--end-year`: End year for loading data (default: 2025)
- `--train-years`: Years to use for training (default: [2022])
- `--test-years`: Years to use for testing (default: [2023, 2024, 2025])
- `--bet-fraction`: Fraction of capital to bet on each race (default: 0.01)
- `--min-ev`: Minimum expected value required to place a bet (default: 1.05)
- `--output-dir`: Directory to save results (default: backtest_results)

## Understanding the Results

### Key Metrics

1. **Total Return**: Overall profit/loss as a percentage of initial capital
2. **Win Rate**: Percentage of bets that resulted in a win
3. **Max Drawdown**: Largest peak-to-trough decline in capital
4. **Sharpe Ratio**: Risk-adjusted return metric (higher is better)

### Interpreting Visualizations

- **Green areas**: Profit periods
- **Red areas**: Loss periods
- **Blue lines**: Average or mean values
- **Dashed lines**: Reference thresholds

### Risk Warnings

The system will display warnings if:
- Total return is negative
- Maximum drawdown exceeds 20%
- Sharpe ratio is below 0.5

## Data Requirements

The backtest system looks for data in the following order:
1. `data_with_payout/` directory (preferred for actual odds)
2. `data/` directory (fallback)

File naming conventions:
- `{year}_with_payout.xlsx` - Files with payout information
- `{year}.xlsx` - Standard data files