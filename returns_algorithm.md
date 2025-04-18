# Returns Algorithm Implementation

## Overview
The Returns Algorithm is a momentum-based portfolio selection strategy that:
1. Selects the top 5 performing assets based on their 3-month rolling returns
2. Allocates equal weights (20% each) to the selected assets
3. Rebalances the portfolio quarterly

## Implementation Details

### Data Processing
- The algorithm takes daily sector returns data as input
- For each rebalancing date, it calculates rolling returns over the prior 3 months (63 trading days)
- Returns are calculated as cumulative returns over the lookback period

### Asset Selection
The `select_top_performers` function:
1. Sorts all assets by their rolling returns in descending order
2. Selects exactly the top 5 performing assets
3. In the rare case where there's no valid data (e.g., at the start of the sample), it randomly selects 5 assets as a last resort

### Portfolio Construction
The `calculate_weights` function:
1. Takes the selected assets and assigns each a 20% weight
2. This equal weighting ensures:
   - No single asset dominates the portfolio
   - The portfolio captures the momentum effect across multiple sectors
   - The strategy remains simple and transparent

### Rebalancing
- The portfolio is rebalanced quarterly
- At each rebalancing date:
  1. Calculate new rolling returns for all assets
  2. Select the new top 5 performers
  3. Reset weights to 20% each
  4. Log the selection and returns information for analysis

## Rationale

### Why Top 5 Assets?
- Limiting to 5 assets provides sufficient diversification while maintaining focus
- Equal weights ensure the strategy isn't overly concentrated in any single sector
- The 20% maximum weight constraint helps manage risk

### Why 3-Month Lookback?
- 3 months (63 trading days) is long enough to capture meaningful trends
- Short enough to be responsive to changing market conditions
- Balances between noise reduction and signal capture

### Why Quarterly Rebalancing?
- Quarterly rebalancing reduces transaction costs
- Provides enough time for momentum to play out
- Aligns with typical institutional rebalancing schedules

### Why Equal Weights?
- Equal weights are simple and transparent
- Avoids overfitting to historical data
- Reduces the impact of estimation error in returns
- Ensures the portfolio captures the pure momentum effect

## Performance Considerations
- The algorithm focuses on relative performance rather than absolute returns
- It will select the best performing assets even if all returns are negative
- The strategy is designed to capture sector rotation and momentum effects
- The quarterly rebalancing helps manage turnover and transaction costs

## Output Information
The algorithm outputs detailed information at each rebalancing:
- Selected assets and their weights
- Returns of selected assets in the prior period
- Next 3 highest-returning assets that weren't selected
- Returns of those near-miss assets

This information helps in:
- Verifying the selection process
- Understanding why certain assets were chosen
- Analyzing the performance of near-miss assets
- Evaluating the effectiveness of the strategy
