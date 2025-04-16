# Weighted Top Five Algorithm Implementation

## Overview
The Weighted Top Five Algorithm is a momentum-based portfolio selection strategy that:
1. Selects the top 5 performing assets based on their 3-month rolling returns
2. Allocates weights as 40%, 25%, 20%, 10%, 5% to the top 5 performers in ranked order
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
1. Takes the selected assets in order of their returns
2. Assigns weights as:
   - 40% to the top performer
   - 25% to the second best
   - 20% to the third best
   - 10% to the fourth best
   - 5% to the fifth best
3. This weighted allocation ensures:
   - Strong momentum tilt towards the best performers
   - Gradual reduction in exposure to lower-ranked assets
   - Maintains some diversification while emphasizing top performers

### Rebalancing
- The portfolio is rebalanced quarterly
- At each rebalancing date:
  1. Calculate new rolling returns for all assets
  2. Select the new top 5 performers
  3. Reset weights according to the 40/25/20/10/5 distribution
  4. Log the selection and returns information for analysis

## Rationale

### Why Top 5 Assets?
- Limiting to 5 assets provides sufficient diversification while maintaining focus
- The weighted allocation allows for stronger emphasis on top performers
- The strategy captures momentum while managing concentration risk

### Why 3-Month Lookback?
- 3 months (63 trading days) is long enough to capture meaningful trends
- Short enough to be responsive to changing market conditions
- Balances between noise reduction and signal capture

### Why Quarterly Rebalancing?
- Quarterly rebalancing reduces transaction costs
- Provides enough time for momentum to play out
- Aligns with typical institutional rebalancing schedules

### Why Weighted Allocation?
- The 40/25/20/10/5 distribution:
  - Strongly emphasizes the best performers
  - Provides meaningful exposure to second and third best
  - Maintains smaller positions in fourth and fifth
  - Creates a natural momentum tilt
  - Balances concentration and diversification

## Performance Considerations
- The algorithm focuses on relative performance rather than absolute returns
- It will select the best performing assets even if all returns are negative
- The strategy is designed to capture sector rotation and momentum effects
- The weighted allocation provides stronger exposure to top performers
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
