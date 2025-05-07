# Example usage of the pricing optimization model

# Initialize system parameters with default values
params = PricingParams()

# Step 1: Run an initial simulation with default pricing
initial_model = PricingSystemModel(params)
initial_results = initial_model.run_simulation()

print("Initial Pricing Strategy Results:")
print(f"Enterprise Price: ${initial_model.enterprise_monthly_price}")
print(f"Pro Price: ${initial_model.pro_monthly_price}")
print(f"Lifetime Deal: ${initial_model.lifetime_deal_price}")
print(f"Consultant Fee Share: {initial_model.consultant_fee_percentage*100:.1f}%")
print(f"Trial Length: {initial_model.trial_length_days} days")
print("\nInitial Results:")
print(f"NPV: ${initial_results['npv']:,.2f}")
print(f"Final MRR: ${initial_results['final_mrr']:,.2f}")
print(f"Final LTV/CAC: {initial_results['final_ltv_cac']:.2f}")
print(f"Final Gross Margin: {initial_results['final_gross_margin']*100:.1f}%")
print(f"Enterprise Users: {initial_results['final_enterprise_users']:.0f}")
print(f"Pro Users: {initial_results['final_pro_users']:.0f}")

# Step 2: Run grid search to find optimal pricing across different objectives
print("\nRunning grid search to find optimal pricing strategies...")
grid_results = grid_search_pricing(params, grid_points=4)

# Step 3: Analyze grid search results
analysis = analyze_grid_search_results(grid_results)

print("\nBest Strategy for NPV:")
best_npv = grid_results['best_results']['npv']
print(f"Enterprise Price: ${best_npv['params']['enterprise']:.2f}")
print(f"Pro Price: ${best_npv['params']['pro']:.2f}")
print(f"Lifetime Deal: ${best_npv['params']['lifetime']:.2f}")
print(f"NPV: ${best_npv['value']:,.2f}")

print("\nBest Strategy for LTV/CAC:")
best_ltv_cac = grid_results['best_results']['ltv_cac']
print(f"Enterprise Price: ${best_ltv_cac['params']['enterprise']:.2f}")
print(f"Pro Price: ${best_ltv_cac['params']['pro']:.2f}")
print(f"Lifetime Deal: ${best_ltv_cac['params']['lifetime']:.2f}")
print(f"LTV/CAC: {best_ltv_cac['value']:.2f}")

print("\nBest Strategy for MRR:")
best_mrr = grid_results['best_results']['mrr']
print(f"Enterprise Price: ${best_mrr['params']['enterprise']:.2f}")
print(f"Pro Price: ${best_mrr['params']['pro']:.2f}")
print(f"Lifetime Deal: ${best_mrr['params']['lifetime']:.2f}")
print(f"MRR: ${best_mrr['value']:,.2f}")

print("\nBest Strategy for Gross Margin:")
best_gm = grid_results['best_results']['gross_margin']
print(f"Enterprise Price: ${best_gm['params']['enterprise']:.2f}")
print(f"Pro Price: ${best_gm['params']['pro']:.2f}")
print(f"Lifetime Deal: ${best_gm['params']['lifetime']:.2f}")
print(f"Gross Margin: {best_gm['value']*100:.1f}%")

# Step 4: Run full optimization for NPV
print("\nRunning full optimization for NPV...")
optimized_results = optimize_pricing_strategy(params, objective='npv')

print("\nOptimized Pricing Strategy:")
print(f"Enterprise Price: ${optimized_results['enterprise_price']:.2f}")
print(f"Pro Price: ${optimized_results['pro_price']:.2f}")
print(f"Lifetime Deal: ${optimized_results['lifetime_price']:.2f}")
print(f"Consultant Fee Share: {optimized_results['consultant_fee']*100:.1f}%")
print(f"Trial Length: {optimized_results['trial_days']:.0f} days")
print("\nOptimized Results:")
print(f"NPV: ${optimized_results['npv']:,.2f}")
print(f"Final MRR: ${optimized_results['final_mrr']:,.2f}")
print(f"Final LTV/CAC: {optimized_results['final_ltv_cac']:.2f}")
print(f"Final Gross Margin: {optimized_results['final_gross_margin']*100:.1f}%")

# Step 5: Visualize results
fig = visualize_pricing_strategies(optimized_results, grid_results)

# Step 6: Print pricing insights and recommendations
print("\nKey Pricing Insights:")
print("1. Enterprise Price Elasticity:")
for price, elasticity in analysis['enterprise_price_elasticity'].items():
    print(f"   At Pro Price ${price:.2f}: {elasticity:.2f}")

print("\n2. Pro Price Elasticity:")
for price, elasticity in analysis['pro_price_elasticity'].items():
    print(f"   At Enterprise Price ${price:.2f}: {elasticity:.2f}")

print("\n3. Price vs Outcome Correlations:")
for price, metrics in analysis['price_outcome_correlations'].items():
    print(f"   {price}:")
    for metric, corr in metrics.items():
        print(f"      {metric}: {corr:.2f}")

print("\n4. Pareto-Optimal Pricing Strategies:")
for i, strategy in enumerate(analysis['pareto_optimal_strategies'][:5]):  # Show top 5
    print(f"   Strategy {i+1}:")
    print(f"      Enterprise: ${strategy['enterprise_price']:.2f}")
    print(f"      Pro: ${strategy['pro_price']:.2f}")
    print(f"      Lifetime: ${strategy['lifetime_price']:.2f}")
    print(f"      NPV: ${strategy['npv']:,.2f}")
    print(f"      LTV/CAC: {strategy['final_ltv_cac']:.2f}")
    print(f"      MRR: ${strategy['final_mrr']:,.2f}")
    print(f"      Gross Margin: {strategy['final_gross_margin']*100:.1f}%")

# Step 7: Implement pricing decision heuristics
print("\nPricing Decision Heuristics:")
print("1. If seeking maximum long-term value (NPV), consider:")
print(f"   - Enterprise Price: ${optimized_results['enterprise_price']:.2f}")
print(f"   - Pro Price: ${optimized_results['pro_price']:.2f}")
print(f"   - Lifetime Deal: ${optimized_results['lifetime_price']:.2f}")
print(f"   - This maximizes long-term profitability but may sacrifice short-term metrics")

print("\n2. If prioritizing customer acquisition and growth:")
print(f"   - Lower Enterprise Price: ${min(best_npv['params']['enterprise'], best_ltv_cac['params']['enterprise']):.2f}")
print(f"   - Lower Pro Price: ${min(best_npv['params']['pro'], best_ltv_cac['params']['pro']):.2f}")
print(f"   - Maintain high consultant fee share (>75%) to attract network participants")
print(f"   - Extend trial period to 21-30 days for enterprise customers")
print(f"   - This approach prioritizes network effects over immediate profitability")

print("\n3. If focusing on sustainability and unit economics:")
print(f"   - Higher Enterprise Price: ${best_gm['params']['enterprise']:.2f}")
print(f"   - Higher Pro Price: ${best_gm['params']['pro']:.2f}")
print(f"   - Reduce acquisition costs by focusing on organic growth")
print(f"   - Target LTV/CAC ratio above 4 for sustainability")
print(f"   - This approach ensures financial stability but may slow growth")

print("\n4. If balancing growth and profitability (recommended):")
print(f"   - Enterprise Price: ${(best_npv['params']['enterprise'] + best_ltv_cac['params']['enterprise'])/2:.2f}")
print(f"   - Pro Price: ${(best_npv['params']['pro'] + best_ltv_cac['params']['pro'])/2:.2f}")
print(f"   - Lifetime Deal: ${best_npv['params']['lifetime']:.2f}")
print(f"   - Trial Length: 14-21 days")
print(f"   - Consultant Fee Share: 70-75%")
print(f"   - This balanced approach optimizes for sustainable growth")

print("\n5. Dynamic pricing considerations:")
print("   - Adjust enterprise pricing based on team size (per-seat model with volume discount)")
print("   - Implement feature-based tiering for Pro users")
print("   - Consider seasonal promotions for lifetime deals")
print("   - Adjust consultant fee share based on marketplace maturity")
print("   - Implement 'free in public' option with clear upgrade paths")

print("\n6. Network effect optimization:")
print("   - Prioritize grower onboarding with zero friction")
print("   - Subsidize initial consultant participation")
print("   - Create incentives for public content creation")
print("   - Consider 'freemium plus' model where basic features are free,")
print("     premium features require payment, but all features are free if output is public")

print("\nRecommended Implementation Strategy:")
print("1. Start with balanced pricing approach (#4)")
print("2. Monitor key metrics weekly: acquisition rates, conversion rates, churn rates")
print("3. Adjust pricing in small increments (±5-10%) based on observed elasticity")
print("4. Prioritize network effect catalysts in early stages")
print("5. Implement more sophisticated price discrimination as user base grows")
print("6. Revisit and re-run model monthly with updated parameters based on real data")

# Display visualization
print("\nRefer to the visualization for detailed system dynamics and pricing impacts.")
plt.show()

# Advanced Usage: Time-Varying Pricing and Network Effect Analysis

## 1. Analyze Price Elasticity and Network Effects

```python
# First, analyze price elasticity and network effects to understand key dynamics
elasticity_analysis = analyze_network_effects_and_elasticity(params)

print("\nPrice Elasticity Analysis:")
print(f"Enterprise Price Elasticity: {elasticity_analysis['enterprise_elasticity']:.2f}")
print(f"Pro Price Elasticity: {elasticity_analysis['pro_elasticity']:.2f}")

print("\nRevenue Maximizing Prices:")
print(f"Enterprise: ${elasticity_analysis['revenue_maximizing_enterprise_price']:.2f}")
print(f"Pro: ${elasticity_analysis['revenue_maximizing_pro_price']:.2f}")

print("\nNPV Maximizing Prices:")
print(f"Enterprise: ${elasticity_analysis['npv_maximizing_enterprise_price']:.2f}")
print(f"Pro: ${elasticity_analysis['npv_maximizing_pro_price']:.2f}")

# Display the visualizations
elasticity_analysis['visualizations']
plt.show()
```

## 2. Hill-Climbing Search for Time-Varying Pricing

```python
# Use hill climbing to find optimal time-varying pricing strategy
print("\nRunning hill-climbing optimization for time-varying pricing...")
heuristic_results = pricing_heuristic_search(
    params,
    iterations=500,
    time_horizon=24,
    smooth_constraint=True,
    max_price_change_pct=10
)

print("\nOptimal Time-Varying Pricing Strategy:")
print(f"Enterprise Prices: {[round(p, 2) for p in heuristic_results['enterprise_price']]}")
print(f"Pro Prices: {[round(p, 2) for p in heuristic_results['pro_price']]}")
print(f"NPV: ${heuristic_results['npv']:,.2f}")
print(f"Final MRR: ${heuristic_results['final_mrr']:,.2f}")
print(f"Final LTV/CAC: {heuristic_results['final_ltv_cac']:.2f}")

# Visualize the time-varying pricing strategy
time_varying_fig = visualize_pricing_strategies(heuristic_results)
plt.show()
```

## 3. CP-SAT Optimization for Constrained Pricing

```python
# For more sophisticated optimization with constraints, use CP-SAT
if ORTOOLS_AVAILABLE:
    print("\nRunning CP-SAT optimization with constraints...")
    cpsat_results = optimize_with_cpsat(
        params,
        objective='npv',
        time_horizon=12,
        price_steps=8,
        max_price_change_pct=15
    )
    
    if cpsat_results:
        print("\nCP-SAT Optimized Pricing Strategy:")
        print(f"Enterprise Prices: {[round(p, 2) for p in cpsat_results['enterprise_price']]}")
        print(f"Pro Prices: {[round(p, 2) for p in cpsat_results['pro_price']]}")
        print(f"NPV: ${cpsat_results['npv']:,.2f}")
        print(f"Final MRR: ${cpsat_results['final_mrr']:,.2f}")
        print(f"Final LTV/CAC: {cpsat_results['final_ltv_cac']:.2f}")
        
        # Visualize CP-SAT optimized strategy
        cpsat_fig = visualize_pricing_strategies(cpsat_results)
        plt.show()
else:
    print("\nGoogle OR-Tools not available. Install with: pip install ortools")
```

## 4. Compare Different Optimization Approaches

```python
# Compare the results from different optimization approaches
print("\nComparison of Optimization Approaches:")
print("                   | Initial  | Static    | Time-Varying | CP-SAT")
print("-------------------|----------|-----------|--------------|--------")
print(f"Enterprise Start   | ${initial_model.enterprise_price[0]:.2f}    | ${optimized_results['enterprise_price']}    | ${heuristic_results['enterprise_price'][0]:.2f}          | {cpsat_results['enterprise_price'][0] if ORTOOLS_AVAILABLE and cpsat_results else 'N/A'}")
print(f"Enterprise End     | ${initial_model.enterprise_price[0]:.2f}    | ${optimized_results['enterprise_price']}    | ${heuristic_results['enterprise_price'][-1]:.2f}          | {cpsat_results['enterprise_price'][-1] if ORTOOLS_AVAILABLE and cpsat_results else 'N/A'}")
print(f"Pro Start          | ${initial_model.pro_price[0]:.2f}     | ${optimized_results['pro_price']}     | ${heuristic_results['pro_price'][0]:.2f}           | {cpsat_results['pro_price'][0] if ORTOOLS_AVAILABLE and cpsat_results else 'N/A'}")
print(f"Pro End            | ${initial_model.pro_price[0]:.2f}     | ${optimized_results['pro_price']}     | ${heuristic_results['pro_price'][-1]:.2f}           | {cpsat_results['pro_price'][-1] if ORTOOLS_AVAILABLE and cpsat_results else 'N/A'}")
print(f"NPV                | ${initial_results['npv']:,.0f} | ${optimized_results['npv']:,.0f} | ${heuristic_results['npv']:,.0f}   | ${cpsat_results['npv']:,.0f if ORTOOLS_AVAILABLE and cpsat_results else 'N/A'}")
print(f"Final MRR          | ${initial_results['final_mrr']:,.0f} | ${optimized_results['final_mrr']:,.0f} | ${heuristic_results['final_mrr']:,.0f}   | ${cpsat_results['final_mrr']:,.0f if ORTOOLS_AVAILABLE and cpsat_results else 'N/A'}")
print(f"Final LTV/CAC      | {initial_results['final_ltv_cac']:.2f}   | {optimized_results['final_ltv_cac']:.2f}   | {heuristic_results['final_ltv_cac']:.2f}     | {cpsat_results['final_ltv_cac']:.2f if ORTOOLS_AVAILABLE and cpsat_results else 'N/A'}")

# Create a comparison visualization
comparison_fig, ax = plt.subplots(figsize=(12, 8))

# Set up x-axis for months
months = list(range(min(36, len(heuristic_results['model'].history['enterprise_users']))))

# Plot initial MRR
ax.plot(months[:len(initial_results['history']['mrr'])], 
        initial_results['history']['mrr'], 
        label='Initial Static Pricing')

# Plot optimized static pricing MRR
ax.plot(months[:len(optimized_results['model'].history['mrr'])], 
        optimized_results['model'].history['mrr'], 
        label='Optimized Static Pricing')

# Plot time-varying pricing MRR
ax.plot(months[:len(heuristic_results['model'].history['mrr'])], 
        heuristic_results['model'].history['mrr'], 
        label='Time-Varying Pricing')

# Plot CP-SAT pricing MRR if available
if ORTOOLS_AVAILABLE and cpsat_results:
    ax.plot(months[:len(cpsat_results['model'].history['mrr'])], 
            cpsat_results['model'].history['mrr'], 
            label='CP-SAT Constrained Pricing')

ax.set_title('MRR Comparison Across Pricing Strategies')
ax.set_xlabel('Month')
ax.set_ylabel('Monthly Recurring Revenue ($)')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
```

## 5. Pricing Strategy Implications

Based on the comprehensive analysis, several key insights emerge:

1. **Network effects are critical** - Our analysis shows that strengthening network effects can increase NPV by ${elasticity_analysis['network_effect_results'][-1]['npv'] - elasticity_analysis['network_effect_results'][0]['npv']:,.0f}, dramatically improving user acquisition and retention.

2. **Price sensitivity differs by segment** - Enterprise customers are less price sensitive (elasticity: {elasticity_analysis['enterprise_elasticity']:.2f}) than Pro users (elasticity: {elasticity_analysis['pro_elasticity']:.2f}), suggesting different pricing approaches for each segment.

3. **Time-varying pricing is superior** - Dynamic pricing strategies outperform static pricing by approximately ${heuristic_results['npv'] - optimized_results['npv']:,.0f} in NPV, due to better alignment with user growth and network effect development.

4. **Entry pricing vs. mature pricing** - Optimal strategy involves starting with lower prices (Enterprise: ${heuristic_results['enterprise_price'][0]:.2f}, Pro: ${heuristic_results['pro_price'][0]:.2f}) to build the network, then gradually increasing as value perception improves (Enterprise: ${heuristic_results['enterprise_price'][-1]:.2f}, Pro: ${heuristic_results['pro_price'][-1]:.2f}).

5. **Revenue diversification matters** - At maturity, subscription revenue represents {100 * (heuristic_results['final_mrr'] - heuristic_results['final_contract_revenue'] - heuristic_results['final_wellspring_revenue']) / heuristic_results['final_mrr']:.1f}% of MRR, with contracts and Wellspring marketplace providing significant additional revenue streams.

## 6. Recommended Implementation Roadmap

1. **Launch Phase (Months 0-6)**
   - Enterprise Price: ${heuristic_results['enterprise_price'][0]:.2f} monthly
   - Pro Price: ${heuristic_results['pro_price'][0]:.2f} monthly
   - Focus on grower onboarding and content generation
   - Offer trial periods of 14-21 days

2. **Growth Phase (Months 7-18)**
   - Enterprise Price: ${heuristic_results['enterprise_price'][6]:.2f} monthly
   - Pro Price: ${heuristic_results['pro_price'][6]:.2f} monthly
   - Introduce lifetime deals for early adopters
   - Launch Wellspring marketplace with high consultant commission (75%)

3. **Mature Phase (Months 19+)**
   - Enterprise Price: ${heuristic_results['enterprise_price'][-1]:.2f} monthly
   - Pro Price: ${heuristic_results['pro_price'][-1]:.2f} monthly
   - Adjust commissions based on marketplace maturity
   - Implement tiered feature-based pricing
   - Optimize "free in public" thresholds based on content value