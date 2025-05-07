import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd
from itertools import product
try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

# Define optimization functions for pricing strategy
def optimize_pricing_strategy(params, objective='npv', constraints=None):
    """
    Optimize pricing strategy using numerical optimization
    
    Parameters:
    -----------
    params : PricingParams
        System parameters
    objective : str
        Optimization objective ('npv', 'ltv_cac', 'mrr', or 'gross_margin')
    constraints : dict
        Constraint parameters (min/max values)
        
    Returns:
    --------
    dict
        Optimized pricing parameters and resulting metrics
    """
    if constraints is None:
        constraints = {
            'enterprise_price_min': 100,
            'enterprise_price_max': 500,
            'pro_price_min': 20,
            'pro_price_max': 150,
            'lifetime_price_min': 500,
            'lifetime_price_max': 3000,
            'consultant_fee_min': 0.5,
            'consultant_fee_max': 0.9,
            'trial_days_min': 7,
            'trial_days_max': 30
        }
    
    # Define objective function for optimization
    def objective_function(x):
        enterprise_price, pro_price, lifetime_price, consultant_fee, trial_days = x
        
        initial_prices = {
            'enterprise': enterprise_price,
            'pro': pro_price,
            'lifetime': lifetime_price,
            'consultant_fee': consultant_fee,
            'trial_days': trial_days
        }
        
        model = PricingSystemModel(params, initial_prices)
        results = model.run_simulation()
        
        # Different objectives to optimize for
        objectives = {
            'npv': -results['npv'],  # Negative because we want to maximize
            'ltv_cac': -results['final_ltv_cac'],
            'mrr': -results['final_mrr'],
            'gross_margin': -results['final_gross_margin']
        }
        
        # Return negative value because scipy.optimize.minimize minimizes function
        return objectives.get(objective, objectives['npv'])
    
    # Initial guess
    x0 = [
        (constraints['enterprise_price_min'] + constraints['enterprise_price_max']) / 2,
        (constraints['pro_price_min'] + constraints['pro_price_max']) / 2,
        (constraints['lifetime_price_min'] + constraints['lifetime_price_max']) / 2,
        (constraints['consultant_fee_min'] + constraints['consultant_fee_max']) / 2,
        (constraints['trial_days_min'] + constraints['trial_days_max']) / 2
    ]
    
    # Define bounds
    bounds = [
        (constraints['enterprise_price_min'], constraints['enterprise_price_max']),
        (constraints['pro_price_min'], constraints['pro_price_max']),
        (constraints['lifetime_price_min'], constraints['lifetime_price_max']),
        (constraints['consultant_fee_min'], constraints['consultant_fee_max']),
        (constraints['trial_days_min'], constraints['trial_days_max'])
    ]
    
    # Run optimization
    result = minimize(
        objective_function,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 50}
    )
    
    # Extract results
    enterprise_price, pro_price, lifetime_price, consultant_fee, trial_days = result.x
    
    # Run final simulation with optimized parameters
    optimized_prices = {
        'enterprise': enterprise_price,
        'pro': pro_price,
        'lifetime': lifetime_price,
        'consultant_fee': consultant_fee,
        'trial_days': trial_days
    }
    
    final_model = PricingSystemModel(params, optimized_prices)
    final_results = final_model.run_simulation()
    
    return {
        'enterprise_price': enterprise_price,
        'pro_price': pro_price,
        'lifetime_price': lifetime_price,
        'consultant_fee': consultant_fee,
        'trial_days': trial_days,
        'npv': final_results['npv'],
        'final_mrr': final_results['final_mrr'],
        'final_ltv_cac': final_results['final_ltv_cac'],
        'final_gross_margin': final_results['final_gross_margin'],
        'model': final_model,
        'history': final_results['history']
    }

def grid_search_pricing(params, grid_points=5):
    """
    Perform a grid search to find optimal pricing across multiple objectives
    
    Parameters:
    -----------
    params : PricingParams
        System parameters
    grid_points : int
        Number of points to test for each parameter
        
    Returns:
    --------
    dict
        Results for different optimization objectives
    """
    # Define parameter ranges for grid search
    enterprise_prices = np.linspace(150, 400, grid_points)
    pro_prices = np.linspace(30, 100, grid_points)
    lifetime_prices = np.linspace(800, 2000, grid_points)
    
    # Fixed for simplicity in grid search
    consultant_fee = 0.7
    trial_days = 14
    
    best_results = {
        'npv': {'value': float('-inf'), 'params': None},
        'ltv_cac': {'value': float('-inf'), 'params': None},
        'mrr': {'value': float('-inf'), 'params': None},
        'gross_margin': {'value': float('-inf'), 'params': None}
    }
    
    all_results = []
    
    # Iterate through parameter combinations
    for ep, pp, lp in product(enterprise_prices, pro_prices, lifetime_prices):
        initial_prices = {
            'enterprise': ep,
            'pro': pp,
            'lifetime': lp,
            'consultant_fee': consultant_fee,
            'trial_days': trial_days
        }
        
        model = PricingSystemModel(params, initial_prices)
        results = model.run_simulation()
        
        result_entry = {
            'enterprise_price': ep,
            'pro_price': pp,
            'lifetime_price': lp,
            'consultant_fee': consultant_fee,
            'trial_days': trial_days,
            'npv': results['npv'],
            'final_mrr': results['final_mrr'],
            'final_ltv_cac': results['final_ltv_cac'],
            'final_gross_margin': results['final_gross_margin'],
            'final_enterprise_users': results['final_enterprise_users'],
            'final_pro_users': results['final_pro_users']
        }
        
        all_results.append(result_entry)
        
        # Update best results for each objective
        if results['npv'] > best_results['npv']['value']:
            best_results['npv']['value'] = results['npv']
            best_results['npv']['params'] = {k: v for k, v in initial_prices.items()}
            
        if results['final_ltv_cac'] > best_results['ltv_cac']['value']:
            best_results['ltv_cac']['value'] = results['final_ltv_cac']
            best_results['ltv_cac']['params'] = {k: v for k, v in initial_prices.items()}
            
        if results['final_mrr'] > best_results['mrr']['value']:
            best_results['mrr']['value'] = results['final_mrr']
            best_results['mrr']['params'] = {k: v for k, v in initial_prices.items()}
            
        if results['final_gross_margin'] > best_results['gross_margin']['value']:
            best_results['gross_margin']['value'] = results['final_gross_margin']
            best_results['gross_margin']['params'] = {k: v for k, v in initial_prices.items()}
    
    return {
        'best_results': best_results,
        'all_results': all_results
    }

def analyze_grid_search_results(grid_results):
    """
    Analyze grid search results to identify pricing patterns and tradeoffs
    
    Parameters:
    -----------
    grid_results : dict
        Results from grid_search_pricing function
        
    Returns:
    --------
    dict
        Analysis of pricing patterns and tradeoffs
    """
    results_df = pd.DataFrame(grid_results['all_results'])
    
    # Calculate price elasticity from the data
    ep_elasticity = {}
    pp_elasticity = {}
    
    # Enterprise price elasticity
    for pp in results_df['pro_price'].unique():
        subset = results_df[results_df['pro_price'] == pp]
        if len(subset) > 1:
            # Calculate log-log elasticity
            log_price = np.log(subset['enterprise_price'])
            log_users = np.log(subset['final_enterprise_users'])
            
            if len(log_price) > 1 and len(log_users) > 1 and not np.isnan(log_users).any():
                try:
                    elasticity = np.polyfit(log_price, log_users, 1)[0]
                    ep_elasticity[pp] = elasticity
                except:
                    pass
    
    # Pro price elasticity
    for ep in results_df['enterprise_price'].unique():
        subset = results_df[results_df['enterprise_price'] == ep]
        if len(subset) > 1:
            # Calculate log-log elasticity
            log_price = np.log(subset['pro_price'])
            log_users = np.log(subset['final_pro_users'])
            
            if len(log_price) > 1 and len(log_users) > 1 and not np.isnan(log_users).any():
                try:
                    elasticity = np.polyfit(log_price, log_users, 1)[0]
                    pp_elasticity[ep] = elasticity
                except:
                    pass
    
    # Identify pareto-optimal pricing strategies
    # (strategies that are not dominated by any other strategy)
    pareto_optimal = []
    for i, row_i in results_df.iterrows():
        is_dominated = False
        for j, row_j in results_df.iterrows():
            if i != j:
                # Check if row_j dominates row_i across all metrics
                if (row_j['npv'] >= row_i['npv'] and
                    row_j['final_ltv_cac'] >= row_i['final_ltv_cac'] and
                    row_j['final_mrr'] >= row_i['final_mrr'] and
                    row_j['final_gross_margin'] >= row_i['final_gross_margin'] and
                    (row_j['npv'] > row_i['npv'] or
                     row_j['final_ltv_cac'] > row_i['final_ltv_cac'] or
                     row_j['final_mrr'] > row_i['final_mrr'] or
                     row_j['final_gross_margin'] > row_i['final_gross_margin'])):
                    is_dominated = True
                    break
        
        if not is_dominated:
            pareto_optimal.append(row_i.to_dict())
    
    # Calculate correlation between price and outcomes
    correlations = {
        'enterprise_price': {
            'npv': results_df[['enterprise_price', 'npv']].corr().iloc[0, 1],
            'ltv_cac': results_df[['enterprise_price', 'final_ltv_cac']].corr().iloc[0, 1],
            'mrr': results_df[['enterprise_price', 'final_mrr']].corr().iloc[0, 1],
            'gross_margin': results_df[['enterprise_price', 'final_gross_margin']].corr().iloc[0, 1]
        },
        'pro_price': {
            'npv': results_df[['pro_price', 'npv']].corr().iloc[0, 1],
            'ltv_cac': results_df[['pro_price', 'final_ltv_cac']].corr().iloc[0, 1],
            'mrr': results_df[['pro_price', 'final_mrr']].corr().iloc[0, 1],
            'gross_margin': results_df[['pro_price', 'final_gross_margin']].corr().iloc[0, 1]
        },
        'lifetime_price': {
            'npv': results_df[['lifetime_price', 'npv']].corr().iloc[0, 1],
            'ltv_cac': results_df[['lifetime_price', 'final_ltv_cac']].corr().iloc[0, 1],
            'mrr': results_df[['lifetime_price', 'final_mrr']].corr().iloc[0, 1],
            'gross_margin': results_df[['lifetime_price', 'final_gross_margin']].corr().iloc[0, 1]
        }
    }
    
    return {
        'enterprise_price_elasticity': ep_elasticity,
        'pro_price_elasticity': pp_elasticity,
        'pareto_optimal_strategies': pareto_optimal,
        'price_outcome_correlations': correlations,
        'best_strategies': grid_results['best_results']
    }

def visualize_pricing_strategies(optimized_results, grid_results=None):
    """Create visualizations of pricing strategies and their impacts"""
    # Plot optimal strategy simulation over time
    fig, axes = plt.subplots(4, 2, figsize=(15, 16))
    
    # Check if we have time-varying prices
    time_varying = optimized_results.get('time_varying', False)
    
    # Users over time
    axes[0, 0].plot(optimized_results['model'].history['enterprise_users'], label='Enterprise Users')
    axes[0, 0].plot(optimized_results['model'].history['pro_users'], label='Pro Users')
    axes[0, 0].plot(optimized_results['model'].history['landscaper_users'], label='Landscaper Users', linestyle='--')
    axes[0, 0].plot(optimized_results['model'].history['free_users'], label='Free Users', linestyle=':')
    axes[0, 0].set_title('User Growth Over Time')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Users')
    axes[0, 0].legend()
    
    # Network effects
    axes[0, 1].plot(optimized_results['model'].history['consultants'], label='Consultants')
    axes[0, 1].plot(optimized_results['model'].history['growers'], label='Growers')
    axes[0, 1].plot(np.array(optimized_results['model'].history['public_content'])/100, label='Public Content (÷100)')
    axes[0, 1].set_title('Network Effect Growth')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].legend()
    
    # Price evolution over time (new plot for time-varying pricing)
    if time_varying:
        axes[1, 0].plot(optimized_results['model'].history['enterprise_price'], label='Enterprise Price', marker='o')
        axes[1, 0].plot(optimized_results['model'].history['pro_price'], label='Pro Price', marker='o')
        axes[1, 0].set_title('Price Evolution Over Time')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Price ($)')
        axes[1, 0].legend()
    else:
        # For static pricing, show price impacts from grid search if available
        if grid_results:
            results_df = pd.DataFrame(grid_results['all_results'])
            axes[1, 0].scatter(results_df['enterprise_price'], results_df['npv'], alpha=0.6, label='Enterprise Price')
            axes[1, 0].scatter(results_df['pro_price'], results_df['npv'], alpha=0.6, label='Pro Price')
            axes[1, 0].set_title('Price vs NPV Relationship')
            axes[1, 0].set_xlabel('Price ($)')
            axes[1, 0].set_ylabel('NPV ($)')
            axes[1, 0].legend()
        else:
            # If no grid results, just show flat price lines
            months = range(len(optimized_results['model'].history['enterprise_users']))
            if isinstance(optimized_results['enterprise_price'], (list, np.ndarray)):
                axes[1, 0].plot(months, optimized_results['enterprise_price'], label='Enterprise Price')
                axes[1, 0].plot(months, optimized_results['pro_price'], label='Pro Price')
            else:
                axes[1, 0].axhline(y=optimized_results['enterprise_price'], color='b', linestyle='-', label='Enterprise Price')
                axes[1, 0].axhline(y=optimized_results['pro_price'], color='g', linestyle='-', label='Pro Price')
            axes[1, 0].set_title('Price Settings')
            axes[1, 0].set_xlabel('Month')
            axes[1, 0].set_ylabel('Price ($)')
            axes[1, 0].legend()
    
    # Revenue breakdown (new plot)
    axes[1, 1].plot(optimized_results['model'].history['mrr'], label='Total MRR')
    if 'contract_revenue' in optimized_results['model'].history and 'wellspring_revenue' in optimized_results['model'].history:
        subscription_revenue = np.array(optimized_results['model'].history['mrr']) - (
            np.array(optimized_results['model'].history['contract_revenue']) + 
            np.array(optimized_results['model'].history['wellspring_revenue'])
        )
        axes[1, 1].plot(subscription_revenue, label='Subscription MRR', linestyle='--')
        axes[1, 1].plot(optimized_results['model'].history['contract_revenue'], label='Contract Revenue', linestyle=':')
        axes[1, 1].plot(optimized_results['model'].history['wellspring_revenue'], label='Wellspring Revenue', linestyle='-.')
    axes[1, 1].set_title('Revenue Breakdown')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Revenue ($)')
    axes[1, 1].legend()
    
    # LTV/CAC
    axes[2, 0].plot(optimized_results['model'].history['ltv_cac_ratio'], label='LTV/CAC')
    axes[2, 0].axhline(y=3, color='r', linestyle='--', label='Target Ratio (3)')
    axes[2, 0].set_title('LTV/CAC Ratio')
    axes[2, 0].set_xlabel('Month')
    axes[2, 0].set_ylabel('Ratio')
    axes[2, 0].legend()
    
    # Churn rate
    axes[2, 1].plot(optimized_results['model'].history['churn_rate'], label='Avg Churn Rate')
    axes[2, 1].set_title('Average Churn Rate')
    axes[2, 1].set_xlabel('Month')
    axes[2, 1].set_ylabel('Monthly Churn %')
    
    # Gross margin
    axes[3, 0].plot(optimized_results['model'].history['gross_margin'], label='Gross Margin')
    axes[3, 0].set_title('Gross Margin')
    axes[3, 0].set_xlabel('Month')
    axes[3, 0].set_ylabel('Margin %')
    
    # NPV accumulation over time (new plot)
    if time_varying:
        # Calculate cumulative NPV
        discount_factor = 1 / (1 + optimized_results['model'].params.discount_rate_monthly)
        monthly_profits = []
        for t in range(len(optimized_results['model'].history['mrr'])):
            # Simplified profit calculation
            if t < len(optimized_results['model'].history['mrr']):
                revenue = optimized_results['model'].history['mrr'][t]
                enterprise_users = optimized_results['model'].history['enterprise_users'][t]
                pro_users = optimized_results['model'].history['pro_users'][t]
                landscaper_users = optimized_results['model'].history['landscaper_users'][t]
                
                costs = (enterprise_users * optimized_results['model'].params.service_cost_per_enterprise +
                        (pro_users + landscaper_users) * optimized_results['model'].params.service_cost_per_pro)
                
                monthly_profit = revenue - costs
                monthly_profits.append(monthly_profit)
        
        # Calculate discounted values
        discounted_profits = [monthly_profits[t] * (discount_factor ** t) for t in range(len(monthly_profits))]
        cumulative_npv = np.cumsum(discounted_profits)
        
        axes[3, 1].plot(cumulative_npv, label='Cumulative NPV')
        axes[3, 1].set_title('NPV Accumulation Over Time')
        axes[3, 1].set_xlabel('Month')
        axes[3, 1].set_ylabel('Cumulative NPV ($)')
    else:
        # Just show a different metric if not time-varying pricing
        catalogs = optimized_results['model'].history['catalog_size']
        axes[3, 1].plot(catalogs, label='Total Plants in Catalog')
        axes[3, 1].set_title('Catalog Growth')
        axes[3, 1].set_xlabel('Month')
        axes[3, 1].set_ylabel('Number of Plants')
    
    plt.tight_layout()
    
    # If we have grid search results, create additional visualization
    if grid_results:
        results_df = pd.DataFrame(grid_results['all_results'])
        
        # Create visualization of price vs outcome relationships
        fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12))
        
        # Enterprise price vs NPV
        axes2[0, 0].scatter(results_df['enterprise_price'], results_df['npv'], alpha=0.6)
        axes2[0, 0].set_title('Enterprise Price vs NPV')
        axes2[0, 0].set_xlabel('Enterprise Price ($)')
        axes2[0, 0].set_ylabel('NPV ($)')
        
        # Pro price vs NPV
        axes2[0, 1].scatter(results_df['pro_price'], results_df['npv'], alpha=0.6)
        axes2[0, 1].set_title('Pro Price vs NPV')
        axes2[0, 1].set_xlabel('Pro Price ($)')
        axes2[0, 1].set_ylabel('NPV ($)')
        
        # Enterprise price vs LTV/CAC
        axes2[1, 0].scatter(results_df['enterprise_price'], results_df['final_ltv_cac'], alpha=0.6)
        axes2[1, 0].set_title('Enterprise Price vs LTV/CAC')
        axes2[1, 0].set_xlabel('Enterprise Price ($)')
        axes2[1, 0].set_ylabel('LTV/CAC Ratio')
        
        # Pro price vs LTV/CAC
        axes2[1, 1].scatter(results_df['pro_price'], results_df['final_ltv_cac'], alpha=0.6)
        axes2[1, 1].set_title('Pro Price vs LTV/CAC')
        axes2[1, 1].set_xlabel('Pro Price ($)')
        axes2[1, 1].set_ylabel('LTV/CAC Ratio')
        
        plt.tight_layout()
        
        return fig, fig2
    
    return fig

def optimize_with_cpsat(params, objective='npv', time_horizon=12, price_steps=10, max_price_change_pct=10):
    """
    Optimize pricing strategy using Google OR-Tools CP-SAT solver.
    
    This approach allows for time-varying pricing strategies and additional constraints.
    
    Parameters:
    -----------
    params : PricingParams
        System parameters
    objective : str
        Optimization objective ('npv', 'ltv_cac', 'mrr', or 'gross_margin')
    time_horizon : int
        Number of months to optimize for
    price_steps : int
        Number of discrete price points to consider
    max_price_change_pct : float
        Maximum percentage price change between consecutive periods
        
    Returns:
    --------
    dict
        Optimized pricing parameters and resulting metrics
    """
    if not ORTOOLS_AVAILABLE:
        print("Google OR-Tools not available. Please install with: pip install ortools")
        return None
    
    # Define price ranges - ensure these are reasonable
    enterprise_price_min = 150
    enterprise_price_max = 400
    pro_price_min = 30
    pro_price_max = 100
    
    # Create price steps
    enterprise_prices = np.linspace(enterprise_price_min, enterprise_price_max, price_steps)
    pro_prices = np.linspace(pro_price_min, pro_price_max, price_steps)
    
    print(f"Enterprise prices: {enterprise_prices}")
    print(f"Pro prices: {pro_prices}")
    
    # Create the CP-SAT model
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    
    # Configure solver for more extensive search
    solver.parameters.max_time_in_seconds = 30  # Increase time limit to 30 seconds
    solver.parameters.num_search_workers = 8     # Increase parallelism
    solver.parameters.log_search_progress = True # Enable search progress logs
    
    # Create integer variables for price indices
    ent_price_idx = [model.NewIntVar(0, price_steps - 1, f'ent_price_idx_{t}') for t in range(time_horizon)]
    pro_price_idx = [model.NewIntVar(0, price_steps - 1, f'pro_price_idx_{t}') for t in range(time_horizon)]
    
    # Add smoothness constraints to limit price changes between periods
    max_idx_change = max(1, int((max_price_change_pct / 100) * price_steps))
    print(f"Max index change between periods: {max_idx_change} (out of {price_steps-1})")
    
    for t in range(time_horizon - 1):
        model.Add(ent_price_idx[t+1] - ent_price_idx[t] <= max_idx_change)
        model.Add(ent_price_idx[t] - ent_price_idx[t+1] <= max_idx_change)
        model.Add(pro_price_idx[t+1] - pro_price_idx[t] <= max_idx_change)
        model.Add(pro_price_idx[t] - pro_price_idx[t+1] <= max_idx_change)
    
    # Precompute metrics for all possible price combinations at each time step
    # This is an approximation, as we can't model the full dynamics directly in CP-SAT
    print("Precomputing metrics for all price combinations...")
    metrics = {}
    
    # Use a simpler simulation approach for precomputation to avoid excessive complexity
    for t in range(time_horizon):
        metrics[t] = {}
        print(f"Computing metrics for month {t+1}/{time_horizon}...")
        
        for e_idx, e_price in enumerate(enterprise_prices):
            for p_idx, p_price in enumerate(pro_prices):
                # Create a mini-simulation just for this price point and time step
                # For simplicity, we'll run a short simulation from time 0 to t with fixed prices
                # then use the specific price combination at time t
                
                # Simplify to reduce computation time - use average prices for earlier periods
                if t == 0:
                    enterprise_price_series = [e_price]
                    pro_price_series = [p_price]
                else:
                    # Use mid-range prices for previous periods to simplify
                    avg_e_price = (enterprise_price_min + enterprise_price_max) / 2
                    avg_p_price = (pro_price_min + pro_price_max) / 2
                    enterprise_price_series = [avg_e_price] * t + [e_price]
                    pro_price_series = [avg_p_price] * t + [p_price]
                
                model_params = {
                    'enterprise': enterprise_price_series,
                    'pro': pro_price_series,
                    'consultant_fee': 0.7,
                    'trial_days': 14
                }
                
                sim_model = PricingSystemModel(params, model_params)
                for step in range(t + 1):
                    sim_model.simulate_month(step)
                
                # Store the metrics for this price combination at this time step
                metrics[t][(e_idx, p_idx)] = {
                    'revenue': sim_model.mrr,
                    'profit': sim_model.mrr - (sim_model.enterprise_users * params.service_cost_per_enterprise + 
                                             (sim_model.pro_users + sim_model.landscaper_users) * params.service_cost_per_pro),
                    'churn': sim_model.history['churn_rate'][-1] if sim_model.history['churn_rate'] else 0.05,
                    'ltv_cac': sim_model.ltv_cac_ratio,
                    'gross_margin': sim_model.gross_margin,
                    'enterprise_users': sim_model.enterprise_users,
                    'pro_users': sim_model.pro_users
                }
    
    # Scale profits for the objective function (CP-SAT works with integers)
    scale = 100  # Scale factor for floating-point values
    profit_vars = []
    
    # Create and link variables for the metrics we care about
    for t in range(time_horizon):
        # Create a profit variable for each time step
        # Increase upper bound to ensure it can handle large profits
        profit_var = model.NewIntVar(0, int(1e7), f'profit_{t}')
        profit_vars.append(profit_var)
        
        # Create an element constraint to select the right profit based on price indices
        # We need to flatten our 2D price combination map to a 1D array for AddElement
        flattened_profits = [int(metrics[t][(e_idx, p_idx)]['profit'] * scale) 
                             for e_idx in range(price_steps) 
                             for p_idx in range(price_steps)]
        
        # Debug: check if we have valid profit values
        min_profit = min(flattened_profits)
        max_profit = max(flattened_profits)
        print(f"Month {t+1} profit range: ${min_profit/scale:.2f} to ${max_profit/scale:.2f}")
        
        # The index into this flattened array is e_idx * price_steps + p_idx
        flat_idx = model.NewIntVar(0, price_steps * price_steps - 1, f'flat_idx_{t}')
        model.Add(flat_idx == ent_price_idx[t] * price_steps + pro_price_idx[t])
        
        # Now use AddElement to select the profit based on the flattened index
        model.AddElement(flat_idx, flattened_profits, profit_var)
        
        # Add constraints for business rules - RELAXED FROM ORIGINAL VERSION
        
        # Only apply LTV/CAC constraint in later periods and with more relaxed values
        if t >= 3:  # After first quarter
            ltv_cac_values = [int(metrics[t][(e_idx, p_idx)]['ltv_cac'] * scale) 
                            for e_idx in range(price_steps) 
                            for p_idx in range(price_steps)]
            
            # Debug: Check the range of LTV/CAC values
            min_ltv_cac = min(ltv_cac_values) / scale
            max_ltv_cac = max(ltv_cac_values) / scale
            print(f"Month {t+1} LTV/CAC range: {min_ltv_cac:.2f} to {max_ltv_cac:.2f}")
            
            ltv_cac_var = model.NewIntVar(0, int(10 * scale), f'ltv_cac_{t}')
            model.AddElement(flat_idx, ltv_cac_values, ltv_cac_var)
            
            # RELAXED: Require LTV/CAC > 1.5 after initial periods (down from 3.0)
            min_ltv_cac_threshold = 1.5
            model.Add(ltv_cac_var >= int(min_ltv_cac_threshold * scale))
            print(f"Applied LTV/CAC constraint: >= {min_ltv_cac_threshold} for month {t+1}")
        
        # Churn constraint - only apply if necessary and with relaxed values
        if t >= 6:  # Only constrain churn in later periods
            churn_values = [int(metrics[t][(e_idx, p_idx)]['churn'] * scale * 100) 
                        for e_idx in range(price_steps) 
                        for p_idx in range(price_steps)]
            
            # Debug: Check the range of churn values
            min_churn = min(churn_values) / (scale * 100)
            max_churn = max(churn_values) / (scale * 100)
            print(f"Month {t+1} churn range: {min_churn:.2%} to {max_churn:.2%}")
            
            churn_var = model.NewIntVar(0, int(0.5 * scale * 100), f'churn_{t}')
            model.AddElement(flat_idx, churn_values, churn_var)
            
            # RELAXED: Limit churn rate to 15% (up from 8%)
            max_churn_threshold = 15.0
            model.Add(churn_var <= int(max_churn_threshold * scale))
            print(f"Applied churn constraint: <= {max_churn_threshold}% for month {t+1}")
    
    # Set the objective to maximize total discounted profit
    discount_factor = 1 / (1 + params.discount_rate_monthly)
    total_profit = model.NewIntVar(0, int(1e8), 'total_profit')
    
    # Calculate discounted profit with decay
    discounted_profits = [model.NewIntVar(0, int(1e7), f'disc_profit_{t}') 
                         for t in range(time_horizon)]
    
    for t in range(time_horizon):
        # Approximate discounting with integers
        discount = int(scale * (discount_factor ** t))
        # discounted_profit = profit * discount / scale
        model.AddDivisionEquality(discounted_profits[t], profit_vars[t] * discount, scale)
    
    model.Add(total_profit == sum(discounted_profits))
    model.Maximize(total_profit)
    
    # Solve the model
    print("Solving optimization model...")
    solver.parameters.log_search_progress = True
    status = solver.Solve(model)
    
    print(f"Solver status: {status}")
    if status == cp_model.OPTIMAL:
        print("Optimal solution found")
    elif status == cp_model.FEASIBLE:
        print("Feasible solution found (may not be optimal)")
    elif status == cp_model.INFEASIBLE:
        print("Problem is infeasible - no solution exists with these constraints")
    elif status == cp_model.MODEL_INVALID:
        print("Model is invalid")
    else:
        print("Unknown solver status")
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # Extract the solution
        enterprise_solution = [enterprise_prices[solver.Value(ent_price_idx[t])] for t in range(time_horizon)]
        pro_solution = [pro_prices[solver.Value(pro_price_idx[t])] for t in range(time_horizon)]
        
        print("Solution found:")
        print(f"Enterprise prices: {enterprise_solution}")
        print(f"Pro prices: {pro_solution}")
        
        # Run a full simulation with the optimized price schedule
        optimized_prices = {
            'enterprise': enterprise_solution,
            'pro': pro_solution,
            'consultant_fee': 0.7,
            'trial_days': 14
        }
        
        final_model = PricingSystemModel(params, optimized_prices)
        final_results = final_model.run_simulation()
        
        return {
            'enterprise_price': enterprise_solution,
            'pro_price': pro_solution,
            'consultant_fee': 0.7,
            'trial_days': 14,
            'npv': final_results['npv'],
            'final_mrr': final_results['final_mrr'],
            'final_ltv_cac': final_results['final_ltv_cac'],
            'final_gross_margin': final_results['final_gross_margin'],
            'time_varying': True,
            'model': final_model,
            'history': final_results['history']
        }
    else:
        # Check for specific constraint issues
        print("No solution found. Analyzing constraints...")
        
        # Check if there's any price combination that satisfies the LTV/CAC requirements
        valid_ltv_combinations = 0
        min_valid_ltv = float('inf')
        
        for t in range(3, time_horizon):  # Check months where LTV/CAC constraint is active
            for e_idx in range(price_steps):
                for p_idx in range(price_steps):
                    ltv_cac = metrics[t][(e_idx, p_idx)]['ltv_cac']
                    if ltv_cac >= 1.5:  # Using our relaxed constraint
                        valid_ltv_combinations += 1
                        min_valid_ltv = min(min_valid_ltv, ltv_cac)
        
        if valid_ltv_combinations == 0:
            print("ERROR: No price combinations satisfy the LTV/CAC constraint in later months")
            print("Try relaxing the LTV/CAC threshold further or changing price ranges")
        else:
            print(f"{valid_ltv_combinations} price combinations satisfy LTV/CAC constraint")
            print(f"Minimum valid LTV/CAC found: {min_valid_ltv:.2f}")
        
        # Check churn constraints similarly
        valid_churn_combinations = 0
        min_valid_churn = float('inf')
        
        for t in range(6, time_horizon):  # Check months where churn constraint is active
            for e_idx in range(price_steps):
                for p_idx in range(price_steps):
                    churn = metrics[t][(e_idx, p_idx)]['churn']
                    if churn <= 0.15:  # Using our relaxed constraint
                        valid_churn_combinations += 1
                        min_valid_churn = min(min_valid_churn, churn)
        
        if valid_churn_combinations == 0:
            print("ERROR: No price combinations satisfy the churn constraint in later months")
            print("Try relaxing the churn threshold further or changing price ranges")
        else:
            print(f"{valid_churn_combinations} price combinations satisfy churn constraint")
            print(f"Minimum valid churn found: {min_valid_churn:.2%}")
        
        # Final fallback: try with just two time periods and very simple constraints
        if time_horizon > 2:
            print("Attempting fallback solution with simplified model (2 time periods)...")
            
            simplified_model = cp_model.CpModel()
            simplified_solver = cp_model.CpSolver()
            
            # Just two periods
            ent_price_idx_simple = [simplified_model.NewIntVar(0, price_steps - 1, f'ent_simple_{t}') for t in range(2)]
            pro_price_idx_simple = [simplified_model.NewIntVar(0, price_steps - 1, f'pro_simple_{t}') for t in range(2)]
            
            # Very relaxed smoothness constraint
            max_idx_change_simple = price_steps - 1  # Allow any change
            simplified_model.Add(ent_price_idx_simple[1] - ent_price_idx_simple[0] <= max_idx_change_simple)
            simplified_model.Add(ent_price_idx_simple[0] - ent_price_idx_simple[1] <= max_idx_change_simple)
            simplified_model.Add(pro_price_idx_simple[1] - pro_price_idx_simple[0] <= max_idx_change_simple)
            simplified_model.Add(pro_price_idx_simple[0] - pro_price_idx_simple[1] <= max_idx_change_simple)
            
            # Just maximize profit without other constraints
            profit_simple = [simplified_model.NewIntVar(0, int(1e7), f'profit_simple_{t}') for t in range(2)]
            
            for t in range(2):
                # Use precomputed metrics from first two periods only
                actual_t = min(t, len(metrics) - 1)
                flattened_profits = [int(metrics[actual_t][(e_idx, p_idx)]['profit'] * scale) 
                                    for e_idx in range(price_steps) 
                                    for p_idx in range(price_steps)]
                
                flat_idx = simplified_model.NewIntVar(0, price_steps * price_steps - 1, f'flat_idx_simple_{t}')
                simplified_model.Add(flat_idx == ent_price_idx_simple[t] * price_steps + pro_price_idx_simple[t])
                simplified_model.AddElement(flat_idx, flattened_profits, profit_simple[t])
            
            # Simple objective: maximize sum of profits
            simplified_model.Maximize(sum(profit_simple))
            
            print("Solving simplified model...")
            simplified_status = simplified_solver.Solve(simplified_model)
            
            if simplified_status == cp_model.OPTIMAL or simplified_status == cp_model.FEASIBLE:
                print("Simplified solution found!")
                
                # Extract simplified solution
                enterprise_simple = [enterprise_prices[simplified_solver.Value(ent_price_idx_simple[t])] for t in range(2)]
                pro_simple = [pro_prices[simplified_solver.Value(pro_price_idx_simple[t])] for t in range(2)]
                
                # Extend to original time horizon with a simple pattern
                enterprise_extended = []
                pro_extended = []
                
                # Fill with a pattern based on the simplified solution
                for t in range(time_horizon):
                    if t < 2:
                        enterprise_extended.append(enterprise_simple[t])
                        pro_extended.append(pro_simple[t])
                    else:
                        # Keep last values or apply a small increase
                        enterprise_extended.append(enterprise_extended[-1] * 1.02)  # 2% increase
                        pro_extended.append(pro_extended[-1] * 1.02)  # 2% increase
                
                print(f"Extended enterprise prices: {enterprise_extended}")
                print(f"Extended pro prices: {pro_extended}")
                
                # Run a full simulation with the extended price schedule
                optimized_prices = {
                    'enterprise': enterprise_extended,
                    'pro': pro_extended,
                    'consultant_fee': 0.7,
                    'trial_days': 14
                }
                
                fallback_model = PricingSystemModel(params, optimized_prices)
                fallback_results = fallback_model.run_simulation()
                
                print("Returning simplified fallback solution")
                return {
                    'enterprise_price': enterprise_extended,
                    'pro_price': pro_extended,
                    'consultant_fee': 0.7,
                    'trial_days': 14,
                    'npv': fallback_results['npv'],
                    'final_mrr': fallback_results['final_mrr'],
                    'final_ltv_cac': fallback_results['final_ltv_cac'],
                    'final_gross_margin': fallback_results['final_gross_margin'],
                    'time_varying': True,
                    'model': fallback_model,
                    'is_fallback': True,  # Mark as fallback solution
                    'history': fallback_results['history']
                }
            else:
                print("Even simplified model failed to find a solution")
        
        print("No solution found.")
        return None

def pricing_heuristic_search(params, iterations=1000, time_horizon=12, smooth_constraint=True, max_price_change_pct=10):
    """
    Use hill climbing to find good pricing strategies
    
    Parameters:
    -----------
    params : PricingParams
        System parameters
    iterations : int
        Number of search iterations
    time_horizon : int
        Number of months to optimize for
    smooth_constraint : bool
        Whether to enforce smooth price transitions
    max_price_change_pct : float
        Maximum percentage price change between consecutive periods
        
    Returns:
    --------
    dict
        Optimized pricing parameters and resulting metrics
    """
    # Initialize at midpoints
    enterprise_price_min = 150
    enterprise_price_max = 400
    pro_price_min = 30
    pro_price_max = 100
    
    # Use random starting point with some structure
    # Start lower and gradually increase
    enterprise_prices = np.linspace(
        enterprise_price_min + (enterprise_price_max - enterprise_price_min) * 0.3, 
        enterprise_price_min + (enterprise_price_max - enterprise_price_min) * 0.7, 
        time_horizon
    )
    
    pro_prices = np.linspace(
        pro_price_min + (pro_price_max - pro_price_min) * 0.3, 
        pro_price_min + (pro_price_max - pro_price_min) * 0.7, 
        time_horizon
    )
    
    # Set up model with current prices
    model = PricingSystemModel(params, {
        'enterprise': enterprise_prices,
        'pro': pro_prices,
        'consultant_fee': 0.7,
        'trial_days': 14
    })
    
    # Evaluate current solution
    results = model.run_simulation()
    best_npv = results['npv']
    best_enterprise_prices = enterprise_prices.copy()
    best_pro_prices = pro_prices.copy()
    
    print(f"Initial NPV: ${best_npv:,.2f}")
    
    # Calculate max allowable price changes for smooth transitions
    max_enterprise_change = (enterprise_price_max - enterprise_price_min) * max_price_change_pct / 100
    max_pro_change = (pro_price_max - pro_price_min) * max_price_change_pct / 100
    
    # Track progress
    iterations_without_improvement = 0
    max_iterations_without_improvement = 50
    
    # Hill climbing
    for i in range(iterations):
        # Create test price arrays before the random choice
        test_enterprise_prices = enterprise_prices.copy()
        test_pro_prices = pro_prices.copy()
        
        # Choose which price to perturb and which time period
        enterprise_perturb = np.random.random() < 0.5
        t = np.random.randint(0, time_horizon)
        
        if enterprise_perturb:
            # Perturb enterprise price
            delta = np.random.uniform(-10, 10)
            
            # Apply smoothness constraint if enabled
            if smooth_constraint and t > 0:
                # Limit change relative to previous period
                delta = np.clip(delta, -max_enterprise_change, max_enterprise_change)
                
                # Ensure the resulting price doesn't violate smoothness from previous period
                if t > 0:
                    min_allowed = enterprise_prices[t-1] - max_enterprise_change
                    max_allowed = enterprise_prices[t-1] + max_enterprise_change
                    new_price = np.clip(enterprise_prices[t] + delta, min_allowed, max_allowed)
                    delta = new_price - enterprise_prices[t]
                
                # Ensure it doesn't create a violation for the next period
                if t < time_horizon - 1:
                    min_next_allowed = enterprise_prices[t] + delta - max_enterprise_change
                    max_next_allowed = enterprise_prices[t] + delta + max_enterprise_change
                    if enterprise_prices[t+1] < min_next_allowed or enterprise_prices[t+1] > max_next_allowed:
                        # Adjust the next period price to maintain smoothness
                        enterprise_prices[t+1] = np.clip(enterprise_prices[t+1], min_next_allowed, max_next_allowed)
            
            test_enterprise_prices[t] = np.clip(test_enterprise_prices[t] + delta, 
                                         enterprise_price_min, 
                                         enterprise_price_max)
        else:
            # Perturb pro price
            delta = np.random.uniform(-5, 5)
            
            # Apply smoothness constraint if enabled
            if smooth_constraint:
                # Limit change relative to previous period
                delta = np.clip(delta, -max_pro_change, max_pro_change)
                
                # Ensure the resulting price doesn't violate smoothness from previous period
                if t > 0:
                    min_allowed = pro_prices[t-1] - max_pro_change
                    max_allowed = pro_prices[t-1] + max_pro_change
                    new_price = np.clip(pro_prices[t] + delta, min_allowed, max_allowed)
                    delta = new_price - pro_prices[t]
                
                # Ensure it doesn't create a violation for the next period
                if t < time_horizon - 1:
                    min_next_allowed = pro_prices[t] + delta - max_pro_change
                    max_next_allowed = pro_prices[t] + delta + max_pro_change
                    if pro_prices[t+1] < min_next_allowed or pro_prices[t+1] > max_next_allowed:
                        # Adjust the next period price to maintain smoothness
                        pro_prices[t+1] = np.clip(pro_prices[t+1], min_next_allowed, max_next_allowed)
            
            test_pro_prices[t] = np.clip(test_pro_prices[t] + delta, 
                                   pro_price_min, 
                                   pro_price_max)
            
        test_model = PricingSystemModel(params, {
            'enterprise': test_enterprise_prices,
            'pro': test_pro_prices,
            'consultant_fee': 0.7,
            'trial_days': 14
        })
        
        test_results = test_model.run_simulation()
        
        # Accept if better
        if test_results['npv'] > best_npv:
            if enterprise_perturb:  # Enterprise price was perturbed
                best_enterprise_prices = test_enterprise_prices.copy()
                enterprise_prices = test_enterprise_prices.copy()
            else:  # Pro price was perturbed
                best_pro_prices = test_pro_prices.copy()
                pro_prices = test_pro_prices.copy()
                
            best_npv = test_results['npv']
            iterations_without_improvement = 0
            
            if i % 50 == 0:
                print(f"Iteration {i}: New best NPV: ${best_npv:,.2f}")
        else:
            iterations_without_improvement += 1
            
            # Occasional random restart to escape local optima
            if iterations_without_improvement >= max_iterations_without_improvement:
                if np.random.random() < 0.3:  # 30% chance of random restart
                    print(f"Iteration {i}: Random restart after {iterations_without_improvement} iterations without improvement")
                    
                    # Keep the best prices discovered so far as a starting point
                    enterprise_prices = best_enterprise_prices.copy()
                    pro_prices = best_pro_prices.copy()
                    
                    # Add random perturbations to all periods
                    for t in range(time_horizon):
                        enterprise_prices[t] += np.random.uniform(-15, 15)
                        enterprise_prices[t] = np.clip(enterprise_prices[t], enterprise_price_min, enterprise_price_max)
                        
                        pro_prices[t] += np.random.uniform(-7, 7)
                        pro_prices[t] = np.clip(pro_prices[t], pro_price_min, pro_price_max)
                    
                    # Apply smoothing if required
                    if smooth_constraint:
                        for t in range(1, time_horizon):
                            # Ensure enterprise price changes are smooth
                            if abs(enterprise_prices[t] - enterprise_prices[t-1]) > max_enterprise_change:
                                if enterprise_prices[t] > enterprise_prices[t-1]:
                                    enterprise_prices[t] = enterprise_prices[t-1] + max_enterprise_change
                                else:
                                    enterprise_prices[t] = enterprise_prices[t-1] - max_enterprise_change
                                    
                            # Ensure pro price changes are smooth
                            if abs(pro_prices[t] - pro_prices[t-1]) > max_pro_change:
                                if pro_prices[t] > pro_prices[t-1]:
                                    pro_prices[t] = pro_prices[t-1] + max_pro_change
                                else:
                                    pro_prices[t] = pro_prices[t-1] - max_pro_change
                    
                    iterations_without_improvement = 0
    
    # Run final simulation with best prices
    final_model = PricingSystemModel(params, {
        'enterprise': best_enterprise_prices,
        'pro': best_pro_prices,
        'consultant_fee': 0.7,
        'trial_days': 14
    })
    
    final_results = final_model.run_simulation()
    
    print(f"Final NPV: ${final_results['npv']:,.2f}")
    print(f"Final Enterprise Prices: {best_enterprise_prices}")
    print(f"Final Pro Prices: {best_pro_prices}")
    
    return {
        'enterprise_price': best_enterprise_prices,
        'pro_price': best_pro_prices,
        'consultant_fee': 0.7,
        'trial_days': 14,
        'npv': final_results['npv'],
        'final_mrr': final_results['final_mrr'],
        'final_ltv_cac': final_results['final_ltv_cac'],
        'final_gross_margin': final_results['final_gross_margin'],
        'time_varying': True,
        'model': final_model,
        'history': final_results['history']
    }

# Model Parameters
class PricingParams:
    def __init__(self):
        # Market size assumptions - Updated based on user's SAM/SOM values
        self.potential_enterprise_customers = 440  # Total addressable market for enterprise (LA firms)
        self.potential_pro_customers = 3800  # Total addressable market for individual pros (LA_ind)
        self.potential_consultants = 200  # Potential consultants for marketplace (Hort consultants)
        self.potential_growers = 200  # Keeping similar to consultants as not specified
        self.potential_landscapers = 8950  # Potential individual landscapers
        
        # Segment-specific parameters
        self.landscape_architects_pct = 0.7  # Percentage of enterprise customers who are landscape architects
        self.consultants_pct = 0.2  # Percentage of enterprise customers who are consultants
        self.growers_in_platform_pct = 0.1  # Initial percentage of growers in the platform
        
        # Conversion rates - Updated based on user input
        self.pro_trial_to_paid = 0.5  # Conversion among individuals (Pro) - 50%
        self.enterprise_demo_to_paid = 0.5  # Conversion of Enterprise - 50%
        self.contracts_platform_usage = 0.05  # Conversion to contracts platform - 5%
        
        # Price elasticity coefficients (sensitivity of demand to price)
        self.enterprise_price_elasticity = -0.7  # Enterprise less price sensitive
        self.pro_price_elasticity = -1.2  # Pro users more price sensitive
        self.lifetime_deal_elasticity = -0.9  # Elasticity for lifetime deals
        
        # Network effect coefficients
        self.catalog_network_effect = 0.15  # Effect of catalog content on user acquisition
        self.consultant_network_effect = 0.12  # Effect of consultants on user acquisition
        self.public_content_network_effect = 0.18  # Effect of public content on acquisition
        
        # Wellspring marketplace parameters - Updated based on user data
        self.schedules_per_seller_per_year = 30  # Average number of schedules per seller per year
        self.avg_substitutions_per_schedule = 6  # Average substitutions per schedule
        self.fee_per_substitution = 1.0  # Fee charged per plant substitution
        self.performance_factor = 0.2  # % of substitutions deemed "performance-related"
        self.avg_contract_size = 20000  # Average commercial landscape contract value
        
        # Baseline acquisition and retention parameters
        self.base_enterprise_acquisition_rate = 0.02  # Monthly rate without network effects
        self.base_pro_acquisition_rate = 0.04  # Monthly rate without network effects
        self.base_enterprise_churn_rate = 0.03  # Monthly churn rate for enterprise
        self.base_pro_churn_rate = 0.08  # Monthly churn rate for pro users
        
        # Cost parameters
        self.enterprise_cac = 1200  # Customer acquisition cost for enterprise
        self.pro_cac = 300  # Customer acquisition cost for pro users
        self.service_cost_per_enterprise = 100  # Monthly cost to service enterprise
        self.service_cost_per_pro = 20  # Monthly cost to service pro user
        
        # Value perception parameters (0-1 scale)
        self.enterprise_value_perception = 0.7  # How much they value the product
        self.pro_value_perception = 0.8  # How much they value the product
        
        # Time parameters
        self.time_horizon_months = 36  # Time horizon for optimization
        self.discount_rate_monthly = 0.01  # Monthly discount rate for NPV calculation
        
        # Conversion rates (some overlap with above, but keeping for compatibility)
        self.trial_to_paid_enterprise = 0.5  # Conversion from trial to paid for enterprise (matching enterprise_demo_to_paid)
        self.trial_to_paid_pro = 0.5  # Conversion from trial to paid for pro (matching pro_trial_to_paid)
        self.free_to_paid_rate = 0.05  # Conversion rate from free to paid
        
        # Marketplace parameters
        self.consultant_fee_share = 0.7  # Share of fees that goes to consultant
        self.avg_advice_price = 50  # Average price per advice transaction
        self.avg_advice_frequency = 0.2  # Average monthly advice transactions per user
        
        # Transaction fee parameters
        self.contract_tx_fee = 0.1  # Contract transaction fee - 10%
        self.wellspring_tx_fee = 0.1  # Wellspring transaction fee - 10%
        
        # Billing multipliers (monthly vs annual)
        self.enterprise_billing_multiplier = 12  # Annual billing
        self.pro_billing_multiplier = 1  # Monthly billing
        
        # Store original SAM values for reference
        self._sam_enterprise_customers = 440
        self._sam_pro_customers = 3800
        self._sam_consultants = 200
        self._sam_growers = 200
        self._sam_landscapers = 8950
    
    def use_som(self, som_percentage=0.1):
        """
        Adjust market sizes to reflect the Serviceable Obtainable Market (SOM)
        
        Parameters:
        -----------
        som_percentage : float
            Percentage of SAM to use (default: 10%)
        """
        self.potential_enterprise_customers = int(self._sam_enterprise_customers * som_percentage)
        self.potential_pro_customers = int(self._sam_pro_customers * som_percentage)
        self.potential_consultants = int(self._sam_consultants * som_percentage)
        self.potential_growers = int(self._sam_growers * som_percentage)
        self.potential_landscapers = int(self._sam_landscapers * som_percentage)
        
        return self
    
    def use_sam(self):
        """
        Reset market sizes to full Serviceable Available Market (SAM)
        """
        self.potential_enterprise_customers = self._sam_enterprise_customers
        self.potential_pro_customers = self._sam_pro_customers
        self.potential_consultants = self._sam_consultants
        self.potential_growers = self._sam_growers
        self.potential_landscapers = self._sam_landscapers
        
        return self

# Define system dynamics model for B2B SaaS with network effects
class PricingSystemModel:
    def __init__(self, params, initial_prices=None):
        self.params = params
        
        # Starting values for prices - now as time series
        if initial_prices is None:
            self.enterprise_price = [200] * params.time_horizon_months
            self.pro_price = [50] * params.time_horizon_months
            self.lifetime_deal_price = 1000
            self.consultant_fee_percentage = 0.7
            self.trial_length_days = 14
        else:
            # Handle both constant prices and time series
            if isinstance(initial_prices.get('enterprise', 200), (int, float)):
                self.enterprise_price = [initial_prices.get('enterprise', 200)] * params.time_horizon_months
            else:
                self.enterprise_price = initial_prices.get('enterprise', [200] * params.time_horizon_months)
                
            if isinstance(initial_prices.get('pro', 50), (int, float)):
                self.pro_price = [initial_prices.get('pro', 50)] * params.time_horizon_months
            else:
                self.pro_price = initial_prices.get('pro', [50] * params.time_horizon_months)
                
            self.lifetime_deal_price = initial_prices.get('lifetime', 1000)
            self.consultant_fee_percentage = initial_prices.get('consultant_fee', 0.7)
            self.trial_length_days = initial_prices.get('trial_days', 14)
        
        # Initialize state variables
        self.enterprise_users = 0
        self.pro_users = 0
        self.consultants = 0
        self.growers = 0
        self.free_users = 0
        self.landscaper_users = 0
        self.public_content = 0
        
        # Track financial metrics
        self.mrr = 0
        self.contract_revenue = 0
        self.wellspring_revenue = 0
        self.ltv_enterprise = 0
        self.ltv_pro = 0
        self.cum_revenue = 0
        self.cum_costs = 0
        self.gross_margin = 0
        self.ltv_cac_ratio = 0
        
        # History arrays for tracking over time
        self.history = {
            'enterprise_users': [],
            'pro_users': [],
            'landscaper_users': [],
            'consultants': [],
            'growers': [],
            'free_users': [],
            'mrr': [],
            'contract_revenue': [],
            'wellspring_revenue': [],
            'churn_rate': [],
            'ltv_cac_ratio': [],
            'gross_margin': [],
            'public_content': [],
            'catalog_size': [],
            'enterprise_price': [],
            'pro_price': []
        }
    
    def calculate_acquisition_rates(self, month):
        """Calculate acquisition rates with network effects and market segments"""
        # Get current prices
        enterprise_monthly_price = self.enterprise_price[month]
        pro_monthly_price = self.pro_price[month]
        
        # Calculate network effect multipliers
        catalog_effect = 1 + (self.params.catalog_network_effect * 
                             np.log1p(self.growers / max(1, self.params.potential_growers)))
        
        consultant_effect = 1 + (self.params.consultant_network_effect * 
                                np.log1p(self.consultants / max(1, self.params.potential_consultants)))
        
        public_effect = 1 + (self.params.public_content_network_effect * 
                            np.log1p(self.public_content / 1000))  # Arbitrary scale for content
        
        # Price effect using elasticity model
        enterprise_price_effect = (enterprise_monthly_price / 200) ** self.params.enterprise_price_elasticity
        pro_price_effect = (pro_monthly_price / 50) ** self.params.pro_price_elasticity
        
        # Trial length effect (positive effect from longer trials for enterprise, less so for pro)
        trial_effect_enterprise = min(2, (self.trial_length_days / 14) ** 0.3)
        trial_effect_pro = min(1.5, (self.trial_length_days / 14) ** 0.1)
        
        # Combine effects following new formulation
        # Enterprise: f₁_ent(P_ent)·[M_LA_firms·Cnv_ent_demo]
        enterprise_acquisition = (self.params.base_enterprise_acquisition_rate * 
                                 catalog_effect * consultant_effect * public_effect * 
                                 enterprise_price_effect * trial_effect_enterprise)
        
        enterprise_acquisition *= self.params.enterprise_demo_to_paid
        
        # Pro: f₁_pro(P_pro)·([M_LA_ind·Cnv_pro] + [M_landscaper_ind·Cnv_pro])
        pro_acquisition_architects = (self.params.base_pro_acquisition_rate * 
                                     catalog_effect * consultant_effect * public_effect * 
                                     pro_price_effect * trial_effect_pro * 
                                     self.params.pro_trial_to_paid)
        
        # Calculate landscaper acquisition (typically less sensitive to catalog size)
        landscaper_acquisition = (self.params.base_pro_acquisition_rate * 0.7 * 
                                 consultant_effect * public_effect *
                                 (pro_monthly_price / 50) ** (self.params.pro_price_elasticity * 1.1) *
                                 trial_effect_pro * self.params.pro_trial_to_paid)
        
        # Early-stage growth boost and late-stage saturation
        time_factor_enterprise = 1 + max(0, (0.5 - month / 36))
        time_factor_pro = 1 + max(0, (0.8 - month / 24))
        
        enterprise_acquisition *= time_factor_enterprise
        pro_acquisition_architects *= time_factor_pro
        landscaper_acquisition *= time_factor_pro
        
        return enterprise_acquisition, pro_acquisition_architects, landscaper_acquisition

    def calculate_churn_rates(self, month):
        """Calculate churn rates based on product value, network effects, and current price"""
        # Get current prices
        enterprise_monthly_price = self.enterprise_price[month]
        pro_monthly_price = self.pro_price[month]
        
        # Value perception increases with network effects
        enterprise_value = self.params.enterprise_value_perception * (
            1 + 0.3 * np.log1p(self.growers / max(1, self.params.potential_growers)) +
            0.2 * np.log1p(self.consultants / max(1, self.params.potential_consultants))
        )
        
        pro_value = self.params.pro_value_perception * (
            1 + 0.2 * np.log1p(self.growers / max(1, self.params.potential_growers)) +
            0.3 * np.log1p(self.consultants / max(1, self.params.potential_consultants)) +
            0.1 * np.log1p(self.public_content / 1000)
        )
        
        # Price-value ratio effect on churn - exponential function: f₂(P) = γe^(δP)
        # With normalized price-value ratio
        enterprise_price_value_ratio = enterprise_monthly_price / (enterprise_value * 300)
        pro_price_value_ratio = pro_monthly_price / (pro_value * 80)
        
        # Calculate actual churn rates
        enterprise_churn = self.params.base_enterprise_churn_rate * min(2, np.exp(1.2 * enterprise_price_value_ratio - 1))
        pro_churn = self.params.base_pro_churn_rate * min(2, np.exp(1.5 * pro_price_value_ratio - 1))
        
        # Minimum churn floor
        enterprise_churn = max(0.01, enterprise_churn)
        pro_churn = max(0.02, pro_churn)
        
        # Landscaper churn (typically higher)
        landscaper_churn = 1.2 * pro_churn
        
        return enterprise_churn, pro_churn, landscaper_churn

    def calculate_growth_and_feedback_effects(self, month, enterprise_churn, pro_churn):
        """Calculate growth of secondary user types and content based on primary users"""
        # Consultant growth based on user base
        potential_new_consultants = 0.01 * (self.enterprise_users + self.pro_users + self.landscaper_users * 0.5) * (
            self.consultant_fee_percentage ** 0.5)  # Higher fee share attracts more consultants
        
        new_consultants = min(
            potential_new_consultants,
            0.05 * (self.params.potential_consultants - self.consultants)
        )
        
        # Grower onboarding rate increases with user base
        potential_new_growers = 0.005 * (self.enterprise_users + self.pro_users * 0.3 + self.landscaper_users * 0.2)
        new_growers = min(
            potential_new_growers,
            0.08 * (self.params.potential_growers - self.growers)
        )
        
        # Free user acquisition from public content
        new_free_users = 0.03 * self.public_content * 0.01
        
        # Free-to-paid conversion
        pro_price = self.pro_price[month]
        free_to_pro_conversions = self.free_users * self.params.free_to_paid_rate * (
            (50 / pro_price) ** 0.5)  # Price impact on conversion
        
        # Content generation rates
        new_public_content = (
            5 * self.free_users * 0.01 +  # Free users contribute more public content
            2 * self.pro_users * 0.005 +
            1 * self.landscaper_users * 0.003 +
            10 * self.enterprise_users * 0.002
        )
        
        # Update feedback assets
        self.consultants += new_consultants
        self.growers += new_growers
        self.free_users += new_free_users - free_to_pro_conversions
        self.public_content += new_public_content
        
        return free_to_pro_conversions

    def calculate_wellspring_revenue(self):
        """Calculate revenue from the Wellspring marketplace"""
        # Marketplace revenue based on consultants, schedules, and substitutions
        wellspring_revenue = (
            self.consultants * 
            (self.params.schedules_per_seller_per_year / 12) *  # Monthly schedules
            self.params.avg_substitutions_per_schedule * 
            self.params.fee_per_substitution * 
            (1 - self.consultant_fee_percentage) *  # Platform's share
            self.params.performance_factor *  # Adjustment for performance-related substitutions
            self.params.wellspring_tx_fee  # Transaction fee percentage
        )
        
        return wellspring_revenue

    def calculate_contract_revenue(self):
        """Calculate revenue from landscape contract transactions"""
        # Contract revenue based on enterprise users using the contracts platform
        contract_revenue = (
            self.enterprise_users * 
            self.params.contracts_platform_usage * 
            (self.params.avg_contract_size * self.params.contract_tx_fee) / 12  # Monthly contract value with tx fee
        )
        
        return contract_revenue

    def simulate_month(self, month):
        """Simulate one month of business operations with time-varying prices"""
        # Store current prices in history
        self.history['enterprise_price'].append(self.enterprise_price[month])
        self.history['pro_price'].append(self.pro_price[month])
        
        # Calculate acquisition and churn rates
        enterprise_acquisition_rate, pro_acquisition_rate, landscaper_acquisition_rate = self.calculate_acquisition_rates(month)
        enterprise_churn_rate, pro_churn_rate, landscaper_churn_rate = self.calculate_churn_rates(month)
        
        # Calculate new users and churned users
        new_enterprise = enterprise_acquisition_rate * (self.params.potential_enterprise_customers - self.enterprise_users)
        new_pro = pro_acquisition_rate * (self.params.potential_pro_customers - self.pro_users)
        new_landscaper = landscaper_acquisition_rate * (self.params.potential_landscapers - self.landscaper_users)
        
        churned_enterprise = self.enterprise_users * enterprise_churn_rate
        churned_pro = self.pro_users * pro_churn_rate
        churned_landscaper = self.landscaper_users * landscaper_churn_rate
        
        # Calculate effects from network and feedback loops
        free_to_pro_conversions = self.calculate_growth_and_feedback_effects(
            month, enterprise_churn_rate, pro_churn_rate)
        
        # Update user counts
        self.enterprise_users += new_enterprise - churned_enterprise
        self.pro_users += new_pro - churned_pro + free_to_pro_conversions * 0.7  # Assume 70% go to Pro
        self.landscaper_users += new_landscaper - churned_landscaper + free_to_pro_conversions * 0.3  # 30% to Landscaper
        
        # Add lifetime deal impact (one-time revenue, but reduced churn)
        lifetime_deal_conversions = 0
        if month < 12:  # Limited time offer for early adopters
            lifetime_price_effect = (self.lifetime_deal_price / 1000) ** self.params.lifetime_deal_elasticity
            lifetime_deal_conversions = 0.01 * pro_acquisition_rate * lifetime_price_effect * (
                self.params.potential_pro_customers - self.pro_users)
            
            # Lifetime deals reduce pro users (they're now in a different category)
            self.pro_users -= lifetime_deal_conversions
        
        # Calculate revenue from different streams
        monthly_enterprise_revenue = self.enterprise_users * self.enterprise_price[month]
        monthly_pro_revenue = self.pro_users * self.pro_price[month]
        monthly_landscaper_revenue = self.landscaper_users * self.pro_price[month] * 0.9  # Slight discount for landscapers
        monthly_lifetime_revenue = lifetime_deal_conversions * self.lifetime_deal_price
        
        # Calculate contract and wellspring revenue
        contract_revenue = self.calculate_contract_revenue()
        wellspring_revenue = self.calculate_wellspring_revenue()
        
        # Calculate total subscription revenue (before additional streams)
        subscription_revenue = monthly_enterprise_revenue + monthly_pro_revenue + monthly_landscaper_revenue
        
        # Calculate costs
        monthly_enterprise_costs = (
            self.enterprise_users * self.params.service_cost_per_enterprise +
            new_enterprise * self.params.enterprise_cac
        )
        
        monthly_pro_costs = (
            (self.pro_users + self.landscaper_users) * self.params.service_cost_per_pro +
            (new_pro + new_landscaper) * self.params.pro_cac
        )
        
        # Update financial KPIs
        monthly_revenue = (
            subscription_revenue + 
            monthly_lifetime_revenue + 
            contract_revenue + 
            wellspring_revenue
        )
        
        monthly_costs = monthly_enterprise_costs + monthly_pro_costs
        monthly_profit = monthly_revenue - monthly_costs
        
        self.mrr = subscription_revenue + wellspring_revenue + contract_revenue
        self.contract_revenue = contract_revenue
        self.wellspring_revenue = wellspring_revenue
        self.cum_revenue += monthly_revenue
        self.cum_costs += monthly_costs
        
        # Gross margin calculation
        if monthly_revenue > 0:
            self.gross_margin = (monthly_revenue - monthly_costs) / monthly_revenue
        else:
            self.gross_margin = 0
            
        # Calculate LTV with discount rate
        discount_factor = 1 / (1 + self.params.discount_rate_monthly)
        
        if enterprise_churn_rate > 0:
            self.ltv_enterprise = self.enterprise_price[month] / enterprise_churn_rate * (
                1 - discount_factor ** (1 / enterprise_churn_rate)) / (1 - discount_factor)
            
            # Add contract revenue to enterprise LTV
            self.ltv_enterprise += (contract_revenue / self.enterprise_users) / enterprise_churn_rate if self.enterprise_users > 0 else 0
        else:
            self.ltv_enterprise = self.enterprise_price[month] * 36  # Capped at time horizon
            
        if pro_churn_rate > 0:
            self.ltv_pro = self.pro_price[month] / pro_churn_rate * (
                1 - discount_factor ** (1 / pro_churn_rate)) / (1 - discount_factor)
        else:
            self.ltv_pro = self.pro_price[month] * 36  # Capped at time horizon
            
        # Weighted average LTV and LTV/CAC
        total_users = self.enterprise_users + self.pro_users + self.landscaper_users
        if total_users > 0:
            avg_ltv = (
                self.enterprise_users * self.ltv_enterprise + 
                self.pro_users * self.ltv_pro +
                self.landscaper_users * self.ltv_pro * 0.9
            ) / total_users
            
            avg_cac = (
                self.enterprise_users * self.params.enterprise_cac + 
                (self.pro_users + self.landscaper_users) * self.params.pro_cac
            ) / total_users
            
            self.ltv_cac_ratio = avg_ltv / avg_cac if avg_cac > 0 else 0
        else:
            self.ltv_cac_ratio = 0
            
        # Save history
        self.history['enterprise_users'].append(self.enterprise_users)
        self.history['pro_users'].append(self.pro_users)
        self.history['landscaper_users'].append(self.landscaper_users)
        self.history['consultants'].append(self.consultants)
        self.history['growers'].append(self.growers)
        self.history['free_users'].append(self.free_users)
        self.history['mrr'].append(self.mrr)
        self.history['contract_revenue'].append(contract_revenue)
        self.history['wellspring_revenue'].append(wellspring_revenue)
        self.history['churn_rate'].append(
            (enterprise_churn_rate * self.enterprise_users + 
             pro_churn_rate * self.pro_users +
             landscaper_churn_rate * self.landscaper_users) / 
            max(1, self.enterprise_users + self.pro_users + self.landscaper_users)
        )
        self.history['ltv_cac_ratio'].append(self.ltv_cac_ratio)
        self.history['gross_margin'].append(self.gross_margin)
        self.history['public_content'].append(self.public_content)
        self.history['catalog_size'].append(self.growers * 100)  # Assuming 100 plants per grower
        
        return monthly_profit

    def run_simulation(self):
        """Run the full simulation over the time horizon"""
        total_npv = 0
        discount_factor = 1 / (1 + self.params.discount_rate_monthly)
        
        for month in range(self.params.time_horizon_months):
            monthly_profit = self.simulate_month(month)
            # Calculate NPV of this month's profit
            total_npv += monthly_profit * (discount_factor ** month)
            
        self.npv = total_npv
        
        return {
            'npv': total_npv,
            'final_mrr': self.mrr,
            'final_contract_revenue': self.contract_revenue,
            'final_wellspring_revenue': self.wellspring_revenue,
            'final_ltv_cac': self.ltv_cac_ratio,
            'final_gross_margin': self.gross_margin,
            'final_enterprise_users': self.enterprise_users,
            'final_pro_users': self.pro_users,
            'final_landscaper_users': self.landscaper_users,
            'final_consultants': self.consultants,
            'final_growers': self.growers,
            'final_free_users': self.free_users,
            'history': self.history
        }

def analyze_network_effects_and_elasticity(params, price_range=0.5, num_points=20):
    """
    Analyze the impact of network effects and price elasticity on the SaaS model.
    
    Parameters:
    -----------
    params : PricingParams
        System parameters
    price_range : float
        Percentage range around base price to test (0.5 = ±50%)
    num_points : int
        Number of price points to test in the range
        
    Returns:
    --------
    dict
        Analysis results including elasticity curves and network effect impact
    """
    print("Analyzing price elasticity and network effects...")
    
    # Base prices
    base_enterprise_price = 250
    base_pro_price = 50
    
    # Define price test ranges
    enterprise_prices = np.linspace(
        base_enterprise_price * (1 - price_range),
        base_enterprise_price * (1 + price_range),
        num_points
    )
    
    pro_prices = np.linspace(
        base_pro_price * (1 - price_range),
        base_pro_price * (1 + price_range),
        num_points
    )
    
    # 1. Price elasticity analysis
    # Test enterprise price elasticity keeping pro price constant
    enterprise_elasticity_results = []
    for ep in enterprise_prices:
        model = PricingSystemModel(params, {
            'enterprise': ep,
            'pro': base_pro_price,
            'consultant_fee': 0.7,
            'trial_days': 14
        })
        results = model.run_simulation()
        enterprise_elasticity_results.append({
            'enterprise_price': ep,
            'enterprise_users': results['final_enterprise_users'],
            'pro_users': results['final_pro_users'],
            'npv': results['npv'],
            'mrr': results['final_mrr']
        })
    
    # Test pro price elasticity keeping enterprise price constant
    pro_elasticity_results = []
    for pp in pro_prices:
        model = PricingSystemModel(params, {
            'enterprise': base_enterprise_price,
            'pro': pp,
            'consultant_fee': 0.7,
            'trial_days': 14
        })
        results = model.run_simulation()
        pro_elasticity_results.append({
            'pro_price': pp,
            'enterprise_users': results['final_enterprise_users'],
            'pro_users': results['final_pro_users'],
            'npv': results['npv'],
            'mrr': results['final_mrr']
        })
    
    # 2. Network effect analysis
    # Create versions with different strength of network effects
    network_effect_results = []
    network_strength_multipliers = [0, 0.5, 1.0, 1.5, 2.0]
    
    for multiplier in network_strength_multipliers:
        # Create a copy of the parameters with modified network effects
        modified_params = PricingParams()
        modified_params.catalog_network_effect = params.catalog_network_effect * multiplier
        modified_params.consultant_network_effect = params.consultant_network_effect * multiplier
        modified_params.public_content_network_effect = params.public_content_network_effect * multiplier
        
        model = PricingSystemModel(modified_params, {
            'enterprise': base_enterprise_price,
            'pro': base_pro_price,
            'consultant_fee': 0.7,
            'trial_days': 14
        })
        results = model.run_simulation()
        
        network_effect_results.append({
            'network_multiplier': multiplier,
            'enterprise_users': results['final_enterprise_users'],
            'pro_users': results['final_pro_users'],
            'consultants': results['final_consultants'],
            'growers': results['final_growers'],
            'npv': results['npv'],
            'mrr': results['final_mrr'],
            'history': {
                'enterprise_users': results['history']['enterprise_users'],
                'pro_users': results['history']['pro_users'],
                'mrr': results['history']['mrr']
            }
        })
    
    # Calculate elasticity metrics
    ep_df = pd.DataFrame(enterprise_elasticity_results)
    pp_df = pd.DataFrame(pro_elasticity_results)
    
    # Calculate point elasticity at each price
    ep_df['log_price'] = np.log(ep_df['enterprise_price'])
    ep_df['log_users'] = np.log(ep_df['enterprise_users'])
    
    pp_df['log_price'] = np.log(pp_df['pro_price'])
    pp_df['log_users'] = np.log(pp_df['pro_users'])
    
    # Estimate elasticity from log-log slope
    enterprise_elasticity = np.polyfit(ep_df['log_price'], ep_df['log_users'], 1)[0]
    pro_elasticity = np.polyfit(pp_df['log_price'], pp_df['log_users'], 1)[0]
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Price vs users (elasticity curves)
    axes[0, 0].plot(ep_df['enterprise_price'], ep_df['enterprise_users'], marker='o')
    axes[0, 0].set_title(f'Enterprise Price Elasticity: {enterprise_elasticity:.2f}')
    axes[0, 0].set_xlabel('Enterprise Price ($)')
    axes[0, 0].set_ylabel('Enterprise Users')
    
    axes[0, 1].plot(pp_df['pro_price'], pp_df['pro_users'], marker='o')
    axes[0, 1].set_title(f'Pro Price Elasticity: {pro_elasticity:.2f}')
    axes[0, 1].set_xlabel('Pro Price ($)')
    axes[0, 1].set_ylabel('Pro Users')
    
    # Network effect strength vs user growth
    net_df = pd.DataFrame([{
        'multiplier': r['network_multiplier'],
        'enterprise_users': r['enterprise_users'],
        'pro_users': r['pro_users']
    } for r in network_effect_results])
    
    axes[1, 0].plot(net_df['multiplier'], net_df['enterprise_users'], marker='o', label='Enterprise')
    axes[1, 0].plot(net_df['multiplier'], net_df['pro_users'], marker='o', label='Pro')
    axes[1, 0].set_title('Network Effect Strength vs User Growth')
    axes[1, 0].set_xlabel('Network Effect Multiplier')
    axes[1, 0].set_ylabel('Users')
    axes[1, 0].legend()
    
    # Network effect strength vs NPV
    net_npv_df = pd.DataFrame([{
        'multiplier': r['network_multiplier'],
        'npv': r['npv']
    } for r in network_effect_results])
    
    axes[1, 1].plot(net_npv_df['multiplier'], net_npv_df['npv'], marker='o')
    axes[1, 1].set_title('Network Effect Strength vs NPV')
    axes[1, 1].set_xlabel('Network Effect Multiplier')
    axes[1, 1].set_ylabel('NPV ($)')
    
    plt.tight_layout()
    
    # Return analysis results
    return {
        'enterprise_elasticity': enterprise_elasticity,
        'pro_elasticity': pro_elasticity,
        'enterprise_elasticity_data': enterprise_elasticity_results,
        'pro_elasticity_data': pro_elasticity_results,
        'network_effect_results': network_effect_results,
        'visualizations': fig,
        'revenue_maximizing_enterprise_price': ep_df.loc[ep_df['mrr'].idxmax(), 'enterprise_price'],
        'revenue_maximizing_pro_price': pp_df.loc[pp_df['mrr'].idxmax(), 'pro_price'],
        'npv_maximizing_enterprise_price': ep_df.loc[ep_df['npv'].idxmax(), 'enterprise_price'],
        'npv_maximizing_pro_price': pp_df.loc[pp_df['npv'].idxmax(), 'pro_price']
    }