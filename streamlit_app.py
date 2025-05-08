import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import plotly.graph_objects as go
import plotly.express as px
import copy  # Add this for deepcopy operations
from main import (
    PricingParams, 
    PricingSystemModel, 
    optimize_pricing_strategy, 
    grid_search_pricing, 
    visualize_pricing_strategies,
    pricing_heuristic_search,
    optimize_with_cpsat,
    analyze_network_effects_and_elasticity
)

# Add PyMoo imports - wrap in try/except to handle cases where it's not installed
try:
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.optimize import minimize
    from pymoo.decomposition.asf import ASF
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.visualization.scatter import Scatter
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False

# Create a multi-objective optimization problem class for PyMoo
class PricingProblem(Problem):
    def __init__(self, pricing_params, objectives, param_ranges):
        self.pricing_params = pricing_params
        self.objectives = objectives  # List of objective names
        self.param_names = list(param_ranges.keys())
        
        # Extract lower and upper bounds for each parameter
        lb = np.array([param_ranges[param]['min'] for param in self.param_names])
        ub = np.array([param_ranges[param]['max'] for param in self.param_names])
        
        # Define problem with n_var variables, n_obj objectives, n_constr constraints
        super().__init__(
            n_var=len(self.param_names),  # Number of parameters to optimize
            n_obj=len(objectives),        # Number of objectives
            n_constr=0,                   # Number of constraints
            xl=lb,                        # Lower bounds
            xu=ub                         # Upper bounds
        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        # Initialize array to store objective values
        f = np.zeros((x.shape[0], len(self.objectives)))
        
        # For each solution in the population
        for i in range(x.shape[0]):
            # Create a copy of params to avoid modifying the original
            params_copy = PricingParams()
            params_copy.__dict__.update(self.pricing_params.__dict__)
            
            # Set parameter values for this solution
            param_values = {}
            for j, param_name in enumerate(self.param_names):
                setattr(params_copy, param_name, x[i, j])
                param_values[param_name] = x[i, j]
            
            # Extract parameters relevant for PricingSystemModel
            enterprise_price = param_values.get('enterprise_price', 299)
            pro_price = param_values.get('pro_price', 99)
            lifetime_price = param_values.get('lifetime_price', 995)
            consultant_fee = param_values.get('consultant_fee', 0.7)
            trial_days = param_values.get('trial_days', 14)
            
            # Create model and run simulation
            model = PricingSystemModel(params_copy, {
                'enterprise': enterprise_price,
                'pro': pro_price,
                'lifetime': lifetime_price,
                'consultant_fee': consultant_fee,
                'trial_days': trial_days
            })
            results = model.run_simulation()
            
            # Calculate additional user revenue if needed
            additional_users = results['final_enterprise_users'] * max(0, params_copy.avg_team_size - params_copy.base_team_size)
            additional_user_revenue = additional_users * params_copy.additional_user_cost
            results['final_mrr'] += additional_user_revenue
            
            # Map objectives according to maximize/minimize direction
            for j, obj in enumerate(self.objectives):
                if obj == 'npv':
                    # Maximize NPV (multiply by -1 for minimization problem)
                    f[i, j] = -results['npv']
                elif obj == 'mrr':
                    # Maximize MRR
                    f[i, j] = -results['final_mrr']
                elif obj == 'ltv_cac':
                    # Maximize LTV/CAC ratio
                    f[i, j] = -results['final_ltv_cac']
                elif obj == 'gross_margin':
                    # Maximize gross margin
                    f[i, j] = -results['final_gross_margin']
                elif obj == 'enterprise_users':
                    # Maximize enterprise users
                    f[i, j] = -results['final_enterprise_users']
                elif obj == 'pro_users':
                    # Maximize pro users
                    f[i, j] = -results['final_pro_users']
                elif obj == 'consultants':
                    # Maximize consultants
                    f[i, j] = -results['final_consultants']
                elif obj == 'churn':
                    # Minimize churn
                    f[i, j] = results.get('churn', 0)
        
        out["F"] = f

# Function to run multi-objective optimization
def run_multiobjective_optimization(pricing_params, objectives, param_ranges, pop_size=100, n_gen=50):
    """
    Run multi-objective optimization using NSGA-III algorithm
    
    Parameters:
    -----------
    pricing_params : PricingParams
        Parameters for the pricing model
    objectives : list
        List of objective names
    param_ranges : dict
        Dictionary of parameter ranges with min and max values
    pop_size : int
        Population size for NSGA-III
    n_gen : int
        Number of generations
    
    Returns:
    --------
    res : Result
        Optimization results
    """
    # Create the problem
    problem = PricingProblem(pricing_params, objectives, param_ranges)
    
    # Create reference directions for NSGA-III
    n_obj = len(objectives)
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
    
    # Create the algorithm
    algorithm = NSGA3(
        pop_size=pop_size,
        ref_dirs=ref_dirs
    )
    
    # Run the optimization
    res = minimize(problem,
                  algorithm,
                  ('n_gen', n_gen),
                  verbose=False)
    
    return res, problem

# Define global helper functions for Wellspring marketplace
def effective_consultants(model, params):
    """Calculate effective number of consultants including dual-role participants"""
    return (
        model.consultants + 
        (model.growers * params.grower_seller_pct) + 
        (model.pro_users * params.landscaper_seller_pct * 0.5)  # Pro users participation factor
    )

def calculate_wellspring_revenue(model, params):
    """Calculate Wellspring marketplace revenue based on credits model"""
    # Base revenue from dedicated consultants
    base_revenue = model.consultants * params.avg_monthly_credits * params.credits_price * (1 - params.consultant_fee_percentage)
    
    # Additional revenue from dual-role participants
    grower_revenue = model.growers * params.grower_seller_pct * params.avg_monthly_credits * 0.7 * params.credits_price * (1 - params.consultant_fee_percentage)
    
    # Calculate total enterprise users (including additional team members)
    total_enterprise_users = model.enterprise_users * params.avg_team_size
    enterprise_additional_users = model.enterprise_users * max(0, params.avg_team_size - params.base_team_size)
    
    # Landscaper revenue includes both individual pros and enterprise additional users
    landscaper_revenue = (model.pro_users + enterprise_additional_users * 0.3) * params.landscaper_seller_pct * params.avg_monthly_credits * 0.5 * params.credits_price * (1 - params.consultant_fee_percentage)
    
    # Total marketplace revenue
    total_revenue = base_revenue + grower_revenue + landscaper_revenue
    
    # Apply liquidity factor (more diverse sellers = more activity)
    seller_diversity = min(1.0, effective_consultants(model, params) / 100)  # Cap at 1.0
    liquidity_multiplier = 1.0 + (seller_diversity * 0.5)  # Max 50% boost from diversity
    
    return total_revenue * liquidity_multiplier

try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="SaaS Pricing Simulator",
    page_icon="💲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create tabs
tab_params, tab_basic, tab_advanced, tab_compare, tab_seedstage, tab_network, tab_wellspring, tab_multiobjective, tab_financial = st.tabs([
    "📊 Parameters", 
    "🔄 Basic Pricing", 
    "📈 Advanced Pricing", 
    "🔍 Compare Models", 
    "🌱 Seed Stage",
    "🌐 Network Effects",
    "🤝 Wellspring",
    "🎯 Multi-Objective",
    "💰 Financial Modeling"
])

# App title and description
st.title("SaaS Pricing Strategy Simulator")
st.markdown("""
This tool helps optimize subscription pricing strategies for SaaS platforms with multiple user segments and network effects.
Configure parameters in the sidebar, then use the tabs to explore different pricing approaches.
""")

# Sidebar for parameters
st.sidebar.header("Model Parameters")

# Add SOM/SAM toggle at the top of the sidebar
st.sidebar.subheader("Market Size")
market_size_option = st.sidebar.radio(
    "Market Size Model", 
    ["Full SAM (100%)", "SOM (10%)", "Custom SOM %"],
    help="Choose between full Serviceable Available Market (SAM) or Serviceable Obtainable Market (SOM). SAM represents your total addressable market, while SOM is the portion you can realistically capture."
)

# Function to create parameters with sidebar inputs
def create_params():
    params = PricingParams()
    
    # Apply market size selection
    if market_size_option == "SOM (10%)":
        params.use_som(0.1)
        st.sidebar.info("Using SOM: 10% of total market")
    elif market_size_option == "Full SAM (100%)":
        params.use_sam()
        st.sidebar.info("Using full SAM: 100% of total market")
    elif market_size_option == "Custom SOM %":
        custom_som_pct = st.sidebar.slider(
            "Custom SOM Percentage", 
            1, 100, 10, 
            help="Percentage of the total market (SAM) you expect to capture. Industry benchmarks: 5-15% for new markets, 10-30% for established markets."
        ) / 100
        params.use_som(custom_som_pct)
        st.sidebar.info(f"Using custom SOM: {custom_som_pct*100:.0f}% of total market")
    
    # Market size parameters - allow fine-tuning with sliders
    st.sidebar.subheader("Market Size (Edit to Customize)")
    
    # Store original values to use as defaults for sliders
    orig_enterprise = params.potential_enterprise_customers
    orig_pro = params.potential_pro_customers
    orig_consultants = params.potential_consultants
    orig_growers = params.potential_growers
    orig_landscapers = params.potential_landscapers
    
    # Sliders for adjusting each market size parameter
    params.potential_enterprise_customers = st.sidebar.slider(
        "Enterprise Customers", 
        10, int(orig_enterprise * 2), int(orig_enterprise), 
        step=10,
        help="Number of potential enterprise customers (landscape architecture firms) who could use your platform."
    )
    
    params.potential_pro_customers = st.sidebar.slider(
        "Pro Users (Architects)", 
        100, int(orig_pro * 2), int(orig_pro), 
        step=100,
        help="Number of potential individual landscape architects who could use pro accounts on your platform."
    )
    
    params.potential_consultants = st.sidebar.slider(
        "Consultants", 
        10, 6000, int(orig_consultants), 
        step=10,
        help="Number of potential consultants available for your marketplace."
    )
    
    params.potential_growers = st.sidebar.slider(
        "Growers", 
        10, 3000, int(orig_growers), 
        step=10,
        help="Number of potential growers who could provide plant catalogs on your platform."
    )
    
    params.potential_landscapers = st.sidebar.slider(
        "Landscapers (Pro Accounts)", 
        100, int(orig_landscapers * 2), int(orig_landscapers), 
        step=100,
        help="Number of potential individual landscapers/contractors who could use pro accounts on your platform (same pricing as architects)."
    )
    
    # Parameters that can still be adjusted
    st.sidebar.subheader("Elasticity Parameters")
    params.enterprise_price_elasticity = st.sidebar.slider(
        "Enterprise Price Elasticity", 
        -1.5, -0.1, params.enterprise_price_elasticity, 0.1,
        help="Price elasticity measures how demand responds to price changes. Values closer to 0 mean less price sensitivity. Enterprise customers typically range from -0.5 to -0.9."
    )
    params.pro_price_elasticity = st.sidebar.slider(
        "Pro Price Elasticity", 
        -2.0, -0.5, params.pro_price_elasticity, 0.1,
        help="Individual users tend to be more price sensitive than enterprises. Values typically range from -0.8 to -1.5 for B2B SaaS products targeted at professionals."
    )
    
    # Network effect parameters
    st.sidebar.subheader("Network Effects")
    network_effect_multiplier = st.sidebar.slider(
        "Network Effect Strength", 
        0.1, 2.0, 1.0, 0.1,
        help="Multiplier for how strongly network effects impact growth. Higher values mean users gain more value as the network grows. Values >1 indicate strong network effects like in marketplaces."
    )
    params.catalog_network_effect = 0.15 * network_effect_multiplier
    params.consultant_network_effect = 0.12 * network_effect_multiplier
    params.public_content_network_effect = 0.18 * network_effect_multiplier
    
    # Wellspring Marketplace parameters
    st.sidebar.subheader("Wellspring Marketplace")
    
    # Add parameters for dual-role participants
    params.grower_seller_pct = st.sidebar.slider(
        "Growers as Sellers (%)", 
        5, 80, 30, 5,
        help="Percentage of growers who also act as sellers on the Wellspring marketplace, providing specialized knowledge."
    ) / 100
    
    params.landscaper_seller_pct = st.sidebar.slider(
        "Landscapers as Sellers (%)", 
        5, 70, 25, 5,
        help="Percentage of landscapers who also act as sellers on the Wellspring marketplace, providing practical advice."
    ) / 100
    
    # Credits system parameters
    credits_price = st.sidebar.number_input(
        "Price per Credit ($)", 
        1.0, 50.0, 10.0, 1.0, 
        help="Price per credit for Wellspring marketplace transactions. Credits are used to pay for advice and consultations."
    )
    
    avg_monthly_credits = st.sidebar.slider(
        "Avg. Monthly Credits per User", 
        1, 50, 5, 1,
        help="Average number of credits consumed per active user each month. Higher credit usage indicates more marketplace activity."
    )
    
    # Consultant fee percentage - this is now the direct platform revenue share
    consultant_fee = st.sidebar.slider(
        "Consultant Fee Share (%)", 
        10, 95, 70, 5,
        help="Percentage of transaction value that goes to the consultant. The platform keeps the remainder (100% - consultant fee %)."
    ) / 100
    
    # Add more Wellspring marketplace parameters from PricingParams
    schedules_per_year = st.sidebar.slider(
        "Schedules per Seller (yearly)", 
        1, 100, int(params.schedules_per_seller_per_year), 1,
        help="Average number of planting schedules created by each seller annually."
    )
    
    substitutions_per_schedule = st.sidebar.slider(
        "Substitutions per Schedule", 
        1, 20, int(params.avg_substitutions_per_schedule), 1,
        help="Average number of plant substitutions made per schedule."
    )
    
    fee_per_substitution = st.sidebar.number_input(
        "Fee per Substitution ($)", 
        0.1, 10.0, params.fee_per_substitution, 0.1,
        help="Fee charged for each plant substitution processed through the platform."
    )
    
    performance_factor = st.sidebar.slider(
        "Performance-Related Factor (%)", 
        1, 100, int(params.performance_factor * 100), 1,
        help="Percentage of substitutions deemed performance-related, affecting revenue calculations."
    ) / 100
    
    # Store these values on params
    params.grower_seller_pct = params.grower_seller_pct
    params.landscaper_seller_pct = params.landscaper_seller_pct
    params.credits_price = credits_price
    params.avg_monthly_credits = avg_monthly_credits
    params.consultant_fee_percentage = consultant_fee  # Updated to store directly
    params.schedules_per_seller_per_year = schedules_per_year
    params.avg_substitutions_per_schedule = substitutions_per_schedule
    params.fee_per_substitution = fee_per_substitution
    params.performance_factor = performance_factor
    
    # No need to redefine these functions as they're now global
    params.calculate_wellspring_revenue = lambda model: calculate_wellspring_revenue(model, params)
    
    # Conversion parameters
    st.sidebar.subheader("Conversion Parameters")
    params.pro_trial_to_paid = st.sidebar.slider(
        "Pro Trial Conversion", 
        0.1, 0.8, params.pro_trial_to_paid, 0.05,
        help="Percentage of trial users who convert to paid. Industry benchmarks: 10-25% is typical, 25-50% is good, >50% is excellent. Higher value = better trial experience or product-market fit."
    )
    params.enterprise_demo_to_paid = st.sidebar.slider(
        "Enterprise Demo Conversion", 
        0.1, 0.8, params.enterprise_demo_to_paid, 0.05,
        help="Percentage of enterprise demos that convert to paid accounts. Industry benchmarks: 20-40% is average, >50% indicates strong product-market fit and sales execution."
    )
    params.contracts_platform_usage = st.sidebar.slider(
        "Contracts Platform Usage", 
        0.01, 0.20, params.contracts_platform_usage, 0.01,
        help="Percentage of enterprise users who utilize the contracts platform. Higher usage generates more transaction revenue but requires additional platform adoption."
    )
    
    # Other key parameters
    st.sidebar.subheader("Other Parameters")
    params.base_enterprise_churn_rate = st.sidebar.slider(
        "Base Enterprise Churn", 
        0.01, 0.1, params.base_enterprise_churn_rate, 0.01,
        help="Monthly churn rate for enterprise customers. Industry benchmarks: 1-2% is excellent, 2-3% is good, 3-5% is average, >5% needs improvement. Annual contracts typically reduce churn."
    )
    params.base_pro_churn_rate = st.sidebar.slider(
        "Base Pro Churn", 
        0.02, 0.2, params.base_pro_churn_rate, 0.01,
        help="Monthly churn rate for pro users (architects and landscapers). Industry benchmarks: 3-5% is good, 5-7% is average, >8% needs improvement. Higher than enterprise due to individual purchasing decisions."
    )
    
    # Transaction fees
    st.sidebar.subheader("Transaction Fees")
    params.contract_tx_fee = st.sidebar.slider(
        "Contract Transaction Fee", 
        0.01, 0.2, params.contract_tx_fee, 0.01,
        help="Percentage fee charged on contract transactions. Industry benchmarks: 1-3% for payment processing only, 5-15% for platforms providing significant value-add services."
    )
    # Removed Wellspring Transaction Fee slider since we don't use it anymore
    
    # Add average contract size
    params.avg_contract_size = st.sidebar.number_input(
        "Average Contract Size ($)",
        1000, 100000, int(params.avg_contract_size), 1000,
        help="Average value of commercial landscape contracts processed through the platform."
    )
    
    # Enterprise Team Parameters - add this section
    st.sidebar.subheader("Enterprise Team Parameters")
    params.base_team_size = st.sidebar.slider(
        "Base Team Size", 
        1, 15, 5, 1,
        help="Number of users included in the base enterprise subscription. Industry benchmarks: 3-5 for basic teams, 5-10 for growing teams."
    )
    
    params.additional_user_cost = st.sidebar.slider(
        "Additional User Cost ($)", 
        10, 100, 49, 5,
        help="Monthly cost for each additional user beyond the base team size. Typically 20-40% of the per-user equivalent price."
    )
    
    params.avg_team_size = st.sidebar.slider(
        "Average Team Size", 
        params.base_team_size, 30, min(params.base_team_size + 5, 15), 1,
        help="Average team size for enterprise customers. This determines how many additional users are paying."
    )
    
    params.max_team_size = st.sidebar.slider(
        "Maximum Team Size", 
        params.avg_team_size, 50, params.avg_team_size + 10, 5,
        help="Maximum allowed team size. Set to unlimited by making it very large."
    )
    
    # Add billing multipliers
    st.sidebar.subheader("Billing Cycles")
    params.enterprise_billing_multiplier = st.sidebar.slider(
        "Enterprise Billing Multiplier",
        1, 12, params.enterprise_billing_multiplier, 1,
        help="Multiplier for enterprise billing cycle. 1=monthly, 12=annual billing."
    )
    
    params.pro_billing_multiplier = st.sidebar.slider(
        "Pro Billing Multiplier",
        1, 12, params.pro_billing_multiplier, 1,
        help="Multiplier for pro billing cycle. 1=monthly, 12=annual billing."
    )
    
    # Ensure avg_team_size is never less than base_team_size
    if params.avg_team_size < params.base_team_size:
        params.avg_team_size = params.base_team_size
    
    # Segment distribution parameters
    st.sidebar.subheader("Customer Segment Distribution")
    params.landscape_architects_pct = st.sidebar.slider(
        "Landscape Architects in Enterprise (%)",
        10, 90, int(params.landscape_architects_pct * 100), 5,
        help="Percentage of enterprise customers who are landscape architecture firms."
    ) / 100
    
    params.consultants_pct = st.sidebar.slider(
        "Consultants in Enterprise (%)",
        5, 50, int(params.consultants_pct * 100), 5,
        help="Percentage of enterprise customers who are consultancy firms."
    ) / 100
    
    params.growers_in_platform_pct = st.sidebar.slider(
        "Initial Growers in Platform (%)",
        1, 50, int(params.growers_in_platform_pct * 100), 5,
        help="Initial percentage of potential growers who join the platform at launch."
    ) / 100
    
    # Add additional elasticity parameter
    st.sidebar.subheader("Additional Elasticity Parameters")
    params.lifetime_deal_elasticity = st.sidebar.slider(
        "Lifetime Deal Elasticity",
        -1.5, -0.1, params.lifetime_deal_elasticity, 0.1,
        help="Price elasticity for lifetime deals. How sensitive demand is to lifetime deal pricing."
    )
    
    # Add value perception parameters
    st.sidebar.subheader("Value Perception")
    params.enterprise_value_perception = st.sidebar.slider(
        "Enterprise Value Perception",
        0.1, 1.0, params.enterprise_value_perception, 0.05,
        help="How highly enterprise users value the product (0.1=low, 1.0=high). Affects churn and willingness to pay."
    )
    
    params.pro_value_perception = st.sidebar.slider(
        "Pro Value Perception",
        0.1, 1.0, params.pro_value_perception, 0.05,
        help="How highly professional users value the product (0.1=low, 1.0=high). Affects churn and willingness to pay."
    )
    
    # Add free-to-paid conversion rate
    st.sidebar.subheader("Additional Conversion Parameters")
    params.free_to_paid_rate = st.sidebar.slider(
        "Free to Paid Conversion Rate",
        0.01, 0.2, params.free_to_paid_rate, 0.01,
        help="Percentage of free users who convert to paid accounts monthly."
    )
    
    # Add advice parameters
    params.avg_advice_price = st.sidebar.number_input(
        "Average Advice Price ($)",
        10, 200, int(params.avg_advice_price), 5,
        help="Average price per advice transaction in the marketplace."
    )
    
    params.avg_advice_frequency = st.sidebar.slider(
        "Average Monthly Advice Transactions",
        0.05, 1.0, params.avg_advice_frequency, 0.05,
        help="Average number of advice transactions per user per month."
    )
    
    # Add acquisition rate parameters to Advanced section
    st.sidebar.subheader("Advanced Acquisition Parameters")
    params.base_enterprise_acquisition_rate = st.sidebar.slider(
        "Base Enterprise Acquisition Rate",
        0.001, 0.1, params.base_enterprise_acquisition_rate, 0.001,
        help="Monthly percentage of potential enterprise customers acquired without network effects."
    )
    
    params.base_pro_acquisition_rate = st.sidebar.slider(
        "Base Pro Acquisition Rate",
        0.001, 0.1, params.base_pro_acquisition_rate, 0.001,
        help="Monthly percentage of potential pro users acquired without network effects."
    )
    
    # Cost structure parameters
    st.sidebar.subheader("Cost Structure")
    params.service_cost_per_enterprise = st.sidebar.number_input(
        "Monthly Cost per Enterprise User ($)", 
        10, 500, int(params.service_cost_per_enterprise), 10,
        help="Direct monthly cost to service an enterprise account, including infrastructure, support, and operations costs."
    )
    params.service_cost_per_pro = st.sidebar.number_input(
        "Monthly Cost per Pro User ($)", 
        1, 100, int(params.service_cost_per_pro), 5,
        help="Direct monthly cost to service a pro account (architect or landscaper), including infrastructure, support, and operations costs."
    )
    
    # Replace CAC inputs with marketing budget allocation fields
    st.sidebar.subheader("Marketing Budget Allocation")
    total_monthly_marketing = st.sidebar.number_input(
        "Total Monthly Marketing Budget ($)", 
        500, 50000, 10000, 500,
        help="Total monthly marketing and user acquisition budget for your seed-stage startup."
    )
    
    enterprise_allocation = st.sidebar.slider(
        "Enterprise Marketing Allocation (%)", 
        10, 90, 40, 5,
        help="Percentage of marketing budget allocated to enterprise customer acquisition (B2B sales, demos, webinars)."
    )
    
    # Calculate implied CAC values based on budget allocation
    params.enterprise_cac = total_monthly_marketing * (enterprise_allocation/100) / (params.base_enterprise_acquisition_rate * params.potential_enterprise_customers * 0.05)
    params.pro_cac = total_monthly_marketing * ((100-enterprise_allocation)/100) / (params.base_pro_acquisition_rate * (params.potential_pro_customers + params.potential_landscapers) * 0.05)
    
    # Show the implied CAC values
    st.sidebar.info(f"Implied Enterprise CAC: ${int(params.enterprise_cac)}")
    st.sidebar.info(f"Implied Pro/Landscaper CAC: ${int(params.pro_cac)}")
    
    # Seed-stage relevant metrics
    st.sidebar.subheader("Seed-Stage Metrics")
    monthly_burn_rate = st.sidebar.number_input(
        "Monthly Burn Rate ($)",
        5000, 100000, 30000, 1000,
        help="Your total monthly expenses including salaries, infrastructure, marketing, and operations."
    )
    
    runway_months = st.sidebar.number_input(
        "Current Runway (months)",
        3, 24, 12, 1,
        help="How many months of runway you currently have at your burn rate."
    )
    
    # Discount rate
    st.sidebar.subheader("Financial Parameters")
    discount_rate_annual = st.sidebar.slider(
        "Annual Discount Rate (%)", 
        5.0, 30.0, 20.0, 1.0,
        help="Annual discount rate used for NPV calculations. Typically 15-30% for early-stage startups with higher risk profiles."
    )
    params.discount_rate_monthly = (1 + discount_rate_annual/100)**(1/12) - 1
    
    return params, monthly_burn_rate, runway_months

# Create parameters in Parameters tab
with tab_params:
    st.header("Parameter Configuration")
    
    # Add expander explaining seed-stage pricing strategies
    with st.expander("Seed-Stage Pricing Strategies", expanded=True):
        st.markdown("""
        ### Effective Pricing Strategies for Pre-Revenue Seed-Stage Startups
        
        **Key Considerations for Early-Stage Products:**
        
        1. **Value-Based Pricing**: Set prices based on the value you deliver, not your costs. Even with limited features, focus on the core problem you solve.
        
        2. **Penetration vs. Premium**: 
           - **Penetration pricing** (lower prices) can accelerate adoption and network effects
           - **Premium pricing** (higher prices) signals quality and attracts serious customers
        
        3. **Experimentation Approach**:
           - Start with premium pricing if you have high-touch service or enterprise focus
           - Start with penetration pricing if network effects are critical
           - Use time-limited offers to test price sensitivity without permanently lowering prices
        
        4. **Early Adopter Specials**:
           - Lifetime deals can provide early cash flow
           - "Founding member" pricing with special benefits
           - Steep annual prepay discounts (40-50%) to improve cash position
        
        5. **Metrics That Matter at Seed Stage**:
           - Revenue growth rate (week-over-week or month-over-month)
           - Conversion rate from free to paid
           - Net dollar retention (are early customers expanding usage?)
           - Cash extension from revenue (how much does each pricing model extend your runway?)
        
        Remember that pricing can and should evolve as you learn more about your market and as your product matures.
        """)
    
    st.markdown("""
    Configure your SaaS business parameters in the sidebar. The model includes:
    
    * **Market Sizes**: Potential customers in each segment
    * **Price Elasticity**: How sensitive each segment is to price changes
    * **Network Effects**: Impact of catalogs, consultants, and content on acquisition
    * **Churn & Conversion**: Base churn and trial conversion rates
    
    After configuring parameters, explore the optimization tabs to analyze different pricing strategies.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Key Customer Segments")
        st.markdown("""
        - **Enterprise Users**: Landscape architecture firms (teams)
        - **Pro Users**: Individual professionals (includes both landscape architects and landscapers)
        - **Consultants**: Horticultural experts on Wellspring marketplace
        - **Growers**: Providers of plant catalogs
        """)
    
    with col2:
        st.subheader("Revenue Streams")
        st.markdown("""
        - **Subscription Revenue**: Monthly/annual plans
        - **Contract Revenue**: Fees from landscape contracts
        - **Wellspring Revenue**: Marketplace transactions
        - **Lifetime Deals**: One-time purchases (early phase)
        """)

# Get parameters
params, monthly_burn_rate, runway_months = create_params()

# Basic Optimization Tab
with tab_basic:
    st.header("Basic Pricing Optimization")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Initial pricing input
        st.subheader("Initial Pricing")
        init_enterprise_price = st.number_input(
            "Enterprise Monthly Price ($)", 
            50, 1000, 299, 10,
            help="Monthly price for enterprise customers. SaaS benchmarks: $200-500 for team tools, $300-1000 for specialized B2B software. Higher prices are justified by strong ROI and enterprise features."
        )
        init_pro_price = st.number_input(
            "Pro Monthly Price ($)", 
            10, 500, 99, 5,
            help="Monthly price for professional individual users (both landscape architects and landscapers). Industry benchmarks: $10-30 for basic tools, $50-150 for professional tools, $100-300 for specialized software with high value."
        )
        init_lifetime_price = st.number_input(
            "Lifetime Deal Price ($)", 
            200, 3000, 995, 50,
            help="One-time payment for lifetime access. Typically 20-40x monthly price. Used for early adoption and cash flow. Note: Can cannibalize subscription revenue if overused."
        )
        init_consultant_fee = st.number_input(
            "Consultant Fee Share", 
            0.5, 0.9, 0.7, 0.05,
            help="Percentage of marketplace fees paid to consultants. Industry benchmarks: 60-80% for contributor/expert marketplaces. Higher values attract more consultants but reduce platform revenue."
        )
        
        # Run initial simulation button
        if st.button("Run Initial Simulation"):
            with st.spinner("Running initial simulation..."):
                initial_model = PricingSystemModel(params, {
                    'enterprise': init_enterprise_price,
                    'pro': init_pro_price,
                    'lifetime': init_lifetime_price,
                    'consultant_fee': init_consultant_fee,
                    'trial_days': 14
                })
                initial_results = initial_model.run_simulation()
                
                # Add the model to the results for revenue calculation
                initial_results['model'] = initial_model
                
                # Calculate Wellspring marketplace metrics using our new function
                wellspring_revenue = calculate_wellspring_revenue(initial_model, params)
                initial_model.wellspring_revenue = wellspring_revenue  # Override the default calculation
                
                # Calculate effective consultant count including dual-role participants
                effective_consultant_count = effective_consultants(initial_model, params)
                initial_results['effective_consultants'] = effective_consultant_count
                
                # Store in session state for comparison tab
                st.session_state.initial_results = {
                    'enterprise_price': init_enterprise_price,
                    'pro_price': init_pro_price,
                    'lifetime_price': init_lifetime_price,
                    'consultant_fee': init_consultant_fee,
                    'trial_days': 14,
                    'npv': initial_results['npv'],
                    'final_mrr': initial_results['final_mrr'],
                    'final_ltv_cac': initial_results['final_ltv_cac'],
                    'final_gross_margin': initial_results['final_gross_margin'],
                    'final_enterprise_users': initial_results['final_enterprise_users'],
                    'final_pro_users': initial_results['final_pro_users'],
                    'final_consultants': initial_results['final_consultants'],
                    'model': initial_model,
                    'history': initial_results['history']
                }
                
                # Calculate additional team users revenue
                additional_users = initial_results['final_enterprise_users'] * max(0, params.avg_team_size - params.base_team_size)
                additional_user_revenue = additional_users * params.additional_user_cost
                
                # Add the additional user revenue to the final MRR for display
                initial_results['final_mrr'] += additional_user_revenue
                initial_model.additional_user_revenue = additional_user_revenue
                st.session_state.initial_results['final_mrr'] = initial_results['final_mrr']
                
                # Display results
                st.subheader("Initial Results")
                st.metric("NPV", f"${initial_results['npv']:,.2f}")
                
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    st.metric("Final MRR", f"${initial_results['final_mrr']:,.2f}")
                with metrics_col2:
                    st.metric("LTV/CAC Ratio", f"{initial_results['final_ltv_cac']:.2f}")
                with metrics_col3:
                    st.metric("Gross Margin", f"{initial_results['final_gross_margin']*100:.1f}%")
                
                # Display static metrics
                st.subheader("Initial Model Results")
                rev_col1, rev_col2, rev_col3, rev_col4 = st.columns(4)
                with rev_col1:
                    # Use model directly for the values
                    contract_rev = initial_model.contract_revenue
                    wellspring_rev = initial_model.wellspring_revenue
                    subscription_rev = initial_results['final_mrr'] - contract_rev - wellspring_rev - additional_user_revenue
                    st.metric("Subscription Revenue", f"${subscription_rev:,.2f}")
                with rev_col2:
                    st.metric("Contract Revenue", f"${contract_rev:,.2f}")
                with rev_col3:
                    st.metric("Wellspring Revenue", f"${wellspring_rev:,.2f}")
                with rev_col4:
                    st.metric("Additional User Revenue", f"${additional_user_revenue:,.2f}")

                # Add enterprise team metrics section
                st.subheader("Enterprise Team Metrics")
                team_col1, team_col2, team_col3 = st.columns(3)
                with team_col1:
                    st.metric("Enterprise Accounts", f"{initial_results['final_enterprise_users']:.0f}")
                with team_col2:
                    additional_users = initial_results['final_enterprise_users'] * max(0, params.avg_team_size - params.base_team_size)
                    additional_user_revenue = additional_users * params.additional_user_cost
                    st.metric("Additional Users", f"{additional_users:.0f}")
                with team_col3:
                    total_enterprise_users = initial_results['final_enterprise_users'] * params.avg_team_size
                    st.metric("Total Enterprise Users", f"{total_enterprise_users:.0f}")
                
                # Add additional user revenue to the revenue breakdown
                st.metric("Additional User Revenue", f"${additional_user_revenue:,.2f}")

    with col2:
        # Optimization
        st.subheader("Pricing Optimization")
        
        # Add informative expander explaining optimization objectives
        with st.expander("Understanding Optimization Objectives", expanded=True):
            st.markdown("""
            ### Choose which metric to optimize for:
            
            **NPV (Net Present Value)**
            - *What it is*: Sum of all future profits, discounted to present value
            - *When to choose*: Best default choice that balances short and long-term value
            - *Optimizes for*: Balance between growth, retention, and profitability
            - *Trade-offs*: May sacrifice some immediate revenue for longer-term gains
            
            **LTV/CAC Ratio**
            - *What it is*: Ratio between customer lifetime value and acquisition cost
            - *When to choose*: When focusing on unit economics and capital efficiency
            - *Optimizes for*: Customer profitability and retention
            - *Trade-offs*: May lead to higher prices and slower growth
            
            **MRR (Monthly Recurring Revenue)**
            - *What it is*: Total predictable monthly revenue from all customers
            - *When to choose*: When prioritizing top-line growth and market share
            - *Optimizes for*: Maximum number of customers and total revenue
            - *Trade-offs*: Might sacrifice margins and efficiency
            
            **Gross Margin**
            - *What it is*: Percentage of revenue remaining after direct costs
            - *When to choose*: When profitability is the primary concern
            - *Optimizes for*: Maximum profitability per customer
            - *Trade-offs*: May lead to highest prices and smallest customer base
            
            **Practical Approach**:
            Run optimizations for multiple objectives to understand the trade-offs, then align with your current business priorities.
            """)
        
        objective = st.selectbox(
            "Optimization Objective", 
            ["npv", "ltv_cac", "mrr", "gross_margin"],
            help="Metric to optimize for. NPV (net present value) balances short and long-term value. LTV/CAC optimizes for unit economics. MRR maximizes monthly revenue. Gross margin optimizes for profitability."
        )
        
        # Price constraints
        st.subheader("Price Constraints")
        min_ent, max_ent = st.slider(
            "Enterprise Price Range ($)", 
            100, 1000, (200, 500),
            help="Range of possible enterprise prices to consider in optimization. Wider ranges give more flexibility but may lead to extreme solutions."
        )
        min_pro, max_pro = st.slider(
            "Pro Price Range ($)", 
            20, 300, (50, 150),
            help="Range of possible pro user prices to consider in optimization. Pro prices typically follow a 1:3 to 1:5 ratio compared to enterprise prices."
        )
        min_lifetime, max_lifetime = st.slider(
            "Lifetime Price Range ($)", 
            500, 3000, (800, 2000),
            help="Range of possible lifetime deal prices. Typically 20-40x monthly price for high-value SaaS products."
        )
        
        constraints = {
            'enterprise_price_min': min_ent,
            'enterprise_price_max': max_ent,
            'pro_price_min': min_pro,
            'pro_price_max': max_pro,
            'lifetime_price_min': min_lifetime,
            'lifetime_price_max': max_lifetime,
            'consultant_fee_min': 0.5,
            'consultant_fee_max': 0.9,
            'trial_days_min': 7,
            'trial_days_max': 30
        }
        
        # Run optimization button
        if st.button("Run Optimization"):
            progress_bar = st.progress(0)
            with st.spinner("Running optimization..."):
                # Display a progress bar simulation
                for i in range(100):
                    # Simulate progress
                    progress_bar.progress(i + 1)
                    time.sleep(0.05)
                
                # Run actual optimization
                optimized_results = optimize_pricing_strategy(params, objective, constraints)
                
                # Store in session state for comparison tab
                st.session_state.optimized_results = optimized_results
                
                # Calculate additional user revenue for optimized model
                opt_additional_users = optimized_results['final_enterprise_users'] * max(0, params.avg_team_size - params.base_team_size)
                opt_additional_user_revenue = opt_additional_users * params.additional_user_cost
                
                # Add the additional user revenue to the final MRR for display
                optimized_results['final_mrr'] += opt_additional_user_revenue
                optimized_results['model'].additional_user_revenue = opt_additional_user_revenue
                
                # Calculate Wellspring marketplace metrics for the optimized model
                optimized_wellspring_revenue = calculate_wellspring_revenue(optimized_results['model'], params)
                optimized_results['model'].wellspring_revenue = optimized_wellspring_revenue  # Override default
                
                # Calculate effective consultant count
                optimized_effective_consultants = effective_consultants(optimized_results['model'], params)
                optimized_results['effective_consultants'] = optimized_effective_consultants
                
                # Display results
                st.subheader("Optimized Results")
                st.metric("NPV", f"${optimized_results['npv']:,.2f}")
                
                opt_col1, opt_col2, opt_col3 = st.columns(3)
                with opt_col1:
                    st.metric("Enterprise Price", f"${optimized_results['enterprise_price']:.2f}")
                with opt_col2:
                    st.metric("Pro Price", f"${optimized_results['pro_price']:.2f}")
                with opt_col3:
                    st.metric("Lifetime Deal", f"${optimized_results['lifetime_price']:.2f}")
                
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    st.metric("Final MRR", f"${optimized_results['final_mrr']:,.2f}")
                with metrics_col2:
                    st.metric("LTV/CAC Ratio", f"{optimized_results['final_ltv_cac']:.2f}")
                with metrics_col3:
                    st.metric("Gross Margin", f"{optimized_results['final_gross_margin']*100:.1f}%")
                
                # Revenue breakdown
                st.subheader("Revenue Breakdown")
                rev_col1, rev_col2, rev_col3, rev_col4 = st.columns(4)
                with rev_col1:
                    # Use model directly for the values
                    contract_rev = optimized_results['model'].contract_revenue
                    wellspring_rev = optimized_results['model'].wellspring_revenue  # Use our calculated value
                    subscription_rev = optimized_results['final_mrr'] - contract_rev - wellspring_rev - opt_additional_user_revenue
                    st.metric("Subscription Revenue", f"${subscription_rev:,.2f}")
                with rev_col2:
                    st.metric("Contract Revenue", f"${contract_rev:,.2f}")
                with rev_col3:
                    st.metric("Wellspring Revenue", f"${wellspring_rev:,.2f}")
                with rev_col4:
                    st.metric("Additional User Revenue", f"${opt_additional_user_revenue:,.2f}")
                    
                # Add enterprise team metrics section
                st.subheader("Enterprise Team Metrics")
                team_col1, team_col2, team_col3 = st.columns(3)
                with team_col1:
                    st.metric("Enterprise Accounts", f"{optimized_results['final_enterprise_users']:.0f}")
                with team_col2:
                    opt_additional_users = optimized_results['final_enterprise_users'] * max(0, params.avg_team_size - params.base_team_size)
                    opt_additional_user_revenue = opt_additional_users * params.additional_user_cost
                    st.metric("Additional Users", f"{opt_additional_users:.0f}")
                with team_col3:
                    opt_total_enterprise_users = optimized_results['final_enterprise_users'] * params.avg_team_size
                    st.metric("Total Enterprise Users", f"{opt_total_enterprise_users:.0f}")
                
                # Add additional user revenue to the revenue breakdown
                st.metric("Additional User Revenue", f"${opt_additional_user_revenue:,.2f}")
                    
                # Plot
                fig = visualize_pricing_strategies(optimized_results)
                st.pyplot(fig)
                
                # Add runway impact section for optimized results
                st.subheader("Runway & Cash Flow Impact")
                
                # Calculate monthly burn after revenue
                monthly_net_burn = monthly_burn_rate - optimized_results['final_mrr']
                
                # Only show extended runway if revenue is positive but less than burn rate
                if monthly_net_burn > 0 and optimized_results['final_mrr'] > 0:
                    new_runway = runway_months * (monthly_burn_rate / monthly_net_burn)
                    runway_extension = new_runway - runway_months
                    
                    runway_col1, runway_col2 = st.columns(2)
                    with runway_col1:
                        st.metric("Current Runway", f"{runway_months} months")
                    with runway_col2:
                        st.metric("Extended Runway", f"{new_runway:.1f} months", f"+{runway_extension:.1f} months")
                
                # Calculate months to cash flow positive
                if monthly_net_burn <= 0:
                    st.success(f"This optimized pricing strategy makes you cash flow positive with ${abs(monthly_net_burn):.2f} monthly surplus!")
                else:
                    # Estimate time to positive cash flow based on MRR growth rate
                    if len(optimized_results['model'].history['mrr']) > 3:
                        # Calculate average monthly growth rate from the last few months
                        recent_mrr = optimized_results['model'].history['mrr'][-3:]
                        if recent_mrr[0] > 0:
                            monthly_growth_rate = (recent_mrr[-1] / recent_mrr[0]) ** (1/3) - 1
                            
                            # Only show projection if we have positive growth
                            if monthly_growth_rate > 0:
                                # Project months until MRR >= burn rate
                                months_to_positive = np.log(monthly_burn_rate / optimized_results['final_mrr']) / np.log(1 + monthly_growth_rate)
                                
                                if months_to_positive > 0 and months_to_positive < 36:
                                    st.info(f"Estimated time to positive cash flow: {months_to_positive:.1f} months at current growth rate ({monthly_growth_rate*100:.1f}% monthly)")
                                elif months_to_positive >= 36:
                                    st.warning(f"At current growth rate ({monthly_growth_rate*100:.1f}% monthly), positive cash flow will take more than 3 years")
                            else:
                                st.warning("MRR growth is flat or negative. Unable to project time to positive cash flow.")

# Advanced Pricing Tab
with tab_advanced:
    st.header("Advanced Time-Varying Pricing")
    
    # Add information about potential unrealistic LTV/CAC ratios
    with st.expander("About LTV/CAC Ratios & Model Calibration", expanded=False):
        st.markdown("""
        ### Understanding Model Results

        **High LTV/CAC Ratios**
        
        If you see unusually high LTV/CAC ratios (e.g., >10-15) in your results, this may be due to:
        
        - **Low churn rates**: The model may be underestimating realistic churn for your market
        - **Network effects**: Strong network effects in the model can create compounding benefits
        - **High pricing power**: The model might assume stronger pricing power than realistic
        - **Fixed CAC**: The model uses fixed CAC values rather than increasing CAC as you penetrate the market
        
        **For More Realistic Results**
        
        If you want more conservative projections:
        
        - Increase base churn rates (try 4-6% for enterprise, 10-15% for pro)
        - Increase CAC values in the underlying parameters
        - Reduce network effect strength (try 0.5-0.7 instead of 1.0)
        - Increase price elasticity if your market is more price-sensitive
        
        Remember that models are simplifications, and it's always good practice to run scenarios with varying parameters to test the robustness of your pricing strategy.
        """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Time-Varying Pricing Parameters")
        time_horizon = st.slider(
            "Time Horizon (months)", 
            6, 36, 24,
            help="Number of months to model in the simulation. Longer horizons (24-36 months) capture full market dynamics but increase complexity. Shorter horizons are more accurate for near-term planning."
        )
        iterations = st.slider(
            "Search Iterations", 
            100, 2000, 500,
            help="Number of iterations for the heuristic search algorithm. More iterations (>500) typically find better solutions but take longer to run. Diminishing returns after ~1000 iterations."
        )
        max_price_change = st.slider(
            "Max Price Change (%)", 
            5, 30, 10,
            help="Maximum percentage price change allowed between consecutive periods. Lower values (5-10%) create smoother price transitions. Higher values allow more aggressive price optimization."
        )
        smooth_constraint = st.checkbox(
            "Enforce Smooth Price Transitions", 
            True,
            help="When enabled, ensures prices change gradually over time. This creates more predictable pricing for customers and reduces churn from price shock."
        )
        
        # Run heuristic search button
        if st.button("Run Heuristic Search"):
            with st.spinner("Running time-varying pricing optimization..."):
                progress_bar = st.progress(0)
                
                # Make sure params has the correct time_horizon_months
                # Without this, the simulation might try to access price indices beyond array length
                params.time_horizon_months = time_horizon
                
                # Progress updates
                for i in range(50):
                    progress_bar.progress(i + 1)
                    time.sleep(0.02)
                
                # Run optimization with the updated time horizon
                try:
                    heuristic_results = pricing_heuristic_search(
                        params,
                        iterations=iterations,
                        time_horizon=time_horizon,  # Pass time_horizon explicitly
                        smooth_constraint=smooth_constraint,
                        max_price_change_pct=max_price_change
                    )
                    
                    # Update progress to 100% after successful completion
                    progress_bar.progress(100)
                    
                    # Store in session state for comparison tab
                    st.session_state.heuristic_results = heuristic_results
                    
                    # Display results
                    st.subheader("Time-Varying Pricing Results")
                    st.metric("NPV", f"${heuristic_results['npv']:,.2f}")
                    
                    # Display price evolution
                    st.subheader("Price Evolution")
                    price_data = pd.DataFrame({
                        'Month': list(range(time_horizon)),
                        'Enterprise Price': heuristic_results['enterprise_price'],
                        'Pro Price': heuristic_results['pro_price']
                    })
                    
                    st.line_chart(price_data.set_index('Month'))
                    
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    with metrics_col1:
                        st.metric("Final MRR", f"${heuristic_results['final_mrr']:,.2f}")
                    with metrics_col2:
                        st.metric("LTV/CAC Ratio", f"{heuristic_results['final_ltv_cac']:.2f}")
                    with metrics_col3:
                        st.metric("Gross Margin", f"{heuristic_results['final_gross_margin']*100:.1f}%")
                    
                    # Plot
                    fig = visualize_pricing_strategies(heuristic_results)
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error running optimization: {str(e)}")
                    st.warning("Try reducing the time horizon or check if prices arrays have sufficient length.")
                    
    with col2:
        st.subheader("Constrained Optimization (CP-SAT)")
        
        if not ORTOOLS_AVAILABLE:
            st.warning("Google OR-Tools not available. Install with: pip install ortools")
        else:
            cpsat_time_horizon = st.slider(
                "CP-SAT Time Horizon", 
                3, 12, 6,
                help="Number of months for constrained optimization. CP-SAT works best with shorter horizons (3-8 months) due to computational complexity."
            )
            price_steps = st.slider(
                "Price Discretization Steps", 
                3, 10, 4,
                help="Number of discrete price points to consider at each time step. More steps give finer-grained optimization but increase complexity exponentially. 3-5 is recommended."
            )
            cpsat_max_price_change = st.slider(
                "CP-SAT Max Price Change (%)", 
                10, 70, 30,
                help="Maximum allowed price change between consecutive periods. Higher values (>30%) make finding solutions easier but may lead to erratic pricing."
            )
            
            # Business constraints
            st.subheader("Business Constraints")
            min_ltv_cac = st.slider(
                "Minimum LTV/CAC After Month 3", 
                1.0, 3.0, 1.5, 0.1,
                help="Minimum required LTV/CAC ratio after the 3rd month. SaaS benchmark: >3 is healthy, 1.5-3 is marginal, <1.5 is unsustainable. Lower values make solutions easier to find."
            )
            max_churn = st.slider(
                "Maximum Churn Rate (%)", 
                5.0, 25.0, 15.0, 0.5,
                help="Maximum allowed monthly churn rate. SaaS benchmarks: <5% is excellent, 5-10% is good, 10-15% is concerning, >15% is problematic. Higher values make solutions easier to find."
            )
            enable_relaxed_constraints = st.checkbox(
                "Enable Automatic Fallback", 
                True, 
                help="If enabled, the system will automatically try simpler models when the full optimization fails to find a solution."
            )
            
            with st.expander("Advanced Settings"):
                st.markdown("""
                **CP-SAT Optimization Tips:**
                - Start with a small time horizon (3-6 months) and few price steps (3-5)
                - Increase max price change (>30%) to give solver more flexibility
                - If you get no solution, try relaxing the business constraints
                - The solver has a built-in fallback mechanism for very difficult problems
                """)
            
            # Run CP-SAT button
            if st.button("Run CP-SAT Optimization"):
                with st.spinner("Running constrained optimization..."):
                    progress_bar = st.progress(0)
                    
                    # Make sure params has the correct time_horizon_months for CP-SAT
                    params.time_horizon_months = cpsat_time_horizon
                    
                    # Simulate progress for precomputation phase
                    for i in range(50):
                        progress_bar.progress(i)
                        time.sleep(0.01)
                    
                    st.text("Precomputing metrics for all price combinations...")
                    
                    for i in range(50, 80):
                        progress_bar.progress(i)
                        time.sleep(0.01)
                    
                    st.text("Solving optimization model...")
                    
                    for i in range(80, 101):
                        progress_bar.progress(i)
                        time.sleep(0.01)
                    
                    try:
                        # Run the actual CP-SAT optimization if available
                        if ORTOOLS_AVAILABLE:
                            st.code("Running CP-SAT with settings:\n" + 
                                  f"Time Horizon: {cpsat_time_horizon} months\n" +
                                  f"Price Steps: {price_steps}\n" +
                                  f"Max Price Change: {cpsat_max_price_change}%\n" +
                                  f"Min LTV/CAC: {min_ltv_cac}\n" +
                                  f"Max Churn: {max_churn}%", language="text")
                            
                            # If no solution is found with the original parameters, try with relaxed constraints
                            cpsat_results = None
                            
                            # Try with original constraints first
                            cpsat_results = optimize_with_cpsat(
                                params,
                                objective='npv',
                                time_horizon=cpsat_time_horizon,
                                price_steps=price_steps,
                                max_price_change_pct=cpsat_max_price_change
                            )
                            
                            # Check if we got a fallback solution
                            if cpsat_results and cpsat_results.get('is_fallback', False):
                                st.warning("Primary optimization failed. Using fallback solution instead.")
                                st.info("Fallback solution uses a simplified model with fewer time periods and more relaxed constraints.")
                            
                            if cpsat_results:
                                # Display results
                                st.subheader("CP-SAT Optimization Results")
                                
                                # Store in session state for comparison tab
                                st.session_state.cpsat_results = cpsat_results
                                
                                # Display results
                                st.metric("NPV", f"${cpsat_results['npv']:,.2f}")
                                
                                # Display price evolution
                                st.subheader("Price Evolution")
                                price_data = pd.DataFrame({
                                    'Month': list(range(cpsat_time_horizon)),
                                    'Enterprise Price': cpsat_results['enterprise_price'],
                                    'Pro Price': cpsat_results['pro_price']
                                })
                                
                                st.line_chart(price_data.set_index('Month'))
                                
                                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                                with metrics_col1:
                                    st.metric("Final MRR", f"${cpsat_results['final_mrr']:,.2f}")
                                with metrics_col2:
                                    st.metric("LTV/CAC Ratio", f"{cpsat_results['final_ltv_cac']:.2f}")
                                with metrics_col3:
                                    st.metric("Gross Margin", f"{cpsat_results['final_gross_margin']*100:.1f}%")
                                
                                # Plot
                                fig = visualize_pricing_strategies(cpsat_results)
                                st.pyplot(fig)
                                
                                # Show the actual price schedules
                                with st.expander("View Price Schedule Details"):
                                    st.write("### Enterprise Pricing Schedule")
                                    st.write(pd.DataFrame({
                                        'Month': list(range(len(cpsat_results['enterprise_price']))),
                                        'Price ($)': [f"${p:.2f}" for p in cpsat_results['enterprise_price']]
                                    }))
                                    
                                    st.write("### Pro Pricing Schedule")
                                    st.write(pd.DataFrame({
                                        'Month': list(range(len(cpsat_results['pro_price']))),
                                        'Price ($)': [f"${p:.2f}" for p in cpsat_results['pro_price']]
                                    }))
                            else:
                                st.error("CP-SAT optimization did not return results.")
                                st.warning("""
                                Try the following:
                                1. Reduce the time horizon (3-4 months)
                                2. Reduce price steps (3-4)
                                3. Increase max price change (>50%)
                                4. Relax business constraints (lower LTV/CAC, higher max churn)
                                5. Use the Heuristic Search instead, which is more flexible
                                """)
                                
                                # Show comparison of methods
                                st.info("""
                                **Recommendation:** The CP-SAT solver works best for simple problems with few variables.
                                For longer time horizons, try the Heuristic Search in the Advanced Time-Varying Pricing tab.
                                """)
                        else:
                            # Use a mock implementation for demo purposes
                            mock_enterprise_prices = [250 + i*5 for i in range(cpsat_time_horizon)]
                            mock_pro_prices = [50 + i*2 for i in range(cpsat_time_horizon)]
                            
                            # Create a mock model with the correct time horizon
                            mock_model = PricingSystemModel(params, {
                                'enterprise': mock_enterprise_prices,
                                'pro': mock_pro_prices,
                                'consultant_fee': 0.7,
                                'trial_days': 14
                            })
                            mock_results = mock_model.run_simulation()
                            
                            cpsat_results = {
                                'enterprise_price': mock_enterprise_prices,
                                'pro_price': mock_pro_prices,
                                'consultant_fee': 0.7,
                                'trial_days': 14,
                                'npv': mock_results['npv'],
                                'final_mrr': mock_results['final_mrr'],
                                'final_ltv_cac': mock_results['final_ltv_cac'],
                                'final_gross_margin': mock_results['final_gross_margin'],
                                'time_varying': True,
                                'model': mock_model
                            }
                            
                            # Display mock results
                            st.subheader("Mock CP-SAT Results (OR-Tools not available)")
                            
                            # Store in session state for comparison tab
                            st.session_state.cpsat_results = cpsat_results
                            
                            # Display results
                            st.metric("NPV", f"${cpsat_results['npv']:,.2f}")
                            
                            # Display price evolution
                            st.subheader("Price Evolution")
                            price_data = pd.DataFrame({
                                'Month': list(range(cpsat_time_horizon)),
                                'Enterprise Price': cpsat_results['enterprise_price'],
                                'Pro Price': cpsat_results['pro_price']
                            })
                            
                            st.line_chart(price_data.set_index('Month'))
                            
                            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                            with metrics_col1:
                                st.metric("Final MRR", f"${cpsat_results['final_mrr']:,.2f}")
                            with metrics_col2:
                                st.metric("LTV/CAC Ratio", f"{cpsat_results['final_ltv_cac']:.2f}")
                            with metrics_col3:
                                st.metric("Gross Margin", f"{cpsat_results['final_gross_margin']*100:.1f}%")
                            
                            # Plot
                            fig = visualize_pricing_strategies(cpsat_results)
                            st.pyplot(fig)
                    
                    except Exception as e:
                        st.error(f"Error running CP-SAT optimization: {str(e)}")
                        st.warning("""
                        Try the following:
                        1. Reduce the time horizon (3-4 months)
                        2. Reduce price steps (3-4)
                        3. Increase max price change (>50%)
                        4. Relax business constraints (lower LTV/CAC, higher max churn)
                        5. Use the Heuristic Search instead, which is more flexible
                        """)

# Network Effects Tab
with tab_network:
    st.header("Network Effects & Elasticity Analysis")
    
    st.markdown("""
    This analysis examines how network effects and price sensitivity impact your business model. It helps you understand:
    
    1. **Price Elasticity** - How changes in price affect user acquisition and retention
    2. **Network Effects** - How the value of your platform increases as more users join
    3. **Revenue-Maximizing Prices** - The optimal price points for different objectives
    """)
    
    if st.button(
        "Analyze Network Effects and Price Elasticity",
        help="Run a comprehensive analysis of how network effects and price elasticity impact your business. This may take a few moments to calculate."
    ):
        with st.spinner("Running elasticity analysis..."):
            progress_bar = st.progress(0)
            
            # Simulate progress
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.05)
            
            # Run analysis
            elasticity_analysis = analyze_network_effects_and_elasticity(params)
            
            # Store in session state for comparison tab
            st.session_state.elasticity_analysis = elasticity_analysis
            
            # Display results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Price Elasticity")
                st.info("""
                **Price Elasticity Interpretation**:
                Values closer to 0 mean less price sensitivity. 
                For example, -0.5 means a 10% price increase reduces demand by only 5%.
                Values below -1.0 are considered elastic (price-sensitive).
                """)
                st.metric(
                    "Enterprise Price Elasticity", 
                    f"{elasticity_analysis['enterprise_elasticity']:.2f}",
                    help="How enterprise customer acquisition responds to price changes. Values -0.5 to -0.8 are typical for enterprise SaaS."
                )
                st.metric(
                    "Pro Price Elasticity", 
                    f"{elasticity_analysis['pro_elasticity']:.2f}",
                    help="How pro user acquisition responds to price changes. Values -0.8 to -1.5 are typical for professional software."
                )
                
                st.subheader("Revenue-Maximizing Prices")
                st.metric(
                    "Enterprise Price", 
                    f"${elasticity_analysis['revenue_maximizing_enterprise_price']:.2f}",
                    help="The enterprise price that maximizes monthly recurring revenue. May sacrifice growth for short-term revenue."
                )
                st.metric(
                    "Pro Price", 
                    f"${elasticity_analysis['revenue_maximizing_pro_price']:.2f}",
                    help="The pro user price that maximizes monthly recurring revenue. May sacrifice growth for short-term revenue."
                )
                
                st.subheader("NPV-Maximizing Prices")
                st.metric(
                    "Enterprise Price", 
                    f"${elasticity_analysis['npv_maximizing_enterprise_price']:.2f}",
                    help="The enterprise price that maximizes net present value. Balances short-term revenue with long-term growth."
                )
                st.metric(
                    "Pro Price", 
                    f"${elasticity_analysis['npv_maximizing_pro_price']:.2f}",
                    help="The pro user price that maximizes net present value. Balances short-term revenue with long-term growth."
                )
            
            with col2:
                st.subheader("Network Effect Impact")
                
                # Create dataframe from network effect results
                network_data = []
                for result in elasticity_analysis['network_effect_results']:
                    network_data.append({
                        'Network Multiplier': result['network_multiplier'],
                        'Enterprise Users': result['enterprise_users'],
                        'Pro Users': result['pro_users'],
                        'NPV': result['npv'],
                        'MRR': result['mrr']
                    })
                
                network_df = pd.DataFrame(network_data)
                
                # Calculate impact
                no_network = network_df[network_df['Network Multiplier'] == 0].iloc[0]
                full_network = network_df[network_df['Network Multiplier'] == 1.0].iloc[0]
                
                st.metric(
                    "NPV Increase from Network Effects", 
                    f"${full_network['NPV'] - no_network['NPV']:,.2f}", 
                    f"{((full_network['NPV'] / no_network['NPV']) - 1) * 100:.1f}%",
                    help="The total value created by network effects (difference between no network effects and standard network effects)."
                )
                
                st.metric(
                    "User Increase from Network Effects", 
                    f"{full_network['Enterprise Users'] + full_network['Pro Users'] - no_network['Enterprise Users'] - no_network['Pro Users']:.0f}", 
                    f"{((full_network['Enterprise Users'] + full_network['Pro Users']) / (no_network['Enterprise Users'] + no_network['Pro Users']) - 1) * 100:.1f}%",
                    help="Additional users acquired due to network effects."
                )
                
                # Plot network effect chart
                st.subheader("Network Effect Strength vs NPV")
                st.info("This chart shows how increasing network effects impact total business value (NPV). Steeper curves indicate stronger network effects.")
                st.line_chart(network_df.set_index('Network Multiplier')['NPV'])
            
            # Display the elasticity visualization
            st.subheader("Elasticity and Network Effect Visualizations")
            st.pyplot(elasticity_analysis['visualizations'])

# Wellspring Marketplace Analysis Tab
with tab_wellspring:
    st.header("Wellspring Marketplace Analysis")
    
    st.markdown("""
    This tab focuses on analyzing the Wellspring marketplace dynamics, including:
    
    - **Marketplace Liquidity** - Balance of buyers and sellers
    - **Credits Economics** - Pricing and consumption patterns
    - **Dual-Role Participants** - Users who both buy and sell advice
    - **Revenue Optimization** - Finding the optimal transaction fee structure
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Marketplace Parameters")
        
        # Use existing parameters for consistency
        wellspring_tx_fee = st.slider(
            "Transaction Fee (%)", 
            1, 30, int(params.wellspring_tx_fee * 100), 1,
            help="Percentage fee charged on marketplace transactions"
        ) / 100
        
        credit_price = st.slider(
            "Credit Price ($)", 
            1, 50, int(params.credits_price), 1,
            help="Price per credit for advice marketplace transactions"
        )
        
        # Calculate implied metrics
        implied_revenue_per_credit = credit_price * wellspring_tx_fee
        st.info(f"Revenue per credit: ${implied_revenue_per_credit:.2f}")
        
        # Credit package options
        st.subheader("Credit Package Options")
        
        credit_packages = {
            "Starter": {"credits": 10, "price": credit_price * 10},
            "Pro": {"credits": 50, "price": credit_price * 50 * 0.9},  # 10% discount
            "Business": {"credits": 200, "price": credit_price * 200 * 0.8},  # 20% discount
        }
        
        # Display as a table
        package_data = []
        for name, details in credit_packages.items():
            package_data.append({
                "Package": name,
                "Credits": details["credits"],
                "Price": f"${details['price']:.2f}",
                "Per Credit": f"${details['price']/details['credits']:.2f}",
                "Savings": f"{(1 - (details['price']/(details['credits']*credit_price))) * 100:.0f}%"
            })
        
        package_df = pd.DataFrame(package_data)
        st.table(package_df)
        
    with col2:
        st.subheader("Marketplace Dynamics")
        
        if st.button("Analyze Marketplace Dynamics"):
            with st.spinner("Analyzing marketplace dynamics..."):
                # Create a simple model for analysis
                base_model = PricingSystemModel(params, {
                    'enterprise': 299,
                    'pro': 99,
                    'lifetime': 995,
                    'consultant_fee': 0.7,
                    'trial_days': 14
                })
                base_results = base_model.run_simulation()
                
                # Calculate metrics specific to the marketplace using global function
                effective_sellers = effective_consultants(base_model, params)
                potential_buyers = base_model.enterprise_users * 5 + base_model.pro_users
                
                # Calculate liquidity metrics
                buyer_seller_ratio = potential_buyers / max(1, effective_sellers)
                liquidity_score = min(10, 10 * (effective_sellers / 100)) * min(10, 10 * (potential_buyers / 1000)) / 10
                
                # Display metrics
                st.metric("Effective Sellers", f"{effective_sellers:.0f}")
                st.metric("Potential Buyers", f"{potential_buyers:.0f}")
                st.metric("Buyer/Seller Ratio", f"{buyer_seller_ratio:.1f}", 
                         help="Ratio of potential buyers to sellers. Optimal range is typically 5-20.")
                
                st.metric("Marketplace Liquidity Score", f"{liquidity_score:.1f}/10",
                         help="Combined score based on marketplace size and balance. Higher is better.")
                
                # Revenue simulation with different fee structures
                st.subheader("Fee Structure Analysis")
                
                fee_range = np.arange(0.1, 0.41, 0.05)  # Range of consultant fee shares (10%-40% platform fee)
                fee_results = []
                
                for fee in fee_range:
                    # Simulate impact of different platform fees on volume
                    # Note: fee here represents platform fee, not consultant fee
                    platform_fee = fee  # This is what platform keeps
                    consultant_fee = 1 - platform_fee  # This is what consultant gets
                    
                    # Adjust for fee elasticity (higher platform fees may reduce volume)
                    fee_elasticity = -0.4  # Assume modest elasticity
                    
                    # Calculate relative change compared to baseline
                    base_platform_fee = 1 - params.consultant_fee_percentage
                    volume_factor = (platform_fee / base_platform_fee) ** fee_elasticity
                    
                    # Create a temporary params copy with the test fee value
                    temp_params = copy.deepcopy(params)
                    temp_params.consultant_fee_percentage = consultant_fee
                    
                    # Calculate revenue with this fee structure
                    adjusted_revenue = calculate_wellspring_revenue(base_model, temp_params) * volume_factor
                    
                    fee_results.append({
                        "Platform Fee": f"{platform_fee*100:.0f}%",
                        "Consultant Fee": f"{consultant_fee*100:.0f}%",
                        "Revenue": adjusted_revenue,
                        "Volume": base_model.consultants * params.avg_monthly_credits * volume_factor
                    })
                
                # Convert to DataFrame and display
                fee_df = pd.DataFrame(fee_results)
                
                # Find optimal fee
                optimal_row = fee_df.loc[fee_df["Revenue"].idxmax()]
                
                st.success(f"Optimal platform fee: {optimal_row['Platform Fee']} (consultant gets {optimal_row['Consultant Fee']}) - maximizes revenue at ${optimal_row['Revenue']:.2f}/mo")
                
                # Show as chart
                st.line_chart(fee_df.set_index("Platform Fee")["Revenue"])
                
                # Show credit consumption patterns
                st.subheader("Credit Consumption Patterns")
                
                # Create sample distribution of credit usage
                st.bar_chart({
                    "Light Users (1-5 credits)": 50,
                    "Medium Users (6-15 credits)": 30,
                    "Heavy Users (16-30 credits)": 15,
                    "Power Users (31+ credits)": 5
                })

# Comparison Tab
with tab_compare:
    st.header("Strategy Comparison")
    
    # Add advice for comparing different strategies
    with st.expander("How to Compare Pricing Strategies", expanded=True):
        st.markdown("""
        ### Comparing Different Pricing Approaches
        
        When comparing results from different optimization strategies, consider these key aspects:
        
        **1. Short-term vs. Long-term Results**
        - Static pricing optimizes for current conditions
        - Time-varying pricing adapts as your market evolves
        - NPV-optimized strategies typically show lower initial MRR but stronger long-term performance
        
        **2. Growth vs. Profitability**
        - MRR-optimized strategies prioritize user growth but may have weaker unit economics
        - Margin-optimized strategies show higher profitability but slower adoption
        - LTV/CAC-optimized strategies balance retention and acquisition efficiency
        
        **3. Risk Assessment**
        - More aggressive strategies (lower prices, higher growth) typically carry higher risk
        - More conservative strategies (higher prices, stronger margins) provide safety but may limit upside
        - Time-varying strategies can adapt to market conditions, potentially reducing risk
        
        **4. Implementation Considerations**
        - Static pricing is simpler to implement and communicate
        - Time-varying pricing requires more explanation to customers and careful timing
        - Consider pricing communication, grandfathering policies, and competitive responses
        
        **5. Sensitivity Analysis**
        - The most robust strategies perform well across multiple scenarios
        - Test how sensitive your optimal pricing is to changes in elasticity, churn, and network effects
        - Prefer strategies that don't rely on perfect execution of every assumption
        """)
    
    st.markdown("""
    To compare different pricing strategies, run the simulations in the other tabs first.
    Then come back to this tab to see a side-by-side comparison.
    
    You can compare:
    
    1. Initial pricing 
    2. Static optimized pricing
    3. Time-varying pricing
    4. Constrained CP-SAT pricing
    """)
    
    # Initialize session state to store results from different tabs if not already done
    if 'initial_results' not in st.session_state:
        st.session_state.initial_results = None
    if 'optimized_results' not in st.session_state:
        st.session_state.optimized_results = None
    if 'heuristic_results' not in st.session_state:
        st.session_state.heuristic_results = None
    if 'cpsat_results' not in st.session_state:
        st.session_state.cpsat_results = None
    
    # Check which models have been run
    available_models = []
    if st.session_state.initial_results is not None:
        available_models.append("Initial Pricing")
    if st.session_state.optimized_results is not None:
        available_models.append("Optimized Static")
    if st.session_state.heuristic_results is not None:
        available_models.append("Time-Varying")
    if st.session_state.cpsat_results is not None:
        available_models.append("CP-SAT Constrained")
    
    if not available_models:
        st.warning("No pricing models have been run yet. Please run simulations in the other tabs first.")
    else:
        st.success(f"Available models for comparison: {', '.join(available_models)}")
        
        # Create comparison chart with actual data
        st.subheader("MRR Comparison")
        
        # Get the maximum time horizon from available models
        max_months = 36  # Default
        actual_months = []
        
        comparison_data = {'Month': list(range(max_months))}
        
        if st.session_state.initial_results is not None:
            initial_history = st.session_state.initial_results['model'].history
            initial_mrr = initial_history['mrr']
            actual_months.append(len(initial_mrr))
            # Pad with last value if needed
            initial_mrr_list = initial_mrr.tolist() if isinstance(initial_mrr, np.ndarray) else list(initial_mrr)
            padded_initial_mrr = initial_mrr_list + [initial_mrr_list[-1]] * (max_months - len(initial_mrr_list)) if initial_mrr_list else [0] * max_months
            comparison_data['Initial Pricing'] = padded_initial_mrr[:max_months]
        
        if st.session_state.optimized_results is not None:
            optimized_history = st.session_state.optimized_results['model'].history
            optimized_mrr = optimized_history['mrr']
            actual_months.append(len(optimized_mrr))
            # Pad with last value if needed
            optimized_mrr_list = optimized_mrr.tolist() if isinstance(optimized_mrr, np.ndarray) else list(optimized_mrr)
            padded_optimized_mrr = optimized_mrr_list + [optimized_mrr_list[-1]] * (max_months - len(optimized_mrr_list)) if optimized_mrr_list else [0] * max_months
            comparison_data['Optimized Static'] = padded_optimized_mrr[:max_months]
        
        if st.session_state.heuristic_results is not None:
            heuristic_history = st.session_state.heuristic_results['model'].history
            heuristic_mrr = heuristic_history['mrr']
            actual_months.append(len(heuristic_mrr))
            # Pad with last value if needed
            heuristic_mrr_list = heuristic_mrr.tolist() if isinstance(heuristic_mrr, np.ndarray) else list(heuristic_mrr)
            padded_heuristic_mrr = heuristic_mrr_list + [heuristic_mrr_list[-1]] * (max_months - len(heuristic_mrr_list)) if heuristic_mrr_list else [0] * max_months
            comparison_data['Time-Varying'] = padded_heuristic_mrr[:max_months]
        
        if st.session_state.cpsat_results is not None:
            cpsat_history = st.session_state.cpsat_results['model'].history
            cpsat_mrr = cpsat_history['mrr']
            actual_months.append(len(cpsat_mrr))
            # Pad with last value if needed
            cpsat_mrr_list = cpsat_mrr.tolist() if isinstance(cpsat_mrr, np.ndarray) else list(cpsat_mrr)
            padded_cpsat_mrr = cpsat_mrr_list + [cpsat_mrr_list[-1]] * (max_months - len(cpsat_mrr_list)) if cpsat_mrr_list else [0] * max_months
            comparison_data['CP-SAT Constrained'] = padded_cpsat_mrr[:max_months]
        
        # Use the actual max months from the data
        if actual_months:
            actual_max_months = min(max(actual_months), max_months)
            # Trim the data to the actual max months
            comparison_data = {k: v[:actual_max_months] for k, v in comparison_data.items()}
            comparison_data['Month'] = list(range(actual_max_months))
        
        # Create a dataframe for charting
        comparison_df = pd.DataFrame(comparison_data)
        
        # Plot the chart
        st.line_chart(comparison_df.set_index('Month'))
        
        # Create comparison table with actual data
        st.subheader("Pricing Strategy Comparison")
        
        # Build comparison table data
        table_data = {
            'Strategy': [],
            'Enterprise Start': [],
            'Enterprise End': [],
            'Pro Start': [],
            'Pro End': [],
            'NPV': [],
            'Final MRR': [],
            'LTV/CAC': []
        }
        
        if st.session_state.initial_results is not None:
            initial = st.session_state.initial_results
            table_data['Strategy'].append("Initial")
            
            # Check if enterprise_price is a list or scalar
            if isinstance(initial['enterprise_price'], (list, np.ndarray)):
                table_data['Enterprise Start'].append(f"${initial['enterprise_price'][0]:.2f}")
                table_data['Enterprise End'].append(f"${initial['enterprise_price'][-1]:.2f}")
            else:
                table_data['Enterprise Start'].append(f"${initial['enterprise_price']:.2f}")
                table_data['Enterprise End'].append(f"${initial['enterprise_price']:.2f}")
                
            # Check if pro_price is a list or scalar
            if isinstance(initial['pro_price'], (list, np.ndarray)):
                table_data['Pro Start'].append(f"${initial['pro_price'][0]:.2f}")
                table_data['Pro End'].append(f"${initial['pro_price'][-1]:.2f}")
            else:
                table_data['Pro Start'].append(f"${initial['pro_price']:.2f}")
                table_data['Pro End'].append(f"${initial['pro_price']:.2f}")
                
            table_data['NPV'].append(f"${initial['npv']:,.0f}")
            table_data['Final MRR'].append(f"${initial['final_mrr']:,.0f}")
            table_data['LTV/CAC'].append(f"{initial['final_ltv_cac']:.1f}")
        
        if st.session_state.optimized_results is not None:
            optimized = st.session_state.optimized_results
            table_data['Strategy'].append("Static Optimized")
            
            # Check if enterprise_price is a list or scalar
            if isinstance(optimized['enterprise_price'], (list, np.ndarray)):
                table_data['Enterprise Start'].append(f"${optimized['enterprise_price'][0]:.2f}")
                table_data['Enterprise End'].append(f"${optimized['enterprise_price'][-1]:.2f}")
            else:
                table_data['Enterprise Start'].append(f"${optimized['enterprise_price']:.2f}")
                table_data['Enterprise End'].append(f"${optimized['enterprise_price']:.2f}")
                
            # Check if pro_price is a list or scalar
            if isinstance(optimized['pro_price'], (list, np.ndarray)):
                table_data['Pro Start'].append(f"${optimized['pro_price'][0]:.2f}")
                table_data['Pro End'].append(f"${optimized['pro_price'][-1]:.2f}")
            else:
                table_data['Pro Start'].append(f"${optimized['pro_price']:.2f}")
                table_data['Pro End'].append(f"${optimized['pro_price']:.2f}")
                
            table_data['NPV'].append(f"${optimized['npv']:,.0f}")
            table_data['Final MRR'].append(f"${optimized['final_mrr']:,.0f}")
            table_data['LTV/CAC'].append(f"{optimized['final_ltv_cac']:.1f}")
        
        if st.session_state.heuristic_results is not None:
            heuristic = st.session_state.heuristic_results
            table_data['Strategy'].append("Time-Varying")
            
            # Time-varying prices are always arrays
            table_data['Enterprise Start'].append(f"${heuristic['enterprise_price'][0]:.2f}")
            table_data['Enterprise End'].append(f"${heuristic['enterprise_price'][-1]:.2f}")
            table_data['Pro Start'].append(f"${heuristic['pro_price'][0]:.2f}")
            table_data['Pro End'].append(f"${heuristic['pro_price'][-1]:.2f}")
            
            table_data['NPV'].append(f"${heuristic['npv']:,.0f}")
            table_data['Final MRR'].append(f"${heuristic['final_mrr']:,.0f}")
            table_data['LTV/CAC'].append(f"{heuristic['final_ltv_cac']:.1f}")
        
        if st.session_state.cpsat_results is not None:
            cpsat = st.session_state.cpsat_results
            table_data['Strategy'].append("CP-SAT")
            
            # CP-SAT prices are always arrays
            table_data['Enterprise Start'].append(f"${cpsat['enterprise_price'][0]:.2f}")
            table_data['Enterprise End'].append(f"${cpsat['enterprise_price'][-1]:.2f}")
            table_data['Pro Start'].append(f"${cpsat['pro_price'][0]:.2f}")
            table_data['Pro End'].append(f"${cpsat['pro_price'][-1]:.2f}")
            
            table_data['NPV'].append(f"${cpsat['npv']:,.0f}")
            table_data['Final MRR'].append(f"${cpsat['final_mrr']:,.0f}")
            table_data['LTV/CAC'].append(f"{cpsat['final_ltv_cac']:.1f}")
        
        # Create the comparison table
        comparison_table = pd.DataFrame(table_data)
        st.table(comparison_table)
        
        # Additional visualization: Price evolution comparison
        if any(model is not None for model in [st.session_state.initial_results, st.session_state.optimized_results, 
                                               st.session_state.heuristic_results, st.session_state.cpsat_results]):
            st.subheader("Enterprise Price Evolution")
            
            price_data = {'Month': list(range(max_months))}
            
            # Add enterprise price data for each available model
            for model_name, model_results in [
                ("Initial", st.session_state.initial_results),
                ("Static Optimized", st.session_state.optimized_results),
                ("Time-Varying", st.session_state.heuristic_results),
                ("CP-SAT", st.session_state.cpsat_results)
            ]:
                if model_results is not None:
                    if isinstance(model_results['enterprise_price'], (list, np.ndarray)):
                        enterprise_prices = model_results['enterprise_price']
                        # Pad with last value if needed
                        enterprise_prices_list = enterprise_prices.tolist() if isinstance(enterprise_prices, np.ndarray) else list(enterprise_prices)
                        padded_prices = enterprise_prices_list + [enterprise_prices_list[-1]] * (max_months - len(enterprise_prices_list))
                        price_data[model_name] = padded_prices[:max_months]
                    else:
                        # Constant price
                        price_data[model_name] = [model_results['enterprise_price']] * max_months
            
            # Trim to actual max months
            if actual_months:
                actual_max_months = min(max(actual_months), max_months)
                price_data = {k: v[:actual_max_months] for k, v in price_data.items()}
                price_data['Month'] = list(range(actual_max_months))
            
            # Create price evolution chart
            price_df = pd.DataFrame(price_data)
            st.line_chart(price_df.set_index('Month'))
        
        # Show insights based on actual data instead of hardcoded insights
        st.subheader("Key Pricing Strategy Insights")
        
        # Generate insights based on actual data
        insights = []
        
        # Check if we have data to compare time-varying vs static
        if st.session_state.heuristic_results is not None and st.session_state.optimized_results is not None:
            heuristic_npv = st.session_state.heuristic_results['npv']
            optimized_npv = st.session_state.optimized_results['npv']
            
            if heuristic_npv > optimized_npv:
                insights.append(f"**Time-varying pricing is superior** - The time-varying pricing strategy outperforms static pricing by {((heuristic_npv/optimized_npv)-1)*100:.1f}% in NPV, showing the value of adapting prices as your market evolves.")
            else:
                insights.append("**Simpler pricing may be sufficient** - Static pricing performs well in this scenario, suggesting market conditions may not require a complex time-varying approach.")
        
        # Check network effects data if available
        if 'elasticity_analysis' in st.session_state and st.session_state.elasticity_analysis is not None:
            network_df = pd.DataFrame([{
                'multiplier': r['network_multiplier'],
                'npv': r['npv']
            } for r in st.session_state.elasticity_analysis['network_effect_results']])
            
            if len(network_df) >= 2:
                no_network = network_df[network_df['multiplier'] == 0].iloc[0]['npv'] if not network_df[network_df['multiplier'] == 0].empty else 0
                full_network = network_df[network_df['multiplier'] == 1.0].iloc[0]['npv'] if not network_df[network_df['multiplier'] == 1.0].empty else 0
                
                if full_network > no_network:
                    insights.append(f"**Network effects are critical** - Network effects increase NPV by {((full_network/no_network)-1)*100:.1f}%, highlighting their importance in user acquisition and retention.")
        
        # Check price elasticity data
        if 'elasticity_analysis' in st.session_state and st.session_state.elasticity_analysis is not None:
            enterprise_elasticity = st.session_state.elasticity_analysis.get('enterprise_elasticity', 0)
            pro_elasticity = st.session_state.elasticity_analysis.get('pro_elasticity', 0)
            
            if abs(enterprise_elasticity) < abs(pro_elasticity):
                insights.append(f"**Price sensitivity differs by segment** - Enterprise customers (elasticity: {enterprise_elasticity:.2f}) are less price sensitive than Pro users (elasticity: {pro_elasticity:.2f}), suggesting different pricing approaches for each segment.")
        
        # Add insight about price evolution if time-varying strategy exists
        if st.session_state.heuristic_results is not None and isinstance(st.session_state.heuristic_results['enterprise_price'], (list, np.ndarray)):
            start_price = st.session_state.heuristic_results['enterprise_price'][0]
            end_price = st.session_state.heuristic_results['enterprise_price'][-1]
            
            if end_price > start_price:
                insights.append(f"**Entry pricing vs. mature pricing** - Optimal strategy involves starting with lower prices (${start_price:.2f}) to build the network, then gradually increasing (to ${end_price:.2f}) as value perception improves.")
        
        # Add insight about revenue diversification
        if any(model is not None for model in [st.session_state.initial_results, st.session_state.optimized_results, 
                                               st.session_state.heuristic_results, st.session_state.cpsat_results]):
            # Get the model with highest NPV
            best_model = None
            best_npv = 0
            for model in [st.session_state.initial_results, st.session_state.optimized_results, 
                          st.session_state.heuristic_results, st.session_state.cpsat_results]:
                if model is not None and model['npv'] > best_npv:
                    best_model = model
                    best_npv = model['npv']
            
            if best_model is not None:
                subscription_rev = best_model['final_mrr'] - best_model['model'].contract_revenue - best_model['model'].wellspring_revenue
                total_rev = best_model['final_mrr']
                
                if total_rev > 0:
                    sub_pct = subscription_rev / total_rev * 100
                    other_pct = 100 - sub_pct
                    
                    if other_pct > 15:
                        insights.append(f"**Revenue diversification matters** - At maturity, subscription revenue ({sub_pct:.1f}%) should be complemented by marketplace and contract revenue streams ({other_pct:.1f}%).")
        
        # If we don't have enough insights, add some generic ones
        if len(insights) < 3:
            insights.append("**Test different network effect strengths** - Adjusting the network effect strength multiplier can help identify the optimal investment in community and network-building features.")
            insights.append("**Balance short-term and long-term objectives** - The optimal strategy balances immediate revenue needs with long-term value creation through network effects.")
        
        # Display insights
        for insight in insights:
            st.markdown(insight)
        
        # Generate implementation roadmap based on actual results
        st.subheader("Recommended Implementation Roadmap")
        
        # Find the best performing model
        best_model = None
        best_npv = 0
        for model_name, model in [
            ("Initial", st.session_state.initial_results),
            ("Static", st.session_state.optimized_results),
            ("Time-Varying", st.session_state.heuristic_results),
            ("CP-SAT", st.session_state.cpsat_results)
        ]:
            if model is not None and model['npv'] > best_npv:
                best_model = model
                best_model_name = model_name
                best_npv = model['npv']
        
        # Generate roadmap based on best model
        if best_model is not None and best_model_name in ["Time-Varying", "CP-SAT"]:
            # Use time-varying prices from the best model
            enterprise_prices = best_model['enterprise_price']
            pro_prices = best_model['pro_price']
            
            # Split into three phases
            phases = min(len(enterprise_prices), 36) // 3
            
            launch_ent = enterprise_prices[0]
            launch_pro = pro_prices[0]
            
            growth_ent = enterprise_prices[phases] if phases < len(enterprise_prices) else enterprise_prices[-1]
            growth_pro = pro_prices[phases] if phases < len(pro_prices) else pro_prices[-1]
            
            mature_ent = enterprise_prices[-1]
            mature_pro = pro_prices[-1]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                ### Launch Phase (Months 0-{phases})
                - Enterprise Price: ${launch_ent:.2f} monthly
                - Pro Price: ${launch_pro:.2f} monthly
                - Focus on grower onboarding and content generation
                - Offer trial periods of 14-21 days
                """)
            
            with col2:
                st.markdown(f"""
                ### Growth Phase (Months {phases+1}-{2*phases})
                - Enterprise Price: ${growth_ent:.2f} monthly
                - Pro Price: ${growth_pro:.2f} monthly
                - Introduce lifetime deals for early adopters
                - Launch Wellspring marketplace with high consultant commission (75%)
                """)
            
            with col3:
                st.markdown(f"""
                ### Mature Phase (Months {2*phases+1}+)
                - Enterprise Price: ${mature_ent:.2f} monthly
                - Pro Price: ${mature_pro:.2f} monthly
                - Adjust commissions based on marketplace maturity
                - Implement tiered feature-based pricing
                - Optimize "free in public" thresholds based on content value
                """)
        else:
            # Use static pricing or default values if we don't have a good time-varying model
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                ### Launch Phase (Months 0-6)
                - Enterprise Price: $225.00 monthly
                - Pro Price: $40.00 monthly
                - Focus on grower onboarding and content generation
                - Offer trial periods of 14-21 days
                """)
            
            with col2:
                st.markdown("""
                ### Growth Phase (Months 7-18)
                - Enterprise Price: $275.00 monthly
                - Pro Price: $55.00 monthly
                - Introduce lifetime deals for early adopters
                - Launch Wellspring marketplace with high consultant commission (75%)
                """)
            
            with col3:
                st.markdown("""
                ### Mature Phase (Months 19+)
                - Enterprise Price: $375.00 monthly
                - Pro Price: $85.00 monthly
                - Adjust commissions based on marketplace maturity
                - Implement tiered feature-based pricing
                - Optimize "free in public" thresholds based on content value
                """)

# Financial Modeling Tab
with tab_financial:
    st.header("Startup Financial Modeling")
    
    st.markdown("""
    This tab helps you model your startup's financial trajectory based on pricing strategy results, 
    incorporating burn rate and runway calculations with your actual cost structure.
    """)
    
    # Seed funding section
    with st.expander("Seed Funding", expanded=True):
        seed_funding = st.number_input("Seed Funding Amount ($)", min_value=0, value=2200000, step=100000, 
                                       help="Current fundraising target or available capital")
        
        st.markdown(f"**Target Seed Round: ${seed_funding:,}**")
    
    # Cost structure
    with st.expander("Cost Structure", expanded=True):
        st.markdown("### Monthly Cost Structure")
        
        # Define default cost structure data
        default_costs_data = {
            "Expense": ["Director Salaries", "Rent", "Software", "Debt equity", "Engineer", 
                       "CMO", "Sales", "Freelance", "Legal", "Customer support","Accounting","travel"],
            "Monthly":[25000, 417, 667, 6250, 16667, 6667, 8333, 5000, 833, 2500, 417, 167],
            "Yearly": [300000, 5000, 8000, 75000, 200000, 80000, 100000, 60000, 10000, 30000, 5000, 2000],
            "Annual Growth":["5.00%", "3.00%", "10.00%", "0.00%", "8.00%", "12.00%", "15.00%", "10.00%", "5.00%", "10.00%", "3.00%", "20.00%"],
            "Contingency": ["10%", "5%", "10%", "0%", "10%", "15%", "15%", "10%", "10%", "10%", "5%", "15%"]
        }
        
        # Create a session state key for the costs data if it doesn't exist
        if 'costs_data' not in st.session_state:
            st.session_state.costs_data = default_costs_data
            
            # Ensure numeric columns have proper types when initializing
            df_temp = pd.DataFrame(default_costs_data)
            df_temp["Monthly"] = pd.to_numeric(df_temp["Monthly"], errors="coerce").fillna(0).astype(int)
            df_temp["Yearly"] = pd.to_numeric(df_temp["Yearly"], errors="coerce").fillna(0).astype(int)
            
            # Update session state with proper types
            st.session_state.costs_data = {
                "Expense": df_temp["Expense"].tolist(),
                "Monthly": df_temp["Monthly"].tolist(),
                "Yearly": df_temp["Yearly"].tolist(),
                "Annual Growth": df_temp["Annual Growth"].tolist(),
                "Contingency": df_temp["Contingency"].tolist()
            }
        
        # Convert to DataFrame for editing
        costs_df = pd.DataFrame(st.session_state.costs_data)
        
        # Ensure all values in the Yearly column are numeric
        costs_df["Yearly"] = pd.to_numeric(costs_df["Yearly"], errors="coerce").fillna(0).astype(int)
        
        # Create an editable dataframe
        edited_costs_df = st.data_editor(
            costs_df,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "Monthly": st.column_config.NumberColumn(
                    "Monthly ($)",
                    min_value=0,
                    format="$%d"
                ),
                "Yearly": st.column_config.NumberColumn(
                    "Yearly ($)",
                    min_value=0,
                    format="$%d",
                    help="Annual salary or cost"
                ),
                "Annual Growth": st.column_config.TextColumn(
                    "Annual Growth",
                    help="Expected annual growth rate"
                ),
                "Contingency": st.column_config.TextColumn(
                    "Contingency",
                    help="Buffer percentage for unexpected costs"
                )
            },
            key="cost_structure_editor"
        )
        
        # Update the session state with the edited data
        st.session_state.costs_data = {
            "Expense": edited_costs_df["Expense"].tolist(),
            "Monthly": edited_costs_df["Monthly"].astype(int).tolist(),
            "Yearly": edited_costs_df["Yearly"].astype(int).tolist(),
            "Annual Growth": edited_costs_df["Annual Growth"].tolist(),
            "Contingency": edited_costs_df["Contingency"].tolist()
        }
        
        # Calculate totals based on the edited data
        monthly_total = sum(edited_costs_df["Monthly"])
        
        # Display totals
        st.metric("Total Monthly Burn (before revenue)", f"${monthly_total:,.2f}")
        st.metric("Total Annual Burn (before revenue)", f"${monthly_total*12:,.2f}")
        
        # Add a button to reset to default values
        if st.button("Reset Cost Structure to Default"):
            st.session_state.costs_data = default_costs_data
            st.rerun()
    
    # Financial projections based on selected pricing model
    st.subheader("Revenue & Runway Projections")
    
    st.markdown("""
    Select a pricing model to project revenue and calculate burn rate and runway.
    Run a simulation in one of the other tabs first to use those results here.
    """)
    
    # Select pricing model for financial calculations
    pricing_model = st.selectbox(
        "Select Pricing Model for Financial Projections",
        ["Initial Model", "Optimized Static Model", "Time-Varying Model", "CP-SAT Model", "None (no model selected)"],
        index=4,
        help="Select which pricing model to use for financial projections"
    )
    
    # Financial projection calculations
    if st.button("Calculate Financial Projections"):
        with st.spinner("Calculating financial projections..."):
            progress_bar = st.progress(0)
            
            # Simulate progress
            for i in range(50):
                progress_bar.progress(i + 1)
                time.sleep(0.02)
                
            selected_model = None
            
            # Get the selected model results
            if pricing_model == "Initial Model" and st.session_state.initial_results is not None:
                selected_model = st.session_state.initial_results
            elif pricing_model == "Optimized Static Model" and st.session_state.optimized_results is not None:
                selected_model = st.session_state.optimized_results
            elif pricing_model == "Time-Varying Model" and st.session_state.heuristic_results is not None:
                selected_model = st.session_state.heuristic_results
            elif pricing_model == "CP-SAT Model" and st.session_state.cpsat_results is not None:
                selected_model = st.session_state.cpsat_results
                
            if selected_model is None:
                st.warning(f"No data available for {pricing_model}. Please run that simulation first.")
                st.stop()
                
            # Use actual data from the model
            months = range(1, min(25, len(selected_model['model'].history['mrr']) + 1))
            monthly_revenue = selected_model['model'].history['mrr'][:24]
            
            # If we have fewer than 24 months of data, pad with the last value
            if len(monthly_revenue) < 24:
                monthly_revenue = monthly_revenue + [monthly_revenue[-1]] * (24 - len(monthly_revenue))
                
            # Update progress
            progress_bar.progress(75)
                
            # Calculate burn rate (expenses - revenue)
            burn_rate = [max(0, monthly_total - rev) for rev in monthly_revenue]
            
            # Calculate cumulative burn and remaining capital
            cumulative_burn = np.cumsum(burn_rate).tolist()
            remaining_capital = [seed_funding - burn for burn in cumulative_burn]
            
            # Calculate runway (months until money runs out)
            runway = 0
            for i, capital in enumerate(remaining_capital):
                if capital > 0:
                    runway = i + 1
                else:
                    break
                    
            if all(cap > 0 for cap in remaining_capital):
                runway = len(remaining_capital)  # If we don't run out within our projection
            
            # Create a dataframe for the financial projections
            financial_data = pd.DataFrame({
                'Month': list(months),
                'Revenue': monthly_revenue,
                'Expenses': [monthly_total] * len(months),
                'Burn Rate': burn_rate,
                'Remaining Capital': remaining_capital
            })
            
            # Complete progress
            progress_bar.progress(100)
            
            # Display financial metrics
            st.subheader(f"Financial Metrics Based on {pricing_model}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Burn Rate", f"${burn_rate[0]:,.2f}/mo")
            with col2:
                st.metric("Runway", f"{runway} months")
            with col3:
                break_even_month = next((i+1 for i, br in enumerate(burn_rate) if br <= 0), None)
                if break_even_month:
                    st.metric("Break-even Month", f"Month {break_even_month}")
                else:
                    st.metric("Break-even Month", "Not within projection")
            
            # Revenue breakdown
            st.subheader("Revenue Breakdown")
            rev_col1, rev_col2, rev_col3, rev_col4 = st.columns(4)
            with rev_col1:
                # Calculate additional user revenue if available
                additional_user_rev = getattr(selected_model['model'], 'additional_user_revenue', 0)
                subscription_rev = monthly_revenue[-1] - selected_model['model'].contract_revenue - selected_model['model'].wellspring_revenue - additional_user_rev
                st.metric("Subscription Revenue", f"${subscription_rev:,.2f}")
            with rev_col2:
                st.metric("Contract Revenue", f"${selected_model['model'].contract_revenue:,.2f}")
            with rev_col3:
                st.metric("Wellspring Revenue", f"${selected_model['model'].wellspring_revenue:,.2f}")
            with rev_col4:
                st.metric("Additional User Revenue", f"${additional_user_rev:,.2f}")
                
            # Display the financial projection table
            st.subheader("Monthly Financial Projections")
            st.dataframe(financial_data)
            
            # Create charts for financial projections
            st.subheader("Financial Projection Charts")
            
            # Revenue vs Expenses
            st.write("Revenue vs Expenses")
            rev_exp_data = pd.DataFrame({
                'Month': list(months),
                'Revenue': monthly_revenue,
                'Expenses': [monthly_total] * len(months)
            })
            st.line_chart(rev_exp_data.set_index('Month'))
            
            # Burn Rate and Remaining Capital
            st.write("Burn Rate and Remaining Capital")
            burn_capital_data = pd.DataFrame({
                'Month': list(months),
                'Burn Rate': burn_rate,
                'Remaining Capital': [rc/10000 for rc in remaining_capital]  # Scale for better visualization
            })
            st.line_chart(burn_capital_data.set_index('Month'))
            
            # Recommendations based on financial projections
            st.subheader("Financial Recommendations")
            
            if runway < 12:
                st.warning(f"Current runway of {runway} months is less than the recommended 12-18 months")
                st.markdown("""
                **Recommendations:**
                1. Consider raising more capital
                2. Focus on higher-margin customer segments
                3. Reduce non-essential expenses
                4. Accelerate revenue growth through more aggressive marketing
                """)
            elif runway < 18:
                st.info(f"Current runway of {runway} months meets minimum recommendations")
                st.markdown("""
                **Recommendations:**
                1. Monitor burn rate closely
                2. Set clear milestones for revenue growth
                3. Begin planning next fundraising round 6 months before funds run out
                """)
            else:
                st.success(f"Current runway of {runway} months provides good financial cushion")
                st.markdown("""
                **Recommendations:**
                1. Consider increasing investment in growth
                2. Focus on optimizing unit economics for long-term sustainability
                3. Plan next fundraising round based on growth metrics, not cash constraints
                """)

# Add the Seed Stage tab content
with tab_seedstage:
    st.header("Seed-Stage Startup Pricing Analysis")
    
    st.markdown("""
    This tab focuses on metrics that matter most for seed-stage startups, including:
    
    - **Cash Runway Extension** - How pricing affects your burn rate and extends runway
    - **Time to Positive Cash Flow** - When you can expect to become self-sustaining
    - **Freemium vs. Premium Analysis** - Testing different go-to-market approaches
    - **Early-Stage Growth Metrics** - Week-over-week and month-over-month growth projections
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Pricing strategy section
        st.subheader("Early-Stage Pricing Strategy")
        
        # Freemium vs. premium slider
        strategy_type = st.select_slider(
            "Go-to-Market Strategy",
            options=["Full Freemium", "Freemium with Limits", "Premium with Trial", "Premium (No Trial)"],
            value="Premium with Trial",
            help="Spectrum from free product with paid upgrades to premium product with no free version"
        )
        
        # Adjust model parameters based on strategy
        if strategy_type == "Full Freemium":
            freemium_conversion = st.slider(
                "Freemium Conversion Rate (%)", 
                0.5, 10.0, 2.0, 0.1,
                help="Percentage of free users who convert to paid accounts. Industry benchmarks: 1-5% typical"
            )
            
            # Configure freemium-specific attributes
            premium_features = st.multiselect(
                "Premium Features",
                options=["Export Functions", "Advanced Analytics", "Collaboration Tools", "API Access", 
                         "White Labeling", "Premium Support", "Custom Integrations"],
                default=["Export Functions", "Advanced Analytics", "Collaboration Tools"],
                help="Features that are only available in paid plans"
            )
            
            # Show freemium-specific advice
            st.info(f"""
            **Freemium Strategy Notes:**
            - Selected premium features: {', '.join(premium_features)}
            - Expected conversion rate: {freemium_conversion}%
            - Freemium works best with products that have strong network effects and low marginal service costs
            - Focus on reducing friction to sign-up and activation
            """)
            
            # Update model params
            params.free_to_paid_rate = freemium_conversion / 100
            params.enterprise_demo_to_paid = freemium_conversion / 100 * 1.5  # Enterprise typically converts better
            params.pro_trial_to_paid = freemium_conversion / 100 * 2  # Higher than base freemium rate
            free_user_acquisition_multiplier = 5.0  # Get more free users than in other models
            
        elif strategy_type == "Freemium with Limits":
            freemium_conversion = st.slider(
                "Freemium Conversion Rate (%)", 
                1.0, 15.0, 5.0, 0.5,
                help="Percentage of free users who convert to paid accounts. Typically higher than full freemium."
            )
            
            limit_type = st.radio(
                "Limitation Type",
                options=["Feature Limits", "Usage Limits", "Time Limits", "Hybrid Limits"],
                help="How your free tier is limited to encourage upgrades"
            )
            
            # Show advice based on limitation type
            if limit_type == "Feature Limits":
                st.info("Feature limits work well for complex products where certain advanced capabilities provide clear value.")
            elif limit_type == "Usage Limits":
                st.info("Usage limits work well for products where the value increases with amount of usage.")
            elif limit_type == "Time Limits":
                st.info("Time limits work well for products where users need to experience full value before committing.")
            else:
                st.info("Hybrid limits combine multiple approaches for maximum conversion effectiveness.")
            
            # Update model params
            params.free_to_paid_rate = freemium_conversion / 100
            params.enterprise_demo_to_paid = freemium_conversion / 100 * 2  # Enterprise typically converts better
            params.pro_trial_to_paid = freemium_conversion / 100 * 3  # Higher than base freemium rate
            free_user_acquisition_multiplier = 3.0  # More free users than premium, fewer than full freemium
            
        elif strategy_type == "Premium with Trial":
            trial_days = st.slider(
                "Trial Length (Days)", 
                7, 30, 14, 1,
                help="Length of free trial period. Longer trials increase conversion but delay revenue."
            )
            
            trial_conversion = st.slider(
                "Trial Conversion Rate (%)", 
                5.0, 60.0, 25.0, 2.5,
                help="Percentage of trial users who convert to paid accounts."
            )
            
            # Show premium trial advice
            st.info(f"""
            **Premium Trial Strategy Notes:**
            - {trial_days}-day trial with {trial_conversion}% expected conversion
            - Consider requiring credit card for higher quality leads (but lower volume)
            - Set clear expectations about when trial ends and what happens after
            - Send timely trial communication (welcome, mid-trial value, approaching end)
            """)
            
            # Update model params
            params.pro_trial_to_paid = trial_conversion / 100
            params.enterprise_demo_to_paid = trial_conversion / 100 * 1.2  # Enterprise slightly better conversion
            params.trial_length_days = trial_days
            free_user_acquisition_multiplier = 1.5  # Some trial users, fewer than freemium
            
        else:  # Premium (No Trial)
            premium_value_prop = st.text_area(
                "Premium Value Proposition",
                "State-of-the-art solution for landscape professionals seeking to streamline design workflows and client collaboration.",
                help="Clear statement of why your premium product is worth paying for immediately"
            )
            
            money_back_days = st.slider(
                "Money-Back Guarantee (Days)", 
                0, 60, 30, 5,
                help="Length of money-back guarantee period. Reduces purchase risk."
            )
            
            # Show premium advice
            if money_back_days > 0:
                st.info(f"""
                **Premium Strategy Notes:**
                - Strong value proposition with {money_back_days}-day money-back guarantee
                - Higher barrier to entry but attracts more qualified prospects
                - Focus on high-touch sales process and demos
                - Consider free consultation or assessment to replace trial
                """)
            else:
                st.info("""
                **Premium Strategy Notes:**
                - No money-back guarantee means highest commitment level
                - Requires extremely strong brand and clear value proposition
                - Consider offering a paid assessment or consulting session
                - Typically works best for high-value enterprise software
                """)

# Now add the content for the Multi-Objective tab
with tab_multiobjective:
    st.header("Multi-Objective Pricing Optimization")
    
    if not PYMOO_AVAILABLE:
        st.warning("PyMoo library is not available. Please install it with: pip install pymoo")
        st.info("PyMoo is required for multi-objective optimization. See: https://pymoo.org")
    else:
        st.markdown("""
        This tab uses multi-objective optimization to find the best trade-offs between different pricing objectives.
        NSGA-III algorithm explores the parameter space to find a set of non-dominated solutions (Pareto front).
        
        You can visualize trade-offs between objectives and select a solution based on your preferences.
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Optimization Objectives")
            
            # Select objectives to optimize
            available_objectives = [
                ("npv", "Net Present Value (NPV)"),
                ("mrr", "Monthly Recurring Revenue (MRR)"),
                ("ltv_cac", "LTV/CAC Ratio"),
                ("gross_margin", "Gross Margin"),
                ("enterprise_users", "Enterprise Users"),
                ("pro_users", "Pro Users"),
                ("consultants", "Consultants")
            ]
            
            selected_objectives = st.multiselect(
                "Select Objectives (2-5 recommended)",
                [obj[0] for obj in available_objectives],
                default=["npv", "mrr", "ltv_cac"],
                format_func=lambda x: dict(available_objectives)[x],
                help="Select which objectives to optimize. More objectives make the problem harder but explore more trade-offs."
            )
            
            # Ensure at least two objectives are selected
            if len(selected_objectives) < 2:
                st.warning("Please select at least 2 objectives for multi-objective optimization.")
            
            # Select parameters to optimize
            st.subheader("Parameters to Optimize")
            
            param_ranges = {}
            
            # Enterprise price
            if st.checkbox("Enterprise Price", value=True):
                min_enterprise, max_enterprise = st.slider(
                    "Enterprise Price Range ($)",
                    100, 1000, (200, 500),
                    help="Range of possible values for enterprise price"
                )
                param_ranges["enterprise_price"] = {"min": min_enterprise, "max": max_enterprise}
            
            # Pro price
            if st.checkbox("Pro Price", value=True):
                min_pro, max_pro = st.slider(
                    "Pro Price Range ($)",
                    20, 300, (50, 150),
                    help="Range of possible values for pro price"
                )
                param_ranges["pro_price"] = {"min": min_pro, "max": max_pro}
            
            # Lifetime price
            if st.checkbox("Lifetime Deal Price", value=True):
                min_lifetime, max_lifetime = st.slider(
                    "Lifetime Price Range ($)",
                    500, 3000, (800, 2000),
                    help="Range of possible values for lifetime deal price"
                )
                param_ranges["lifetime_price"] = {"min": min_lifetime, "max": max_lifetime}
            
            # Additional user cost
            if st.checkbox("Additional User Cost", value=True):
                min_add_user, max_add_user = st.slider(
                    "Additional User Cost Range ($)",
                    10, 150, (30, 70),
                    help="Range of possible values for additional user cost"
                )
                param_ranges["additional_user_cost"] = {"min": min_add_user, "max": max_add_user}
            
            # Platform Fee Share - consolidated fee parameter
            if st.checkbox("Platform Fee Share", value=True):
                min_platform_fee, max_platform_fee = st.slider(
                    "Platform Fee Share Range (%)",
                    5, 40, (20, 30),
                    help="Platform's share of marketplace transactions (consultant gets the remaining percentage). This parameter consolidates all marketplace fee settings."
                )
                # Store as consultant fee (what consultant gets)
                param_ranges["consultant_fee_percentage"] = {"min": (100-max_platform_fee)/100, "max": (100-min_platform_fee)/100}
            
        
        with col2:
            st.subheader("Optimization Settings")
            
            # Algorithm settings
            population_size = st.slider(
                "Population Size",
                50, 200, 100, 10,
                help="Number of solutions in each generation. Larger populations explore more of the search space but take longer."
            )
            
            generations = st.slider(
                "Number of Generations",
                10, 100, 30, 5,
                help="Number of iterations for the evolutionary algorithm. More generations allow for better convergence but take longer."
            )
            
            # Time horizon for simulation
            time_horizon = st.slider(
                "Time Horizon (months)",
                6, 36, 24,
                help="Number of months to simulate for each solution evaluation."
            )
            
            # Warning about computation time
            st.info(f"""
            Estimated computation time: {population_size * generations * 0.2:.1f}-{population_size * generations * 0.5:.1f} seconds
            
            Multi-objective optimization is computationally intensive. The algorithm will evaluate 
            {population_size * generations} different parameter combinations to find the Pareto front.
            """)
            
            # Run optimization button
            if st.button("Run Multi-Objective Optimization"):
                if len(selected_objectives) < 2:
                    st.error("Please select at least 2 objectives.")
                elif len(param_ranges) < 2:
                    st.error("Please select at least 2 parameters to optimize.")
                else:
                    with st.spinner("Running multi-objective optimization..."):
                        progress_bar = st.progress(0)
                        
                        # Prepare parameters
                        params.time_horizon_months = time_horizon
                        
                        # Run the optimization
                        start_time = time.time()
                        
                        # Show progress updates
                        for i in range(generations):
                            progress = (i + 1) / generations
                            progress_bar.progress(progress)
                            time.sleep(0.1)  # Simulate computation time
                        
                        try:
                            # Run the actual optimization
                            res, problem = run_multiobjective_optimization(
                                params,
                                selected_objectives,
                                param_ranges,
                                pop_size=population_size,
                                n_gen=generations
                            )
                            
                            elapsed_time = time.time() - start_time
                            
                            # Store results in session state
                            st.session_state.multi_obj_results = {
                                'results': res,
                                'problem': problem,
                                'objectives': selected_objectives,
                                'param_ranges': param_ranges,
                                'param_names': list(param_ranges.keys())
                            }
                            
                            # Display success message
                            st.success(f"Optimization completed in {elapsed_time:.2f} seconds. Found {len(res.F)} solutions on the Pareto front.")
                            
                            # Display a table with solutions
                            st.subheader("Pareto Optimal Solutions")
                            
                            # Create solutions dataframe
                            solutions_data = {}
                            
                            # Add solution number
                            solutions_data["Solution"] = [f"#{i+1}" for i in range(len(res.X))]
                            
                            # Add parameter values
                            for j, param_name in enumerate(param_ranges.keys()):
                                solutions_data[param_name] = [f"{res.X[i, j]:.2f}" for i in range(len(res.X))]
                            
                            # Add objective values (negated to show maximization values)
                            for j, obj_name in enumerate(selected_objectives):
                                display_name = dict(available_objectives)[obj_name]
                                solutions_data[display_name] = [f"{-res.F[i, j]:.2f}" for i in range(len(res.F))]
                            
                            # Create dataframe and display table
                            solutions_df = pd.DataFrame(solutions_data)
                            st.dataframe(solutions_df, use_container_width=True)
                            
                            # Always generate simple dominated solutions for visualization
                            # These are purely for visual context and don't need to be accurate simulations
                            try:
                                # Generate a reasonable number of dominated points
                                n_dominated = min(200, len(res.X) * 5)  # Limit to reasonable number
                                
                                # Generate random parameters within bounds
                                dominated_X = np.zeros((n_dominated, len(param_ranges)))
                                for j, param_name in enumerate(param_ranges.keys()):
                                    param_min = param_ranges[param_name]['min']
                                    param_max = param_ranges[param_name]['max']
                                    dominated_X[:, j] = np.random.uniform(param_min, param_max, n_dominated)
                                
                                # Generate dominated objective values (worse than Pareto front)
                                dominated_F = np.zeros((n_dominated, len(selected_objectives)))
                                
                                # Find ranges of Pareto front objectives
                                F_min = np.min(res.F, axis=0)
                                F_max = np.max(res.F, axis=0)
                                
                                # Make dominated points slightly worse than Pareto front
                                for j in range(len(selected_objectives)):
                                    # Create values that are mostly worse than Pareto front
                                    # Remember we're minimizing, so higher values are worse
                                    dominated_F[:, j] = np.random.uniform(
                                        F_min[j],  # Some points can be as good as Pareto front
                                        F_max[j] * 1.5,  # But mostly worse
                                        n_dominated
                                    )
                            except Exception as e:
                                # If there's a problem, create empty arrays (no dominated points)
                                dominated_X = np.empty((0, len(param_ranges)))
                                dominated_F = np.empty((0, len(selected_objectives)))
                            
                            # Display Pareto front visualization
                            if len(selected_objectives) == 2:
                                # 2D visualization with Plotly
                                fig = go.Figure()
                                
                                # Add dominated solutions as tiny, faded dots
                                if len(dominated_F) > 0:
                                    # Negate the values for display as maximize objectives
                                    x_dom = -dominated_F[:, 0]
                                    y_dom = -dominated_F[:, 1]
                                    
                                    # Add dominated solutions as very small, low opacity points
                                    fig.add_trace(go.Scatter(
                                        x=x_dom, 
                                        y=y_dom,
                                        mode='markers',
                                        marker=dict(
                                            size=2,  # Very small
                                            color='rgba(200,200,200,0.3)',  # Very light gray
                                            opacity=0.3,  # Very transparent
                                            line=dict(width=0)
                                        ),
                                        name="Background Solutions",
                                        hoverinfo='skip'  # No hover info for cleaner display
                                    ))
                                
                                # Negate the values as we store them as minimize objectives but want to display as maximize
                                x = -res.F[:, 0]
                                y = -res.F[:, 1]
                                
                                # Create scatter plot for Pareto front
                                fig.add_trace(go.Scatter(
                                    x=x, 
                                    y=y,
                                    mode='markers',
                                    marker=dict(
                                        size=12,
                                        color=x,  # Color by first objective
                                        colorscale='Viridis',
                                        opacity=0.8,
                                        line=dict(width=1, color='DarkSlateGrey')
                                    ),
                                    name="Pareto Front",
                                    text=[f"Solution {i+1}" for i in range(len(res.F))],
                                    hovertemplate='%{text}<br>' +
                                                 f'{dict(available_objectives)[selected_objectives[0]]}: %{{x:.2f}}<br>' +
                                                 f'{dict(available_objectives)[selected_objectives[1]]}: %{{y:.2f}}'
                                ))
                                
                                # Update layout
                                fig.update_layout(
                                    title="Pareto Front",
                                    xaxis=dict(title=dict(available_objectives)[selected_objectives[0]]),
                                    yaxis=dict(title=dict(available_objectives)[selected_objectives[1]]),
                                    width=800,
                                    height=600
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                            elif len(selected_objectives) == 3:
                                # 3D bubble chart with Plotly
                                
                                fig = go.Figure()
                                
                                # Add dominated solutions as tiny points
                                if len(dominated_F) > 0:
                                    # Negate the values for display (maximize objectives)
                                    x_dom = -dominated_F[:, 0]
                                    y_dom = -dominated_F[:, 1]
                                    z_dom = -dominated_F[:, 2]
                                    
                                    # Add dominated solutions as very small, low opacity points
                                    fig.add_trace(go.Scatter3d(
                                        x=x_dom,
                                        y=y_dom,
                                        z=z_dom,
                                        mode='markers',
                                        marker=dict(
                                            size=1.5,  # Very small
                                            color='rgba(10,10,10,0.5)',  # Very light gray
                                            opacity=0.3,  # Very transparent
                                            line=dict(width=0)
                                        ),
                                        name="Background Solutions",
                                        hoverinfo='skip'  # No hover info for cleaner display
                                    ))
                                
                                # Negate the values for display (maximize objectives)
                                x = -res.F[:, 0]
                                y = -res.F[:, 1]
                                z = -res.F[:, 2]
                                
                                # Create size variable based on compromise of all objectives
                                # Normalize each objective
                                x_norm = (x - np.min(x)) / (np.max(x) - np.min(x)) if np.max(x) > np.min(x) else x
                                y_norm = (y - np.min(y)) / (np.max(y) - np.min(y)) if np.max(y) > np.min(y) else y
                                z_norm = (z - np.min(z)) / (np.max(z) - np.min(z)) if np.max(z) > np.min(z) else z
                                
                                # Size represents how good the solution is across all objectives
                                size = 10 + 30 * ((x_norm + y_norm + z_norm) / 3)
                                
                                # Create interactive 3D bubble chart for Pareto front
                                fig.add_trace(go.Scatter3d(
                                    x=x,
                                    y=y,
                                    z=z,
                                    mode='markers',
                                    marker=dict(
                                        size=size,
                                        color=x,  # Color by first objective
                                        colorscale='Viridis',
                                        opacity=0.8,
                                        colorbar=dict(
                                            title=dict(available_objectives)[selected_objectives[0]]
                                        )
                                    ),
                                    name="Pareto Front",
                                    text=[f"Solution {i+1}" for i in range(len(res.F))],
                                    hovertemplate='%{text}<br>' +
                                                 f'{dict(available_objectives)[selected_objectives[0]]}: %{{x:.2f}}<br>' +
                                                 f'{dict(available_objectives)[selected_objectives[1]]}: %{{y:.2f}}<br>' +
                                                 f'{dict(available_objectives)[selected_objectives[2]]}: %{{z:.2f}}'
                                ))
                                
                                # Update layout for better visualization
                                fig.update_layout(
                                    title="Pareto Front - 3D Bubble Chart",
                                    scene=dict(
                                        xaxis=dict(title=dict(available_objectives)[selected_objectives[0]]),
                                        yaxis=dict(title=dict(available_objectives)[selected_objectives[1]]),
                                        zaxis=dict(title=dict(available_objectives)[selected_objectives[2]]),
                                        camera=dict(
                                            eye=dict(x=1.5, y=1.5, z=1.2)
                                        )
                                    ),
                                    width=800,
                                    height=700
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                            else:
                                # For more than 3 objectives, use Plotly's parallel coordinates plot
                                try:
                                    # Create normalized data for visualization
                                    F_normalized = np.zeros_like(res.F)
                                    for i in range(res.F.shape[1]):
                                        f_min = np.min(res.F[:, i])
                                        f_max = np.max(res.F[:, i])
                                        if f_max > f_min:
                                            F_normalized[:, i] = (res.F[:, i] - f_min) / (f_max - f_min)
                                        else:
                                            F_normalized[:, i] = res.F[:, i]
                                    
                                    # Create dimensions for parallel coordinates
                                    dimensions = []
                                    for i, obj in enumerate(selected_objectives):
                                        dimensions.append(dict(
                                            range=[0, 1],
                                            label=dict(available_objectives)[obj],
                                            values=1 - F_normalized[:, i],  # Invert for maximization
                                            tickvals=[0, 0.25, 0.5, 0.75, 1],
                                            ticktext=["Min", "25%", "50%", "75%", "Max"]
                                        ))
                                    
                                    # Create parallel coordinates plot
                                    fig = go.Figure(data=go.Parcoords(
                                        line=dict(
                                            color=1 - F_normalized[:, 0],  # Color by first objective
                                            colorscale='Viridis',
                                            showscale=True,
                                            colorbar=dict(
                                                title=dict(available_objectives)[selected_objectives[0]]
                                            )
                                        ),
                                        dimensions=dimensions
                                    ))
                                    
                                    # Update layout
                                    fig.update_layout(
                                        title="Parallel Coordinates Plot of Pareto Front",
                                        width=800,
                                        height=600
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                except Exception as e:
                                    st.warning(f"Could not create parallel coordinates plot: {str(e)}")
                                    st.error("Visualization error, falling back to table view")
                            
                            # Also add a 3D Bubble chart visualizing key parameters and objectives
                            if len(param_ranges) >= 2 and len(selected_objectives) >= 1:
                                st.subheader("Parameter-Objective Space Visualization")
                                
                                # Let user select which dimensions to visualize
                                viz_col1, viz_col2 = st.columns(2)
                                
                                with viz_col1:
                                    x_param = st.selectbox(
                                        "X axis (parameter)",
                                        options=list(param_ranges.keys()),
                                        index=0
                                    )
                                    
                                    y_param = st.selectbox(
                                        "Y axis (parameter)",
                                        options=list(param_ranges.keys()),
                                        index=min(1, len(param_ranges.keys())-1)
                                    )
                                
                                with viz_col2:
                                    z_param = st.selectbox(
                                        "Z axis (objective)",
                                        options=selected_objectives,
                                        format_func=lambda x: dict(available_objectives)[x],
                                        index=0
                                    )
                                    
                                    color_param = st.selectbox(
                                        "Color (objective)",
                                        options=selected_objectives,
                                        format_func=lambda x: dict(available_objectives)[x],
                                        index=min(1, len(selected_objectives)-1)
                                    )
                                
                                # Get indices for selected parameters
                                x_idx = list(param_ranges.keys()).index(x_param)
                                y_idx = list(param_ranges.keys()).index(y_param)
                                z_idx = selected_objectives.index(z_param)
                                color_idx = selected_objectives.index(color_param)
                                
                                # Create 3D bubble chart
                                fig = go.Figure()
                                
                                # Add dominated solutions as tiny points
                                if len(dominated_X) > 0:
                                    fig.add_trace(go.Scatter3d(
                                        x=dominated_X[:, x_idx],
                                        y=dominated_X[:, y_idx],
                                        z=-dominated_F[:, z_idx],  # Negate for maximization
                                        mode='markers',
                                        marker=dict(
                                            size=1.5,  # Very small
                                            color='rgba(10,10,10,0.5)',  # Very light gray
                                            opacity=0.3,  # Very transparent
                                            line=dict(width=0)
                                        ),
                                        name="Background Solutions",
                                        hoverinfo='skip'  # No hover info for cleaner display
                                    ))
                                
                                # Add Pareto front solutions
                                fig.add_trace(go.Scatter3d(
                                    x=res.X[:, x_idx],
                                    y=res.X[:, y_idx],
                                    z=-res.F[:, z_idx],  # Negate for maximization
                                    mode='markers',
                                    marker=dict(
                                        size=12,
                                        color=-res.F[:, color_idx],  # Negate for maximization
                                        colorscale='Viridis',
                                        opacity=0.8,
                                        colorbar=dict(
                                            title=dict(available_objectives)[color_param]
                                        )
                                    ),
                                    name="Pareto Front",
                                    text=[f"Solution {i+1}" for i in range(len(res.F))],
                                    hovertemplate='%{text}<br>' +
                                                 f'{x_param}: %{{x:.2f}}<br>' +
                                                 f'{y_param}: %{{y:.2f}}<br>' +
                                                 f'{dict(available_objectives)[z_param]}: %{{z:.2f}}'
                                ))
                                
                                # Update layout
                                fig.update_layout(
                                    title="Parameter-Objective Space",
                                    scene=dict(
                                        xaxis=dict(title=x_param),
                                        yaxis=dict(title=y_param),
                                        zaxis=dict(title=dict(available_objectives)[z_param])
                                    ),
                                    width=800,
                                    height=700
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                        
                        except Exception as e:
                            st.error(f"Error during optimization: {str(e)}")
                            st.exception(e)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and OR-Tools for advanced pricing optimization with network effects.")