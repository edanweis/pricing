import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
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
tab_params, tab_basic, tab_advanced, tab_compare, tab_seedstage, tab_network, tab_financial = st.tabs([
    "📊 Parameters", 
    "🔄 Basic Pricing", 
    "📈 Advanced Pricing", 
    "🔍 Compare Models", 
    "🌱 Seed Stage",
    "🌐 Network Effects",
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
        "Pro Users", 
        100, int(orig_pro * 2), int(orig_pro), 
        step=100,
        help="Number of potential professional users (individual landscape architects) who could use your platform."
    )
    
    params.potential_consultants = st.sidebar.slider(
        "Consultants", 
        10, int(orig_consultants * 2), int(orig_consultants), 
        step=10,
        help="Number of potential consultants available for your marketplace."
    )
    
    params.potential_growers = st.sidebar.slider(
        "Growers", 
        10, int(orig_growers * 2), int(orig_growers), 
        step=10,
        help="Number of potential growers who could provide plant catalogs on your platform."
    )
    
    params.potential_landscapers = st.sidebar.slider(
        "Landscapers", 
        100, int(orig_landscapers * 2), int(orig_landscapers), 
        step=100,
        help="Number of potential individual landscapers/contractors who could use your platform."
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
        help="Monthly churn rate for pro users. Industry benchmarks: 3-5% is good, 5-7% is average, >8% needs improvement. Higher than enterprise due to individual purchasing decisions."
    )
    
    # Transaction fees
    st.sidebar.subheader("Transaction Fees")
    params.contract_tx_fee = st.sidebar.slider(
        "Contract Transaction Fee", 
        0.01, 0.2, params.contract_tx_fee, 0.01,
        help="Percentage fee charged on contract transactions. Industry benchmarks: 1-3% for payment processing only, 5-15% for platforms providing significant value-add services."
    )
    params.wellspring_tx_fee = st.sidebar.slider(
        "Wellspring Transaction Fee", 
        0.01, 0.2, params.wellspring_tx_fee, 0.01,
        help="Percentage fee charged on Wellspring marketplace transactions. Marketplace fees typically range from 5-30% depending on value provided and industry standards."
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
        help="Direct monthly cost to service a pro/landscaper account, including infrastructure, support, and operations costs."
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
        - **Pro Users**: Individual landscape architects
        - **Landscapers**: Individual contractors/landscapers
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
            help="Monthly price for professional individual users. Industry benchmarks: $10-30 for basic tools, $50-150 for professional tools, $100-300 for specialized software with high value."
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
                rev_col1, rev_col2, rev_col3 = st.columns(3)
                with rev_col1:
                    # Use model directly for the values
                    contract_rev = initial_model.contract_revenue
                    wellspring_rev = initial_model.wellspring_revenue
                    subscription_rev = initial_results['final_mrr'] - contract_rev - wellspring_rev
                    st.metric("Subscription Revenue", f"${subscription_rev:,.2f}")
                with rev_col2:
                    st.metric("Contract Revenue", f"${contract_rev:,.2f}")
                with rev_col3:
                    st.metric("Wellspring Revenue", f"${wellspring_rev:,.2f}")
                    
                # User metrics
                users_col1, users_col2, users_col3 = st.columns(3)
                with users_col1:
                    st.metric("Enterprise Users", f"{initial_results['final_enterprise_users']:.0f}")
                with users_col2:
                    st.metric("Pro Users", f"{initial_results['final_pro_users']:.0f}")
                with users_col3:
                    st.metric("Consultants", f"{initial_results['final_consultants']:.0f}")
                
                # Plot
                fig = visualize_pricing_strategies({'model': initial_model, 'enterprise_price': init_enterprise_price, 'pro_price': init_pro_price})
                st.pyplot(fig)
                
                # Add runway impact section
                st.subheader("Runway & Cash Flow Impact")
                
                # Calculate monthly burn after revenue
                monthly_net_burn = monthly_burn_rate - initial_results['final_mrr']
                
                # Only show extended runway if revenue is positive but less than burn rate
                if monthly_net_burn > 0 and initial_results['final_mrr'] > 0:
                    new_runway = runway_months * (monthly_burn_rate / monthly_net_burn)
                    runway_extension = new_runway - runway_months
                    
                    runway_col1, runway_col2 = st.columns(2)
                    with runway_col1:
                        st.metric("Current Runway", f"{runway_months} months")
                    with runway_col2:
                        st.metric("Extended Runway", f"{new_runway:.1f} months", f"+{runway_extension:.1f} months")
                
                # Calculate months to cash flow positive
                if monthly_net_burn <= 0:
                    st.success(f"This pricing strategy makes you cash flow positive with ${abs(monthly_net_burn):.2f} monthly surplus!")
                else:
                    # Estimate time to positive cash flow based on MRR growth rate
                    if len(initial_model.history['mrr']) > 3:
                        # Calculate average monthly growth rate from the last few months
                        recent_mrr = initial_model.history['mrr'][-3:]
                        if recent_mrr[0] > 0:
                            monthly_growth_rate = (recent_mrr[-1] / recent_mrr[0]) ** (1/3) - 1
                            
                            # Only show projection if we have positive growth
                            if monthly_growth_rate > 0:
                                # Project months until MRR >= burn rate
                                months_to_positive = np.log(monthly_burn_rate / initial_results['final_mrr']) / np.log(1 + monthly_growth_rate)
                                
                                if months_to_positive > 0 and months_to_positive < 36:
                                    st.info(f"Estimated time to positive cash flow: {months_to_positive:.1f} months at current growth rate ({monthly_growth_rate*100:.1f}% monthly)")
                                elif months_to_positive >= 36:
                                    st.warning(f"At current growth rate ({monthly_growth_rate*100:.1f}% monthly), positive cash flow will take more than 3 years")
                            else:
                                st.warning("MRR growth is flat or negative. Unable to project time to positive cash flow.")
    
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
                rev_col1, rev_col2, rev_col3 = st.columns(3)
                with rev_col1:
                    # Use model directly for the values, rather than looking for keys in optimized_results
                    contract_rev = optimized_results['model'].contract_revenue
                    wellspring_rev = optimized_results['model'].wellspring_revenue
                    subscription_rev = optimized_results['final_mrr'] - contract_rev - wellspring_rev
                    st.metric("Subscription Revenue", f"${subscription_rev:,.2f}")
                with rev_col2:
                    st.metric("Contract Revenue", f"${contract_rev:,.2f}")
                with rev_col3:
                    st.metric("Wellspring Revenue", f"${wellspring_rev:,.2f}")
                
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
            "Expense": ["Rent", "Software", "Office", "Director Salaries (2x)", "Debt equity", 
                       "Engineer", "CMO", "Sales", "Freelance", "Legal", "Customer support"],
            "Monthly": [5000, 4000, 600, 25000, 75000, 16667, 8333, 8333, 8000, 0, 7500],
            "Yearly": ["", "", "", 300000, "", 200000, 100000, 100000, "", "", 90000],
            "Annual Growth": ["5.00%", "20.00%", "5.00%", "2.00%", "0.00%", "5.00%", "5.00%", "5.00%", "0.00%", "0.00%", "2.00%"],
            "Contingency": ["25%", "200%", "5%", "25%", "0%", "20%", "20%", "20%", "25%", "25%", "20%"]
        }
        
        # Create a session state key for the costs data if it doesn't exist
        if 'costs_data' not in st.session_state:
            st.session_state.costs_data = default_costs_data
        
        # Convert to DataFrame for editing
        costs_df = pd.DataFrame(st.session_state.costs_data)
        
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
                "Yearly": st.column_config.TextColumn(
                    "Yearly ($)",
                    help="Annual salary or cost (leave empty if monthly is used)"
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
            "Monthly": edited_costs_df["Monthly"].tolist(),
            "Yearly": edited_costs_df["Yearly"].tolist(),
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
            rev_col1, rev_col2, rev_col3 = st.columns(3)
            with rev_col1:
                subscription_rev = monthly_revenue[-1] - selected_model['model'].contract_revenue - selected_model['model'].wellspring_revenue
                st.metric("Subscription Revenue", f"${subscription_rev:,.2f}")
            with rev_col2:
                st.metric("Contract Revenue", f"${selected_model['model'].contract_revenue:,.2f}")
            with rev_col3:
                st.metric("Wellspring Revenue", f"${selected_model['model'].wellspring_revenue:,.2f}")
                
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

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and OR-Tools for advanced pricing optimization with network effects.")