import streamlit as st
# rest of the libraries needed for the app to work
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns


# --- Black-Scholes Class ---
class Black_Sholes:
    '''
    Initializing all the variables/components needed to solve for the call price and put price
    * S is the current price of the asset, SPX in this case
    * K is the strike price we are trying to find the price for. this determins the moneyness of the option
    * r is the risk free rate, this is the 3 month treasury bill yeild, usually arount 3 to 4 percent.
    * t is the time to maturity in terms of years.
    * sigma is the volitility of the stock/underling asset and is the standard derivitive, we use the VIX instead of calculating it outselves.
    '''
    def __init__(self, S, K, r, t, sigma):
        self.S = S
        self.K = K
        self.r = r
        self.t = t
        self.sigma = sigma
    '''
    d1 and d2 are represent the chances of the option expiring in the money(d2).
    For a call option it implies the strike price price is lower than the current price, meaning we can buy the stock at a discount. 
    With a put option we want the strike price to be higher than the current price so we can sell the asset at a premium.
    d1 is the expected value of the option when exercised.
    the Black-Sholes formula uses the cumulative distribution formula with d1 and d2 as inputs.
    '''
    def _d1_d2(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.t) / (self.sigma * np.sqrt(self.t))
        d2 = d1 - self.sigma * np.sqrt(self.t)
        return d1, d2

    def call_put_price(self):
        d1, d2 = self._d1_d2()
        call = self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.t) * norm.cdf(d2)
        put = self.K * np.exp(-self.r * self.t) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        return call, put

    def greeks(self):
        d1, d2 = self._d1_d2()
        nd1 = norm.pdf(d1)
        Nd1 = norm.cdf(d1)
        Nd2 = norm.cdf(d2)

        delta_call = Nd1
        delta_put = Nd1 - 1
        gamma = nd1 / (self.S * self.sigma * np.sqrt(self.t))
        vega = self.S * nd1 * np.sqrt(self.t) / 100
        theta_call = (-self.S * nd1 * self.sigma / (2 * np.sqrt(self.t))
                      - self.r * self.K * np.exp(-self.r * self.t) * Nd2) / 365
        theta_put = (-self.S * nd1 * self.sigma / (2 * np.sqrt(self.t))
                     + self.r * self.K * np.exp(-self.r * self.t) * norm.cdf(-d2)) / 365
        rho_call = self.K * self.t * np.exp(-self.r * self.t) * Nd2 / 100
        rho_put = -self.K * self.t * np.exp(-self.r * self.t) * norm.cdf(-d2) / 100

        return {
            "Delta Call": delta_call,
            "Delta Put": delta_put,
            "Gamma": gamma,
            "Vega": vega,
            "Theta Call": theta_call,
            "Theta Put": theta_put,
            "Rho Call": rho_call,
            "Rho Put": rho_put
        }

    def heatmap_data(self, metric):
        strikes = np.linspace(self.S * 0.8, self.S * 1.2, 10)
        volatilities = np.linspace(self.sigma * 0.8, self.sigma * 1.2, 10)
        data = []
        for K in strikes:
            row = []
            for vol in volatilities:
                temp_bs = Black_Sholes(self.S, K, self.r, self.t, vol)
                if metric == "call":
                    val, _ = temp_bs.call_put_price()
                elif metric == "put":
                    _, val = temp_bs.call_put_price()
                else:
                    g = temp_bs.greeks()
                    val = g[[k for k in g.keys() if metric in k.lower()][0]]
                row.append(val)
            data.append(row)

        df = pd.DataFrame(data, index=np.round(strikes, 2), columns=np.round(volatilities, 3))
        return df
    
    import numpy as np



# --- Data Fetch ---
@st.cache_data
def load_data():
    end = datetime.today()
    start = end - timedelta(days=30)
    vix = pd.read_csv("https://stooq.com/q/l/?s=vi.f&f=sd2t2ohlcv&h&e=csv")
    fed = (pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3")
           .set_index("observation_date")
           .dropna())
    spx_url = f"https://stooq.com/q/d/l/?s=^spx&d1={start:%Y%m%d}&d2={end:%Y%m%d}&i=d"
    spx = (pd.read_csv(spx_url, parse_dates=["Date"])
           .set_index("Date")
           .sort_index())
    return vix, fed, spx



def home():
    st.title("📈 Options Pricing & Analysis App")
    st.subheader("Powered by the Black-Scholes Model")
    st.image("./pics/option_trading.jpg")
    st.markdown("""
    Welcome to the **Options Pricing & Analysis App**.  
    This tool lets you:
    - Calculate European call & put prices
    - Visualize option Greeks in heatmaps
    - Generate Profit/Loss (P/L) curves for various strategies
    - Experiment with custom inputs for strike price, volatility, and more
    """)

    st.divider()

    # About Black-Scholes
    st.header("About the Black-Scholes Model")
    st.markdown("""
    The **Black-Scholes model** is a groundbreaking mathematical framework for pricing European-style options.  
    Developed in 1973 by **Fischer Black**, **Myron Scholes**, and **Robert Merton**,  
    it transformed financial markets by providing a formula to estimate the fair value of an option.
    """)

    # Formula
    st.latex(r"C = S_0 N(d_1) - K e^{-rt} N(d_2)")
    st.latex(r"P = K e^{-rt} N(-d_2) - S_0 N(-d_1)")
    st.latex(r"d_1 = \frac{\ln(S_0 / K) + (r + \frac{\sigma^2}{2})t}{\sigma\sqrt{t}}, \quad d_2 = d_1 - \sigma\sqrt{t}")

    st.markdown("""
    **Where:**
    - \(C\) = Call option price
    - \(P\) = Put option price
    - \(S_0\) = Current underlying price
    - \(K\) = Strike price
    - \(t\) = Time to maturity (in years)
    - \(r\) = Risk-free interest rate
    - \(sigma\) = Volatility of the underlying asset(It is the "o" looking greek letter)
    - \(N(x)\) = Cumulative distribution function of the standard normal distribution
    """)

    st.divider()

    # Assumptions
    st.subheader("Model Assumptions")
    st.markdown("""
    - European options only (can only be exercised at maturity)
    - No dividends during the life of the option
    - Constant volatility & interest rate
    - No transaction costs or taxes
    - Efficient markets (no arbitrage opportunities)
    """)


    st.divider()

    st.subheader("Option Price Calculator")
    st.subheader("Please remember that this a model and should not be used to make finacial desision.")
    st.subheader("Options are extreamly risky so it is really easy to lose alot of money, really fast.")
    st.subheader("here is a [PDF](%s) of just on how risky it can get" % 'https://www.theocc.com/getmedia/a151a9ae-d784-4a15-bdeb-23a029f50b70/riskstoc.pdf')
    st.image("./pics/losing_money.jpg")
    st.markdown("""
    Options are financial contracts that give you the right, but not the obligation, to buy (Call) or sell (Put) an underlying asset at a predetermined price (strike price) before or at a specific date (expiration).

    The value of an option depends on several factors:

    * Underlying Price (S) — current price of the asset

    * Strike Price (K) — price at which the option can be exercised

    * Time to Expiration (t) — how long until the option expires

    * Volatility (sigma ) — how much the asset's price moves

    * Risk-Free Rate (r) — typically the yield on 3 Month Treasury bills
                
    

                    
    """)
    vix, fed, spx = load_data()
    rfr = fed['DTB3'].iloc[-1]
    spx_vol = spx["Close"].pct_change().dropna().std() * np.sqrt(252)
    vix_vol = vix["Close"][0] / 100
    current_price = spx['Close'].iloc[-1]

    st.header(f"""Current Values based on the nearest closing date are:
              Current SPX Price: {current_price}\n
              VIX(volitility proxy): {vix_vol}\n
              risk-free rate (3 month T-Bill yeild): {rfr}""")
    
    
    

    st.sidebar.header("Input Parameters")
    strategy = st.sidebar.selectbox("Select Strategy for PnL Calculation", 
            ["Long Call", "Long Put", "Covered Call", "Protective Put", "Bull Call Spread"])
    strike_price = st.sidebar.number_input("Strike Price", value=int(current_price))
    days_to_exp = st.sidebar.number_input("Days to Expiration", value=30)
    time_to_maturity = days_to_exp / 365
    use_vix = st.sidebar.selectbox("Volatility Source", ["VIX", "SPX Realized"])
    volatility = vix_vol if use_vix == "VIX" else spx_vol
    calc_button = st.sidebar.button("Calculate")

    
    if calc_button:
        bs = Black_Sholes(current_price, strike_price, rfr/100, time_to_maturity, volatility)
        call_price, put_price = bs.call_put_price()

        st.subheader("Option Prices")
        st.write(f"**Call Price:** {call_price:.2f}")
        st.write(f"**Put Price:** {put_price:.2f}")


        st.subheader("Greeks")
        greeks_dict = bs.greeks()
        for g, val in greeks_dict.items():
            st.write(f"**{g}:** {val:.4f}")

        st.subheader("Heatmaps")
        metrics = ["call", "put", "delta", "gamma", "vega", "theta", "rho"]
        cols = st.columns(2)
        for i, metric in enumerate(metrics):
            df = bs.heatmap_data(metric)
            fig, ax = plt.subplots()
            sns.heatmap(df, cmap="coolwarm", ax=ax)
            ax.set_title(metric.capitalize())
            ax.set_xlabel("Volatility")
            ax.set_ylabel("Strike Price")
            cols[i % 2].pyplot(fig)

        st.subheader("📊 Strategy PnL at Expiration")
        # Description of all the option strategies stored as a dictionary.
        strategy_descriptions = {
    "Long Call": """
    **Long Call** : Buying a call option to profit from an upward move in the underlying price.  
    **Max Loss** = Premium paid.  
    **Max Gain** = Unlimited.  
    **Breakeven** = Strike Price + Premium.
    """,
    "Long Put": """
    **Long Put** : Buying a put option to profit from a downward move in the underlying price.  
    **Max Loss** = Premium paid.  
    **Max Gain** = Strike Price − Premium (if price goes to zero).  
    **Breakeven** = Strike Price − Premium.
    """,
    "Covered Call": """
    **Covered Call** : Owning the stock and selling a call option.  
    Earn premium income but cap upside potential.  
    **Max Loss** = Stock value drop offset by premium.  
    **Max Gain** = Strike Price − Stock Price + Premium.
    """,
    "Protective Put": """
    **Protective Put** : Owning the stock and buying a put option as insurance.  
    **Max Loss** = Strike Price − Stock Price + Premium.  
    **Max Gain** = Unlimited upside.
    """,
    "Bull Call Spread": """
    **Bull Call Spread** : Buying a call at a lower strike, selling a call at a higher strike.  
    Cheaper than a long call but caps profits.  
    **Max Loss** = Net Premium Paid.  
    **Max Gain** = Strike Difference − Net Premium.
    """
    }
        
        S_range = np.linspace(current_price * 0.7, current_price * 1.3, 100)

        def payoff_long_call(S_range, K, premium):
            return np.maximum(S_range - K, 0) - premium

        def payoff_long_put(S_range, K, premium):
            return np.maximum(K - S_range, 0) - premium

        def payoff_covered_call(S_range, stock_price, call_strike, call_premium):
            stock_payoff = S_range - stock_price
            call_payoff = -np.maximum(S_range - call_strike, 0) + call_premium
            return stock_payoff + call_payoff

        def payoff_protective_put(S_range, stock_price, put_strike, put_premium):
            stock_payoff = S_range - stock_price
            put_payoff = np.maximum(put_strike - S_range, 0) - put_premium
            return stock_payoff + put_payoff

        def payoff_bull_call_spread(S_range, K1, premium1, K2, premium2):
            long_call = np.maximum(S_range - K1, 0) - premium1
            short_call = -np.maximum(S_range - K2, 0) + premium2
            return long_call + short_call

        if strategy == "Long Call":
            premium = call_price
            pnl = payoff_long_call(S_range, strike_price, premium)

        elif strategy == "Long Put":
            premium = put_price
            pnl = payoff_long_put(S_range, strike_price, premium)

        elif strategy == "Covered Call":
            call_premium = call_price
            pnl = payoff_covered_call(S_range, current_price, strike_price, call_premium)

        elif strategy == "Protective Put":
            put_premium = put_price
            pnl = payoff_protective_put(S_range, current_price, strike_price, put_premium)

        elif strategy == "Bull Call Spread":
            premium1 = call_price
            K2 = strike_price + 100
            premium2 = Black_Sholes(current_price, K2, rfr/100, time_to_maturity, volatility).call_put_price()[0]
            pnl = payoff_bull_call_spread(S_range, strike_price, premium1, K2, premium2)
        st.write(strategy_descriptions[strategy])
        # Plot PnL Curve
        fig, ax = plt.subplots()
        ax.plot(S_range, pnl, label="PnL", color="blue")
        ax.axhline(0, color="black", linewidth=1)
        ax.axvline(current_price, color="red", linestyle="--", label="Current Price")
        ax.set_xlabel("Underlying Price at Expiration")
        ax.set_ylabel("Profit / Loss")
        ax.set_title(f"{strategy} - PnL at Expiration")
        ax.legend()
        st.pyplot(fig)     
            
   

if __name__ == "__main__":
    home()
