import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyomo.environ import Var
from src.MILP.BatteryMILP import BatteryMILP

price_data = pd.read_csv("./data/PGF1_2_PDRP88-APND_predicted_prices.csv")

# declare battery config
battery_power = 10  # MW
battery_capacity = 20  # MWh

# declare intial soc
soc = 0.5 * battery_capacity
current_cycle = 0
remaining_capacity = 100
previous_ep_power = 0

# Instaniate MILP battery object with price data
a = BatteryMILP(battery_power, battery_capacity)

# pass daily prices for optmisation
for day_idx in range(0, len(price_data), 168):
    print(day_idx)

    # call optmise method - storing pyomo model
    mod = a.optimise(
        price_data[day_idx : day_idx + 168],
        soc,
        current_cycle,
        remaining_capacity,
        previous_ep_power,
    )

    model_results = {}

    # loop through the vars
    for idx, v in enumerate(mod.component_objects(Var, active=True)):
        # print(idx, v.getname())

        var_val = getattr(mod, str(v))

        model_results[f"{v.getname()}"] = var_val[:].value

    # store data in dataframe
    if day_idx == 0:
        df = pd.DataFrame(model_results)
    else:
        df = pd.concat([df, pd.DataFrame(model_results)])

    # store soc for next 'episode'
    soc = df["soc"].iloc[-1]

    # store cycle to carry forward for cumlative calculation
    current_cycle = df["cumlative_cycle_rate"].iloc[-1]

    a.previous_cap = remaining_capacity

    # update alpha degradation after 'episode'
    remaining_capacity = a.update_remaining_cap(current_cycle)

    previous_ep_power = abs(
        np.sum(df["energy_in"].iloc[-168:] + df["energy_out"].iloc[-168:])
    )


# update cumlative profit so ensure profits carried between episodes
df["cumlative_profit"] = df["profit_timeseries"].cumsum()

# save profits for runtime duration (for comparison with DQN models)
df.to_csv("results/timeseries_results_MILP_predict.csv")

plt.plot(df["cumlative_profit"].values)
plt.show()

df.to_clipboard()
