import requests
import os
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = Path(current_dir).parent.parent / 'data' / 'raw'

base_url = "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/"

def download_balance(years=None):
    if not years:
        years = range(2016, 2024)

    filepaths = []
    for year in years:
        first = f"EIA930_BALANCE_{year}_Jan_Jun.csv"
        second = f"EIA930_BALANCE_{year}_Jul_Dec.csv"

        if not os.path.exists(f"{data_dir}/{first}"):

            url = f"{base_url}{first}"
            print(f"Fetching url {url}")
            first_response = requests.get(url)

            with open(f"{data_dir}/{first}", "wb") as f:
                f.write(first_response.content)

        if not os.path.exists(f"{data_dir}/{second}"):

            url = f"{base_url}{second}"
            print(f"Fetching url {url}")
            second_response = requests.get(url)

            with open(f"{data_dir}/{second}", "wb") as f:
                f.write(second_response.content)

        filepaths.append(os.path.join(f"{data_dir}/{first}"))
        filepaths.append(os.path.join(f"{data_dir}/{second}"))

    return filepaths
