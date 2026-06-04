import requests
import pandas as pd
from typing import Union
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..config.config import *

BASE_URL = "https://apigw.fiingroup.vn/FXPF/Corporate/GetBasicInfo"

class BasicInfor:
    def __init__(self, token_provider: callable, tickers: Union[list[str],str], max_workers: int = 10):

        self.token_provider = token_provider
        self.base_url = BASE_URL
        self.org_mapping = ticker_org
        self.max_workers = max_workers
        if isinstance(tickers, str):
            self.tickers = [tickers]
        else:
            self.tickers = tickers
        self.data = self.get()

    def _get_each_ticker(self, organizationId: str) -> dict:
        retries = 0
        while retries < 5:
            try:
                headers = {
                    "Authorization": f"Bearer {self.token_provider()}",
                    "X-FG-APP": "FiinQuant"
                }
                payload = {"OrganizationId": organizationId}
                response = requests.get(url=self.base_url, headers=headers, params=payload, timeout=10)

                if response.status_code == 200:
                    result = response.json()
                    data = result.get("data", {})
                    return {
                        k: v for k, v in data.items()
                        if k in ["ticker", "companyName", "exchange", "sector"]
                    }

                elif response.status_code in [500, 502, 503, 504, 401]:
                    retries += 1
                    time.sleep(3)
                else:
                    raise ConnectionError(f"Error {response.status_code}: {response.text}")

            except requests.exceptions.RequestException as e:
                retries += 1
                if retries >= 5:
                    raise ConnectionError(f"Request failed after {5} retries: {e}")
                time.sleep(3)

        raise ConnectionError(f"Failed to fetch data for OrganizationId={organizationId} after {5} retries")

    def get(self) -> pd.DataFrame:

        # mapping ticker -> org_id
        org_ids = {}
        for ticker in self.tickers:
            org_id = self.org_mapping.get(ticker)
            if not org_id:
                raise ValueError(f"Ticker {ticker} not found in mapping file")
            org_ids[ticker] = org_id

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(self._get_each_ticker, org_id): ticker
                for ticker, org_id in org_ids.items()
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    results.append(data)
                except Exception as e:
                    print(f"Error fetching {ticker}: {e}")

        return pd.DataFrame(results)
    
    def __repr__(self):
        return (self.data)


