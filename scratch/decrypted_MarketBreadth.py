import requests
import pandas as pd
from datetime import datetime
from urllib.parse import urlparse
import base64, gzip
import json, time
from typing import Union
from requests.exceptions import RequestException

BASE_URL = "https://apigw.fiingroup.vn/FTMA/MarketInDepth/GetLatestIndices"

class MarketBreadth:
    def __init__(self, token_provider: callable):
        self.url = BASE_URL
        self.token_provider = token_provider

    def _decode_jwt(self, token):
        header_b64, payload_b64, _ = token.split('.')
        header = json.loads(base64.urlsafe_b64decode(header_b64 + '=='))
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + '=='))
        return header, payload

    def _decode_list_api(self,compressed_data):
        compressed_bytes = base64.b64decode(compressed_data)
        decompressed_bytes = gzip.decompress(compressed_bytes)
        return decompressed_bytes.decode('utf-8')
    
    def _is_valid_api(self, url, token):
        parsed = urlparse(url)
        endpoint = parsed.path.lstrip("/")
        _, payload = self._decode_jwt(token)

        if 'list_api' in payload:
            list_api_decoded = self._decode_list_api(payload['list_api']).split(",")
            return endpoint in list_api_decoded
        else:
            list_api_decoded = []
            print("No api list!")
            return False

    def get(self, tickers: Union[str, list[str], None] = None) -> pd.DataFrame:
        retries = 0
        while retries < 5:
            try:
                token = self.token_provider()
                if not self._is_valid_api(self.url, token):
                    raise PermissionError("You do not have permission to access MarketBreadth.")
                
                headers = {
                    "Authorization": f"Bearer {token}",
                    "X-FG-APP": "FiinQuant"
                }

                payload = {
                    "pageSize": 1,
                    "status": 1
                }

                response = requests.get(url=self.url, headers=headers, params=payload, timeout=10)

                if response.status_code == 200:
                    result = response.json()
                    data = result.get("items", [])

                    results = []
                    for item in data:
                        row = {}
                        for k, v in item.items():
                            if k in [
                                "comGroupCode",
                                "totalStockUpPrice",
                                "totalStockDownPrice",
                                "totalStockNoChangePrice",
                                "totalStockUnderFloor",
                                "totalStockOverCeiling"
                            ]:
                                row[k] = v
                            elif k == "tradingDate" and v:
                                try:
                                    dt = datetime.fromisoformat(v)
                                except Exception:
                                    dt = datetime.strptime(v[:19], "%Y-%m-%dT%H:%M:%S")
                                dt_midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
                                row[k] = dt_midnight.strftime("%Y-%m-%d %H:%M")
                        results.append(row)

                    df = pd.DataFrame(results)

                    if tickers:
                        if isinstance(tickers, str):
                            tickers = [tickers]
                        tickers = [t.upper() for t in tickers]
                        df = df[df["comGroupCode"].isin(tickers)]
                    return df

                elif response.status_code in [500, 502, 503, 504, 401]:
                    retries += 1
                    time.sleep(3)
                else:
                    raise ConnectionError(f"Error {response.status_code}: {response.text}")

            except RequestException as e:
                retries += 1
                if retries >= 5:
                    raise ConnectionError(f"Request failed after {5} retries: {e}")
                time.sleep(3)

        raise ConnectionError(f"Failed to fetch data from {self.url} after {5} retries")
