# This code is encrypted and does not contain viruses or Trojans.
from typing import Union, List, Optional, Literal

from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import requests
import pandas as pd
import json
import base64
import gzip
from urllib.parse import urlparse
from datetime import datetime
from ..config.config import *
import time
warnings.filterwarnings("ignore")


RATIOS_URL="https://apigw.fiingroup.vn/FXPF/Corporate/GetFinancialDataRatio"

INCOMESTATEMENT_URL = "https://apigw.fiingroup.vn/FXPF/Corporate/GetFinancialDataIncomeStatement"

FA_URL = "https://apigw.fiingroup.vn/FQBV/"

# FA_URL = "http://localhost:8081/"

class FundamentalAnalysis:
    def __init__(self, access_token: callable) -> None:
        self.url = FA_URL
        self.ratios_url = RATIOS_URL
        self.income_statement_url = INCOMESTATEMENT_URL
        self.access_token_func = access_token if callable(access_token) else lambda: access_token
        self.package_list = self._get_package()
        self.valid_fields = next((package_ratio_infor[k] for k in package_ratio_infor if k in self.package_list), []) if "FiinQuantFARatio.Advance" not in self.package_list else package_ratio_infor["FiinQuantFARatio.Advance"]
        self.ticker_org = ticker_org
        self.ratio_map = ratio_map

    def _get_header(self):
        return {
            'Authorization': f'Bearer {self.access_token_func()}',
            'X-FG-APP': 'FiinQuant'
        }


    # def __fetch_data(
    #     self,
    #     statement: Literal['balancesheet', 'incomestatement', 'cashflow', 'full'],
    #     tickers: List[str],
    #     years: Union[int, List[int]],
    #     audited: bool,
    #     type: str,
    #     quarters: Union[List[int], int, None],
    #     sections_str: Optional[str],
    #     max_retries: int = 5,
    #     backoff: int = 3
    # ):
    #     params = {
    #         'tickers': ','.join(tickers),
    #         'years': str(years) if isinstance(years, int) else ','.join(str(y) for y in years),
    #         'audited': audited,
    #         'type': type.lower()
    #     }
    #     if statement == "balancesheet":
    #         api_statement = "GetBalanceSheet"
    #     elif statement == "incomestatement":
    #         api_statement = "GetIncomeStatement"
    #     elif statement == "cashflow":
    #         api_statement = "GetCashFlow"
    #     elif statement == "full":
    #         api_statement = "GetFull"
    #     else:
    #         raise ValueError(f"Invalid statement type: {statement}")

    #     if quarters:
    #         params['quarters'] = str(quarters) if isinstance(quarters, int) else ','.join(str(q) for q in quarters)
    #     if sections_str:
    #         params['sections'] = sections_str

    #     retries = 0
    #     while retries < max_retries:
    #         try:
    #             header = self._get_header()
    #             response = requests.get(
    #                 f"{self.url}BVFinancialStatement/{api_statement}",
    #                 params=params,
    #                 headers=header,
    #                 verify=False,
    #                 timeout=30
    #             )
                
    #             status = response.status_code
    #             if status == 200:
    #                 return response.json().get("data", [])

    #             elif status == 400:
    #                 message = response.json().get("message")
    #                 raise ValueError(f"[400]: {message}")

    #             elif status == 401:
    #                 message = response.json().get("message")
    #                 raise ValueError(f"[401]: {message}")

    #             elif status == 403:
    #                 try:
    #                     message = response.json().get("message") 
    #                     if "Some provided tickers are not part of VN30" in message:
    #                         invalidTickers = response.json().get("invalidTickers")
    #                         allowedVN30Tickers = response.json().get("allowedVN30Tickers")
    #                         raise ValueError(f"[403]: {message}, 'invalidTickers': {invalidTickers}, 'allowedVN30Tickers':{allowedVN30Tickers}")
    #                     else:
    #                         raise ValueError(f"[403]: {message}")
    #                 except:
    #                     message = response.json()
    #                     raise ValueError(f"[403]: {message}")

    #             elif status in [429, 500, 502, 503, 504]:
    #                 retries += 5
    #                 time.sleep(backoff)
    #                 continue

    #             else:
    #                 raise ValueError(f"Unexpected error [{status}]: {response.text}")

    #         except requests.exceptions.RequestException as e:
    #             retries += 1
    #             if retries >= max_retries:
    #                 raise ValueError(f"Request failed after {max_retries} retries: {e}")
    #             time.sleep(backoff)

    #     raise ConnectionError(f"Failed to fetch {statement} after {max_retries} retries")


    def __fetch_data(
        self,
        statement: str,
        tickers: list,
        years: Union[int, list],
        audited: bool,
        type: str,
        quarters: Union[List[int], int, None],
        sections_str: Optional[str],
        batch_size: int = 50,
        max_workers: int = 5,
        max_retries: int = 5,
        backoff: int = 3
    ):
        def fetch_batch(batch):
            params = {
                'tickers': ','.join(batch),
                'years': str(years) if isinstance(years, int) else ','.join(str(y) for y in years),
                'audited': audited,
                'type': type.lower()
            }
            if statement == "balancesheet":
                api_statement = "GetBalanceSheet"
            elif statement == "incomestatement":
                api_statement = "GetIncomeStatement"
            elif statement == "cashflow":
                api_statement = "GetCashFlow"
            elif statement == "full":
                api_statement = "GetFull"
            else:
                raise ValueError(f"Invalid statement type: {statement}")

            if quarters:
                params['quarters'] = str(quarters) if isinstance(quarters, int) else ','.join(str(q) for q in quarters)
            if sections_str:
                params['sections'] = sections_str

            retries = 0
            while retries < max_retries:
                try:
                    header = self._get_header()
                    response = requests.get(
                        f"{self.url}BVFinancialStatement/{api_statement}",
                        params=params,
                        headers=header,
                        verify=False,
                        timeout=30
                    )
                    if response.status_code == 200:
                        return response.json().get("data", [])
                    elif response.status_code in [429, 500, 502, 503, 504]:
                        retries += 1
                        time.sleep(backoff)
                        continue
                    else:
                        response.raise_for_status()
                except requests.exceptions.RequestException:
                    retries += 1
                    time.sleep(backoff)
            raise ValueError(f"Failed batch {batch} after {max_retries} retries")

        batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {executor.submit(fetch_batch, b): b for b in batches}
            for future in as_completed(future_to_batch):
                try:
                    results.extend(future.result())
                except Exception as e:
                    print(f"{future_to_batch[future]}: {e}")

        return results


    def __process_statement(
            self,
            statement: Literal['balancesheet', 'incomestatement', 'cashflow', 'full'],
            tickers: Union[str, list],
            years: Union[int, List[int]],
            type: Literal['consolidated', 'separate'],
            audited: bool,
            quarters: Union[List[int], int, None] = None,
            fields: Union[str, list, None] = None):
        
        if type not in ['consolidated', 'separate']:
            raise ValueError("'type' must be 'consolidated' or 'separate'.")

        if statement.lower() not in ['balancesheet', 'incomestatement', 'cashflow', 'full']:
            raise ValueError("statement must be one of 'balancesheet', 'incomestatement', 'cashflow', or 'full'.")
    
        if isinstance(tickers, str):
            tickers = [t.strip() for t in tickers.split(',')]
        else:
            tickers = [t.strip() for t in tickers]

        sections_str = ','.join(fields) if isinstance(fields, list) else fields
        years = [years] if isinstance(years, int) else years
        if quarters is None:
            quarter_list = [None]
        elif isinstance(quarters, int):
            quarter_list = [quarters]
        elif isinstance(quarters, (list, tuple, set)):
            quarter_list = list(quarters)
        else:
            raise ValueError("quarters must be None, int, or list of int")
        all_results = {
            f"{ticker}_{year}_{quarter or 0}": {
                "ticker": ticker,
                "year": year,
                "quarter": quarter,
                "reportType": type.lower(),
                "audited": audited,
                "financialStatement": {}
            }
            for ticker in tickers
            for year in years
            for quarter in quarter_list
        }

        data = self.__fetch_data(statement, tickers, years, audited, type, quarters, sections_str)

        fs_type_map = {
            'balancesheet': 'balanceSheet',
            'incomestatement': 'incomeStatement',
            'cashflow': 'cashFlow'
        }
        api_key_map = {
            'balancesheet': 'balanceSheet',
            'incomestatement': 'incomeStatement',
            'cashflow': 'cashFlows',  
            'full': 'financialStatement'  
        }

        for item in data:
            ticker = item.get('ticker')
            year = item.get('year')
            quarter = item.get('quarter')
            report_key = f"{ticker}_{year}_{quarter or 0}"

            if report_key not in all_results:
                continue  
            result = all_results[report_key]

            cleaned_item = item.copy()
            for key in ['ticker', 'year', 'reportType', 'audited', 'companyType', 'quarter', 'businessTypeId', 'financialStatement']:
                cleaned_item.pop(key, None)

            if sections_str:
                result.pop("financialStatement")
                result.update(cleaned_item)
            else:
                if statement.lower() == 'full':
                    result["financialStatement"] = {}
                    for api_key, result_key in [('balancesheet', 'balanceSheet'), ('incomestatement', 'incomeStatement'), ('cashflow', 'cashFlow')]:
                        if api_key in cleaned_item and cleaned_item[api_key]:
                            result["financialStatement"][result_key] = [cleaned_item[api_key]]
                        else:
                            result["financialStatement"][result_key] = []
                else:
                    fs_type = fs_type_map[statement.lower()]
                    api_key = api_key_map[statement.lower()]
                    result["financialStatement"] = {}
                    if api_key in cleaned_item and cleaned_item[api_key]:
                        result["financialStatement"][fs_type] = [cleaned_item[api_key]]
                    else:
                        result["financialStatement"][fs_type] = []

            if "companyType" in item:
                result["companyType"] = item["companyType"]
            if "businessTypeId" in item:
                result["businessTypeId"] = item["businessTypeId"]

        for result in all_results.values():
            if "financialStatement" in result:
                empty_keys = [k for k, v in result["financialStatement"].items() if not v]
                for k in empty_keys:
                    del result["financialStatement"][k]
                if not result["financialStatement"]:
                    del result["financialStatement"]

        result_list = list(all_results.values())

        return result_list
    
    def get_financial_statement(
        self,
        statement: Literal['balancesheet', 'incomestatement', 'cashflow', 'full'],
        tickers: Union[str, list],
        years: Union[int, List[int]],
        type: Literal['consolidated', 'separate'],
        audited: Union[bool, None] = None,
        quarters: Union[List[int], int, None] = None,
        fields: Union[str, list, None] = None):

        result = []
        if audited is None:
            result_unaudited = self.__process_statement(statement, tickers, years, type, audited=False, quarters=quarters, fields=fields)
            result_audited = self.__process_statement(statement, tickers, years, type, audited=True, quarters=quarters, fields=fields)
            result = result_unaudited + result_audited
        else:
            result = self.__process_statement(statement, tickers, years, type, audited=audited, quarters=quarters, fields=fields)

        required_keys = {"ticker", "year", "quarter", "reportType", "audited"}
        cleaned_result = []
        for item in result:
            item_keys = set(item.keys())
            if item_keys == required_keys:
                print(f"[Warning] No actual data for {item['ticker']} - Q{item['quarter'] or '-'} {item['year']} "
                    f"({'audited' if item['audited'] else 'unaudited'})")
                continue
            cleaned_result.append(item)

        return cleaned_result
    
    #Ratios
    def _get_package(self):
            token = self.access_token_func()
            _, payload = self._decode_jwt(token)
            if 'list_package' in payload:
                list_package_decoded = payload["list_package"].split(",")
            else:
                list_package_decoded = []
                print("No package list!")
            return list_package_decoded
    
    def _decode_jwt(self, token):
        header_b64, payload_b64, _ = token.split('.')
        header = json.loads(base64.urlsafe_b64decode(header_b64 + '==').decode('utf-8'))
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + '==').decode('utf-8'))
        return header, payload

    def _decode_list_api(self, compressed_data):
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

    def _filter_ratios(self, data: dict, fields: list):
        result = {
            "organizationId": data.get("organizationId"),
            "ticker": data.get("ticker"),
            "year": data.get("year"),
            "quarter": data.get("quarter"),
        }

        ratio_keys = set(data.get("ratios", {}).keys())
        if all(f in ratio_keys for f in fields):
            result["ratios"] = {k: v for k, v in data["ratios"].items() if k in fields}
            return result

        for f in fields:
            if "." in f:  
                cat, key = f.split(".", 1)
                if cat in data.get("ratios", {}) and key in data["ratios"][cat]:
                    result[key] = data["ratios"][cat][key]
            else:
                for cat, ratios in data.get("ratios", {}).items():
                    if f in ratios:
                        result[f] = ratios[f]
                        break

        return result

    def _fetch_ratio_data(
        self,
        OrganizationId: int,
        TimeFilter: Literal["Yearly", "Quarterly"],
        NumberOfPeriod: int,
        LatestYear: int,
        Consolidated: bool,
        ratio_map: dict,
        fields: Optional[list] = None
    ) -> list[dict]:
        payloads = {
            "OrganizationId": OrganizationId,
            "TimeFilter": TimeFilter,
            "NumberOfPeriod": NumberOfPeriod,
            "LatestYear": LatestYear,
            "Consolidated": Consolidated
        }
        token = self.access_token_func()

        if not self._is_valid_api(self.ratios_url, token):
            raise PermissionError("You do not have permission to access this API (Ratios).")

        headers = {
            "Authorization": f"Bearer {token}",
            "X-FG-APP": "FiinQuant"
        }

        response_ratios = requests.get(url=self.ratios_url, headers=headers, params=payloads)
        response_income_statement = requests.get(url=self.income_statement_url, headers=headers, params=payloads)

        if response_ratios.status_code != 200:
            raise ConnectionError(f"Error {response_ratios.status_code}: {response_ratios.text}")
        elif response_income_statement.status_code != 200:
            print(f"Warning {response_income_statement.status_code}: {response_income_statement.text}")

        result_ratios = response_ratios.json()
        result_income_statement = response_income_statement.json()
        data_ratios = result_ratios.get("data", [])
        data_income_statement = result_income_statement.get("data", [])
        cleaned_data = []

        for ratio_item, income_statemt_item in zip(data_ratios, data_income_statement):
            if ratio_item["year"] >= datetime.now().year - 5 and income_statemt_item["year"] >= datetime.now().year - 5:
                ratio_result = {"ratios": {}}
                for k, v in ratio_item.items():
                    if not v:
                        continue
                    
                    if (self.valid_fields == "All" or k.upper() in self.valid_fields) and k not in ["isTTM", "organizationId", "ticker", "year", "quarter"]:
                        ratio_name = ratio_map[k.upper()]["indicator"]
                        parent = ratio_map[k.upper()]["parent"]
                        if parent not in ratio_result["ratios"].keys():
                            ratio_result["ratios"][parent] = {}
                        ratio_result["ratios"][parent][ratio_name] = v
                    elif k in ["isTTM", "organizationId", "ticker", "year", "quarter"]:
                        ratio_result[k] = v
                    else:
                        continue
                if income_statemt_item["iS1"] or income_statemt_item["y"]:
                    ratio_result["ratios"]["Revenue"] = income_statemt_item["iS1"] if income_statemt_item["iS1"] else income_statemt_item["y"]
                else:
                    ratio_result["ratios"]["NetInterestIncome"] = income_statemt_item["iS27"]
                
                ratio_result["ratios"]["AttributeToParentCompany"] = income_statemt_item["iS21"]
                EBIT_calc_values = [income_statemt_item.get(k) for k in ["iS5", "iS10", "iS11"]]
                if all(v is not None for v in EBIT_calc_values):
                    ratio_result["ratios"]["EBIT"] = sum(EBIT_calc_values)
                if fields:
                    ratio_result = self._filter_ratios(data=ratio_result, fields=fields)

                cleaned_data.append(ratio_result)
        
        for item in cleaned_data:
            if "ratios" in item and not item["ratios"]:
                item.pop("ratios", None)
            
            if TimeFilter == "Yearly":
                item.pop("quarter", None)
                item.pop("organizationId", None)
            else:
                item.pop("organizationId", None)  
        return cleaned_data

    def _process_ratio(
        self,
        OrganizationId: int,
        years: list,
        quarters: Optional[list],
        Consolidated: bool,
        ratio_map: dict,
        fields: Optional[list] = None
    ) -> list[dict]:
        results = []

        if not quarters:
            number_of_periods = max(years) - min(years) + 1
            data = self._fetch_ratio_data(OrganizationId=OrganizationId, TimeFilter="Yearly", NumberOfPeriod=number_of_periods, LatestYear=max(years), Consolidated=Consolidated, ratio_map=ratio_map, fields=fields)
            results.extend([d for d in data if d.get("year") in years])
        else:
            number_of_periods = (max(years) - min(years) + 1) * 4
            data = self._fetch_ratio_data(OrganizationId=OrganizationId, TimeFilter="Quarterly", NumberOfPeriod=number_of_periods, LatestYear=max(years), Consolidated=Consolidated, ratio_map=ratio_map, fields=fields)
            results.extend([d for d in data if d.get("year") in years and d.get("quarter") in quarters])

        return results

    def get_ratios(
        self,
        tickers: Union[list[str], str],
        years: list,
        quarters: Optional[list] = None,
        type: Literal["consolidated", "separate"] = "consolidated",
        fields: Optional[list] = None
    ) -> list[dict]:
        if isinstance(tickers, str):
            tickers = [tickers]

        def fetch_one(ticker: str):
            organization_id = self.ticker_org[ticker]
            return self._process_ratio(
                OrganizationId=organization_id,
                years=years,
                quarters=quarters,
                Consolidated=(type == "consolidated"),
                ratio_map=self.ratio_map,
                fields=fields
            )

        results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(fetch_one, t): t for t in tickers}
            for future in as_completed(futures):
                try:
                    results.extend(future.result())
                except Exception as e:
                    ticker = futures[future]
                    print(f"Error fetching ratios for {ticker}: {e}")

        return results

