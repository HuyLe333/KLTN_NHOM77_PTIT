import requests
import json
import pandas as pd
from urllib.parse import urlparse
import base64, gzip, time
from typing import Literal, Union
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import asyncio, aiohttp
from ..config.config import *

BASE_URL = {
    "Overview": "https://apigw.fiingroup.vn/FXMA/EquityTrading/PriceStatistic/Overview",
    "Foreign": "https://apigw.fiingroup.vn/FXMA/EquityTrading/PriceStatistic/Foreign",
    "Freefloat": "https://apigw.fiingroup.vn/FXSE/DataExplorer/TradingData/Corporate/Query",
    "CeilingFloor": "https://apigw.fiingroup.vn/FTTE/PriceData/GetPriceData"
}

class PriceStatistics(object):
    def __init__(self, access_token: callable):
        self.df_overview = pd.DataFrame()
        self.df_foreign = pd.DataFrame()
        self.access_token_func = access_token if callable(access_token) else lambda: access_token

        self.base_url = BASE_URL
        self.time_frequency = {
            "Overview": "TimeFilter",
            "Foreign": "TimeFrequency"
        }

        self.fields = {
            "Overview": {
                "Daily": ["ticker", "tradingDateId", "percentPriceChange", "totalMatchVolume", "totalMatchValue", "totalDealVolume", "totalDealValue", "marketCap"],
                "Weekly": ["ticker", "year", "week", "percentPriceChange", "totalMatchVolume", "totalMatchValue", "totalDealVolume", "totalDealValue", "marketCap"],
                "Monthly": ["ticker", "year", "month", "percentPriceChange", "totalMatchVolume", "totalMatchValue", "totalDealVolume", "totalDealValue", "marketCap"],
                "Quarterly": ["ticker", "year", "quarter", "percentPriceChange", "totalMatchVolume", "totalMatchValue", "totalDealVolume", "totalDealValue", "marketCap"],
                "Yearly": ["ticker", "year", "percentPriceChange", "totalMatchVolume", "totalMatchValue", "totalDealVolume", "totalDealValue", "marketCap"]
            },
            "Foreign": {
                "Daily": ["ticker", "tradingDateId", "foreignBuyVolumeTotal", "foreignSellVolumeTotal", "foreignBuyValueTotal", "foreignSellValueTotal", "foreignNetVolumeTotal", "foreignNetValueTotal", "foreignBuyVolumeMatched", "foreignSellVolumeMatched", "foreignBuyValueMatched", "foreignSellValueMatched", "foreignBuyVolumeDeal", "foreignSellVolumeDeal", "foreignBuyValueDeal", "foreignSellValueDeal", "foreignCurrentRoom", "foreignTotalRoom", "percentForeignTotalRoom", "foreignOwned", "percentForeignOwned"],
                "Weekly": ["ticker", "year", "week", "foreignBuyVolumeTotal", "foreignSellVolumeTotal", "foreignBuyValueTotal", "foreignSellValueTotal", "foreignNetVolumeTotal", "foreignNetValueTotal", "foreignBuyVolumeMatched", "foreignSellVolumeMatched", "foreignBuyValueMatched", "foreignSellValueMatched", "foreignBuyVolumeDeal", "foreignSellVolumeDeal", "foreignBuyValueDeal", "foreignSellValueDeal", "foreignCurrentRoom", "foreignTotalRoom", "percentForeignTotalRoom", "foreignOwned", "percentForeignOwned"],
                "Monthly": ["ticker", "year", "month", "foreignBuyVolumeTotal", "foreignSellVolumeTotal", "foreignBuyValueTotal", "foreignSellValueTotal", "foreignNetVolumeTotal", "foreignNetValueTotal", "foreignBuyVolumeMatched", "foreignSellVolumeMatched", "foreignBuyValueMatched", "foreignSellValueMatched", "foreignBuyVolumeDeal", "foreignSellVolumeDeal", "foreignBuyValueDeal", "foreignSellValueDeal", "foreignCurrentRoom", "foreignTotalRoom", "percentForeignTotalRoom", "foreignOwned", "percentForeignOwned"],
                "Quarterly": ["ticker", "year", "quarter", "foreignBuyVolumeTotal", "foreignSellVolumeTotal", "foreignBuyValueTotal", "foreignSellValueTotal", "foreignNetVolumeTotal", "foreignNetValueTotal", "foreignBuyVolumeMatched", "foreignSellVolumeMatched", "foreignBuyValueMatched", "foreignSellValueMatched", "foreignBuyVolumeDeal", "foreignSellVolumeDeal", "foreignBuyValueDeal", "foreignSellValueDeal", "foreignCurrentRoom", "foreignTotalRoom", "percentForeignTotalRoom", "foreignOwned", "percentForeignOwned"],
                "Yearly": ["ticker", "year", "foreignBuyVolumeTotal", "foreignSellVolumeTotal", "foreignBuyValueTotal", "foreignSellValueTotal", "foreignNetVolumeTotal", "foreignNetValueTotal", "foreignBuyVolumeMatched", "foreignSellVolumeMatched", "foreignBuyValueMatched", "foreignSellValueMatched", "foreignBuyVolumeDeal", "foreignSellVolumeDeal", "foreignBuyValueDeal", "foreignSellValueDeal", "foreignCurrentRoom", "foreignTotalRoom", "percentForeignTotalRoom", "foreignOwned", "percentForeignOwned"]
            },
            "CeilingFloor": ["code", "tradingDate", "floorValue", "ceilingValue", "iNav", "iIndex",
                            "totalTrade", "totalBuyTrade", "totalSellTrade", "totalBuyTradeVolume",
                            "totalSellTradeVolume", "shareIssue", "foreignIndividualBuyTradingMatchValue",
                            "foreignInstitutionalBuyTradingMatchValue", "foreignIndividualSellTradingMatchValue",
                            "foreignInstitutionalSellTradingMatchValue", "foreignIndividualBuyTradingMatchVolume",
                            "foreignInstitutionalBuyTradingMatchVolume", "foreignIndividualSellTradingMatchVolume",
                            "foreignInstitutionalSellTradingMatchVolume", "localIndividualBuyValue", "localIndividualBuyVolume",
                            "localIndividualSellVolume", "localIndividualSellValue", "localInstitutionalBuyVolume",
                            "localInstitutionalBuyValue", "localInstitutionalSellVolume", "localInstitutionalSellValue",
                            "localIndividualBuyMatchVolume", "localIndividualBuyMatchValue", "localIndividualSellMatchVolume",
                            "localIndividualSellMatchValue", "localInstitutionalBuyMatchVolume", "localInstitutionalBuyMatchValue",
                            "localInstitutionalSellMatchVolume", "localInstitutionalSellMatchValue", "netProprietaryMatchVolume",
                            "netProprietaryMatchValue", "netInstitutionMatchVolume", "netInstitutionMatchValue",
                            "totalMatchBuyTradeVolume", "totalMatchBuyTradeValue", "totalMatchSellTradeValue",
                            "totalMatchSellTradeVolume", "proprietaryTotalMatchBuyTradeVolume", "proprietaryTotalMatchBuyTradeValue",
                            "proprietaryTotalMatchSellTradeValue", "proprietaryTotalMatchSellTradeVolume",
                            "proprietaryTotalBuyTradeVolume", "proprietaryTotalBuyTradeValue", "proprietaryTotalSellTradeValue",
                            "proprietaryTotalSellTradeVolume", "proprietaryTotalDealBuyTradeVolume", "proprietaryTotalDealBuyTradeValue",
                            "proprietaryTotalDealSellTradeValue", "proprietaryTotalDealSellTradeVolume"]
        }

        self.organization_mapping = ticker_org

    def _decode_jwt(self, token):
        header_b64, payload_b64, _ = token.split('.')
        header = json.loads(base64.urlsafe_b64decode(header_b64 + '=='))
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + '=='))
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
            print("No api list!")
            return False
        
    def _process_tradingdate(self, column: pd.Series, timezone: str = None) -> pd.Series:
        utc_time = pd.to_datetime(column, utc=True)
        utc_time = utc_time.dt.tz_convert(timezone) if timezone else utc_time
        timestamp = utc_time.dt.strftime("%Y-%m-%d %H:%M")
        return timestamp

    async def _get_each_page_overview_foreign_data_async(
        self,
        session: aiohttp.ClientSession,
        organization_id: int,
        type: str,
        time_filter: Literal["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"],
        from_date: str,
        to_date: str,
        page: int = 1,
        get_total_page: bool = False,
    ) -> Union[pd.DataFrame, dict]:
        base_url = self.base_url[type]
        token = self.access_token_func()
        if not self._is_valid_api(base_url, token):
            raise PermissionError(f"You do not have permission to access {type}.")

        payloads = {
            "OrganizationId": organization_id,
            self.time_frequency[type]: time_filter,
            "StartDate": from_date,
            "EndDate": to_date,
            "SortField": "tradingDateId",
            "SortOrder": 0,
            "Page": page,
            "PageSize": 100
        }

        headers = {
            "Authorization": f"Bearer {token}",
            "X-FG-APP": "FiinQuant"
        }

        while True:
            async with session.get(url=base_url, headers=headers, params=payloads) as response:
                if response.status == 200:
                    result = await response.json()
                    data = result.get("data", [])
                    if data:
                        TotalPage = result["totalPage"]
                        fields = self.fields[type][time_filter]
                        cleaned_data = [
                            {k: v for k, v in item.items() if k in fields}
                            for item in data
                        ]
                        df_cleaned_data = pd.DataFrame(cleaned_data)

                        if time_filter == "Daily":
                            df_cleaned_data["timestamp"] = self._process_tradingdate(df_cleaned_data["tradingDateId"], timezone="Asia/Bangkok")
                            if type == "Overview":
                                df_cleaned_data = df_cleaned_data[
                                    ["ticker", "timestamp", "percentPriceChange",
                                     "totalMatchVolume", "totalMatchValue",
                                     "totalDealVolume", "totalDealValue", "marketCap"]
                                ]
                            else:
                                df_cleaned_data = df_cleaned_data[
                                    ["ticker", "timestamp",
                                     "foreignBuyVolumeTotal", "foreignSellVolumeTotal",
                                     "foreignBuyValueTotal", "foreignSellValueTotal",
                                     "foreignNetVolumeTotal", "foreignNetValueTotal",
                                     "foreignBuyVolumeMatched", "foreignSellVolumeMatched",
                                     "foreignBuyValueMatched", "foreignSellValueMatched",
                                     "foreignBuyVolumeDeal", "foreignSellVolumeDeal",
                                     "foreignBuyValueDeal", "foreignSellValueDeal",
                                     "foreignCurrentRoom", "foreignTotalRoom",
                                     "percentForeignTotalRoom", "foreignOwned",
                                     "percentForeignOwned"]
                                ]

                        if get_total_page:
                            return {"total_page": TotalPage, "data": df_cleaned_data}
                        return df_cleaned_data
                    else:
                        if time_filter == "Daily":
                            if type == "Overview":
                                overview_columns = ["ticker", "timestamp", "percentPriceChange",
                                                    "totalMatchVolume", "totalMatchValue",
                                                    "totalDealVolume", "totalDealValue", "marketCap"]
                                df_cleaned_data = pd.DataFrame(columns=overview_columns)
                            else:
                                foreign_columns = ["ticker", "timestamp",
                                                    "foreignBuyVolumeTotal", "foreignSellVolumeTotal",
                                                    "foreignBuyValueTotal", "foreignSellValueTotal",
                                                    "foreignNetVolumeTotal", "foreignNetValueTotal",
                                                    "foreignBuyVolumeMatched", "foreignSellVolumeMatched",
                                                    "foreignBuyValueMatched", "foreignSellValueMatched",
                                                    "foreignBuyVolumeDeal", "foreignSellVolumeDeal",
                                                    "foreignBuyValueDeal", "foreignSellValueDeal",
                                                    "foreignCurrentRoom", "foreignTotalRoom",
                                                    "percentForeignTotalRoom", "foreignOwned",
                                                    "percentForeignOwned"]
                                df_cleaned_data = pd.DataFrame(columns=foreign_columns)
                        if get_total_page:
                            return {"total_page": 0, "data": df_cleaned_data}
                        return df_cleaned_data

                elif response.status == 403:
                    response_json = await response.json()
                    message = "".join(response_json.get("Errors", []))
                    raise PermissionError(message)

                elif response.status == 429:
                    await asyncio.sleep(2)
                    continue

                else:
                    raise ConnectionError(f"Error {response.status}: {await response.text()} with url {response.url}")

    async def _get_each_page_ceiling_floor_data_async(self, ticker: str, session: aiohttp.ClientSession, from_date: str, to_date: str, page: int, get_total_page: bool = False) -> Union[list[dict], dict]:
        base_url = self.base_url["CeilingFloor"]
        token = self.access_token_func()
        if not self._is_valid_api(base_url, token):
            raise PermissionError(f"You do not have permission to access {type}.")
        headers = {
            "Authorization": f"Bearer {token}",
            "X-FG-APP": "FiinQuant"
        }
        
        payloads = {
            "Code": ticker,
            "Frequently": "Daily",
            "From": from_date,
            "To": to_date,
            "Page": page,
            "PageSize": 60
        }
        while True:
            async with session.get(url=base_url, headers=headers, params=payloads) as response:
                if response.status == 200:
                    result = await response.json()
                    data = result.get("items", [])
                    TotalPage = 0
                    cleaned_data = []
                    if data:
                        TotalCount = result["totalCount"]
                        TotalPage = TotalCount // 60 + 1
                        cleaned_data = [
                            {k: v for k, v in item.items() if k in self.fields["CeilingFloor"]}
                            for item in data
                        ]

                    if get_total_page:
                        return {"TotalPage": TotalPage, "Data": cleaned_data}
                    return cleaned_data
                elif response.status == 403:
                    response_json = await response.json()
                    message = "".join(response_json.get("Errors", []))
                    raise PermissionError(message)
                elif response.status == 429:
                    await asyncio.sleep(2)
                    continue
                else:
                    raise ConnectionError(f"Error {response.status}: {await response.text()} with url {response.url}")

    async def _get_each_ticker_overview_foreign_data_async(self, session: aiohttp.ClientSession, organization_id: int, time_filter: Literal["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"], from_date: str, to_date: str, type: str) -> pd.DataFrame:
        first_result = await self._get_each_page_overview_foreign_data_async(session, organization_id=organization_id, type=type, time_filter=time_filter, from_date=from_date, to_date=to_date, get_total_page=True)
        df = first_result["data"]
        total_page = first_result["total_page"]
        if total_page > 1:
            tasks = [
                self._get_each_page_overview_foreign_data_async(session, organization_id=organization_id, type=type, time_filter=time_filter, from_date=from_date, to_date=to_date, page=page)
                for page in range(2, total_page + 1)
            ]
            sub_dfs = await asyncio.gather(*tasks, return_exceptions=True)
            for sub_df in sub_dfs:
                if isinstance(sub_df, pd.DataFrame):
                    df = pd.concat([df, sub_df], ignore_index=True)
                else:
                    print(f"Error fetching page: {sub_df}")
            df = df.reset_index(drop=True)

        return df

    async def _get_each_ticker_ceiling_floor_data_async(self, ticker: str, session: aiohttp.ClientSession, from_date: str, to_date: str) -> pd.DataFrame:
        first_result = await self._get_each_page_ceiling_floor_data_async(ticker=ticker, session=session, from_date=from_date, to_date=to_date, page=1, get_total_page=True)
        price_data = first_result["Data"]
        TotalPage = first_result["TotalPage"]
        if TotalPage > 1:
            tasks = [
                self._get_each_page_ceiling_floor_data_async(ticker=ticker, session=session, from_date=from_date, to_date=to_date, page=page)
                for page in range(2, TotalPage + 1)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, list):
                    price_data.extend(res)
                else:
                    print(f"Error fetching page: {res}")
        return price_data

    async def _fetch_all_overview_foreign_tickers(self, tickers: list, time_filter: Literal["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"], from_date: str, to_date: str, type: str) -> pd.DataFrame:
        async with aiohttp.ClientSession() as session:
            tasks = []
            for ticker in tickers:
                if ticker not in self.organization_mapping:
                    print(f"Mã {ticker} không có trong danh sách lấy dữ liệu")
                    continue
                tasks.append(
                    self._get_each_ticker_overview_foreign_data_async(
                        session,
                        organization_id=self.organization_mapping.get(ticker),
                        time_filter=time_filter,
                        from_date=from_date,
                        to_date=to_date,
                        type=type
                    )
                )

            results = await asyncio.gather(*tasks, return_exceptions=True)
            df = pd.DataFrame()
            for ticker, result in zip(tickers, results):
                if isinstance(result, pd.DataFrame):
                    try:
                        df = pd.concat([df, result], ignore_index=True)
                    except Exception as e:
                        print(result)
                        raise IndexError(e)
                else:
                    print(f"Error fetching ticker data for {ticker}: {result}")
            return df.reset_index(drop=True)

    async def _fetch_all_ceiling_floor_tickers(self, tickers: list, from_date: str, to_date: str) -> pd.DataFrame:

        async with aiohttp.ClientSession() as session:
            tasks = [
                self._get_each_ticker_ceiling_floor_data_async(ticker=ticker, session=session, from_date=from_date, to_date=to_date)
                for ticker in tickers
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            price_data = []
            for res in results:
                if isinstance(res, list):
                    price_data.extend(res)
                else:
                    print(f"Error fetching ticker data: {res}")
            df = pd.DataFrame(price_data)
            if not df.empty:
                df["tradingDate"] = self._process_tradingdate(df["tradingDate"])
                df = df.rename(columns={"tradingDate": "timestamp"})
            return df

    def get_overview(self, tickers:Union[list,str], time_filter: Literal["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"] = "Daily", from_date: str = str((datetime.now() - relativedelta(years=2)).year) + "-01-01", to_date: str = datetime.now().strftime("%Y-%m-%d")) -> pd.DataFrame:
        if isinstance(tickers, str):
            tickers = [tickers]
        return asyncio.run(self._fetch_all_overview_foreign_tickers(tickers, time_filter, from_date, to_date, type="Overview"))
    
    def get_foreign(self, tickers:Union[list,str], time_filter: Literal["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"] = "Daily", from_date: str = str((datetime.now() - relativedelta(years=2)).year) + "-01-01", to_date: str = datetime.now().strftime("%Y-%m-%d")) -> pd.DataFrame:
        if isinstance(tickers, str):
            tickers = [tickers]
        return asyncio.run(self._fetch_all_overview_foreign_tickers(tickers, time_filter, from_date, to_date, type="Foreign"))
    
    def get_ceilingfloor(self, tickers:Union[list,str], from_date: str, to_date: str = datetime.now().strftime("%Y-%m-%d")) -> pd.DataFrame:
        if isinstance(tickers, str):
            tickers = [tickers]
        return asyncio.run(self._fetch_all_ceiling_floor_tickers(tickers=tickers, from_date=from_date, to_date=to_date))

    def get_freefloat(self, tickers: Union[list,str], from_date: str, to_date = datetime.now().strftime("%Y-%m-%d")) -> pd.DataFrame:
        if isinstance(tickers, str):
            tickers = [tickers]
        base_url = self.base_url["Freefloat"]
        token = self.access_token_func()
        if not self._is_valid_api(base_url, token):
            raise PermissionError("You do not have permission to access Freefloat")

        from_date = (datetime.strptime(from_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%dT17:00:00Z")
        to_date += "T16:59:59Z"

        data_ids = [
            {"id": self.organization_mapping[ticker], "dataType": "Corporate"}
            for ticker in tickers
        ]
        
        headers = {
            "Authorization": f"Bearer {token}",
            "X-FG-APP": "FiinQuant",
            "Content-Type": "application/json"
        }
        payload = {
            "dataIds": data_ids,
            "conditionGroups": [
                {
                    "groupId": "297d4165-8b93-3425-9381-755068579595",
                    "order": 0,
                    "conditionType": "DE04_SC01",
                    "condition": {"timeRange": "Specific", "fromDate": from_date, "toDate": to_date},
                    "indicators": [{"order": 0, "indicatorId": 30045, "alias": 30045, "indicatorType": "", "condition": {}}]
                }
            ]
        }
        
        retries = 0
        while retries < 5:
            try:
                response = requests.post(url=base_url, headers=headers, json=payload, timeout=15)
                if response.status_code == 200:
                    result = response.json()
                    data = result.get("data", [])
                    cleaned_data = [
                        {"ticker": item["ticker"], "tradingdate": item["tradingDate"], "freefloat": item["items"]["30045"]}
                        for item in data
                    ]
                    df_cleaned_data = pd.DataFrame(cleaned_data)
                    if not df_cleaned_data.empty:
                        df_cleaned_data["timestamp"] = self._process_tradingdate(df_cleaned_data["tradingdate"], timezone="Asia/Bangkok")
                        df_cleaned_data = df_cleaned_data[["ticker", "timestamp", "freefloat"]]
                    return df_cleaned_data

                elif response.status_code in [500, 502, 503, 504, 429]:
                    retries += 1
                    time.sleep(3)
                elif response.status_code == 403:
                    raise PermissionError("You do not have permission to access Freefloat")
                else:
                    raise ConnectionError(f"Error {response.status_code}: {response.text}")

            except requests.exceptions.RequestException as e:
                retries += 1
                if retries >= 5:
                    raise ConnectionError(f"Request failed after {5} retries: {e}")
                time.sleep(3)

        raise ConnectionError(f"Failed to fetch Freefloat data after {5} retries")

