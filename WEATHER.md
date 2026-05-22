针对 **山东黄瓜价格预测** 的气温数据方案，采用 **多城市平均法** 来代表全省气温状况。该方案能更好地捕捉山东省内主要蔬菜产区和消费区的气候特征，尤其覆盖了 **潍坊寿光** 这一核心蔬菜基地，同时实现简单、数据量小，完全适合单机一天完成。

---

### 一、代表城市选择与理由
选取以下 4 个城市（涵盖鲁中、鲁西北、鲁南、胶东）：
| 城市 | 经纬度 | 代表意义 |
|------|--------|----------|
| 济南 | 36.67°N, 116.98°E | 省会，鲁中集散地 |
| 潍坊 | 36.71°N, 119.16°E | **寿光所在地**，全国最大蔬菜生产与交易中心 |
| 临沂 | 35.10°N, 118.35°E | 鲁南重要农产品产区 |
| 烟台 | 37.54°N, 121.39°E | 胶东沿海，影响物流及部分蔬菜供应 |

```python
import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": <LAT HERE>,
	"longitude": <LAT HERE>,
	"start_date": "2022-01-01",
	"end_date": "2026-04-30",
	"hourly": ["temperature_2m", "rain"],
}
responses = openmeteo.weather_api(url, params = params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_rain = hourly.Variables(1).ValuesAsNumpy()

hourly_data = {
	"date": pd.date_range(
		start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
		end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = hourly.Interval()),
		inclusive = "left"
	)
}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["rain"] = hourly_rain

hourly_dataframe = pd.DataFrame(data = hourly_data)
print("\nHourly data\n", hourly_dataframe)
```
