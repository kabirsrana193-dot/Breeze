from breeze_connect import BreezeConnect
import datetime

app_key = "68`47N89970w1dH7u1s5347j8403f287"
secret_key = "5v9k141093cf4361528$z24Q7(Yv2839"
session_token = "53705299"



breeze = BreezeConnect(api_key=app_key)

# Generate Session

breeze.generate_session(api_secret=secret_key,
                        session_token=session_token)




iso_date_string = datetime.datetime.strptime("28/02/2021","%d/%m/%Y").isoformat()[:10] + 'T05:30:00.000Z'
iso_date_time_string = datetime.datetime.strptime("28/02/2021 23:59:59","%d/%m/%Y %H:%M:%S").isoformat()[:19] + '.000Z'


breeze.get_historical_data_v2(interval="30minute",
                    from_date= "2025-02-03T09:20:00.000Z",
                    to_date= "2025-02-13T09:22:00.000Z",
                    stock_code="RELIND",
                    exchange_code="NSE",
                    product_type="cash")

breeze.get_portfolio_holdings(exchange_code="NFO",
                    from_date="2025-10-14T06:00:00.000Z",
                    to_date="2025-11-15T06:00:00.000Z",
                    stock_code="NIFTY",
                    portfolio_type="")
