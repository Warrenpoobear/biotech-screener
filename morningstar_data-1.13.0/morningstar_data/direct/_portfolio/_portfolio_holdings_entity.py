from __future__ import annotations

from typing import Any, Dict, List, Optional

import simplejson as json
from pandas import DataFrame

from ..._base import _logger
from .._config import _Config
from .._error_messages import (
    BAD_REQUEST_ERROR_NO_PORTFOLIO_HOLDING_ID,
    BAD_REQUEST_INVALID_PORTFOLIO_HOLDING_DATA_FRAME,
    BAD_REQUEST_PORTFOLIO_WEIGHT_ERROR,
)
from .._exceptions import BadRequestException
from .._utils import _format_date
from ._common import PortfolioDataApiBackend, PortfolioJSONEncoder

_portfolio_api = PortfolioDataApiBackend()
_config = _Config()


class Holding:
    holding_id: str
    weight: Optional[float]
    market_value: float
    quantity: float
    price: float

    def __init__(self, holding_id: str, weight: Optional[float] = None) -> None:
        self.holding_id = holding_id
        self.weight = weight


class HoldingsEntity:
    holdings: List[Holding]
    portfolio_date: str
    total_amount: float = 10000

    def __init__(self, holdings: List, portfolio_date: str, total_amount: float = 10000) -> None:
        self.holdings = holdings
        self.total_amount = total_amount
        self.portfolio_date = _format_date(portfolio_date)

        self._validate_holdings_id_and_weight()
        self._validate_holdings_total_weight()

    def _validate_holdings_id_and_weight(self) -> None:
        """
        Validate portfolio holdings must have id and weight.
        """
        all_weight_is_none = True
        _logger.info("Validating portfolio holdings id and weight")
        for holding in self.holdings:
            if not holding.holding_id:
                # Find out holding without holding id.This is an invalid case.
                _logger.debug(f"The portfolio date {self.portfolio_date} holdings Holding_Id is missing")
                raise BadRequestException(BAD_REQUEST_ERROR_NO_PORTFOLIO_HOLDING_ID) from None
            if holding.weight is not None:
                all_weight_is_none = False

        if all_weight_is_none:
            # All holdings without weight.This is an valid case.
            _logger.info(
                f"The portfolio date {self.portfolio_date} all holdings weight is none.divide these holding weight by 100."
            )
            new_weight = 100 / len(self.holdings)
            for holding in self.holdings:
                holding.weight = new_weight
            _logger.debug(
                f"The portfolio date {self.portfolio_date} all holdings weight is None,set holdings weight: {new_weight}"
            )

    def _validate_holdings_total_weight(self) -> None:
        """
        Validation portfolio holdings total weight must equal 100.0 .
        """
        total_weight_error_msg = BAD_REQUEST_PORTFOLIO_WEIGHT_ERROR

        total_weight = sum(c.weight or 0 for c in self.holdings)  # get total weight.
        _logger.info(f"Validating the portfolio date {self.portfolio_date} holdings total weight: {total_weight}%.")
        if round(total_weight, 2) > 100:
            # Holdings total_weight>100.This is an invalid case.
            error_message = (
                f"The portfolio date: {self.portfolio_date}. {total_weight_error_msg}subtract {round(total_weight - 100, 2)}%."
            )
            _logger.debug(error_message)
            raise BadRequestException(error_message) from None
        if round(total_weight, 2) < 100:
            # Holdings total_weight<100.This is an invalid case.
            error_message = (
                f"The portfolio date: {self.portfolio_date}. {total_weight_error_msg}add {round(100 - total_weight, 2)}%."
            )
            _logger.debug(error_message)
            raise BadRequestException(error_message) from None

        for holding in self.holdings:  # Set holding market_value with weight.
            if holding.weight is not None:
                _logger.info("# Set holding market_value with weight.")
                holding.market_value = self.total_amount * holding.weight / 100
        return None

    def __str__(self) -> str:
        return f"{json.dumps(self.to_json(), ignore_nan=True, cls=PortfolioJSONEncoder, ensure_ascii=False)}"

    def to_json(self) -> Dict:
        # Set holding price and quantity
        self._set_holding_price_quantity()
        dict_holdings = []
        _logger.info("Generate holdings entity request json")
        for holding in self.holdings:
            dict_holdings.append(
                {
                    "secId": holding.holding_id,
                    "weight": holding.weight,
                    "marketValue": holding.market_value,
                    "quantity": holding.quantity,
                    "price": holding.price,
                }
            )
        to_return = {"portfolioDate": self.portfolio_date, "holdings": dict_holdings}
        return to_return

    def _set_holding_price_quantity(self) -> None:
        """
        Get and set portfolio holdings price and quantity .
        """
        _logger.info("Get and set the portfolio date price for holdings.")
        post_body = {
            "priceDate": self.portfolio_date,
            "investments": list(map(lambda x: {"id": x.holding_id}, self.holdings)),
        }
        _logger.debug(f"Get portfolio date price post body..{post_body}")
        url = f"{_config.securitydata_service_url()}/v1/securities/price"
        response_json = _portfolio_api.do_post_request(url, json.dumps(post_body, ignore_nan=True))
        _logger.debug(f"The portfolio date price for holdings..{response_json}")

        _logger.info("Set price and quantity for each holding.")
        for holding in self.holdings:
            # Get holding details from response by hoding_id.
            holding_info = [item for item in response_json if item.get("id") == holding.holding_id]
            price = float(holding_info[0].get("displayprice", 1))
            holding.price = price
            # get quantity by price.
            holding.quantity = holding.market_value / price
            _logger.debug(f"The {holding.holding_id} price:{price},quantity:{holding.quantity}")


class HoldingsEntityRequestBuilder:
    portfolio_id: Optional[str]

    def __init__(self, holdings_data_frame: DataFrame) -> None:
        self.holdings_entitys = self._data_frame_to_holdings_entitys(holdings_data_frame)

    def _data_frame_to_holdings_entitys(self, holdings_data_frame: DataFrame) -> List[HoldingsEntity]:
        """
        Convert portfolio holdings dataframe to holdings entitys.
        """
        _logger.info("Portfolio holdings dataframe to holdings entitys.")
        if not set(["Portfolio Date", "HoldingId", "Weight"]).issubset(holdings_data_frame.columns):
            # Holdings DataFrame columns must have Portfolio Date ,HoldingId, Weight.This is an invalid case.
            _logger.debug(
                f"Holdings data frame is missing required columns. It only has :{','.join(holdings_data_frame.columns)}"
            )
            raise BadRequestException(BAD_REQUEST_INVALID_PORTFOLIO_HOLDING_DATA_FRAME)

        # Group by Portfolio Date
        portfolio_dates = holdings_data_frame.groupby("Portfolio Date")
        # holdings dataframe to holdings entitys.
        holdings_entitys = []
        for portfolio_date, rows in portfolio_dates:
            holdings_entitys.append(
                HoldingsEntity(
                    portfolio_date=portfolio_date,
                    holdings=[(Holding(holding_id=row.HoldingId, weight=row.Weight)) for index, row in rows.iterrows()],
                )
            )
        return holdings_entitys

    def update_portfolio_holdings(self) -> Any:
        """
        Update portfolio holdings with holdings entitys.
        """
        _logger.info("Saving holdings to portfolio.")
        _logger.debug(f"Portfolio id: {self.portfolio_id}")
        url = f"{_config.portfolio_service_url()}portfoliodataservice/v1/portfolios/{self.portfolio_id}/holdings"
        response_json = _portfolio_api.do_post_request(
            url, json.dumps(self.holdings_entitys, cls=PortfolioJSONEncoder, ignore_nan=True)
        )
        _logger.debug("Update portfolio holdings response:{response}")
        return response_json
