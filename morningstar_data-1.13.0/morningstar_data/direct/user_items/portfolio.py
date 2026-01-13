from typing import Optional

import pandas as pd
from pandas import DataFrame

from ..._base import _logger
from .. import _decorator, _error_messages
from .._config import _Config
from .._error_messages import (
    BAD_REQUEST_INVALID_PORTFOLIO_NAME,
    BAD_REQUEST_INVALID_PORTFOLIO_TYPE,
    BAD_REQUEST_PORTFOLIO_NAME_ALREADY_EXISTS,
    CLIENT_ACCOUNTS_PORTFOLIO_TYPE_USED,
)
from .._exceptions import (
    AccessDeniedError,
    BadRequestException,
    InternalServerError,
    ResourceNotFoundError,
)
from .._portfolio import HoldingsEntityRequestBuilder, PortfolioSetting, PortfolioType
from .._portfolio._common import PortfolioDataApiBackend, PortfolioNotFoundError

_config = _Config()
_portfolio_api = PortfolioDataApiBackend()


@_decorator.not_null
@_decorator.typechecked
def get_portfolios(portfolio_type: Optional[str] = None) -> DataFrame:
    """Returns all portfolios for the given portfolio type.

    Args:
        portfolio_type(:obj:`str`): The portfolio type. The available options are: model_portfolios, custom_benchmarks.

    :Returns:
        DataFrame: A DataFrame object with portfolio details. DataFrame columns include:

        * PortfolioId
        * Name
        * Type

    :Examples:
        Get client account portfolios.
    ::

        import morningstar_data as md

        df = md.direct.user_items.get_portfolios(portfolio_type = "model_portfolios")
        df

    :Output:
        =======================================     ==========    ===============
        PortfolioId                                 Name          Type
        =======================================     ==========    ===============
        3d25b613-ae42-4cb6-bd05-7ba7c74883f3;MD     Portfolio1    model_portfolios
        5d25b613-ae42-4cb6-bd05-7ba7c74883f3;MD     Portfolio2    model_portfolios
        ...
        =======================================     ==========    ===============

    :Errors:
        AccessDeniedError: Raised when the user is not authenticated.

        BadRequestError: Raised when the user does not provide a properly formatted request.

        ForbiddenError: Raised when the user does not have permission to access the requested resource.

        InternalServerError: Raised when the server encounters an unhandled error.

        NetworkExceptionError: Raised when the request fails to reach the server due to a network error.

        ResourceNotFoundError: Raised when the requested resource does not exist in Direct.

    """
    all_data_frame = DataFrame()
    if portfolio_type == "client_accounts":
        raise BadRequestException(CLIENT_ACCOUNTS_PORTFOLIO_TYPE_USED)
    if portfolio_type is None:
        types = ",".join([PortfolioType.model_portfolios.key, PortfolioType.custom_benchmarks.key])
    else:
        types = _portfolio_type_map(portfolio_type)
    try:
        url = f"{_config.portfolio_service_url()}/portfoliodataservice/v3/portfolios?types={types}"
        _logger.debug(f"Fetching portfolio data from: {url}")
        response_json = _portfolio_api.do_get_request(url)
        df = DataFrame(response_json["snapshots"])
        new_data_frame = df.drop(columns=["dataPoints", "permission", "folderId"])
        new_data_frame.columns = ["PortfolioId", "Name", "Type"]
        new_data_frame["Type"] = new_data_frame["Type"].map(_mapping_portfolio_type_full_name)
        if portfolio_type is not None:
            _logger.debug(f"Portfolio type is {portfolio_type}. Filtering data to match this portfolio.")
            new_data_frame = new_data_frame.loc[new_data_frame["Type"] == portfolio_type]
        all_data_frame = pd.concat([all_data_frame, new_data_frame])
    except PortfolioNotFoundError:
        _logger.debug(f"There are no portfolio(type={types}) in the Direct account.")
    except AccessDeniedError as e:
        raise AccessDeniedError(str(e)) from None
    except BadRequestException as e:
        raise BadRequestException(str(e)) from None
    except Exception as e:
        raise InternalServerError(str(e)) from None

    if all_data_frame.empty:
        if len(types) == 2:  # i.e. if only one portfolio type was selected
            _logger.debug(f"Requested resource not found for portfolio type `{portfolio_type}`")
            raise ResourceNotFoundError(_error_messages.RESOURCE_NOT_FOUND_ERROR_PORTFOLIO_ACCOUNT) from None
        raise ResourceNotFoundError(_error_messages.RESOURCE_NOT_FOUND_ERROR_PORTFOLIO_ACCOUNTS) from None
    else:
        all_data_frame = all_data_frame.reset_index(drop=True)

    return all_data_frame


def _portfolio_type_map(portfolio_type: str) -> str:
    try:
        return PortfolioType[portfolio_type].key
    except KeyError:
        raise BadRequestException(BAD_REQUEST_INVALID_PORTFOLIO_TYPE)


def _mapping_portfolio_type_full_name(val: str) -> str:
    return PortfolioType.get_full_name_by_abbr(val)


@_decorator.typechecked
def save_portfolio(
    portfolio_name: str, portfolio_type: PortfolioType, holdings: Optional[DataFrame] = None, overwrite_if_exists: bool = False
) -> DataFrame:
    """Save or update a portfolio.

    Args:
        portfolio_name (:obj:`str`): Name of the portfolio.
        portfolio_type (:obj:`PortfolioType`):Type of the portfolio. Supported portfolio types: model_portfolios, custom_benchmarks.
        overwrite_if_exists(:obj:`bool`): If true, an existing portfolio will be overwritten with new name and holdings.
        holdings:(:obj:`DataFrame`, `optional`) : A DataFrame object containing holdings to save to this portfolio.

    :Returns:
        DataFrame: A DataFrame object, columns include:

        * Name
        * Portfolio Id

    :Examples:

    Save a new portfolio.
    ::

        import pandas as pd
        import morningstar_data as md
        from morningstar_data.direct.user_items.portfolio import PortfolioType

        holdings_data_frame = pd.DataFrame(
            [
                ["FOUSA00DFS;FO", "2020-10-31", 33.33],
                ["0P000002RH;ST", "2020-10-31", 66.67]
            ],
            columns=["HoldingId", "Portfolio Date", "Weight"],
        )

        df = md.direct.user_items.save_portfolio(
            portfolio_name = "new_portfolio",
            portfolio_type = PortfolioType.model_portfolios,
            holdings = holdings_data_frame,
            overwrite_if_exists = False,
        )

        df

    :Output:
        ===============  ====================================
        Name             Portfolio Id
        ===============  ====================================
        new_portfolio    295F7E59-15AC-4424-958E-3BD8B0A733EE
        ===============  ====================================

        Update an existing portfolio.

        ::

            import pandas as pd
            import morningstar_data as md
            from morningstar_data.direct.user_items.portfolio import PortfolioType

            # Updated holdings data
            holdings_data_frame = pd.DataFrame(
                [
                ["FOUSA00DFS;FO", "2020-05-31", 50],
                ["0P000002RH;ST", "2020-05-31", 50]
                ],
                columns=["HoldingId", "Portfolio Date", "Weight"]
            )

            df = md.direct.user_items.save_portfolio(
                portfolio_name = "new_portfolio",
                portfolio_type = PortfolioType.model_portfolios,
                holdings = holdings_data_frame,
                overwrite_if_exists = True # Update the existing portfolio
            )

            df


        =============== ====================================
        Name             Portfolio Id
        =============== ====================================
        new_portfolio   295F7E59-15AC-4424-958E-3BD8B0A733EE
        =============== ====================================

    """
    _logger.info("Validating portfolio name")
    if portfolio_name is None or portfolio_name.strip() == "":
        _logger.debug(f"Invalid portfolio name: {portfolio_name}. Raising exception.")
        raise BadRequestException(BAD_REQUEST_INVALID_PORTFOLIO_NAME) from None

    _logger.info("Validating portfolio type")
    if portfolio_type not in [PortfolioType.custom_benchmarks, PortfolioType.model_portfolios]:
        _logger.debug(f"Invalid portfolio type: {portfolio_type}")
        raise BadRequestException(BAD_REQUEST_INVALID_PORTFOLIO_TYPE) from None

    _logger.info("Validating portfolio holdings")
    if holdings is not None:
        _logger.info("Holdings is not None, creating holdings entity request builder ,and validate holdings.")
        holdings_entitys_builder = HoldingsEntityRequestBuilder(holdings)

    _logger.info("Validating against existing portfolio names.")

    _logger.info("Get all user created portfolios")
    user_portfolios = get_portfolios(portfolio_type.name)

    portfolio_id = None
    if (portfolio_name in user_portfolios.values) and overwrite_if_exists is False:
        # Portfolio with same name exists and user is not overwriting it. This is an invalid case.

        _logger.info("Portfolio name already exists and overwrite is set to False. Throwing error.")
        _logger.debug(f"Portfolio name: {portfolio_name}. Overwrite if exists: {overwrite_if_exists}")
        raise BadRequestException(BAD_REQUEST_PORTFOLIO_NAME_ALREADY_EXISTS) from None

    if (portfolio_name in user_portfolios.values) and overwrite_if_exists:
        # Portfolio with same name exists and user is overwriting it. This is an valid case.

        _logger.info("Portfolio name already exists and overwrite is set to True")
        _logger.info("Fetching matching portfolio id.")
        portfolio_id = user_portfolios.loc[user_portfolios["Name"].str.lower() == portfolio_name.lower(), "PortfolioId"].iloc[0]

        _logger.debug(f"Got matching portfolio id: {portfolio_id}")

    if portfolio_id is None:
        _logger.info("Portfolio id is None, so creating a new portfolio")
        # create new portfolio with default settings.
        portfolio_setting = PortfolioSetting(name=portfolio_name, portfolio_type=portfolio_type.key, portfolio_id=portfolio_id)

        portfolio_setting.save_portfolio_settings()  # Fix this in DO API to return portfolio id

        # The above statement does not return newly created portfolio id.
        # So we will call get_portfolios one more time to get the newly created portfolio id.
        _logger.info("Fetching user portfolio that match the newly created portfolio name.")
        user_portfolios = get_portfolios(portfolio_type.name)

        if portfolio_name in user_portfolios.values:
            portfolio_id = user_portfolios.loc[user_portfolios["Name"] == portfolio_name, "PortfolioId"].iloc[0]
            _logger.debug(f"Got newly created portfolio id: {portfolio_id}")

    # Assign portfolio id to the holdings and update portfolio
    if holdings_entitys_builder is not None:
        _logger.info("Holdings entity builder is not None, updating portfolio with holdings.")
        holdings_entitys_builder.portfolio_id = portfolio_id
        holdings_entitys_builder.update_portfolio_holdings()

    data = {"Name": [portfolio_name], "Portfolio Id": [portfolio_id]}

    return DataFrame(data)
