import uuid
from typing import Any, Dict, List

import simplejson as json
from pandas import DataFrame

from ..._base import _logger
from .. import _decorator, investment
from .._base_api import APIBackend
from .._config import _Config
from .._error_messages import (
    BAD_REQUEST_ERROR_INVALID_INVESTMENT_LIST_ID,
    RESOURCE_NOT_FOUND_ERROR_INVESTMENT_List,
)
from .._exceptions import (
    ApiResponseException,
    BadRequestException,
    ResourceNotFoundError,
    ValueErrorException,
)
from .._utils import _reduce_list_data

_config = _Config()


class InvestmentListAPIBackend(APIBackend):
    """
    Subclass to call the Holding API and handle any HTTP errors that occur.
    """

    def __init__(self) -> None:
        super().__init__()

    def _handle_custom_http_errors(self, res: Any) -> Any:
        """
        Handle HTTP errors with custom error messages
        """
        response_message = res.json().get("message")
        if res.status_code == 404:
            _logger.debug(f"Resource Not Found Error: {res.status_code} {response_message}")
            raise ResourceNotFoundError(RESOURCE_NOT_FOUND_ERROR_INVESTMENT_List) from None


_investment_list_api_request = InvestmentListAPIBackend()


@_decorator.not_null
@_decorator.typechecked
def get_investment_list(list_id: str) -> DataFrame:
    """Returns all investments that belong to a saved investment list in Morningstar Direct.

    Args:
        list_id (:obj:`str`): The unique identifier of a saved investment list in Morningstar
            Direct, e.g., "EBE416A3-03E0-4215-9B83-8D098D2A9C0D". Use the `get_investment_lists <#morningstar_data.direct.user_items.get_investment_lists>`_ function to discover saved lists.

    :Returns:
        DataFrame: A DataFrame object with investments that belong to the specified list. DataFrame columns include:

        * secid
        * masterportfolioid
        * tradingsymbol
        * name
        * securitytype
        * exchangeid
        * category

    :Examples:
        Get investment list by ID.

    ::

        import morningstar_data as md

        df = md.direct.user_items.get_investment_list(list_id="385349FE-01D6-4064-B297-64EAA28BD4E9") # Replace with a valid List ID
        df

    :Output:
        ==========  =================  =============  =============================  ============  ==========  ========
        secid       masterportfolioid  tradingsymbol  name                           securitytype  exchangeid  category
        ==========  =================  =============  =============================  ============  ==========  ========
        XIUSA000KQ  24729              NaN            Russell 2000 Growth TR USD XI  NaN           Small       Growth
        ...
        ==========  =================  =============  =============================  ============  ==========  ========

    """
    try:
        # sourcetype should be "Desktop" to retrieve secids for list items
        # The alternative is "Cloud", which retrieves performanceids.
        # We need secids for joining with Datalake tables
        _validate_investment_list_id(list_id)
        _logger.debug(f"Getting Direct List: {list_id}")
        url = f"{_config.object_service_url()}objectapi/v1/lists/{list_id}?sourcetype=Desktop"
        response_json: Dict[str, Any] = _investment_list_api_request.do_get_request(url)

        # The Lists API will respond with a 200 if the list is not found.
        # Instead of relying on HTTP status code, let's check their JSON response for any errors
        if response_json["_meta"]["response_status"] != "200100":
            raise ApiResponseException(response_json["_meta"]["hint"])

        list_json: Dict[str, Any] = response_json["lists"][0]

        ids = list(map(lambda x: f"{x['id']}", list_json["idList"]))
        ids_types = list(map(lambda x: f"{x['3xType']}", list_json["idList"]))

        if not ids:
            list_df = DataFrame(
                {
                    "secid": [],
                    "masterportfolioid": [],
                    "tradingsymbol": [],
                    "name": [],
                    "securitytype": [],
                    "exchangeid": [],
                    "category": [],
                }
            )
        else:
            list_df = investment.get_investment_data(
                investments=ids,
                data_points=[
                    {"datapointId": "OS00I", "isTsdp": False},
                    {"datapointId": "DC09A", "isTsdp": False},
                    {"datapointId": "OS385", "isTsdp": False},
                    {"datapointId": "OS01W", "isTsdp": False},
                    {"datapointId": "OS010", "isTsdp": False},
                    {"datapointId": "OS01Z", "isTsdp": False},
                    {"datapointId": "OS245", "isTsdp": False},
                ],
            )
            list_df = list_df.drop(["Id"], axis=1)
            list_df.columns = [
                "secid",
                "masterportfolioid",
                "tradingsymbol",
                "name",
                "securitytype",
                "exchangeid",
                "category",
            ]
            list_df["secid"] = ids
            list_df["securitytype"] = ids_types
        # We are getting float values for masterportfolioid from the api and the fact that this statement is trying
        # to convert it to integer and convert Na/NaN values to python None object.
        list_df["masterportfolioid"] = (
            list_df["masterportfolioid"].fillna(-1).astype(int).astype(str).replace({"-1": None, "None": None})
        )

        if list_df.empty:
            _logger.debug("Returning empty DataFrame.")
        _logger.debug(f"Returning DataFrame for List: {list_id}")
        return list_df

    except Exception as e:
        _logger.error(repr(e))
        error_message = str(e)
    if error_message:
        raise ValueErrorException(error_message)


@_decorator.typechecked
def get_investment_lists() -> DataFrame:
    """Returns all investment lists saved or shared to a user in Morningstar Direct. Also includes Morningstar
    pre-defined investment lists.

    :Returns:
        DataFrame: A DataFrame object containing investment list details. DataFrame columns include:

        * id
        * name

    :Examples:

    ::

        import morningstar_data as md

        df = md.direct.user_items.get_investment_lists()
        df

    :Output:
        ====================================  ==============================
        id                                    name
        ====================================  ==============================
        EBE416A3-03E0-4215-9B83-8D098D2A9C0D  Morningstar Open Index Project
        858BD493-68B3-4D44-9DF0-333D3CC88A1C  Morningstar Prospects
        ...
        ====================================  ==============================

    """
    try:
        _logger.debug("Getting All Direct Lists")
        url = f"{_config.object_service_url()}objectapi/v1/lists"
        response_json: Dict[str, Any] = _investment_list_api_request.do_get_request(url)

        # The Lists API will respond with a 200 if the list is not found.
        # Instead of relying on HTTP status code, let's check their JSON response for any errors
        if response_json["_meta"]["response_status"] != "200100":
            raise ApiResponseException(response_json["_meta"]["hint"])

        props_dict: Dict[str, str] = {"id": "id", "name": "name"}
        all_investment_list: List[Dict[str, Any]] = _reduce_list_data(response_json["lists"], props_dict)
        return DataFrame(all_investment_list)
    except Exception as e:
        _logger.error(repr(e))
        error_message = str(e)
    if error_message:
        raise ValueErrorException(error_message)


@_decorator.typechecked
def save_investment_list(list_name: str, investment_ids: List[str], overwrite_if_exists: bool) -> DataFrame:
    """Saves or updates an investment list.

    Args:
        list_name (:obj:`str`): Name of the list
        investment_ids (:obj:`list`): List of investment IDs (SecId). For example: ["F00000YVYF","FOUSA00CFV"]
        overwrite_if_exists(:obj:`bool`): If True the list will be overwritten

    :Returns:
        DataFrame: A DataFrame object with the new investment list data.

    :Examples:

    Example 1: Save a new investment list.

    Output DataFrame columns include:

    * Status
    * Name
    * List Id
    * Investments
    * Created Date

    ::

        import morningstar_data as md

        df = md.direct.user_items.save_investment_list(
            list_name="new_investments",
            investment_ids=["F00000YVYF", "FOUSA00CFV"],
            overwrite_if_exists=False
        )
        df

    :Output:
        =========================  ===============  ====================================  ========================  ===================
        Status                     Name             List Id                               Investments               Created Date
        =========================  ===============  ====================================  ========================  ===================
        List successfully created  new_investments  295F7E59-15AC-4424-958E-3BD8B0A733EE  [F00000YVYF, FOUSA00CFV]  2022-03-29T14:43:00
        ...
        =========================  ===============  ====================================  ========================  ===================

        Example 2: Update an existing list.

        Output DataFrame columns include:

        * Status
        * Name
        * List Id
        * Modified Date

        ::

            import morningstar_data as md

            df = md.direct.user_items.save_investment_list(
                list_name="new_investments",
                investment_ids=["F00000YVYF", "FOUSA00CFV"],
                overwrite_if_exists=True
            )
            df


        =========================  ===============  ====================================  ===================
        Status                     Name             List Id                               Modified Date
        =========================  ===============  ====================================  ===================
        List successfully updated  new_investments  295F7E59-15AC-4424-958E-3BD8B0A733EE  2022-03-29T14:45:00
        ...
        =========================  ===============  ====================================  ===================

    """

    try:
        if list_name is None or list_name.strip() == "":
            raise BadRequestException("Please specify a list name to proceed.") from None
        if not investment_ids:
            raise BadRequestException("Please specify investment ids to proceed.") from None

        get_item = get_investment_lists()

        data: Dict[str, Any]
        if (list_name in get_item.values) and overwrite_if_exists is False:
            raise BadRequestException("The list name already exists.") from None
        if (list_name in get_item.values) and overwrite_if_exists:
            list_id = get_item.loc[get_item["name"] == list_name, "id"].iloc[0]
            url = f"{_config.object_service_url()}objectapi/v1/lists/{list_id}?sourcetype=Desktop"
            data = {"name": list_name, "listType": "ID_LIST", "idList": investment_ids}
            result = _investment_list_api_request.do_put_request(url, json.dumps(data, ignore_nan=True))
            if result["_meta"]["response_status"] != "200102":
                raise ApiResponseException(result["_meta"]["hint"]) from None
            data = {
                "Status": [result["_meta"]["hint"]],
                "Name": [result["lists"][0]["name"]],
                "List Id": [result["lists"][0]["id"]],
                "Modified Date": [result["lists"][0]["lastModifiedOn"]],
            }
            return DataFrame(data)
        else:
            url = f"{_config.object_service_url()}objectapi/v1/lists?sourcetype=Desktop"
            data = {"name": list_name, "listType": "ID_LIST", "idList": investment_ids}
            result = _investment_list_api_request.do_post_request(url, json.dumps(data, ignore_nan=True))
            ids = list(map(lambda x: f"{x['id']}", result["lists"][0]["idList"]))
            data = {
                "Status": [result["_meta"]["hint"]],
                "Name": [result["lists"][0]["name"]],
                "List Id": [result["lists"][0]["id"]],
                "Investments": [ids],
                "Created Date": [result["lists"][0]["createdOn"]],
            }
            return DataFrame(data)
    except Exception as e:
        _logger.error(repr(e))
        error_message = str(e)
        if error_message:
            raise ValueErrorException(error_message)


def _validate_investment_list_id(list_id: str) -> None:
    try:
        uuid.UUID(list_id)
    except ValueError:
        _logger.debug(f"Investment list id {list_id} is not valid.")
        raise BadRequestException(BAD_REQUEST_ERROR_INVALID_INVESTMENT_LIST_ID) from None
