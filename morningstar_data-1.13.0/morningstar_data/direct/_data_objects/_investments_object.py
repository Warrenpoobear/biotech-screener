import functools
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from .. import _error_messages
from .._api import _direct_api_request
from .._config import _Config
from .._error_messages import (
    BAD_REQUEST_ERROR_INVALID_CATEGORY_ID,
    BAD_REQUEST_ERROR_NO_INVESTMENT_SOURCE,
)
from .._exceptions import BadRequestException, ResourceNotFoundError
from ..data_type import InvestmentIdentifier
from ..user_items.search_criteria import execute_search

_config = _Config()

SearchObjectType = Optional[Dict[str, Any]]
InvestmentsListType = Optional[List[str]]
UUIDOrInitType = Optional[str]
InvestmentType = Union[SearchObjectType, UUIDOrInitType, InvestmentsListType, List[InvestmentIdentifier]]
CategoryType = Optional[str]


class Investments:
    def __init__(self, investment_object: InvestmentType) -> None:
        self.investment_ids: InvestmentsListType = None
        self.investment_identifiers: Optional[List[InvestmentIdentifier]] = None
        self.search_criteria: UUIDOrInitType = None
        self.list_id: UUIDOrInitType = None
        self.search_criteria_object: SearchObjectType = None
        self.category_id: CategoryType = None
        self._parse_investment_object(investment_object)

    def _parse_investment_object(self, investment_object: InvestmentType) -> None:
        if isinstance(investment_object, list):
            if all(isinstance(item, str) for item in investment_object):
                # NOTE: Though its being explicitly checked whether all items are strings, mypy is not able to infer this. So we are ignoring the type error here.
                self.investment_ids = investment_object  # type: ignore
            elif all(isinstance(item, InvestmentIdentifier) for item in investment_object):
                # NOTE: Though its being explicitly checked whether all items are InvesgtmentIdentifier, mypy is not able to infer this. So we are ignoring the type error here.
                self.investment_identifiers = investment_object  # type: ignore
        elif isinstance(investment_object, dict):
            self.search_criteria_object = investment_object
        elif self._is_int(investment_object):
            self.search_criteria = investment_object
        elif self._is_guid(investment_object):
            self.list_id = investment_object
        elif isinstance(investment_object, str) and investment_object is not None and len(investment_object.strip()) > 0:
            self.category_id = investment_object
        else:
            raise ValueError("Invalid investment object")

    def _is_int(self, value: InvestmentType) -> bool:
        try:
            if not isinstance(value, str):
                raise ValueError("Invalid investment object: search criteria must be a string")
            int(value)
            return True
        except ValueError:
            return False

    def _is_guid(self, value: InvestmentType) -> bool:
        try:
            if not isinstance(value, str):
                raise ValueError("Invalid investment object: list ID must be a string")
            UUID(value)
            return True
        except ValueError:
            return False

    @functools.lru_cache()
    def get_investment_ids(self) -> list:
        id_list = []
        if self.investment_ids:
            id_list = self.investment_ids

        elif self.investment_identifiers:
            # NOTE: Mypy is not able to infer list of InvestmentIdentifier objects can be assigned to id_list. So, ignoring the type error.
            id_list = self.investment_identifiers  # type: ignore

        elif self.list_id is not None and len(self.list_id.strip()) > 0:
            id_list = self._get_investment_ids_by_list_id()
            if not id_list:
                raise ResourceNotFoundError(_error_messages.RESOURCE_NOT_FOUND_INVESTMENT_DATA_LIST_ID)

        elif self.search_criteria is not None and len(self.search_criteria.strip()) > 0:
            id_list = self._get_investment_ids_by_search_id()
            if not id_list:
                raise ResourceNotFoundError(_error_messages.RESOURCE_NOT_FOUND_INVESTMENT_DATA_SEARCH_CRITERIA_ID)

        elif self.search_criteria_object:
            id_list = self._get_investments_by_search_criteria_object()
            if not id_list:
                raise ResourceNotFoundError(_error_messages.RESOURCE_NOT_FOUND_INVESTMENT_DATA_SEARCH_CRITERIA_CONDITION)

        return id_list

    def _get_investment_ids_by_list_id(self) -> list:
        investment_list_dict = self._get_investment_list()
        investment_id_list = investment_list_dict.get("idList", []) if investment_list_dict is not None else []
        investment_ids = []
        if investment_id_list:
            investment_ids = list(map(lambda x: self._concat_id_and_type(x), investment_id_list))
        return investment_ids

    def _get_investment_ids_by_search_id(self) -> list:
        investment_list = self._get_investments_by_search_id()
        if investment_list:
            return [investment.get("id", "") for investment in investment_list]
        return []

    def _get_investment_list(self) -> Dict[str, Any]:
        # sourcetype should be "Desktop" to retrieve secids for list items
        # The alternative is "Cloud", which retrieves performanceids.
        # We need secids for joining with Datalake tables
        url = f"{_config.object_service_url()}objectapi/v1/lists/{self.list_id}?sourcetype=Desktop"
        response_json: Dict[str, Any] = _direct_api_request("GET", url)

        investment_list: List[Dict[str, Any]] = response_json.get("lists", [])
        if len(investment_list) > 0:
            return investment_list[0]
        return dict()

    def _get_investments_by_search_id(self) -> list:
        url = f"{_config.searches_service_url()}v1/searches/{self.search_criteria}"
        response_json = _direct_api_request("GET", url)
        search_criteria = response_json.get("content", dict())
        if len(search_criteria) > 0:
            search_result: list = execute_search(search_criteria)
            return search_result
        return []

    def _get_investments_by_search_criteria_object(self) -> list:
        if self.search_criteria_object is not None and len(self.search_criteria_object) > 0:
            if "universeId" not in self.search_criteria_object.keys() or self.search_criteria_object["universeId"] == "":
                raise BadRequestException("'universeId' is a required property in the search criteria object.")
            if "criteria" not in self.search_criteria_object.keys() or self.search_criteria_object["criteria"] == []:
                raise BadRequestException("'criteria' is a required property in the search criteria object.")
            investment_data = execute_search(self.search_criteria_object)
            return [investmentId["id"] for investmentId in investment_data if "id" in investmentId.keys()]
        return []

    def _concat_id_and_type(self, investment: dict) -> str:
        investment_id = investment.get("id", "").strip()
        investment_type = investment.get("3xType", None)
        investment_type = investment_type.strip() if investment_type else ""
        if investment_type and investment_type.upper() != "CA":
            return f"{investment_id};{investment_type}"
        return f"{investment_id}"

    def generate_investment_source(self) -> dict:
        investment_source = {}
        if self.category_id is not None and len(self.category_id.strip()) > 0:
            if ";" not in self.category_id:
                raise BadRequestException(BAD_REQUEST_ERROR_INVALID_CATEGORY_ID)
            investment_source["type"] = "MORNINGSTAR_CATEGORY"
            investment_source["id"] = self.category_id

        elif self.investment_ids:
            investment_source["type"] = "SECIDS"
            investment_source["id"] = ",".join(self.investment_ids)

        elif self.list_id is not None and len(self.list_id.strip()) > 0:
            investment_source["type"] = "LIST"
            investment_source["id"] = self.list_id

        elif self.search_criteria is not None and len(self.search_criteria.strip()) > 0:
            investment_source["type"] = "SEARCH"
            investment_source["id"] = self.search_criteria

        if not investment_source:
            raise BadRequestException(BAD_REQUEST_ERROR_NO_INVESTMENT_SOURCE)

        return investment_source
