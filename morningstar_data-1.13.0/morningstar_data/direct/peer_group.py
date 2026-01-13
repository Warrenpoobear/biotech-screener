from typing import Any, Dict, List, Optional, Union

from pandas import DataFrame

from . import _decorator
from ._config import _Config
from ._data_objects import DataPoints, Investments
from ._investment._peer_group_data import PeerGroupProvider
from .data_type import Order, PeerGroupMethodology

_config = _Config()
peer_group_provider = PeerGroupProvider()


@_decorator.typechecked
def get_peer_group_breakpoints(
    investments: Union[List[str], str],
    data_points: Union[List[Dict[str, Any]], DataFrame],
    order: Order = Order.ASC,
    percentiles: Optional[List[int]] = list(range(1, 101)),
    methodology: Optional[PeerGroupMethodology] = None,
) -> DataFrame:
    """Returns peer group breakpoints for the specified list of investments and data points.

    Args:
        investments (:obj:`Union`, `required`): Defines the investments to fetch. Input can be:

            * Investment IDs (:obj:`list`, `optional`): Investment identifiers, in the format of SecId;Universe or just SecId. E.g., ["F00000YOOK;FO","FOUSA00CFV;FO"] or ["F00000YOOK","FOUSA00CFV"]. Use the `investments <./lookup.html#morningstar_data.direct.investments>`_ function to discover identifiers.
            * Investment List ID (:obj:`str`, `optional`): Saved investment list in Morningstar Direct. Use the `get_investment_lists <./lists.html#morningstar_data.direct.user_items.get_investment_lists>`_ function to discover saved lists.
            * Search Criteria  ID (:obj:`str`, `optional`): Saved search criteria in Morningstar Direct. Use the `get_search_criteria <./search_criteria.html#morningstar_data.direct.user_items.get_search_criteria>`_ function to discover saved search criteria.
            * Search Criteria Condition (:obj:`dict`, `optional`): Search criteria definition. See details in the Reference section of `get_investment_data <./investment.html#morningstar_data.direct.get_investment_data>`_ or use the `get_search_criteria_conditions <./search_criteria.html#morningstar_data.direct.user_items.get_search_criteria_conditions>`_ function to discover the definition of a saved search criteria.

        data_points (:obj:`Union`, `optional`): Defines the data points to fetch. If not provided and investments are specified with a list ID or search criteria ID, the corresponding bound dataset will be used.

            * Data Point IDs (:obj:`List[Dict]`, `optional`): A list of dictionaries, each defining a data point and its (optional) associated settings. The optional `alias` attribute can be added to each data point and will be used in the response, e.g., [{"datapointId": "41", "alias": "Z1"}. Use `get_data_set_details <./data_set.html#morningstar_data.direct.user_items.get_data_set_details>`_ to discover data point identifiers from a saved data set.
            * Data Point Settings (:obj:`DataFrame`, `optional`): A DataFrame of data point identifiers and their associated settings. Use the `get_data_set_details <./data_set.html#morningstar_data.direct.user_items.get_data_set_details>`_ function to discover data point settings from a saved data set.

        order (:obj:`md.direct.data_type.Order`, `optional`): Breakpoint order, can be set to ``md.direct.data_type.Order.DESC`` (descending) or ``md.direct.dats_type.Order.ASC`` (ascending, default).
        percentiles (:obj:`list`, `optional`): Percentiles default to a list [1,2,3,...,100] if not provided, values should be within 1-100 range.
        methodology (:obj:`md.direct.data_type.PeerGroupMethodology`, `optional`): Breakpoint calculation methodology, can be set to ``md.direct.data_type.PeerGroupMethodology.MORNINGSTAR`` or ``md.direct.data_type.PeerGroupMethodology.SIMPLE``. Defaults to the global setting "Custom Peer Group Ranking" in Morningstar Direct if not provided.

    :Returns:
        DataFrame: A DataFrame object with peer group breakpoint data. The columns include the `alias` that the user
        input in the `data_points` parameter.

    :Examples:

    Get peer group breakpoint data for the standard deviation data point.

    ::

        import morningstar_data as md

        df = md.direct.get_peer_group_breakpoints(
                investments='740284aa-fcd3-43f6-99d1-8f3d4a179fcc',
                data_points=[
                    {"datapointId": "41", "alias": "Z1"},
                    {"datapointId": "41", "alias": "Z2", "startDate": "2021-07-01", "endDate": "2021-12-31", "windowType": "2", "windowSize": "3", "stepSize": "2"}
                ],
                order=md.direct.data_type.Order.ASC,
                percentiles=[25, 50, 75, 100]
            )
        df

    :Output:
        ======  ===========  ==========  ==========  ==========  ==========  ==========
        Alias   StartDate    EndDate     25          50          75          100
        ======  ===========  ==========  ==========  ==========  ==========  ==========
        Z1      2019-04-01   2022-03-31  17.301437   12.720889   7.055372    -3.460187
        Z2      2021-07-01   2021-09-30  1.827371    -0.804269   -4.899745   -52.143678
        Z2      2021-09-01   2021-11-30  -0.030321   -4.336051   -10.618009  -40.980480
        ======  ===========  ==========  ==========  ==========  ==========  ==========

    :Errors:
        AccessDeniedError: Raised when the user lacks permission or is not authorized to access the resource.

        BadRequestException: Raised due to invalid/incorrect request, malformed request syntax, or deceptive request routing.

        NetworkExceptionError: Raised when there is an issue with the internet connection or if the request is made from an unsecure network.

        ResourceNotFoundError: Raised when the requested resource does not exist in Direct.

    """
    investment_param = Investments(investments)
    data_point_param = DataPoints(data_points)

    peer_group_req = peer_group_provider.build_request(
        investment_param, data_point_param, order=order, percentiles=percentiles, methodology=methodology
    )
    peer_group_resp = peer_group_provider.run_request(peer_group_req)
    return peer_group_provider.build_data_frame(peer_group_resp)
