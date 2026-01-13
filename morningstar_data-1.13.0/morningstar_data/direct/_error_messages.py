ACCESS_DENIED_ERROR = "Unauthorized request! Please login with valid credentials. For more information, please contact Analytics Lab support team at Morningstar."
FORBIDDEN_ERROR = "Forbidden request! You do not have permission to access this resource. For more information, please contact Analytics Lab support team at Morningstar."
NETWORK_ERROR = "Network connection error. Please check your internet connection."
TIMEOUT_ERROR = "Request timed out. Please try again or contact support at morningstardirect@moringstar.com"
BAD_REQUEST_ERROR = "Bad request! Server could not process your request at this time. This error can occur due to multiple reasons (for example; bad request syntax, invalid message structure/framing, or incorrect/corrupt request).\nPlease refer morningstar data package documentation to make a valid request: https://docs-analyticslab.morningstar.com/latest/morningstar_data.html"
CLIENT_ERROR = "Bad request! Server could not process your request at this time. This error can occur due to multiple reasons (for example; bad request syntax, invalid message structure/framing, or incorrect/corrupt request).\nPlease refer morningstar data package documentation to make a valid request: https://docs-analyticslab.morningstar.com/latest/morningstar_data.html"
INTERNAL_SERVER_ERROR = "Something went wrong on our side!  For more details, please contact Analytics Lab support team at Morningstar. We apologize for any inconvenience this may have caused"

# Add query limit error messages here
QUERY_LIMIT_ERROR = "You have reached your daily query limit. Please fill out this form for further assistance: https://go.morningstar.com/LP=6390"
QUERY_LIMIT_ERROR_SHOW_LIMIT = "You have reached your daily query limit of $query cells. Please fill out this form for further assistance: https://go.morningstar.com/LP=6808"

# AMS Flagged Features
DELIVERY_ACCESS_ERROR = (
    "You don't have access to this feature. Please fill out this form for further assistance: https://go.morningstar.com/LP=6809"
)

# Add custom bad request error messages here
BAD_REQUEST_ERROR_NO_START_AND_END_DATE = "Specify `start_date` and `end_date` to proceed with your query."
BAD_REQUEST_ERROR_NO_END_DATE = "Specify the `end_date` to proceed with your query."
BAD_REQUEST_ERROR_NO_START_DATE = "Specify the `start_date` to proceed with your query."
BAD_REQUEST_ERROR_INCLUDE_ALL_DATE = "Specify either date or both start date and end date to proceed with your query."
BAD_REQUEST_ERROR_INVALID_DATE_FORMAT = "Specify date in the format yyyy-MM-dd to proceed with your query."
BAD_REQUEST_ERROR_NO_INVESTMENT_IDS = "Specify the `investments` to proceed with your query."
BAD_REQUEST_ERROR_NO_INVESTMENT_IDS_FOR_DRY_RUN = "Specify the `investments` when using dry_run parameter."
BAD_REQUEST_ERROR_NO_PORTFOLIO_DATA = "No matching portfolio data for the given date or date range."
BAD_REQUEST_ERROR_INVALID_INVESTMENT_ID = "One or more ID's, in the provided list of `investments`, are invalid!"
BAD_REQUEST_ERROR_INVALID_INVESTMENT_LIST_ID = (
    "Specify the valid `list_id` to proceed with your query. A `list_id` is a global unique identifier i.e. uuid."
)
BAD_REQUEST_ERROR_INVALID_PORTFOLIO_ID = (
    "Specify the valid `portfolio_id` to proceed with your query. A `portfolio_id` is a global unique identifier i.e. uuid."
)
BAD_REQUEST_ERROR_INVALID_LOOKTHROUGH_HOLDING_TYPE = "Indexes are not supported as a valid Look-Through holding."
BAD_REQUEST_ERROR_NO_START_AND_END_DATE = "Please specify `start_date` and `end_date` to proceed with your query."
BAD_REQUEST_ERROR_NO_END_DATE = "Please specify the `end_date` to proceed with your query."
BAD_REQUEST_ERROR_DATA_SET = "Please enter a valid format for `data_set_id`. To retrieve your data sets, execute `md.direct.user_items.get_data_sets()` and select a valid `data_set_id`. To retrieve morningstar data sets, execute `md.direct.lookup.get_morningstar_data_sets()` "
BAD_REQUEST_ERROR_INVALID_DELIVER_CONFIG = "Delivery config is not valid. Check the docs for examples of accepted inputs."

# Add custom resource not found error messages here
RESOURCE_NOT_FOUND_ERROR = (
    "Requested resource does not exist. Please make sure that the provided input belongs to a resource in Direct."
)
RESOURCE_NOT_FOUND_ERROR_SEARCH_CRITERIA = "Requested resource not found with the provided `search_criteria_id`. Please make sure that the provided `search_criteria_id` is valid and exists. \nTo view all the available search criterias, execute - `md.direct.user_items.get_search_criteria()`."
RESOURCE_NOT_FOUND_ERROR_NO_SEARCH_CRITERIA = "Requested resource not found. Please make sure that you have search criterias available in your Direct account. \nTo view search criterias, log in to Direct and navigate to `Workspace >> Search Criteria >> My Search Criteria`.\nSearch tutorial: https://morningstardirect.morningstar.com/clientcomm/Searches.pdf"
RESOURCE_NOT_FOUND_ERROR_SEARCH_CRITERIA_ID = "Requested resource not found for search_criteria_id `{}`. Please make sure that provided search_criteria_id exists in your Direct account. To get all the search criterias, execute `md.direct.user_items.get_search_criteria()`"
RESOURCE_NOT_FOUND_ERROR_SEARCH_CRITERIA_NO_INVESTMENTS = "Could not find any investments in the provided `search criteria`. Please make sure that provided `search criteria` is valid. To get all the search criterias, execute `md.direct.user_items.get_search_criteria()`"

RESOURCE_NOT_FOUND_ERROR_NO_DATA_SETS = "No data sets found. To create a new data set, please open Direct and navigate to Workspace -> Data Sets -> My Data Sets -> New Data Set."
RESOURCE_NOT_FOUND_PORTFOLIO_DATA_SET = "The requested `data_set_id` does not exist. To retrieve your data sets, execute `md.direct.user_items.get_data_sets()` and select a valid `data_set_id`. To retrieve morningstar data sets, execute `md.direct.lookup.get_morningstar_data_sets()`"
BAD_REQUEST_ERROR_PORTFOLIO_DATA_SET = "Please input a valid `data_set_id` to proceed with your query. To retrieve all the data sets, execute `md.direct.portfolio.get_data_sets()` and select a valid `data_set_id`."
RESOURCE_NOT_FOUND_ERROR_PORTFOLIO_ACCOUNT = "No portfolios found for the given portfolio_type. \nTo create or see all of your existing portfolio accounts, log in to Direct and navigate to `Portfolio Management >> Accounts`. \nPortfolio management tutorial: https://morningstardirect.morningstar.com/clientcomm/Portfolios.pdf"
RESOURCE_NOT_FOUND_ERROR_PORTFOLIO_ACCOUNTS = "Requested resources not found. Please make sure that you have created at least one portfolio account in Direct. \nTo create or see all of your existing portfolio accounts, log in to Direct and navigate to `Portfolio Management >> Accounts`. \nPortfolio management tutorial: https://morningstardirect.morningstar.com/clientcomm/Portfolios.pdf"
RESOURCE_NOT_FOUND_ERROR_PORTFOLIO_ID = "Requested resource not found for portfolio ID `{}`. Please make sure that a portfolio account exists in your Direct account with the provided ID. To get the holding dates for all your portfolio accounts, execute `md.direct.portfolio.get_holding_dates()`."
RESOURCE_NOT_FOUND_ERROR_PORTFOLIO_IDS = "Requested resources not found for provided portfolio IDs. Please make sure that portfolio accounts exist, for the provided list of IDs, in your Direct account. To get the holding dates for all your portfolio accounts, execute `md.direct.portfolio.get_holding_dates()`."
RESOURCE_NOT_FOUND_ERROR_PORTFOLIO_HOLDING = "Requested resource not found for portfolio ID `{}` and dates. Please make sure that, in Direct, a portfolio account exists with the provided ID and has holdings during provided dates. To get the holding dates for all your portfolio accounts, execute `md.direct.portfolio.get_holding_dates()`."
RESOURCE_NOT_FOUND_ERROR_PORTFOLIO_HOLDINGS = "Requested resources not found for portfolio IDs and dates. Please make sure that, in Direct, portfolio accounts exist with the provided IDs and have holdings during provided dates. To get the holding dates for all your portfolio accounts, execute `md.direct.portfolio.get_holding_dates()`."
RESOURCE_NOT_FOUND_ERROR_PORTFOLIO_GET_DATA_RETURN = (
    "Requested resource not found for portfolio ID `{}`. There is no return data available for the provided portfolio ID."
)

# Add custom resource not found error messages here for Asset Flow module
RESOURCE_NOT_FOUND_ASSET_FLOW = "The requested resource does not exist. Please make sure to provide a valid market_id."

# Add custom resource not found error messages here for Investments module
RESOURCE_NOT_FOUND_INVESTMENT_DATA = "No investments found for provided `investments` object. Please make sure that provided investments are valid and exist in your Direct account."
RESOURCE_NOT_FOUND_INVESTMENT_DATA_LIST_ID = "No investments found for the provided `list_id`. Please execute `md.direct.user_items.get_investment_lists()` to retrieve all the investment lists and select a valid `list_id`."
RESOURCE_NOT_FOUND_INVESTMENT_DATA_SEARCH_CRITERIA_ID = "No investments found for the provided `search_criteria_id`. Please execute `md.direct.user_items.get_search_criteria()` to retrieve all the available search criterias and select a valid `id`."
RESOURCE_NOT_FOUND_INVESTMENT_DATA_SEARCH_CRITERIA_CONDITION = "No investments found for the provided `search_criteria_condition`. Please execute `md.direct.user_items.get_search_criteria_conditions(search_criteria_id=:str)` with a valid search_criteria_id to build a search criteria condition."

# Add custom resource not found error messages here for Performance module
RESOURCE_NOT_FOUND_ERROR_PERFORMANCE_REPORT = (
    "The requested resource does not exist. Please make sure that the provided `report_id` is valid and exists."
)

# Add custom resource not found error messages here for Returns module
RESOURCE_NOT_FOUND_ERROR_NO_RETURNS_RETRIEVED = "Failed to get returns for provided `investments` argument."
RESOURCE_NOT_FOUND_ERROR_NO_RETURNS_RETRIEVED_FOR_INVESTMENT_LIST = "Could not fetch return for specified investment list. Please make sure that provided investments are valid and exist in your Direct account."
RESOURCE_NOT_FOUND_ERROR_NO_RETURNS_RETRIEVED_FOR_BENCHMARK_ID = "Could not fetch return for specified `benchmark_sec_id`. Please make sure that provided `benchmark_sec_id` is valid and exist in your Direct account."

# Add custom resource not found error messages here for Holding module
RESOURCE_NOT_FOUND_ERROR_HOLDING = "The requested resource does not exist. Make sure that the provided `investment_ids` are valid and have holdings within the specified date or date range."

# Add custom error messages here for Peer group module
BAD_REQUEST_ERROR_NO_DATA_POINT_ID_OR_ALIAS = "Specify datapointId and alias for each data_point."
BAD_REQUEST_ERROR_ALIAS_DUPLICATED = "The alias can't be duplicated."
BAD_REQUEST_ERROR_INCLUDE_NON_CALCULATION_DATA_POINTS = "Only calculated data points are supported."
BAD_REQUEST_ERROR_INCLUDE_DIFF_CALCULATION_DATA_POINTS = "Calculation is supported only on same data points."
BAD_REQUEST_ERROR_NO_DATA_POINTS = "No data_points."
BAD_REQUEST_ERROR_INVALID_CATEGORY_ID = "Specify category_id in the format 'category_id;universe' to proceed with your query."
BAD_REQUEST_ERROR_NO_INVESTMENT_SOURCE = "Specify the `investments` to proceed with your query."
BAD_REQUEST_ERROR_INVALID_PERCENTILES = "The percentiles should be integer if not null and range from 1 to 100."

# Add custom error messages here for investment list
RESOURCE_NOT_FOUND_ERROR_INVESTMENT_List = (
    "The requested resource does not exist. Please make sure that the provided `list_id` is valid and exists."
)

# Add custom error messages here for portfolio list
RESOURCE_NOT_FOUND_ERROR_PORTFOLIO_List = "The requested resource does not exist. Make sure that the portfolios with provided `portfolio_type` is valid and exist in your Direct account."


# Add custom error messages here for Lookup
RESOURCE_NOT_FOUND_ERROR_LOOKUP_KEYWORD = "Requested resource does not exist for the given `keyword`."
RESOURCE_NOT_FOUND_ERROR_LOOKUP_KEYWORD_UNIVERSE = "Requested resource does not exist for the given `keyword` and `universe`."


# Add custom error messages here for custom database
RESOURCE_NOT_FOUND_ERROR_CUSTOM_DATABASE = "Requested resource does not exist for the given `database_type`."


# Error messages related to 'save portfolio'
BAD_REQUEST_ERROR_NO_PORTFOLIO_HOLDING_ID = "Holding id is missing, specify a valid holding_id."
BAD_REQUEST_INVALID_PORTFOLIO_NAME = "Specify a valid portfolio name."
BAD_REQUEST_PORTFOLIO_NAME_ALREADY_EXISTS = "The portfolio name already exists and 'overwrite_if_exists' is False."
BAD_REQUEST_PORTFOLIO_WEIGHT_ERROR = "Holdings total weight is not 100%."
BAD_REQUEST_INVALID_PORTFOLIO_TYPE = (
    "The specified portfolio_type is not valid. Portfolio type must be either 'model_portfolios' or 'custom_benchmarks'."
)
CLIENT_ACCOUNTS_PORTFOLIO_TYPE_USED = "The 'client_accounts' portfolio_type no longer exists. Portfolios that previously had the type 'client_accounts' now have the type 'model_portfolios'"
BAD_REQUEST_INVALID_PORTFOLIO_HOLDING_DATA_FRAME = "Portfolio holdings data frame is missing required columns. Portfolio holdings data frame should have the following columns: 'Portfolio Date, 'HoldingId', 'Weight'"

# Add custom error messages here for data lookup
RESOURCE_NOT_FOUND_ERROR_DATA_SET_LOOKUP = "Requested resource does not exist for the given `universe`."
RESOURCE_NOT_FOUND_ERROR_DATA_POINT_LOOKUP = "Requested resource does not exist for the given `data_point_ids`."

# Add custom errors for environment
MALFORMED_JWT_ERROR = "Invalid JWT format. JWT should look like 'header.payload.signature' where header and payload are base64 encoded strings and signature is a hex string."

# Add custom errors for delivery
FAILED_DELIVERY_ERROR = "The delivery has failed. Please check your delivery configuration for job_id: {}. Please contact Analytics Lab support team at Morningstar for additional support."
