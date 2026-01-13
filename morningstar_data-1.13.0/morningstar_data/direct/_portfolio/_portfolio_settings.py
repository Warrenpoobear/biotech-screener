import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

import simplejson as json

from ..._base import _logger
from .._config import _Config
from ._common import PortfolioDataApiBackend

_portfolio_api = PortfolioDataApiBackend()
_config = _Config()


class PerformanceSource(Enum):
    Calculated_Based_On_Underlying_Positions = 1
    Imported_Account_Level_Performance = 2


class RebalanceOnType(Enum):
    Calendar_Period_End = 0
    Rolling_Period_Based_On_Portfolio_Date = 1


class RebalanceFrequency(Enum):
    Monthly = 2
    Quarterly = 3
    Semi_Annually = 4
    Annually = 5
    Buy_And_Hold = 7
    Daily = 8


class CalculationType(Enum):
    Earliest_Common = 1
    First_Portfolio_Date = 2
    Earliest_Available = 3


class PerformanceSeriesFrequency(Enum):
    Monthly = 4
    Quarterly = 5
    Daily = 6


class ManagementFeeEffectivePeriodType(Enum):
    Entire_Period = 0
    Custom_Period = 1


class ManagementFeeFrequency(Enum):
    Daily = 1
    Monthly = 2
    Quarterly = 3
    Semi_Annually = 4
    Annually = 5


class BenchmarkType(Enum):
    Morningstar_Security = 1
    User_Defined_Security = 2


class MissingTNAHandling(Enum):
    Roll_Previous_Value_Forward = 1
    Infer_Forward_In_Time = 2
    Infer_Backward_In_Time = 3


class Benchmark:
    benchmark_id: str
    benchmark_type_id: int = 1
    investment_id: str

    def __init__(self, investment_id: str, benchmark_id: str) -> None:
        self.investment_id = investment_id
        self.benchmark_type_id = 1
        self.benchmark_id = benchmark_id


class PortfolioSetting:
    notes: None
    benchmark_id: str
    missing_tna_handling_id: MissingTNAHandling = MissingTNAHandling.Infer_Forward_In_Time
    management_fee_effective_period_type: ManagementFeeEffectivePeriodType = ManagementFeeEffectivePeriodType.Entire_Period
    management_fee_frequency: ManagementFeeFrequency = ManagementFeeFrequency.Annually
    management_fee_for_performance: Optional[float] = None
    risk_free_proxy_id: str = "XIUSA000OC"
    rebalance_frequency_id: RebalanceFrequency = RebalanceFrequency.Buy_And_Hold
    start_date_calculation_type_id: CalculationType = CalculationType.First_Portfolio_Date
    rebalance_based_on: RebalanceOnType = RebalanceOnType.Calendar_Period_End
    performance_source_id: PerformanceSource = PerformanceSource.Calculated_Based_On_Underlying_Positions
    performance_series_frequency_id: PerformanceSeriesFrequency = PerformanceSeriesFrequency.Quarterly
    portfolio_to_id: None
    model_portfolio_id: None
    model_portfolio_name: None
    use_gross_return: bool = False
    real_sma_flag: bool = False
    is_combined_series: bool = False
    combined_series: None
    base_currency: str = "USD"
    secondary_benchmark_id: str = "XIUSA04FMS"
    portfolio_type: Optional[str]
    portfolio_id: Optional[str]
    name: Optional[str]
    benchmarks: List[Benchmark]

    def __init__(
        self,
        name: Optional[str] = None,
        portfolio_type: Optional[str] = None,
        portfolio_id: Optional[str] = None,
    ) -> None:
        """
        Build portfolio settings with default value.
        """
        self.notes = None
        self.benchmark_id = "XIUSA04G92"
        self.missing_tna_handling_id = MissingTNAHandling.Infer_Forward_In_Time
        self.management_fee_effective_period_type = ManagementFeeEffectivePeriodType.Entire_Period
        self.management_fee_frequency = ManagementFeeFrequency.Annually
        self.management_fee_for_performance = None
        self.risk_free_proxy_id = "XIUSA000OC"
        self.rebalance_frequency_id = RebalanceFrequency.Buy_And_Hold
        self.start_date_calculation_type_id = CalculationType.First_Portfolio_Date
        self.rebalance_based_on = RebalanceOnType.Calendar_Period_End
        self.performance_source_id = PerformanceSource.Calculated_Based_On_Underlying_Positions
        self.performance_series_frequency_id = PerformanceSeriesFrequency.Quarterly
        self.portfolio_to_id = None
        self.model_portfolio_id = None
        self.model_portfolio_name = None
        self.use_gross_return = False
        self.real_sma_flag = False
        self.is_combined_series = False
        self.combined_series = None
        self.base_currency = "USD"
        self.secondary_benchmark_id = "XIUSA04FMS"
        self.portfolio_type = portfolio_type
        self.portfolio_id = portfolio_id
        self.name = name

    def get_formatted_json_body(self) -> Dict[Any, Any]:
        """
        Build portfolio settings post body.
        """
        # Generate portfolio benchmark objects request dict.
        _logger.info("Generate portfolio benchmark objects request.")
        benchmark_uuid = str(uuid.uuid4())
        secondary_benchmark_uuid = str(uuid.uuid4())
        risk_free_proxy_uuid = str(uuid.uuid4())

        # Each benchmarks need a UUID.
        self.benchmarks = [
            Benchmark(self.benchmark_id, benchmark_uuid),
            Benchmark(self.secondary_benchmark_id, secondary_benchmark_uuid),
            Benchmark(self.risk_free_proxy_id, risk_free_proxy_uuid),
        ]

        benchmarks = []
        # Generate benchmark dict list.
        for benchmark in self.benchmarks:
            benchmarks.append(
                {
                    "securityId": benchmark.investment_id,
                    "benchmarkTypeId": benchmark.benchmark_type_id,
                    "benchmarkId": benchmark.benchmark_id,
                }
            )
        # Generate portfolio settings with default values .
        _logger.info("Generate portfolio settings with default values")
        to_return = {
            "settings": {
                "portfolio": {
                    "portfolioType": self.portfolio_type,
                    "notes": self.notes,
                    "portfolioId": self.portfolio_id,
                    "benchmarkId": benchmark_uuid,
                    "baseCurrency": self.base_currency,
                    "missingTNAHandlingId": self.missing_tna_handling_id.value,
                    "managementFeeEffectivePeriodType": self.management_fee_effective_period_type.value,
                    "managementFeeFrequency": self.management_fee_frequency.value,
                    "managementFeeForPerformance": self.management_fee_for_performance,
                    "riskFreeProxyId": secondary_benchmark_uuid,
                    "rebalanceFrequencyId": self.rebalance_frequency_id.value,
                    "secondaryBenchmarkId": risk_free_proxy_uuid,
                    "startDateCalculationTypeId": self.start_date_calculation_type_id.value,
                    "rebalanceBasedOn": self.rebalance_based_on.value,
                    "performanceSourceId": self.performance_source_id.value,
                    "performanceSeriesFrequencyId": self.performance_series_frequency_id.value,
                    "portfolio2Id": self.portfolio_to_id,
                    "modelPortfolioId": self.model_portfolio_id,
                    "modelPortfolioName": self.model_portfolio_name,
                    "useGrossReturn": self.use_gross_return,
                    "realSMAFlag": self.real_sma_flag,
                    "isCombinedSeries": self.is_combined_series,
                    "combinedSeries": self.combined_series,
                    "name": self.name,
                },
                "benchmarks": benchmarks,
            }
        }
        _logger.debug(f"Generate portfolio settings with default values. {to_return}")
        return to_return

    def save_portfolio_settings(self) -> Any:
        url = f"{_config.portfolio_service_url()}portfoliodataservice/v1/portfolios/update-portfolio-settings"
        _logger.debug(f"Save portfolio settings with portfolio API: {url}")
        response_json = _portfolio_api.do_post_request(url, json.dumps(self.get_formatted_json_body(), ignore_nan=True))
        _logger.debug(f"Save portfolio API response: {response_json}")
        return response_json
