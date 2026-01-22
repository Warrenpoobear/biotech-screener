"""
Schema validation utilities for production hardening.

Provides JSON Schema-like validation without external dependencies.
Uses stdlib-only implementation for zero-dependency core requirement.

Version: 1.0.0
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


class SchemaType(Enum):
    """Supported schema types."""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"


@dataclass
class ValidationError:
    """Single validation error."""
    path: str
    message: str
    value: Any = None

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


@dataclass
class ValidationResult:
    """Result of schema validation."""
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid

    def error_messages(self) -> List[str]:
        return [str(e) for e in self.errors]


@dataclass
class FieldSchema:
    """Schema for a single field."""
    type: Union[SchemaType, List[SchemaType]]
    required: bool = False
    nullable: bool = False
    minimum: Optional[Union[int, float, Decimal]] = None
    maximum: Optional[Union[int, float, Decimal]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    enum: Optional[List[Any]] = None
    items: Optional["FieldSchema"] = None  # For arrays
    properties: Optional[Dict[str, "FieldSchema"]] = None  # For objects
    additional_properties: bool = True  # Allow extra keys in objects
    description: Optional[str] = None
    default: Any = None


class SchemaValidator:
    """
    Validates data against a schema definition.

    Example:
        schema = {
            "financial_weight": FieldSchema(
                type=SchemaType.NUMBER,
                required=True,
                minimum=0,
                maximum=1,
            ),
            "clinical_weight": FieldSchema(
                type=SchemaType.NUMBER,
                required=True,
                minimum=0,
                maximum=1,
            ),
        }

        validator = SchemaValidator(schema)
        result = validator.validate({"financial_weight": 0.25, "clinical_weight": 0.40})
    """

    def __init__(
        self,
        schema: Dict[str, FieldSchema],
        allow_additional: bool = True,
        strict_types: bool = False,
    ):
        """
        Initialize validator.

        Args:
            schema: Dict mapping field names to FieldSchema
            allow_additional: Allow fields not in schema
            strict_types: Strict type checking (no coercion)
        """
        self.schema = schema
        self.allow_additional = allow_additional
        self.strict_types = strict_types

    def validate(self, data: Any, path: str = "$") -> ValidationResult:
        """
        Validate data against schema.

        Args:
            data: Data to validate
            path: JSON path for error messages

        Returns:
            ValidationResult with valid flag and errors
        """
        errors: List[ValidationError] = []

        if not isinstance(data, dict):
            errors.append(ValidationError(
                path=path,
                message=f"Expected object, got {type(data).__name__}",
                value=data,
            ))
            return ValidationResult(valid=False, errors=errors)

        # Check required fields
        for field_name, field_schema in self.schema.items():
            if field_schema.required and field_name not in data:
                errors.append(ValidationError(
                    path=f"{path}.{field_name}",
                    message="Required field missing",
                ))

        # Check each field
        for field_name, value in data.items():
            field_path = f"{path}.{field_name}"

            if field_name in self.schema:
                field_errors = self._validate_field(
                    value, self.schema[field_name], field_path
                )
                errors.extend(field_errors)
            elif not self.allow_additional:
                errors.append(ValidationError(
                    path=field_path,
                    message="Additional property not allowed",
                    value=value,
                ))

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def _validate_field(
        self,
        value: Any,
        schema: FieldSchema,
        path: str
    ) -> List[ValidationError]:
        """Validate a single field."""
        errors: List[ValidationError] = []

        # Handle null
        if value is None:
            if schema.nullable:
                return errors
            else:
                errors.append(ValidationError(
                    path=path,
                    message="Field cannot be null",
                    value=value,
                ))
                return errors

        # Type checking
        expected_types = schema.type if isinstance(schema.type, list) else [schema.type]
        type_valid = False

        for expected_type in expected_types:
            if self._check_type(value, expected_type):
                type_valid = True
                break

        if not type_valid:
            type_names = [t.value for t in expected_types]
            errors.append(ValidationError(
                path=path,
                message=f"Expected type {type_names}, got {type(value).__name__}",
                value=value,
            ))
            return errors  # Skip further validation if type is wrong

        # Numeric constraints
        if schema.minimum is not None:
            if isinstance(value, (int, float, Decimal)) and value < schema.minimum:
                errors.append(ValidationError(
                    path=path,
                    message=f"Value {value} below minimum {schema.minimum}",
                    value=value,
                ))

        if schema.maximum is not None:
            if isinstance(value, (int, float, Decimal)) and value > schema.maximum:
                errors.append(ValidationError(
                    path=path,
                    message=f"Value {value} above maximum {schema.maximum}",
                    value=value,
                ))

        # String constraints
        if isinstance(value, str):
            if schema.min_length is not None and len(value) < schema.min_length:
                errors.append(ValidationError(
                    path=path,
                    message=f"String length {len(value)} below minimum {schema.min_length}",
                    value=value,
                ))

            if schema.max_length is not None and len(value) > schema.max_length:
                errors.append(ValidationError(
                    path=path,
                    message=f"String length {len(value)} above maximum {schema.max_length}",
                    value=value,
                ))

            if schema.pattern is not None:
                import re
                if not re.match(schema.pattern, value):
                    errors.append(ValidationError(
                        path=path,
                        message=f"String does not match pattern {schema.pattern}",
                        value=value,
                    ))

        # Enum constraint
        if schema.enum is not None and value not in schema.enum:
            errors.append(ValidationError(
                path=path,
                message=f"Value not in allowed values: {schema.enum}",
                value=value,
            ))

        # Array validation
        if isinstance(value, list) and schema.items is not None:
            for i, item in enumerate(value):
                item_errors = self._validate_field(
                    item, schema.items, f"{path}[{i}]"
                )
                errors.extend(item_errors)

        # Nested object validation
        if isinstance(value, dict) and schema.properties is not None:
            nested_validator = SchemaValidator(
                schema.properties,
                allow_additional=schema.additional_properties,
            )
            nested_result = nested_validator.validate(value, path)
            errors.extend(nested_result.errors)

        return errors

    def _check_type(self, value: Any, expected: SchemaType) -> bool:
        """Check if value matches expected type."""
        if expected == SchemaType.STRING:
            return isinstance(value, str)
        elif expected == SchemaType.NUMBER:
            return isinstance(value, (int, float, Decimal)) and not isinstance(value, bool)
        elif expected == SchemaType.INTEGER:
            return isinstance(value, int) and not isinstance(value, bool)
        elif expected == SchemaType.BOOLEAN:
            return isinstance(value, bool)
        elif expected == SchemaType.ARRAY:
            return isinstance(value, list)
        elif expected == SchemaType.OBJECT:
            return isinstance(value, dict)
        elif expected == SchemaType.NULL:
            return value is None
        return False


# =============================================================================
# Pre-defined Schemas for biotech-screener
# =============================================================================

# Schema for Module 5 scoring parameters
MODULE5_PARAMS_SCHEMA = {
    "financial_weight": FieldSchema(
        type=SchemaType.NUMBER,
        required=True,
        minimum=0,
        maximum=1,
        description="Weight for financial health score",
    ),
    "clinical_weight": FieldSchema(
        type=SchemaType.NUMBER,
        required=True,
        minimum=0,
        maximum=1,
        description="Weight for clinical development score",
    ),
    "catalyst_weight": FieldSchema(
        type=SchemaType.NUMBER,
        required=True,
        minimum=0,
        maximum=1,
        description="Weight for catalyst score",
    ),
    "momentum_weight": FieldSchema(
        type=SchemaType.NUMBER,
        required=False,
        nullable=True,
        minimum=0,
        maximum=1,
        description="Weight for momentum signal",
    ),
    "smart_money_weight": FieldSchema(
        type=SchemaType.NUMBER,
        required=False,
        nullable=True,
        minimum=0,
        maximum=1,
        description="Weight for smart money signal",
    ),
    "pos_weight": FieldSchema(
        type=SchemaType.NUMBER,
        required=False,
        nullable=True,
        minimum=0,
        maximum=1,
        description="Weight for probability of success",
    ),
}

# Schema for risk gates parameters
RISK_GATES_SCHEMA = {
    "max_concentration_pct": FieldSchema(
        type=SchemaType.NUMBER,
        required=True,
        minimum=0,
        maximum=100,
        description="Maximum position concentration percentage",
    ),
    "min_liquidity_usd": FieldSchema(
        type=SchemaType.NUMBER,
        required=True,
        minimum=0,
        description="Minimum daily trading volume in USD",
    ),
    "max_volatility": FieldSchema(
        type=SchemaType.NUMBER,
        required=False,
        nullable=True,
        minimum=0,
        maximum=10,
        description="Maximum allowed volatility",
    ),
}

# Schema for pipeline configuration
PIPELINE_CONFIG_SCHEMA = {
    "data_paths": FieldSchema(
        type=SchemaType.OBJECT,
        required=True,
        properties={
            "universe": FieldSchema(type=SchemaType.STRING, required=True),
            "financial": FieldSchema(type=SchemaType.STRING, required=True),
            "trials": FieldSchema(type=SchemaType.STRING, required=True),
        },
    ),
    "market_cap_min_mm": FieldSchema(
        type=SchemaType.NUMBER,
        required=False,
        minimum=0,
        description="Minimum market cap in millions",
    ),
    "enhancements": FieldSchema(
        type=SchemaType.OBJECT,
        required=False,
        properties={
            "pos_engine": FieldSchema(type=SchemaType.BOOLEAN, required=False),
            "short_interest": FieldSchema(type=SchemaType.BOOLEAN, required=False),
            "regime_detection": FieldSchema(type=SchemaType.BOOLEAN, required=False),
        },
    ),
}


def validate_params(
    params: Dict[str, Any],
    schema_name: str = "module5"
) -> ValidationResult:
    """
    Validate parameters against a predefined schema.

    Args:
        params: Parameters dict to validate
        schema_name: Name of schema ("module5", "risk_gates", "pipeline")

    Returns:
        ValidationResult
    """
    schemas = {
        "module5": MODULE5_PARAMS_SCHEMA,
        "risk_gates": RISK_GATES_SCHEMA,
        "pipeline": PIPELINE_CONFIG_SCHEMA,
    }

    if schema_name not in schemas:
        return ValidationResult(
            valid=False,
            errors=[ValidationError(
                path="$",
                message=f"Unknown schema: {schema_name}. Available: {list(schemas.keys())}",
            )],
        )

    validator = SchemaValidator(schemas[schema_name])
    return validator.validate(params)


def validate_weights_sum(
    params: Dict[str, Any],
    weight_fields: List[str],
    expected_sum: float = 1.0,
    tolerance: float = 0.01,
) -> ValidationResult:
    """
    Validate that weight fields sum to expected value.

    Args:
        params: Parameters dict
        weight_fields: List of weight field names
        expected_sum: Expected sum (default 1.0)
        tolerance: Allowed deviation from expected sum

    Returns:
        ValidationResult
    """
    errors: List[ValidationError] = []

    weights = []
    for field in weight_fields:
        if field in params and params[field] is not None:
            try:
                weights.append(float(params[field]))
            except (TypeError, ValueError) as e:
                errors.append(ValidationError(
                    path=f"$.{field}",
                    message=f"Cannot convert to number: {e}",
                    value=params.get(field),
                ))

    if not errors and weights:
        total = sum(weights)
        if abs(total - expected_sum) > tolerance:
            errors.append(ValidationError(
                path="$",
                message=f"Weights sum to {total:.4f}, expected {expected_sum} (±{tolerance})",
                value=weights,
            ))

    return ValidationResult(valid=len(errors) == 0, errors=errors)


__all__ = [
    "SchemaType",
    "FieldSchema",
    "ValidationError",
    "ValidationResult",
    "SchemaValidator",
    "validate_params",
    "validate_weights_sum",
    "MODULE5_PARAMS_SCHEMA",
    "RISK_GATES_SCHEMA",
    "PIPELINE_CONFIG_SCHEMA",
]
