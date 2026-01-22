"""
Tests for schema validation utilities.
"""

import pytest

from common.schema_validation import (
    SchemaType,
    FieldSchema,
    ValidationError,
    ValidationResult,
    SchemaValidator,
    validate_params,
    validate_weights_sum,
    MODULE5_PARAMS_SCHEMA,
)


class TestFieldSchema:
    """Tests for FieldSchema dataclass."""

    def test_basic_string_field(self):
        schema = FieldSchema(type=SchemaType.STRING, required=True)
        assert schema.type == SchemaType.STRING
        assert schema.required is True

    def test_numeric_field_with_bounds(self):
        schema = FieldSchema(
            type=SchemaType.NUMBER,
            minimum=0,
            maximum=100,
        )
        assert schema.minimum == 0
        assert schema.maximum == 100


class TestSchemaValidator:
    """Tests for SchemaValidator."""

    def test_validate_simple_object(self):
        schema = {
            "name": FieldSchema(type=SchemaType.STRING, required=True),
            "age": FieldSchema(type=SchemaType.INTEGER, required=True),
        }
        validator = SchemaValidator(schema)

        result = validator.validate({"name": "Alice", "age": 30})
        assert result.valid
        assert len(result.errors) == 0

    def test_validate_missing_required_field(self):
        schema = {
            "name": FieldSchema(type=SchemaType.STRING, required=True),
            "age": FieldSchema(type=SchemaType.INTEGER, required=True),
        }
        validator = SchemaValidator(schema)

        result = validator.validate({"name": "Alice"})
        assert not result.valid
        assert len(result.errors) == 1
        assert "age" in str(result.errors[0])

    def test_validate_wrong_type(self):
        schema = {
            "count": FieldSchema(type=SchemaType.INTEGER, required=True),
        }
        validator = SchemaValidator(schema)

        result = validator.validate({"count": "not a number"})
        assert not result.valid
        assert "type" in str(result.errors[0]).lower()

    def test_validate_numeric_bounds(self):
        schema = {
            "score": FieldSchema(
                type=SchemaType.NUMBER,
                minimum=0,
                maximum=100,
            ),
        }
        validator = SchemaValidator(schema)

        # Valid
        assert validator.validate({"score": 50}).valid
        assert validator.validate({"score": 0}).valid
        assert validator.validate({"score": 100}).valid

        # Invalid
        assert not validator.validate({"score": -1}).valid
        assert not validator.validate({"score": 101}).valid

    def test_validate_string_length(self):
        schema = {
            "ticker": FieldSchema(
                type=SchemaType.STRING,
                min_length=1,
                max_length=5,
            ),
        }
        validator = SchemaValidator(schema)

        assert validator.validate({"ticker": "AAPL"}).valid
        assert not validator.validate({"ticker": ""}).valid
        assert not validator.validate({"ticker": "TOOLONG"}).valid

    def test_validate_enum(self):
        schema = {
            "status": FieldSchema(
                type=SchemaType.STRING,
                enum=["active", "inactive", "pending"],
            ),
        }
        validator = SchemaValidator(schema)

        assert validator.validate({"status": "active"}).valid
        assert not validator.validate({"status": "unknown"}).valid

    def test_validate_nullable_field(self):
        schema = {
            "optional": FieldSchema(
                type=SchemaType.STRING,
                nullable=True,
            ),
        }
        validator = SchemaValidator(schema)

        assert validator.validate({"optional": None}).valid
        assert validator.validate({"optional": "value"}).valid

    def test_validate_non_nullable_field(self):
        schema = {
            "required": FieldSchema(
                type=SchemaType.STRING,
                nullable=False,
            ),
        }
        validator = SchemaValidator(schema)

        assert not validator.validate({"required": None}).valid

    def test_validate_additional_properties_allowed(self):
        schema = {
            "name": FieldSchema(type=SchemaType.STRING),
        }
        validator = SchemaValidator(schema, allow_additional=True)

        result = validator.validate({"name": "test", "extra": "allowed"})
        assert result.valid

    def test_validate_additional_properties_rejected(self):
        schema = {
            "name": FieldSchema(type=SchemaType.STRING),
        }
        validator = SchemaValidator(schema, allow_additional=False)

        result = validator.validate({"name": "test", "extra": "not allowed"})
        assert not result.valid

    def test_validate_nested_object(self):
        schema = {
            "config": FieldSchema(
                type=SchemaType.OBJECT,
                properties={
                    "enabled": FieldSchema(type=SchemaType.BOOLEAN, required=True),
                },
            ),
        }
        validator = SchemaValidator(schema)

        assert validator.validate({"config": {"enabled": True}}).valid
        assert not validator.validate({"config": {}}).valid  # missing required

    def test_validate_array_items(self):
        schema = {
            "scores": FieldSchema(
                type=SchemaType.ARRAY,
                items=FieldSchema(type=SchemaType.NUMBER, minimum=0, maximum=100),
            ),
        }
        validator = SchemaValidator(schema)

        assert validator.validate({"scores": [80, 90, 100]}).valid
        assert not validator.validate({"scores": [80, 150]}).valid

    def test_validate_string_pattern(self):
        schema = {
            "date": FieldSchema(
                type=SchemaType.STRING,
                pattern=r"^\d{4}-\d{2}-\d{2}$",
            ),
        }
        validator = SchemaValidator(schema)

        assert validator.validate({"date": "2026-01-15"}).valid
        assert not validator.validate({"date": "01/15/2026"}).valid


class TestValidateParams:
    """Tests for validate_params function."""

    def test_validate_module5_params_valid(self):
        params = {
            "financial_weight": 0.25,
            "clinical_weight": 0.40,
            "catalyst_weight": 0.15,
        }
        result = validate_params(params, "module5")
        assert result.valid

    def test_validate_module5_params_missing_required(self):
        params = {
            "financial_weight": 0.25,
            # Missing clinical_weight and catalyst_weight
        }
        result = validate_params(params, "module5")
        assert not result.valid

    def test_validate_module5_params_out_of_range(self):
        params = {
            "financial_weight": 1.5,  # > 1
            "clinical_weight": 0.40,
            "catalyst_weight": 0.15,
        }
        result = validate_params(params, "module5")
        assert not result.valid

    def test_validate_unknown_schema(self):
        result = validate_params({}, "nonexistent")
        assert not result.valid
        assert "Unknown schema" in str(result.errors[0])


class TestValidateWeightsSum:
    """Tests for validate_weights_sum function."""

    def test_weights_sum_valid(self):
        params = {
            "w1": 0.25,
            "w2": 0.35,
            "w3": 0.40,
        }
        result = validate_weights_sum(params, ["w1", "w2", "w3"], expected_sum=1.0)
        assert result.valid

    def test_weights_sum_invalid(self):
        params = {
            "w1": 0.25,
            "w2": 0.35,
            "w3": 0.50,  # Sum = 1.10
        }
        result = validate_weights_sum(params, ["w1", "w2", "w3"], expected_sum=1.0)
        assert not result.valid

    def test_weights_sum_with_tolerance(self):
        params = {
            "w1": 0.33,
            "w2": 0.33,
            "w3": 0.33,  # Sum = 0.99
        }
        result = validate_weights_sum(
            params, ["w1", "w2", "w3"], expected_sum=1.0, tolerance=0.02
        )
        assert result.valid

    def test_weights_sum_skips_none(self):
        params = {
            "w1": 0.50,
            "w2": None,  # Should be skipped
            "w3": 0.50,
        }
        result = validate_weights_sum(params, ["w1", "w2", "w3"], expected_sum=1.0)
        assert result.valid
