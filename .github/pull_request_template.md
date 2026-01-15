## Summary

Brief description of the changes in this PR.

## Changes

- Change 1
- Change 2

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update
- [ ] Refactoring (no functional changes)

## Testing

- [ ] Tests pass locally (`pytest tests/`)
- [ ] New tests added for new functionality
- [ ] Manual testing performed

## PIT Compliance Checklist

For changes affecting screening/backtesting logic:

- [ ] No look-ahead bias introduced
- [ ] Deterministic output verified (re-runs produce identical results)
- [ ] Date filters use `<=` as_of_date
- [ ] State snapshots are PIT-safe

## Additional Notes

Any additional context or screenshots.
