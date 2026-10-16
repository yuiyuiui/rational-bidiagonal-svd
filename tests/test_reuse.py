"""Cached exact checks must preserve isolation and all budget dependencies."""
from dataclasses import replace
from rational_bidiagonal_svd._budget import allocate_gram, checks


def test_exact_budget_results_are_detached_and_parameter_sensitive():
    budget = allocate_gram(2, 191)
    original = checks(budget)
    assert all(original.values())
    original['domain'] = False
    assert checks(budget)['domain']
    unsafe = replace(budget, scan_bits=1)
    assert not checks(unsafe)['scan_condition']
    assert all(checks(budget).values())
