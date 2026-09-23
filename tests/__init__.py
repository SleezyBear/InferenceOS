"""Test package for InferenceOS.

Making ``tests`` a package lets test modules import shared synthetic fixtures via
``from tests.diagnosis_fixtures import ...`` while keeping ``mypy``'s module
resolution unambiguous.
"""
