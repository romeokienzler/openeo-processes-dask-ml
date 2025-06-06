import xarray as xr
from pystac.extensions.mlm import ProcessingExpression
import importlib

from openeo_processes_dask_ml.process_implementations.exceptions import (
    ExpressionEvaluationException
)


def _raise_format_not_implemented(f: str):
    raise NotImplementedError(
        f"Execution of Processing Expression of format {f} is currently not available"
    )


def _run_python(dc: xr.DataArray, expression: str):
    module_name, function_name = expression.split(":")

    module = importlib.import_module(module_name)
    fn = module.__getattribute__(function_name)
    return fn(dc)


def run_process_expression(dc: xr.DataArray, proc: ProcessingExpression):
    v = ["gdal-calc", "openeo", "rio-calc", "python", "docker", "uri"]

    p_format = proc.format
    p_expression = proc.expression

    if p_format == "python":
        try:
            new_dc = _run_python(dc, p_expression)
        except ModuleNotFoundError as e:
            raise ExpressionEvaluationException(
                f"Could not execute python expression: {str(e)}"
            )
        except AttributeError as e:
            raise ExpressionEvaluationException(
                f"Could not execute python expression: {str(e)}"
            )
        return new_dc

    if p_format == "uri":
        _raise_format_not_implemented(p_format)

    if p_format == "docker":
        _raise_format_not_implemented(p_format)

    if p_format == "rio-calc":
        _raise_format_not_implemented(p_format)

    if p_format == "openeo":
        _raise_format_not_implemented(p_format)

    if p_format == "gdal-calc":
        _raise_format_not_implemented(p_format)

    _raise_format_not_implemented(p_format)