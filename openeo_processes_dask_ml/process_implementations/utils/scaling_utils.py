import xarray as xr

from pystac.extensions.mlm import ValueScaling, ValueScalingType


def _raise_value_error(scale_type: str, req_props: list[str]):
    """
    Raise a ValueError stating that rquried parameters are missing for a scale type
    :param scale_type: the scale type
    :param req_props: list of rquired parameters
    :raise ValueError:
    """
    raise ValueError(
        f"ValueScaling Type {scale_type} requires the following vlaues: "
        f"{', '.join(req_props)}"
    )


def _test_req_props(scale_obj: ValueScaling, props: list[str]):
    """
    Test whether a given ValueScaling object has certain properties set
    :param scale_obj: The ValueScaling object to be tested
    :param props: list of property names to be tested
    :raise ValueError: if the object does not contain the required parameters
    """
    prop_available = [scale_obj.properties.get(p) is not None for p in props]
    valid = all(prop_available)
    if not valid:
        _raise_value_error(scale_obj.type, props)


def _validate_scaling_obj(scale_obj: ValueScaling):
    """
    Test whether a given ValueScaling object is valid, i.e. certain properties are set
    that are required for a given ValueScaling type
    :param scale_obj: The ValueScaling object to be tested
    """
    if scale_obj.type in [ValueScalingType.MIN_MAX, ValueScalingType.CLIP]:
        req_props = ["minimum", "maximum"]

    elif scale_obj.type == ValueScalingType.Z_SCORE:
        req_props = ["mean", "stddev"]

    elif scale_obj.type == ValueScalingType.CLIP_MIN:
        req_props = ["minimum"]

    elif scale_obj.type == ValueScalingType.CLIP_MAX:
        req_props = ["maximum"]

    elif scale_obj.type in [ValueScalingType.OFFSET, ValueScalingType.SCALE]:
        req_props = ["value"]

    elif scale_obj.type == ValueScalingType.PROCESSING:
        req_props = ["format", "expression"]

    else:
        raise ValueError(f"Unspported ValueScaling type: {scale_obj.type}")

    _test_req_props(scale_obj, req_props)


def scale_datacube(dc: xr.DataArray, scale_obj: ValueScaling) -> xr.DataArray:
    """
    Scale a datacube using a ValueScaling object
    :param dc: The datacube to be scaled
    :param scale_obj: the ValueScaling object
    :return: the scaled datacube
    """

    _validate_scaling_obj(scale_obj)

    if scale_obj.type == ValueScalingType.MIN_MAX:
        return (dc - scale_obj.minimum) / (scale_obj.maximum - scale_obj.minimum)

    if scale_obj.type == ValueScalingType.Z_SCORE:
        return (dc - scale_obj.mean) / scale_obj.stddev

    if scale_obj.type == ValueScalingType.CLIP:
        return dc.clip(scale_obj.minimum, scale_obj.maximum, keep_attrs=True)

    if scale_obj.type == ValueScalingType.CLIP_MIN:
        return dc.clip(min=scale_obj.minimum, keep_attrs=True)

    if scale_obj.type == ValueScalingType.CLIP_MAX:
        return dc.clip(max=scale_obj.maximum, keep_attrs=True)

    if scale_obj.type == ValueScalingType.OFFSET:
        return dc - scale_obj.value

    if scale_obj.type == ValueScalingType.SCALE:
        return dc / scale_obj.value

    if scale_obj.type == ValueScalingType.PROCESSING:
        # todo: implement this
        raise NotImplementedError(
            "Custom Processing Expression for Value Scaling are not implemented."
        )

    raise ValueError(f"Invalue ValueScaling Type: {scale_obj.type}")
