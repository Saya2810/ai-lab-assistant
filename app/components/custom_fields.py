import streamlit as st


def find_matching_option(
    value,
    available_options,
):
    if value is None:
        return None

    normalized = str(value).strip().lower()

    for option in available_options:
        if str(option).strip().lower() == normalized:
            return option

    return None


def find_ai_measurement(
    field_name,
    ai_measurements,
):
    for name, measurement in ai_measurements.items():
        if (
            name.strip().lower()
            == field_name.strip().lower()
        ):
            return measurement

    return None


def render_number_field(
    field_name,
    field_data,
    current_value,
    ai_unit,
):
    units = field_data.get(
        "units",
        [],
    )

    current_unit = (
        ai_unit
        or field_data.get("unit")
    )

    try:
        numeric_value = float(
            current_value
        )
    except (ValueError, TypeError):
        numeric_value = 0.0

    if units:
        value_column, unit_column = (
            st.columns([3, 1])
        )

        with value_column:
            value = st.number_input(
                field_name,
                value=numeric_value,
                key=f"field_{field_name}",
            )

        with unit_column:
            unit_index = (
                units.index(current_unit)
                if current_unit in units
                else 0
            )

            unit = st.selectbox(
                "Unit",
                options=units,
                index=unit_index,
                key=f"unit_{field_name}",
            )

    else:
        value = st.number_input(
            field_name,
            value=numeric_value,
            key=f"field_{field_name}",
        )

        unit = current_unit

    result = field_data.copy()
    result["value"] = str(value)

    if unit is not None:
        result["unit"] = unit

    return result


def render_checkbox_field(
    field_name,
    field_data,
    current_value,
):
    checked = (
        str(current_value).lower()
        in ("1", "true", "yes")
    )

    value = st.checkbox(
        field_name,
        value=checked,
        key=f"field_{field_name}",
    )

    result = field_data.copy()
    result["value"] = (
        "1" if value else "0"
    )

    return result


def render_select_field(
    field_name,
    field_data,
    current_value,
):
    options = field_data.get(
        "options",
        [],
    )

    if not options:
        value = st.text_input(
            field_name,
            value=str(current_value),
            key=f"field_{field_name}",
        )

    else:
        matched = find_matching_option(
            current_value,
            options,
        )

        index = (
            options.index(matched)
            if matched
            else 0
        )

        value = st.selectbox(
            field_name,
            options=options,
            index=index,
            key=f"field_{field_name}",
        )

    result = field_data.copy()
    result["value"] = value

    return result


def render_text_field(
    field_name,
    field_data,
    current_value,
):
    value = st.text_input(
        field_name,
        value=str(current_value),
        key=f"field_{field_name}",
    )

    result = field_data.copy()
    result["value"] = value

    return result


def render_custom_field(
    field_name,
    field_data,
    ai_measurements,
):
    field_type = field_data.get(
        "type",
        "text",
    )

    current_value = field_data.get(
        "value",
        "",
    )

    ai_unit = None

    measurement = find_ai_measurement(
        field_name,
        ai_measurements,
    )

    if measurement is not None:
        if isinstance(
            measurement,
            dict,
        ):
            current_value = measurement.get(
                "value",
                current_value,
            )

            ai_unit = measurement.get(
                "unit"
            )

        else:
            current_value = measurement

    if field_type == "number":
        return render_number_field(
            field_name,
            field_data,
            current_value,
            ai_unit,
        )

    if field_type == "checkbox":
        return render_checkbox_field(
            field_name,
            field_data,
            current_value,
        )

    if field_type in (
        "select",
        "dropdown",
    ):
        return render_select_field(
            field_name,
            field_data,
            current_value,
        )

    return render_text_field(
        field_name,
        field_data,
        current_value,
    )