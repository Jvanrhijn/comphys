def generate_latex_table(left_column, right_column, header_left, header_right, rounding=None) -> str:
    """Generate a LaTeX table from two lists

    :param left_column: list with values to list in left column
    :param right_column: list with values to list in right column
    :param header_left: left header
    :param header_right: right header
    :param rounding: digits to round values to, defaults to no rounding
    :return: string containing LaTeX formatted table
    """
    table = "{0} & {1}\\\\\\hline\n".format(header_left, header_right)

    for left, right in zip(left_column, right_column):
        if rounding is not None:
            left = round(left, rounding)
            right = round(right, rounding)
        table += str(left) + " & " + str(right) + "\\\\ \n"

    return table
