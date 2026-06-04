from typing import Any

import matplotlib.pyplot as plt
import numpy as np

CEND = "\33[0m"
CBOLD = "\33[1m"
CITALIC = "\33[3m"
CURL = "\33[4m"
CBLINK = "\33[5m"
CBLINK2 = "\33[6m"
CSELECTED = "\33[7m"

CBLACK = "\33[30m"
CRED = "\33[31m"
CGREEN = "\33[32m"
CYELLOW = "\33[33m"
CBLUE = "\33[34m"
CCYAN = '\33[96m'
CMAGENTA = '\033[35m'
CVIOLET = "\33[35m"
CBEIGE = "\33[36m"
CWHITE = "\33[37m"

CBLACKBG = "\33[40m"
CREDBG = "\33[41m"
CGREENBG = "\33[42m"
CYELLOWBG = "\33[43m"
CBLUEBG = "\33[44m"
CVIOLETBG = "\33[45m"
CBEIGEBG = "\33[46m"
CWHITEBG = "\33[47m"

CGREY = "\33[90m"
CRED2 = "\33[91m"
CGREEN2 = "\33[92m"
CYELLOW2 = "\33[93m"
CBLUE2 = "\33[94m"
CCYAN2 = "\033[36m"
CVIOLET2 = "\33[95m"
CBEIGE2 = "\33[96m"
CWHITE2 = "\33[97m"

CGREYBG = "\33[100m"
CREDBG2 = "\33[101m"
CGREENBG2 = "\33[102m"
CYELLOWBG2 = "\33[103m"
CBLUEBG2 = "\33[104m"
CVIOLETBG2 = "\33[105m"
CBEIGEBG2 = "\33[106m"
CWHITEBG2 = "\33[107m"


def rgb_color_sequence(r: int | float, g: int | float, b: int | float,
                       *, format_type: str = 'foreground') -> str:
    """
    generates a color-codes, that change the color of text in console outputs.

    rgb values must be numbers between 0 and 255 or 0.0 and 1.0.

    :param r:               red value.
    :param g:               green value
    :param b:               blue value

    :param format_type:     specifies weather the foreground-color or the background-color shall be adjusted.
                            valid options: 'foreground','background'
    :return:                a string that contains the color-codes.
    """
    # type: ignore # noqa: F401
    if format_type == 'foreground':
        f = '\033[38;2;{};{};{}m'.format  # font rgb format
    elif format_type == 'background':
        f = '\033[48;2;{};{};{}m'.format  # font background rgb format
    else:
        raise ValueError(f"format {format_type} is not defined. Use 'foreground' or 'background'.")
    rgb = [r, g, b]

    if isinstance(r, int) and isinstance(g, int) and isinstance(b, int):
        if min(rgb) < 0 and max(rgb) > 255:
            raise ValueError("rgb values must be numbers between 0 and 255 or 0.0 and 1.0")
        return f(r, g, b)
    if isinstance(r, float) and isinstance(g, float) and isinstance(b, float):
        if min(rgb) < 0 and max(rgb) > 1:
            raise ValueError("rgb values must be numbers between 0 and 255 or 0.0 and 1.0")
        return f(*[int(n * 255) for n in [r, g, b]])


def wrap_with_color_codes(s: object, /, r: int | float, g: int | float, b: int | float, **kwargs) \
        -> str:
    """
    stringify an object and wrap it with console color codes. It adds the color control sequence in front and one
    at the end that resolves the color again.

    rgb values must be numbers between 0 and 255 or 0.0 and 1.0.

    :param s: the object to stringify and wrap
    :param r: red value.
    :param g: green value.
    :param b: blue value.
    :param kwargs: additional argument for the 'DisjunctiveGraphJspVisualizer.rgb_color_sequence'-method.
    :return:
    """
    return f"{rgb_color_sequence(r, g, b, **kwargs)}" \
           f"{s}" \
           f"{CEND}"


def wrap_evenly_spaced_color(s: Any, n_of_item: int, n_classes: int, c_map="rainbow") -> str:
    """
    Wraps a string with a color scale (a matplotlib c_map) based on the n_of_item and n_classes.
    This function is used to color code the available actions in the MCTS tree visualisation.
    The children of the MCTS tree are colored based on their action for a clearer visualisation.

    :param s: the string (or object) to be wrapped. objects are converted to string (using the __str__ function).
    :param n_of_item: the index of the item to be colored. In a mcts tree, this is the (parent-)action of the node.
    :param n_classes: the number of classes (or items) to be colored. In a mcts tree, this is the number of available actions.
    :param c_map: the colormap to be used (default is 'rainbow').
                  The colormap can be any matplotlib colormap, e.g. 'viridis', 'plasma', 'inferno', 'magma', 'cividis'.
    :return: a string that contains the color-codes (prefix and suffix) and the string s in between.
    """
    if s is None or n_of_item is None or n_classes is None:
        return s

    c_map = plt.cm.get_cmap(c_map)  # select the desired cmap
    arr = np.linspace(0, 1, n_classes + 1)  # create a list with numbers from 0 to 1 with n items

    color_vals = c_map(arr[n_of_item])[:-1]
    color_asni = rgb_color_sequence(*color_vals, format_type='foreground')

    return f"{color_asni}{s}{CEND}"


def wrap_with_color_scale(s: str, value: float, min_val: float, max_val: float, c_map=None) -> str:
    """
    Wraps a string with a color scale (a matplotlib c_map) based on the value, min_val, and max_val.

    :param s: the string to be wrapped
    :param value: the value to be mapped to a color
    :param min_val: the minimum value of the scale
    :param max_val: the maximum value of the scale
    :param c_map: the colormap to be used (default is 'rainbow')
    :return:
    """
    if s is None or min_val is None or max_val is None or min_val >= max_val:
        return s

    if c_map is not None:
        import matplotlib
        c_map = matplotlib.colormaps[c_map]  # select the desired cmap
    else:
        from matplotlib.colors import LinearSegmentedColormap
        colors = [
            np.array([255 / 255, 100 / 255, 128 / 255, 1.0]),  # RGBA values
            np.array([63 / 255, 197 / 255, 161 / 255, 1.0]),  # RGBA values
        ]
        c_map = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

    color_vals = c_map((value - min_val) / (max_val - min_val))[:-1]
    color_asni = rgb_color_sequence(*color_vals, format_type='foreground')

    return f"{color_asni}{s}{CEND}"



example_board_sw=f"""
┌───────────────┬───┬────────────────┬───┬────────────────┬───┬────────────────┐
│    Population │ 11│     Sanitation │ 11│     Production │ 11│                │
│               └───┘                └───┘                └───┘                │
│                                      ▲                    ▲      Environment │
│                                      │                    │              ┌───┤
│ Politics                             │                    │              │ 11│
├───┐                                  │                    │              └───┤
│-11│                                  │                    │                  │
├───┘             ┌────────────────────┼────────────────────┤                  │
│                 │                    │                    │        Education │
│                 │                    │                    │              ┌───┤
│                 │                    │                    ├─────────────▶│ 11│
│                 │                    │                    │              └───┤
│                 │                    ▼                    ▼                  │
│               ┌───┐ Action         ┌───┐ Population     ┌───┐ Quality        │
│               │ 11│ Points         │ 11│ Growth         │ 11│ of Life        │
└───────────────┴───┴────────────────┴───┴────────────────┴───┴────────────────┘
"""

example_population_value = 11
example_sanitation_value = 11
example_production_value = 11
example_environment_value = 11
example_education_value = 11
example_quality_of_life_value = 11
example_growth_value = 11
example_action_points_value = 8
example_politics_value = 10
example_round = 17


def _number_padding(num: int, total_length: int=3) -> str:
    num_str = str(abs(num))
    padding_length = total_length - len(num_str) - (1 if num < 0 else 0)
    padded = ' ' * max(padding_length, 0) + num_str
    return ('-' if num < 0 else '') + padded

example_board_sw_dynamic_values=f"""
┌───────────────┬───┬────────────────┬───┬────────────────┬───┬────────────────┐
│    Population │{_number_padding(example_population_value)}│     Sanitation │{_number_padding(example_sanitation_value)}│     Production │{_number_padding(example_production_value)}│                │
│               └───┘                └───┘                └───┘                │
│                                      ▲                    ▲      Environment │
│                                      │                    │              ┌───┤
│ Politics                             │                    │              │{_number_padding(example_environment_value)}│
├───┐                                  │                    │              └───┤
│{_number_padding(example_politics_value)}│                                  │                    │                  │
├───┘             ┌────────────────────┼────────────────────┤                  │
│                 │                    │                    │        Education │
│                 │                    │                    │              ┌───┤
│                 │                    │                    ├─────────────▶│{_number_padding(example_education_value)}│
│                 │                    │                    │              └───┤
│                 │                    ▼                    ▼                  │
│               ┌───┐ Action         ┌───┐ Population     ┌───┐ Quality        │
│               │{_number_padding(example_action_points_value)}│ Points         │{_number_padding(example_population_value)}│ Growth         │{_number_padding(example_quality_of_life_value)}│ of Life        │
└───────────────┴───┴────────────────┴───┴────────────────┴───┴────────────────┘
"""
borad_countour = CGREY
example_board_dynamic_colored = f"""
{borad_countour}┌───────────────┬───┬────────────────┬───┬────────────────┬───┬────────────────┐{CEND}
{borad_countour}│    {CBLUE}Population{CEND} {borad_countour}│{wrap_with_color_scale(_number_padding(example_population_value), value=example_population_value, min_val=1, max_val=48, c_map="coolwarm")}{borad_countour}│     {CBLUE}Sanitation{CEND} {borad_countour}│{wrap_with_color_scale(_number_padding(example_sanitation_value), value=example_sanitation_value,min_val=1, max_val=29, c_map="RdYlGn")}{borad_countour}│     {CBLUE}Production{CEND} {borad_countour}│{wrap_with_color_scale(_number_padding(example_production_value), value=example_production_value, min_val=1, max_val=29, c_map="rainbow")}{borad_countour}│                │{CEND}
{borad_countour}│               └───┘                └───┘                └───┘                │{CEND}
{borad_countour}│                 {CVIOLET}                     ▲                    ▲{CEND}      {CCYAN}Environment{CEND} {borad_countour}│{CEND}
{borad_countour}│                 {CVIOLET}                     │                    │{CEND}              {borad_countour}┌───┤{CEND}
{borad_countour}│ {CCYAN}Politics{CEND}        {CVIOLET}                     │                    │{CEND}              {borad_countour}│{wrap_with_color_scale(_number_padding(example_environment_value), value=example_environment_value, min_val=1, max_val=29, c_map="RdYlGn")}{borad_countour}│{CEND}
{borad_countour}├───┐             {CVIOLET}                     │                    │{CEND}              {borad_countour}└───┤{CEND}
{borad_countour}│{wrap_with_color_scale(_number_padding(example_politics_value), value=example_politics_value, min_val=-10, max_val=37, c_map="RdYlGn")}{borad_countour}│             {CVIOLET}                     │                    │{CEND}                  {borad_countour}│{CEND}
{borad_countour}├───┘             {CVIOLET}┌────────────────────┼────────────────────┤{CEND}                  {borad_countour}│{CEND}
{borad_countour}│                 {CVIOLET}│                    │                    │{CEND}        {CBLUE}Education{CEND} {borad_countour}│{CEND}
{borad_countour}│                 {CVIOLET}│                    │                    │{CEND}              {borad_countour}┌───┤{CEND}
{borad_countour}│                 {CVIOLET}│                    │                    ├─────────────▶{CEND}{borad_countour}│{wrap_with_color_scale(_number_padding(example_education_value), value=example_education_value, min_val=1, max_val=29, c_map="RdYlGn")}{borad_countour}│{CEND}
{borad_countour}│                 {CVIOLET}│                    │                    │{CEND}              {borad_countour}└───┤{CEND}
{borad_countour}│                 {CVIOLET}│                    ▼                    ▼{CEND}                  {borad_countour}│{CEND}
{borad_countour}│               ┌───┐ {CVIOLET}Action{CEND}         {borad_countour}┌───┐ {CYELLOW}Population{CEND}     {borad_countour}┌───┐ {CBLUE}Quality{CEND}        {borad_countour}│{CEND}
{borad_countour}│               │{CVIOLET}{_number_padding(example_action_points_value)}{CEND}{borad_countour}│ {CVIOLET}Points{CEND}         {borad_countour}│{wrap_with_color_scale(_number_padding(example_population_value), value=example_population_value, min_val=1, max_val=29, c_map="rainbow")}{borad_countour}│ {CYELLOW}Growth{CEND}         {borad_countour}│{wrap_with_color_scale(_number_padding(example_quality_of_life_value), value=example_quality_of_life_value, min_val=1, max_val=29, c_map="RdYlGn")}{borad_countour}│ {CBLUE}of Life{CEND}        {borad_countour}│{CEND}
{borad_countour}├───────────────┴───┴────────────────┴───┴────────────────┴───┴────────────────┤{CEND}
{borad_countour}│ {CCYAN}Round:{CCYAN2}{_number_padding(example_round)}{CEND}                                                                    {borad_countour}│{CEND}
{borad_countour}└──────────────────────────────────────────────────────────────────────────────┘{CEND}
"""



def render_asni(env: 'OekoEnv') -> str:
    borad_countour = CGREY

    population_value = env.V[env.POPULATION]
    sanitation_value = env.V[env.SANITATION]
    production_value = env.V[env.PRODUCTION]
    environment_value = env.V[env.ENVIRONMENT]
    education_value = env.V[env.EDUCATION]
    quality_of_life_value = env.V[env.QUALITY_OF_LIFE]
    growth_value = env.V[env.POPULATION_GROWTH]
    action_points = env.V[env.POINTS]
    politics_value = env.V[env.POLITICS]
    round = env.V[env.ROUND]

    def _calc_additional_population_points(education_level):
        if education_level in range(21, 24):
            return 3
        elif education_level in range(24, 28):
            return 4
        elif education_level in range(28, 30):
            return 5
        else:
            return 0
    
    board = f"""
{borad_countour}┌───────────────┬───┬────────────────┬───┬────────────────┬───┬────────────────┐{CEND}
{borad_countour}│    {CBLUE}Population{CEND} {borad_countour}│{wrap_with_color_scale(_number_padding(population_value), value=population_value, min_val=1, max_val=48, c_map="coolwarm")}{borad_countour}│     {CBLUE}Sanitation{CEND} {borad_countour}│{wrap_with_color_scale(_number_padding(sanitation_value), value=sanitation_value,min_val=1, max_val=29, c_map="RdYlGn")}{borad_countour}│     {CBLUE}Production{CEND} {borad_countour}│{wrap_with_color_scale(_number_padding(production_value), value=production_value, min_val=1, max_val=29, c_map="rainbow")}{borad_countour}│                │{CEND}
{borad_countour}│               └───┘                └───┘                └───┘                │{CEND}
{borad_countour}│                 {CVIOLET}                     ▲                    ▲{CEND}      {CCYAN}Environment{CEND} {borad_countour}│{CEND}
{borad_countour}│                 {CVIOLET}                     │                    │{CEND}              {borad_countour}┌───┤{CEND}
{borad_countour}│ {CCYAN}Politics{CEND}        {CVIOLET}                     │                    │{CEND}              {borad_countour}│{wrap_with_color_scale(_number_padding(environment_value), value=environment_value, min_val=1, max_val=29, c_map="RdYlGn")}{borad_countour}│{CEND}
{borad_countour}├───┐             {CVIOLET}                     │                    │{CEND}              {borad_countour}└───┤{CEND}
{borad_countour}│{wrap_with_color_scale(_number_padding(politics_value), value=politics_value, min_val=-10, max_val=37, c_map="RdYlGn")}{borad_countour}│             {CVIOLET}                     │                    │{CEND}                  {borad_countour}│{CEND}
{borad_countour}├───┘             {CVIOLET}┌────────────────────┼────────────────────┤{CEND}                  {borad_countour}│{CEND}
{borad_countour}│                 {CVIOLET}│                    │                    │{CEND}        {CBLUE}Education{CEND} {borad_countour}│{CEND}
{borad_countour}│                 {CVIOLET}│                    │                    │{CEND}              {borad_countour}┌───┤{CEND}
{borad_countour}│                 {CVIOLET}│                    │                    ├─────────────▶{CEND}{borad_countour}│{wrap_with_color_scale(_number_padding(education_value), value=education_value, min_val=1, max_val=29, c_map="RdYlGn")}{borad_countour}│{CEND}
{borad_countour}│                 {CVIOLET}│                    │                    │{CEND}              {borad_countour}└───┤{CEND}
{borad_countour}│                 {CVIOLET}│                    ▼                    ▼{CEND}                  {borad_countour}│{CEND}
{borad_countour}│               ┌───┐ {CVIOLET}Action{CEND}         {borad_countour}┌───┐ {CYELLOW}Population{CEND}     {borad_countour}┌───┐ {CBLUE}Quality{CEND}        {borad_countour}│{CEND}
{borad_countour}│               │{CVIOLET}{_number_padding(action_points)}{CEND}{borad_countour}│ {CVIOLET}Points{CEND}         {borad_countour}│{wrap_with_color_scale(_number_padding(growth_value), value=growth_value, min_val=1, max_val=29, c_map="rainbow")}{borad_countour}│ {CYELLOW}Growth{CEND}         {borad_countour}│{wrap_with_color_scale(_number_padding(quality_of_life_value), value=quality_of_life_value, min_val=1, max_val=29, c_map="RdYlGn")}{borad_countour}│ {CBLUE}of Life{CEND}        {borad_countour}│{CEND}
{borad_countour}├───────────────┴───┴────────────────┴───┴────────────────┴───┴────────────────┤{CEND}
{borad_countour}│ {CCYAN}Round:{CCYAN2}{_number_padding(round)}{CEND}                           {CYELLOW}Population Extra Points{CEND}: {CYELLOW2}{_number_padding(_calc_additional_population_points(education_value),2)}{CEND}              {borad_countour}│{CEND}
{borad_countour}└──────────────────────────────────────────────────────────────────────────────┘{CEND}
"""
    return board

round_transition_example_sw = f"""
┌────────────────────────┬────────────────────────────┬────────────────────────┐
│ Sanitation:   11       │   Production:         11   │  Education:       11   │
├────────────────────────┼────────────────────────────┼────────────────────────┤
│ Used Points:  11/11    │   Population Growth:  11   │  Quality of Life: 11   │
└────────────────────────┴────────────────────────────┴────────────────────────┘
"""


sanitation = 11
production = -11
education = 11
used_points = 11
available_points = 11
population_growth = 11
quality_of_life = 11

round_transition_example_color = f"""
{borad_countour}┌────────────────────────┬────────────────────────────┬────────────────────────┐{CEND}
{borad_countour}│ {CBLUE}Sanitation{CEND}:   {CGREEN}{_number_padding(sanitation,2)}{CEND}       {borad_countour}│{CEND}   {CBLUE}Production{CEND}:        {CGREEN}{_number_padding(production,3)}{CEND}   {borad_countour}│{CEND}  {CBLUE}Education{CEND}:       {CGREEN}{_number_padding(education,2)}{CEND}   {borad_countour}│{CEND}
{borad_countour}├────────────────────────┼────────────────────────────┼────────────────────────┤{CEND}
{borad_countour}│ {CVIOLET}Used Points{CEND}:  {CVIOLET2}{_number_padding(used_points,2)}{CEND}/{CVIOLET}{_number_padding(available_points,2)}{CEND}    {borad_countour}│{CEND}   {CYELLOW}Population Growth{CEND}: {CGREEN}{_number_padding(population_growth,3)}{CEND}   {borad_countour}│{CEND}  {CBLUE}Quality of Life{CEND}: {CGREEN}{_number_padding(quality_of_life,2)}{CEND}   {borad_countour}│{CEND}
{borad_countour}└────────────────────────┴────────────────────────────┴────────────────────────┘{CEND}
"""

def render_round_transition_asni(
        sanitation: int,
        production: int,
        education: int,
        available_points: int,
        used_points: int,
        population_growth: int,
        quality_of_life: int) -> str:
    round_transition = f"""
{borad_countour}┌────────────────────────┬────────────────────────────┬────────────────────────┐{CEND}
{borad_countour}│ {CBLUE}Sanitation{CEND}:   {CGREEN}{_number_padding(sanitation,2)}{CEND}       {borad_countour}│{CEND}   {CBLUE}Production{CEND}:        {CGREEN}{_number_padding(production,3)}{CEND}   {borad_countour}│{CEND}  {CBLUE}Education{CEND}:       {CGREEN}{_number_padding(education,2)}{CEND}   {borad_countour}│{CEND}
{borad_countour}├────────────────────────┼────────────────────────────┼────────────────────────┤{CEND}
{borad_countour}│ {CVIOLET}Used Points{CEND}:  {CVIOLET2}{_number_padding(used_points,2)}{CEND}/{CVIOLET}{_number_padding(available_points,2)}{CEND}    {borad_countour}│{CEND}   {CYELLOW}Population Growth{CEND}: {CGREEN}{_number_padding(population_growth,3)}{CEND}   {borad_countour}│{CEND}  {CBLUE}Quality of Life{CEND}: {CGREEN}{_number_padding(quality_of_life,2)}{CEND}   {borad_countour}│{CEND}
{borad_countour}└────────────────────────┴────────────────────────────┴────────────────────────┘{CEND}"""
    return round_transition


if __name__ == '__main__':
    print(example_board_dynamic_colored)
    print(round_transition_example_color)