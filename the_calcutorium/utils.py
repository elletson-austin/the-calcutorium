from typing import Callable, Dict, Tuple

from .scene import MathFunction


def safe_regenerate(math_function: MathFunction, new_equation: str) -> Tuple[bool, str]:
    old_equation = getattr(math_function, "equation_str", None)
    try:
        math_function.regenerate(new_equation)
        math_function.name = new_equation
        return True, ""
    except Exception as e:
        # try to revert
        try:
            if old_equation is not None:
                math_function.regenerate(old_equation)
        except Exception as revert_err:
            return False, f"Regenerate failed: {e}; revert failed: {revert_err}"
        return False, str(e)


def sync_function_editors(scene, 
                          function_editors: Dict[MathFunction, object], 
                          layout, 
                          editor_factory: Callable[[MathFunction], object]):
    
    scene_funcs = [obj for obj in scene.objects if isinstance(obj, MathFunction)]

    # remove editors for removed functions
    for func in list(function_editors):
        if func not in scene_funcs:
            w = function_editors.pop(func)
            w.setParent(None)
            w.deleteLater()

    # create editors for new functions
    for func in scene_funcs:
        if func not in function_editors:
            function_editors[func] = editor_factory(func)

    # clear layout
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget:
            widget.setParent(None)

    # add editors newest-first
    for func in reversed(scene_funcs):
        layout.insertWidget(0, function_editors[func])

    layout.addStretch(1)
    return function_editors
