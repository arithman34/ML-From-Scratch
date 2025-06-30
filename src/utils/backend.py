from typing import Type, Union


EPSILON = 1e-15  # Provide numerical stability


def initialize_component(name: str, component: Union[str, Type], mapping: dict, expected_type: Type) -> Type:
    if isinstance(component, str):
        try:
            return mapping[component.lower()]()
        except KeyError:
            raise ValueError(f"Unknown {name}: '{component}'. Available options are: {list(mapping.keys())}.")
        
    if isinstance(component, expected_type):
        return component
    
    raise TypeError(f"{name.capitalize()} must be a string or an instance of {expected_type.__name__}.")
