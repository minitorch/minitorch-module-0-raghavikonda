from __future__ import annotations  # Importing future annotations for type hints

from typing import Any, Dict, Optional, Sequence, Tuple  # Importing necessary typing modules


class Module:
    """
    Modules form a tree that store parameters and other
    submodules. They make up the basis of neural network stacks
    Attributes:
        _modules : Storage of the child modules
        _parameters : Storage of the module's parameters
        training : Whether the module is in training mode or evaluation mode
    """

    _modules: Dict[str, Module]  # Dictionary to store child modules
    _parameters: Dict[str, Parameter]  # Dictionary to store parameters
    training: bool  # Boolean flag indicating whether the module is in training mode or evaluation mode

    def __init__(self) -> None:
        # Initialize dictionaries to store modules and parameters, and set training mode to True
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self) -> Sequence[Module]:
        "Return the direct child modules of this module."
        m: Dict[str, Module] = self.__dict__["_modules"]  # Get the dictionary of child modules
        return list(m.values())  # Return a list of child modules

    def train(self) -> None:
        "Set the mode of this module and all descendant modules to `train`."
        # Set the training mode to True for self and all descendant modules
        for module in [self] + self.modules():
            module.training = True

    def eval(self) -> None:
        "Set the mode of this module and all descendant modules to `eval`."
        # Set the training mode to False for self and all descendant modules
        for module in [self] + self.modules():
            module.training = False

    def named_parameters(self) -> Sequence[Tuple[str, Parameter]]:
        """
        Collect all the parameters of this module and its descendants.

        Returns:
            The name and `Parameter` of each parameter.
        """
        # Collect parameters from this module and its descendants
        params = []
        for parameter, value in self._parameters.items():
            params.append((parameter, value))

        for mod_name, module in self._modules.items():
            for parameter, value in module._parameters.items():
                params.append((f'{mod_name}.'+parameter, value))

        parameters: Sequence[Tuple[str, Parameter]] = params
        return parameters

    def parameters(self) -> Sequence[Parameter]:
        "Enumerate over all the parameters of this module and its descendants."
        # Enumerate over parameters from this module and its descendants
        params = []
        for module in [self] + self.modules():
            for value in module._parameters.values():
                params.append(value)

        parameters: Sequence[Parameter] = params
        return parameters
        
    def add_parameter(self, k: str, v: Any) -> Parameter:
        """
        Manually add a parameter. Useful helper for scalar parameters.

        Args:
            k: Local name of the parameter.
            v: Value for the parameter.

        Returns:
            Newly created parameter.
        """
        val = Parameter(v, k)  # Create a new Parameter object with provided value and name
        self.__dict__["_parameters"][k] = val  # Add the parameter to the dictionary of parameters
        return val

    def __setattr__(self, key: str, val: Parameter) -> None:
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val  # If the value is a Parameter, add it to the dictionary of parameters
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val  # If the value is a Module, add it to the dictionary of modules
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Any:
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]  # If the key is found in the dictionary of parameters, return the corresponding value

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]  # If the key is found in the dictionary of modules, return the corresponding value
        
        return None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)  # Call the forward method of the module

    def __repr__(self) -> str:
        def _addindent(s_: str, numSpaces: int) -> str:
            s2 = s_.split("\n")
            if len(s2) == 1:
                return s_
            first = s2.pop(0)
            s2 = [(numSpaces * " ") + line for line in s2]
            s = "\n".join(s2)
            s = first + "\n" + s
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class Parameter:
    """
    A Parameter is a special container stored in a :class:`Module`.

    It is designed to hold a :class:`Variable`, but we allow it to hold
    any value for testing.
    """

    def __init__(self, x: Any, name: Optional[str] = None) -> None:
        self.value = x
        self.name = name
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def update(self, x: Any) -> None:
        "Update the parameter value."
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def __repr__(self) -> str:
        return repr(self.value)

    def __str__(self) -> str:
        return str(self.value)

