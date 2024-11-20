import os
import yaml
import logging
import importlib


def write_params_to_file(params: dict, out_dir: str):
    """Save experiment parameters to a text file

    Args:
        params (dict): Loaded parameter dictionary
        out_dir (string): Path to the experiment log file
    """
    filename = os.path.join(out_dir, "parameters.txt")
    with open(filename, "w") as file:
        for arg, value in params.items():
            file.write(f"{arg}: {value}\n")


def check_parameters(params: dict):
    """Checks the parameter objects

    Args:
        params (dict): Load yaml parameter file
    """

    file_keys = list(params.keys())
    base_message = "check_parameters (AssertionError)]: "
    param_groups = ["model", "experiment", "dataset"]

    if type(params) is not dict:
        raise AssertionError(base_message + "params should be a dictionary")

    for mdx, p_group in enumerate(param_groups):
        if mdx > 2:
            if p_group not in file_keys:
                params[p_group] = None
            continue
        if p_group not in file_keys:
            raise AssertionError(
                base_message + 'Parameter file should contain keyword "' + p_group + '"!')

        required_keywords = ['name', 'parameters']
        for i in required_keywords:
            if i not in params[p_group].keys():
                raise AssertionError(
                    base_message
                    + f'{p_group} in parameter file should contain keyword "'
                    + p_group
                    + f'": {i}!'
                )
    return params


def load_parameters(path: str):
    """Load a yaml parameter file

    Args:
        path (str): Path to the yaml file.

    Returns:
        dict: The loaded parameter dictionary
    """
    try:
        stream_file = open(path, "r")
        parameters = yaml.load(stream_file, Loader=yaml.FullLoader)
        check_parameters(parameters)
        logging.info("Success: Loaded parameter file at: {}".format(path))
    except AssertionError as e:
        logging.error(e)
        exit()
    return parameters


def load_fnc(module_name: str, func_name):
    try:
        module_ = importlib.import_module(module_name)
        if not hasattr(module_, func_name):
            raise AttributeError(
                f"Func '{func_name}' not found in module '{module_name}'")

        fnc = getattr(module_, func_name)
        return fnc
    except Exception as e:
        raise Exception(f"An error occurred: {e}")


def instanciate_cls(module_name: str, class_name: str, params: dict = None):
    """Instantiate a class

    Args:
        module_name (str): The name of the module holding the class
        class_name (str): The name of the class to instanciate
        params (dict): The parameters dictionary to feed to the class

    Raises:
        AttributeError: Wrong parameters
        ModuleNotFoundError: Module does not exist
    """
    try:
        module_ = importlib.import_module(module_name)
        if not hasattr(module_, class_name):
            raise AttributeError(
                f"Class '{class_name}' not found in module '{module_name}'")

        class_ = getattr(module_, class_name)
        if params:
            mod = class_(**params)
        else:
            mod = class_()
        return mod
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Module '{module_name}' not found")
    except AttributeError as e:
        raise e
    except Exception as e:
        raise Exception(f"An error occurred: {e}")
