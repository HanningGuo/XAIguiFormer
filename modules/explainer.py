import torch.nn as nn
from modules import explainer_wrapper


class Explainer(nn.Module):
    """
    A unified explainer to call different XAI algorithm interprets
    single/multi layer of the model

    :param model: Callable, the forward function of to be explained model
    :param layer: torch.nn.Module or torch.nn.ModuleList, Layer or layers for which attributions are computed
    :param explainer_type: str (optional), specify which XAI algorithm is used to interpret model
            (default: DeepLift)
    :param device_ids: list[int] (optional), Device ID list,
            necessary only if forward_func applies a DataParallel model
            (default: None)
    :param multiply_by_inputs: bool (optional), Indicates whether to factor model inputs’ multiplier
            in the final attribution scores, also known as local vs global attribution.
            if multiply_by_inputs is set to True,
            final sensitivity scores are being multiplied by layer activations for inputs.
            (default: True)
    """

    explainer_types = ['Activation', 'GradientXActivation', 'InternalInfluence', 'DeepLift', 'IntegratedGradients', 'DeepLiftShap', 'GradCam']

    def __init__(self, model, layer, explainer_type='DeepLift', device_ids=None, multiply_by_inputs=True):
        super().__init__()
        self.explainer_type = explainer_type
        assert self.explainer_type in self.explainer_types, '{} cannot be supported. Please choose the explainer type: {}'.format(self.explainer_type, self.explainer_types)

        if self.explainer_type in ['Activation', 'InternalInfluence', 'GradCam']:
            self.explainer = getattr(explainer_wrapper, self.explainer_type)(model, layer, device_ids)
        elif self.explainer_type in ['GradientXActivation', 'IntegratedGradients']:
            self.explainer = getattr(explainer_wrapper, self.explainer_type)(model, layer, device_ids, multiply_by_inputs)
        elif self.explainer_type in ['DeepLift', 'DeepLiftShap']:
            self.explainer = getattr(explainer_wrapper, self.explainer_type)(model, layer, multiply_by_inputs)

    def forward(self, x, target, baselines=None, attribute_to_layer_input=True):
        if self.explainer_type == 'Activation':
            explanation = self.explainer(x, attribute_to_layer_input)
        elif self.explainer_type in ['GradientXActivation', 'GradCam']:
            explanation = self.explainer(x, target=target, attribute_to_layer_input=attribute_to_layer_input)
        elif self.explainer_type in ['InternalInfluence', 'DeepLift', 'IntegratedGradients', 'DeepLiftShap']:
            explanation = self.explainer(x, baselines=baselines, target=target, attribute_to_layer_input=attribute_to_layer_input)

        return explanation
