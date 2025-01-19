import torch
import torch.nn as nn
import captum.attr as attr


class Activation(nn.Module):
    def __init__(self, model, layer, device_ids=None):
        super().__init__()
        # convert ModuleList to list
        if isinstance(layer, nn.ModuleList):
            layer = list(layer)
        self.explainer = attr.LayerActivation(model, layer, device_ids=device_ids)

    def forward(self, x, attribute_to_layer_input=True):
        return self.explainer.attribute(x, attribute_to_layer_input=attribute_to_layer_input)


class GradientXActivation(nn.Module):
    def __init__(self, model, layer, device_ids=None, multiply_by_inputs=True):
        super().__init__()
        # convert ModuleList to list
        if isinstance(layer, nn.ModuleList):
            layer = list(layer)
        self.explainer = attr.LayerGradientXActivation(model, layer, device_ids=device_ids, multiply_by_inputs=multiply_by_inputs)

    def forward(self, x, target=None, attribute_to_layer_input=True):
        return self.explainer.attribute(x, target=target, attribute_to_layer_input=attribute_to_layer_input)


class InternalInfluence(nn.Module):
    def __init__(self, model, layer, device_ids=None):
        super().__init__()
        # if layer is a ModuleList, initialize an explainer list
        if isinstance(layer, nn.ModuleList):
            self.explainer = [
                attr.InternalInfluence(model, single_layer, device_ids=device_ids)
                for single_layer in layer
            ]
        else:
            self.explainer = attr.InternalInfluence(model, layer, device_ids=device_ids)

    def forward(self, x, baselines=None, target=None, n_steps=50, method='gausslegendre', internal_batch_size=None, attribute_to_layer_input=True):
        if baselines is None:
            if isinstance(x, tuple):
                baselines = []
                for elem in x:
                    baselines.append(torch.zeros_like(elem))
                baselines = tuple(baselines)
            else:
                baselines = torch.zeros_like(x)

        if isinstance(self.explainer, list):
            explanation = [
                single_explainer.attribute(x, baselines=baselines, target=target, n_steps=n_steps, method=method, internal_batch_size=internal_batch_size, attribute_to_layer_input=attribute_to_layer_input)
                for single_explainer in self.explainer
            ]
        else:
            explanation = self.explainer.attribute(x, baselines=baselines, target=target, n_steps=n_steps, method=method, internal_batch_size=internal_batch_size, attribute_to_layer_input=attribute_to_layer_input)

        return explanation


class DeepLift(nn.Module):
    def __init__(self, model, layer, multiply_by_inputs=True):
        super().__init__()
        # if layer is a ModuleList, initialize an explainer list
        if isinstance(layer, nn.ModuleList):
            self.explainer = [
                attr.LayerDeepLift(model, single_layer, multiply_by_inputs=multiply_by_inputs)
                for single_layer in layer
            ]
        else:
            self.explainer = attr.LayerDeepLift(model, layer, multiply_by_inputs=multiply_by_inputs)

    def forward(self, x, baselines=None, target=None, return_convergence_delta=False, attribute_to_layer_input=True):
        if baselines is None:
            if isinstance(x, tuple):
                baselines = []
                for elem in x:
                    baselines.append(torch.zeros_like(elem))
                baselines = tuple(baselines)
            else:
                baselines = torch.zeros_like(x)

        if isinstance(self.explainer, list):
            explanation = [
                single_explainer.attribute(x, baselines=baselines, target=target, return_convergence_delta=return_convergence_delta, attribute_to_layer_input=attribute_to_layer_input)
                for single_explainer in self.explainer
            ]
        else:
            explanation = self.explainer.attribute(x, baselines=baselines, target=target, return_convergence_delta=return_convergence_delta, attribute_to_layer_input=attribute_to_layer_input)

        return explanation


class DeepLiftShap(nn.Module):
    def __init__(self, model, layer, multiply_by_inputs=True):
        super().__init__()
        if isinstance(layer, nn.ModuleList):
            self.explainer = [
                attr.LayerDeepLiftShap(model, single_layer, multiply_by_inputs=multiply_by_inputs)
                for single_layer in layer
            ]
        else:
            self.explainer = attr.LayerDeepLiftShap(model, layer, multiply_by_inputs=multiply_by_inputs)

    def forward(self, x, baselines=None, target=None, return_convergence_delta=False, attribute_to_layer_input=True):
        if baselines is None:
            if isinstance(x, tuple):
                baselines = []
                for elem in x:
                    baselines.append(torch.normal(0, 1, size=(50, ) + elem.shape[1:]).to(elem.device))
                baselines = tuple(baselines)
            else:
                baselines = torch.normal(0, 1, size=(50,) + x.shape[1:]).to(x.device)

        if isinstance(self.explainer, list):
            explanation = []
            for single_explainer in self.explainer:
                single_explanation = single_explainer.attribute(x, baselines=baselines, target=target, return_convergence_delta=return_convergence_delta, attribute_to_layer_input=attribute_to_layer_input)
                explanation.append((single_explanation[0].detach(), single_explanation[1].detach()))
        else:
            explanation = self.explainer.attribute(x, baselines=baselines, target=target, return_convergence_delta=return_convergence_delta, attribute_to_layer_input=attribute_to_layer_input)

        return explanation


class IntegratedGradients(nn.Module):
    def __init__(self, model, layer, device_ids=None, multiply_by_inputs=True):
        super().__init__()
        if isinstance(layer, nn.ModuleList):
            self.explainer = [
                attr.LayerIntegratedGradients(model, single_layer, device_ids=device_ids, multiply_by_inputs=multiply_by_inputs)
                for single_layer in layer
            ]
        else:
            self.explainer = attr.LayerIntegratedGradients(model, layer, device_ids=device_ids, multiply_by_inputs=multiply_by_inputs)

    def forward(self, x, baselines=None, target=None, n_steps=50, method='gausslegendre', internal_batch_size=None, return_convergence_delta=False, attribute_to_layer_input=True):
        if baselines is None:
            if isinstance(x, tuple):
                baselines = []
                for elem in x:
                    baselines.append(torch.zeros_like(elem))
                baselines = tuple(baselines)
            else:
                baselines = torch.zeros_like(x)

        if isinstance(self.explainer, list):
            explanation = []
            for single_explainer in self.explainer:
                single_explanation = single_explainer.attribute(x, baselines=baselines, target=target, n_steps=n_steps, method=method, internal_batch_size=internal_batch_size, return_convergence_delta=return_convergence_delta, attribute_to_layer_input=attribute_to_layer_input)
                explanation.append((single_explanation[0].detach(), single_explanation[1].detach()))
        else:
            explanation = self.explainer.attribute(x, baselines=baselines, target=target, n_steps=n_steps, method=method, internal_batch_size=internal_batch_size, return_convergence_delta=return_convergence_delta, attribute_to_layer_input=attribute_to_layer_input)

        return explanation


class GradCam(nn.Module):
    def __init__(self, model, layer, device_ids=None):
        super().__init__()
        if isinstance(layer, nn.ModuleList):
            self.explainer = [
                attr.LayerGradCam(model, single_layer, device_ids=device_ids)
                for single_layer in layer
            ]
        else:
            self.explainer = attr.LayerGradCam(model, layer, device_ids=device_ids)

    def forward(self, x, target=None, attribute_to_layer_input=True, attr_dim_summation=False):
        if isinstance(self.explainer, list):
            explanation = []
            for single_explainer in self.explainer:
                single_explanation = single_explainer.attribute(x, target=target, attribute_to_layer_input=attribute_to_layer_input, attr_dim_summation=attr_dim_summation)
                explanation.append((single_explanation[0].detach(), single_explanation[1].detach()))
        else:
            explanation = self.explainer.attribute(x, target=target, attribute_to_layer_input=attribute_to_layer_input, attr_dim_summation=attr_dim_summation)

        return explanation
