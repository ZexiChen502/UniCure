import torch.nn as nn
from lora_pytorch import LoRA

class UCE_MHAttenLoRA(LoRA):
    def __init__(self, module: nn.Module, lora_module: nn.Module = None, enabled: bool = True, target_layer_indices=None):
        """
        Args:
            module: The module to be wrapped.
            lora_module: The optional LoRA module (usually None).
            enabled: Whether LoRA is enabled during training.
            target_layer_indices: A list of indices for transformer layers where LoRA will be applied (e.g., [31, 32]).
        """
        super().__init__(module, lora_module, enabled)
        layer_indices = list(range(33))
        self.target_layer_indices = target_layer_indices or layer_indices

    @classmethod
    def from_module(cls, module: nn.Module, rank: int, enabled: bool = True, is_root: bool = True, current_path=""):
        """
        Overrides the from_module method to apply LoRA only to specific transformer layers.
        """
        full_path = current_path

        # Check if the current module is a MultiheadAttention
        if isinstance(module, nn.MultiheadAttention):
            # Extract the transformer layer index if applicable
            layer_index = cls.extract_layer_index(full_path)
            # Only apply LoRA if the layer index is in the target list
            if layer_index is not None and layer_index in cls.target_layer_indices:
                return cls._from_multihead_attention(module, rank)

        # Recursively process all child modules
        for name, child in module.named_children():
            child_path = f"{full_path}.{name}" if full_path else name
            new_child = cls.from_module(child, rank, enabled=enabled, is_root=False, current_path=child_path)
            module._modules[name] = new_child

        # At the root level, return a LoRA wrapper
        if is_root:
            return cls(module, None, enabled=enabled, target_layer_indices=cls.target_layer_indices)
        else:
            return module

    @staticmethod
    def extract_layer_index(path: str):
        """
        Extracts a layer index from a layer path (e.g., 'transformer_encoder.layers.31.self_attn' -> 31).
        """
        if "transformer_encoder.layers." in path:
            # Split the path string to isolate the layer index
            parts = path.split(".")
            try:
                index = int(parts[2])  # The third section in the path corresponds to the layer index
                return index
            except (IndexError, ValueError):
                pass
        return None


