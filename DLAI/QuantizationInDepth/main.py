import torch
from helper import plot_quantization_errors
from dataclasses import dataclass, field

@dataclass(frozen=True)
class QuantizationConfig:
    scale: float = field(default=None)
    zero_point: int = field(default=None)
    dtype: torch.dtype = field(default=torch.int8)

class Quantization:
    def __init__(self, tensor: torch.Tensor, config: QuantizationConfig):
        """
        Initialize the Quantization object.

        :param tensor: The original tensor to be quantized.
        :param config: The quantization configuration containing scale, zero_point, and dtype.
        """
        self.tensor = tensor
        self.config = config
    
    def linear_q_with_scale_zero_point(self) -> torch.Tensor:
        """
        Quantize the tensor using the provided scale and zero point.

        :return: The quantized tensor.
        """
        if self.config.scale is None or self.config.zero_point is None:
            raise ValueError("Scale and zero point must be set before quantizing the tensor.")
        
        scaled_and_shifted_tensor = self.tensor / self.config.scale + self.config.zero_point
        rounded_tensor = torch.round(scaled_and_shifted_tensor)
    
        q_min = torch.iinfo(self.config.dtype).min
        q_max = torch.iinfo(self.config.dtype).max
    
        q_tensor = rounded_tensor.clamp(q_min, q_max).to(self.config.dtype)
    
        return q_tensor
    
    def get_q_scale_and_zero_point(self) -> None:
        """
        Calculate and set the scale and zero point for quantization.
        """
        q_min, q_max = torch.iinfo(self.config.dtype).min, torch.iinfo(self.config.dtype).max
        r_min, r_max = self.tensor.min().item(), self.tensor.max().item()
    
        scale = (r_max - r_min) / (q_max - q_min)
        zero_point = q_min - (r_min / scale)
        
        # Clip the zero_point to fall within [quantized_min, quantized_max]
        if zero_point < q_min:
            zero_point = q_min
        elif zero_point > q_max:
            zero_point = q_max
        else:
            zero_point = int(round(zero_point))
        
        # Update config with calculated scale and zero_point
        object.__setattr__(self.config, 'scale', scale)
        object.__setattr__(self.config, 'zero_point', zero_point)
        
    def get_q_scale_symmetric(self, tensor: torch.Tensor) -> float:
        r_max = tensor.abs().max().item()
        q_max = torch.iinfo(self.config.dtype).max
    
    def linear_q_symmetric(tensor, dtype=torch.int8):
    scale = get_q_scale_symmetric(tensor)
    
    quantized_tensor = linear_q_with_scale_and_zero_point(tensor,
                                                     scale=scale,
                   # in symmetric quantization zero point is = 0    
                                                    zero_point=0,
                                                      dtype=dtype)
    
    return quantized_tensor, scale
    
    def linear_quantization(self) -> torch.Tensor:
        """
        Perform linear quantization on the tensor.

        :return: The quantized tensor.
        """
        self.get_q_scale_and_zero_point()
        quantized_tensor = self.linear_q_with_scale_zero_point()
        return quantized_tensor

    def linear_dequantization(self, quantized_tensor: torch.Tensor) -> torch.Tensor:
        """
        Dequantize the quantized tensor.

        :param quantized_tensor: The quantized tensor to be dequantized.
        :return: The dequantized tensor.
        """
        return self.config.scale * (quantized_tensor.float() - self.config.zero_point)

# Example tensor to test the implementation
test_tensor = torch.tensor(
    [[191.6, -13.5, 728.6],
     [92.14, 295.5, -184],
     [0, 684.6, 245.5]]
)

# Initialize Quantization object with config
config = QuantizationConfig()
quantization = Quantization(test_tensor, config)

# Perform quantization and dequantization
quantized_tensor = quantization.linear_quantization()
dequantized_tensor = quantization.linear_dequantization(quantized_tensor)

# Plot the results
plot_quantization_errors(test_tensor, quantized_tensor, dequantized_tensor)

# Calculate and print dequantization error
dequantized_error = (dequantized_tensor - test_tensor).square().mean().item()
print(f"Dequantization error: {dequantized_error:.6f}")
