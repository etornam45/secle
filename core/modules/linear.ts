import { Tensor } from "../tensor.ts";
import { Module } from "../modules/module.ts";
import { matmul, add } from "../ops.ts";


/**
 * Linear layer. This is a module that implements the linear layer.
 * :math:`y = xW + b`
 * 
 * :math:`W` is the weights matrix, :math:`b` is the bias vector.
 * :math:`x` is the input tensor, :math:`y` is the output tensor.
 */
export class Linear extends Module {
  weights: Tensor;
  bias: Tensor;

  constructor(input_size: number, output_size: number) {
    super(`Linear(in_features=${input_size}, out_features=${output_size})`);
    this.weights = Tensor.randn([input_size, output_size], true);
    this.bias = Tensor.randn([1, output_size], true);

    this.register_parameters();
  }

  override forward(x: Tensor): Tensor {
    return add(matmul(x, this.weights), this.bias);
  }
}

/**
 * Linear layer. This is a convenience function for the Linear class.
 * It is equivalent to the Linear class but is more convenient to use.
 * :math:`y = xW + b`
 * 
 * :math:`W` is the weights matrix, :math:`b` is the bias vector.
 * :math:`x` is the input tensor, :math:`y` is the output tensor.
 */
export function linear(x: Tensor, weights: Tensor, bias: Tensor): Tensor {
  return add(matmul(x, weights), bias);
}