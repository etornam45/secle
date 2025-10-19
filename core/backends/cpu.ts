import { Tensor } from "../tensor.ts";

export const CPU = {
  name: "cpu",

  matmul: (a: Tensor, b: Tensor): Tensor => {
    if (a.shape.length !== 2 || b.shape.length !== 2) {
      throw new Error("Both tensors must be 2-dimensional for matmul." + " Got shapes " + a.shape + " and " + b.shape);
    }
    if (a.shape[1] !== b.shape[0]) {
      throw new Error(`Incompatible tensor shapes for matmul. Got shapes ${a.shape} and ${b.shape}.`);
    }

    const result = Tensor.zeros([a.shape[0], b.shape[1]], a.requires_grad || b.requires_grad);
    for (let i = 0; i < a.shape[0]; i++) {
      for (let j = 0; j < b.shape[1]; j++) {
        for (let k = 0; k < a.shape[1]; k++) {
          result.data[i * b.shape[1] + j] += a.data[i * a.shape[1] + k] * b.data[k * b.shape[1] + j];
        }
      }
    }
    return result;
  },

  add: (a: Tensor, b: Tensor): Tensor => {
    if (a.size() !== b.size()) {
      throw new Error(`Tensors must be the same size for addition. Got sizes ${a.size()} and ${b.size()}.`);
    }

    const result = Tensor.zeros(a.shape, a.requires_grad || b.requires_grad);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = a.data[i] + b.data[i];
    }
    return result;
  },

  add_a_number: (a: Tensor, b: number): Tensor => {
    const result = Tensor.zeros(a.shape, a.requires_grad);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = a.data[i] + b;
    }
    return result;
  },

  sub: (a: Tensor, b: Tensor): Tensor => {
    if (a.size() !== b.size()) {
      throw new Error(`Tensors must be the same size for subtraction. Got sizes ${a.size()} and ${b.size()}.`);
    }

    const result = Tensor.zeros(a.shape, a.requires_grad || b.requires_grad);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = a.data[i] - b.data[i];
    }
    return result;
  },

  mul: (a: Tensor, b: Tensor): Tensor => {
    if (a.size() !== b.size()) {
      throw new Error(`Tensors must be the same size for multiplication. Got sizes ${a.size()} and ${b.size()}.`);
    }

    const result = Tensor.zeros(a.shape, a.requires_grad || b.requires_grad);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = a.data[i] * b.data[i];
    }
    return result;
  },

  /**
   * Multiply a tensor by a scalar.
   * @param a {Tensor} - The tensor to multiply by a scalar.
   * @param b {number} - The scalar to multiply the tensor by.
   * @returns {Tensor} - The tensor multiplied by the scalar.
   */
  mul_scalar: (a: Tensor, b: number): Tensor => {
    const result = Tensor.zeros(a.shape, a.requires_grad);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = a.data[i] * b;
    }
    return result;
  },

  div: (a: Tensor, b: Tensor): Tensor => {
    if (a.size() !== b.size()) {
      throw new Error(`Tensors must be the same size for division. Got sizes ${a.size()} and ${b.size()}.`);
    }

    const result = Tensor.zeros(a.shape, a.requires_grad || b.requires_grad);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = a.data[i] / b.data[i];
    }
    return result;
  },

  neg: (a: Tensor): Tensor => {
    const result = Tensor.zeros(a.shape, a.requires_grad);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = -a.data[i];
    }
    return result;
  },



  sum: (a: Tensor): Tensor => {
    let result = 0;
    for (let i = 0; i < a.size(); i++) {
      result += a.data[i];
    }
    return new Tensor([result], [1, 1], a.requires_grad);
  },

  mean: (a: Tensor): Tensor => {
    const result = CPU.sum(a);
    return new Tensor([result.data[0] / a.size()], [1, 1], a.requires_grad);
  },

  /**
   * Raise a tensor to a power. This is a CPU implementation of the power operation.
   * @param a {Tensor} - The tensor to raise to a power.
   * @param b {number} - The power to raise the tensor to.
   * @returns {Tensor} - The tensor raised to the power.
   */
  pow: (a: Tensor, b: number): Tensor => {
    const result = Tensor.zeros(a.shape, a.requires_grad);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = Math.pow(a.data[i], b);
    }
    return result;
  },

  relu: (a: Tensor): Tensor => {
    const result = Tensor.zeros(a.shape, a.requires_grad);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = Math.max(0, a.data[i]);
    }
    return result;
  },

  sigmoid: (a: Tensor): Tensor => {
    const result = Tensor.zeros(a.shape, a.requires_grad);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = 1 / (1 + Math.exp(-a.data[i]));
    }
    return result;
  },
}