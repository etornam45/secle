import { Tensor } from "./tensor.ts";

export const CPU = {
  name: "cpu",

  matmul: (a: Tensor, b: Tensor): Tensor => {
    if (a.shape.length !== 2 || b.shape.length !== 2) {
      throw new Error("Both tensors must be 2-dimensional for matmul." + " Got shapes " + a.shape + " and " + b.shape);
    }
    if (a.shape[1] !== b.shape[0]) {
      throw new Error(`Incompatible tensor shapes for matmul. Got shapes ${a.shape} and ${b.shape}.`);
    }

    const result = Tensor.zeros([a.shape[0], b.shape[1]]);
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

    const result = Tensor.zeros([a.shape[0], b.shape[1]]);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = a.data[i] + b.data[i];
    }
    return result;
  },

  sub: (a: Tensor, b: Tensor): Tensor => {
    if (a.size() !== b.size()) {
      throw new Error(`Tensors must be the same size for subtraction. Got sizes ${a.size()} and ${b.size()}.`);
    }

    const result = Tensor.zeros([a.shape[0], b.shape[1]]);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = a.data[i] - b.data[i];
    }
    return result;
  },

  mul: (a: Tensor, b: Tensor): Tensor => {
    if (a.size() !== b.size()) {
      throw new Error(`Tensors must be the same size for multiplication. Got sizes ${a.size()} and ${b.size()}.`);
    }

    const result = Tensor.zeros([a.shape[0], b.shape[1]]);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = a.data[i] * b.data[i];
    }
    return result;
  },

  div: (a: Tensor, b: Tensor): Tensor => {
    if (a.size() !== b.size()) {
      throw new Error(`Tensors must be the same size for division. Got sizes ${a.size()} and ${b.size()}.`);
    }

    const result = Tensor.zeros([a.shape[0], b.shape[1]]);
    for (let i = 0; i < a.size(); i++) {
      result.data[i] = a.data[i] / b.data[i];
    }
    return result;
  },
}