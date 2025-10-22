import { mean, pow, sub } from "./ops.ts";
import { Tensor } from "./tensor.ts";

export interface LossFunction {
  (output: Tensor, y: Tensor): Tensor;
}

export function MSELoss(): LossFunction {
  function criterion(output: Tensor, y: Tensor): Tensor {
    return mean(pow(sub(output, y), 2))
  }
  return criterion
}