import { Tensor } from "./tensor.ts";

export interface Optimizer {
  zero_grad(): void;
  step(): void;
}

export class SGD implements Optimizer {
  private readonly parameters: Tensor[];
  private readonly learningRate: number;
  private readonly momentum: number;
  // Per-parameter velocity buffers when momentum > 0
  private readonly velocities: Map<Tensor, Float32Array> = new Map();

  constructor(parameters: Tensor[], learningRate: number, momentum: number = 0) {
    this.parameters = parameters;
    this.learningRate = learningRate;
    this.momentum = momentum;
  }

  zero_grad(): void {
    for (const parameter of this.parameters) {
      if (parameter._grad) {
        parameter._grad.data.fill(0);
      }
    }
  }

  step(): void {
    for (const parameter of this.parameters) {
      if (!parameter._grad) continue;
      const grad = parameter._grad.data;
      const paramData = parameter.data;
      // console.log(`parameter: ${parameter.toString()}`);
      if (this.momentum > 0) {
        let velocity = this.velocities.get(parameter);
        if (!velocity) {
          velocity = new Float32Array(paramData.length);
          this.velocities.set(parameter, velocity);
        }

        // v = momentum * v + grad
        for (let i = 0; i < paramData.length; i++) {
          velocity[i] = this.momentum * velocity[i] + grad[i];
          paramData[i] -= this.learningRate * velocity[i];
        }
      } else {
        // Plain SGD: param -= lr * grad
        for (let i = 0; i < paramData.length; i++) {
          let t = `${paramData[i]} -= ${this.learningRate} * ${grad[i]}`;
          paramData[i] -= this.learningRate * grad[i];
          t += ` = ${paramData[i]}`;
          // console.log(t);
        }
      }
    }
  }
}
