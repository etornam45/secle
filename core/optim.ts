import { Tensor } from "./tensor.ts";

export interface Optimizer {
  zero_grad(): void;
  step(): void;
}

export class SGD implements Optimizer {
  private readonly parameters: Tensor[];
  private readonly learningRate: number;
  private readonly momentum: number;
  private readonly maxGradNorm: number;
  // Per-parameter velocity buffers when momentum > 0
  private readonly velocities: Map<Tensor, Float32Array> = new Map();

  constructor(parameters: Tensor[], learningRate: number, momentum: number = 0, maxGradNorm: number = 1.0) {
    this.parameters = parameters;
    this.learningRate = learningRate;
    this.momentum = momentum;
    this.maxGradNorm = maxGradNorm;
  }

  zero_grad(): void {
    for (const parameter of this.parameters) {
      parameter.zero_grad()
    }
  }

  step(): void {
    // Calculate total gradient norm for clipping
    let totalNorm = 0;
    for (const parameter of this.parameters) {
      if (!parameter._grad) continue;
      const grad = parameter._grad.data;
      for (let i = 0; i < grad.length; i++) {
        totalNorm += grad[i] * grad[i];
      }
    }
    totalNorm = Math.sqrt(totalNorm);

    // Apply gradient clipping
    const clipCoeff = this.maxGradNorm / (totalNorm + 1e-8);
    const shouldClip = totalNorm > this.maxGradNorm;

    for (const parameter of this.parameters) {
      if (!parameter._grad) continue;
      const grad = parameter._grad.data;
      const paramData = parameter.data;
      
      if (shouldClip) {
        // Apply clipping
        for (let i = 0; i < grad.length; i++) {
          grad[i] *= clipCoeff;
        }
      }
      
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
          paramData[i] -= this.learningRate * grad[i];
        }
      }
    }
  }
}
