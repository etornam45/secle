import { Tensor } from "./tensor.ts";
import { WebGPU } from "./backends/webgpu.ts";

export interface Optimizer {
  zero_grad(): void;
  step(): void;
}

export class SGD implements Optimizer {
  private readonly parameters: Tensor[];
  private readonly learningRate: number;
  private readonly momentum: number;
  // private readonly maxGradNorm: number;
  // Per-parameter velocity buffers when momentum > 0
  private readonly velocities: Map<Tensor, Float32Array | GPUBuffer> = new Map();

  constructor(parameters: Tensor[], learningRate: number, momentum: number = 0, _maxGradNorm: number = 1.0) {
    this.parameters = parameters;
    this.learningRate = learningRate;
    this.momentum = momentum;
    // this.maxGradNorm = maxGradNorm;
  }

  zero_grad(): void {
    for (const parameter of this.parameters) {
      parameter.zero_grad()
    }
  }

  step(): void {
    for (const parameter of this.parameters) {
      if (!parameter._grad) continue;
      
      if (parameter.device === "webgpu") {
        // Use efficient WebGPU SGD - no data transfers!
        if (this.momentum > 0) {
          // Handle momentum case - need to create velocity buffer if not exists
          let velocityBuffer = this.velocities.get(parameter);
          if (!velocityBuffer || !(velocityBuffer instanceof GPUBuffer)) {
            const device = WebGPU.getDevice();
            velocityBuffer = device.createBuffer({
              size: parameter.data.byteLength,
              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
              mappedAtCreation: true,
            });
            // Initialize velocity to zeros
            new Float32Array(velocityBuffer.getMappedRange()).fill(0);
            velocityBuffer.unmap();
            this.velocities.set(parameter, velocityBuffer);
          }
          WebGPU.sgdMomentumStep(parameter, parameter._grad, velocityBuffer, this.learningRate, this.momentum);
        } else {
          // Plain SGD on GPU
          WebGPU.sgdStep(parameter, parameter._grad, this.learningRate);
        }
      } else {
        // CPU path - use original logic
        const gradData = parameter._grad.data;
        const paramData = parameter.data;
        
        if (this.momentum > 0) {
          let velocity = this.velocities.get(parameter);
          if (!velocity || !(velocity instanceof Float32Array)) {
            velocity = new Float32Array(paramData.length);
            this.velocities.set(parameter, velocity);
          }

          // v = momentum * v + grad
          for (let i = 0; i < paramData.length; i++) {
            velocity[i] = this.momentum * velocity[i] + gradData[i];
            paramData[i] -= this.learningRate * velocity[i];
          }
        } else {
          // Plain SGD: param -= lr * grad
          for (let i = 0; i < paramData.length; i++) {
            paramData[i] -= this.learningRate * gradData[i];
          }
        }
      }
    }
  }
}
