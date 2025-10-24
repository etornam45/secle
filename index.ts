/**
 *  Torch-TS - A simple ml framework built for `typescript`
 *  Author: Benjamin Etornam (https://github.com/etornam45)
 *  License: MIT
 */

export { WebGPU } from "./core/backends/webgpu.ts";
export { CPU } from "./core/backends/cpu.ts";
export * from "./core/tensor.ts"
export * from "./core/modules/index.ts"
export * from "./core/ops.ts";
export * from "./plot.ts"
export { MSELoss } from "./core/loss.ts"
export { SGD, type Optimizer } from "./core/optim.ts";