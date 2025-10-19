/**
 *  Torch-TS - A simple ml framework built for `typescript`
 *  Author: Benjamin Etornam (https://github.com/etornam45)
 *  License: MIT
 */

export { Tensor } from "./core/tensor.ts"
export { Module } from "./core/module.ts";
export { Linear } from "./core/linear.ts";
export * from "./core/ops.ts";
export { SGD, type Optimizer } from "./core/optim.ts";