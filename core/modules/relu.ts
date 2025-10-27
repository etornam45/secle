import { Module } from "./module.ts";
import { relu } from "../ops.ts";
import { Tensor } from "../tensor.ts";


export class Relu extends Module {
  constructor() {
    super(`Relu()`)
  }

  override forward(_input: Tensor): Tensor {
    return relu(_input)
  }
}