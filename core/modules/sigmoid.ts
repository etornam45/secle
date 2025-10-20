import { Module } from "./module.ts";
import { sigmoid } from "../ops.ts";
import { Tensor } from "../tensor.ts";


export class Sigmoid extends Module {
  constructor() {
    super()
    // this.register_parameters()
  }

  override forward(_input: Tensor): Tensor {
    return sigmoid(_input)
  }
}