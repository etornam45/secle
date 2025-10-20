import { Module } from "../../index.ts";
import { Tensor } from "../tensor.ts";

export class Sequencial extends Module {
  modules: Module[] = [];
  constructor(modules: Module[]) {
    super()
    this.modules = modules
    this.register_parameters(this.modules)
  }

  override forward(_input: Tensor): Tensor {
    let res: Tensor = _input
    for (let i = 0; i < this.modules.length; i++) {
      res = this.modules[i].$(res)
    }
    return res
  }
}