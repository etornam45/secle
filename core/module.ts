import { Tensor } from "./tensor.ts";

export class Module {
  _name?: string;
  parameters: Tensor[] = [];

  constructor(_name?: string) {
    this._name = _name;
    // add all derived class properties with the type Tensor to the parameters array
    for (const key in this) {
      if (this[key] instanceof Tensor && this[key].requires_grad) {
        this.parameters.push(this[key]);
      } else if (this[key] instanceof Module) {
        // Recursively register parameters from nested modules
        this.register_parameters(this[key].parameters);
      }
    }
  }

  register_parameters(_parameters?: Tensor[] | Module[]): void {
    if (!_parameters) {
      for (const key in this) {
        if (this[key] instanceof Tensor && this[key].requires_grad) {
          this.parameters.push(this[key]);
        } else if (this[key] instanceof Module) {
          // Recursively register parameters from nested modules
          this.register_parameters(this[key].parameters);
        }
      }
      return;
    };
    for (const parameter of _parameters) {
      if (parameter instanceof Tensor) {
        if (parameter.requires_grad) {
          this.parameters.push(parameter);
        } else {
          throw new Error("Parameter does not require gradients.");
        }
      } else if (parameter instanceof Module) {
        this.register_parameters(parameter.parameters);
      } else {
        throw new Error("Parameter is not a Tensor or Module.");
      }
    }
  }

  forward(_input: Tensor): Tensor {
    throw new Error("Method not implemented.");
  }

  backward(_grad: Tensor) {
    throw new Error("Method not implemented.");
  }

  /**
   * This ia a wrapper for the forward method.
   * @param _input - The input to the module.
   * @returns The output of the module.
   */
  $(_input: Tensor): Tensor {
    return this.forward(_input);
  }

  summary(_indent: number = 0): string {
    let summary = "";
    for (const key in this) {
      if (this[key] instanceof Module) {
        summary += `${" ".repeat(_indent * 2)}${this[key]._name ?? "Module"}: ${this[key].summary(_indent + 1)}\n`;
      } else if (this[key] instanceof Tensor) {
        summary += `${" ".repeat(_indent * 2)}${this[key]._name ?? "Parameter"}: ${this[key].shape.join("x")}\n`;
      }
    }
    return summary.trim();
  }
}